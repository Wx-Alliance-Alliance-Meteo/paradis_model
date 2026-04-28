import os
import shutil

import dask
import dask.array as da
import numpy
import xarray
from lightning.pytorch.callbacks import BasePredictionWriter
from numcodecs import BitRound, Blosc

from utils.mhuaes import mhuaes3


# Conservative, relatively fast compressor
compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)


def _enc_f32(bits: int):
    """Float32 + BitRound + Blosc."""
    return {
        "compressor": compressor,
        "filters": [BitRound(keepbits=bits)],
        "dtype": "f4",
    }


def _enc_uint8():
    return {"compressor": compressor, "dtype": "uint8"}


def _get_sorted_time_info(dataset):
    """
    Returns:
        sorted_times: dataset times in ascending order
        sample_index_to_sorted_pos: array mapping original dataset index -> sorted position
    """

    raw_times = numpy.asarray(dataset.time)
    sorted_order = numpy.argsort(raw_times)
    sorted_times = raw_times[sorted_order]

    sample_index_to_sorted_pos = numpy.empty_like(sorted_order)
    sample_index_to_sorted_pos[sorted_order] = numpy.arange(len(sorted_order))

    return sorted_times, sample_index_to_sorted_pos


def _replace_variable_names(cfg):
    """Apply the same post-processing variable renames used in forecasting."""
    atmospheric_vars = list(cfg.features.output.atmospheric)
    surface_vars = list(cfg.features.output.surface)

    def replace(items, old, new):
        return [new if x == old else x for x in items]

    atmospheric_vars = replace(atmospheric_vars, "wind_x", "u_component_of_wind")
    atmospheric_vars = replace(atmospheric_vars, "wind_y", "v_component_of_wind")
    atmospheric_vars = replace(atmospheric_vars, "wind_z", "vertical_velocity")

    surface_vars = replace(surface_vars, "wind_x_10m", "10m_u_component_of_wind")
    surface_vars = replace(surface_vars, "wind_y_10m", "10m_v_component_of_wind")

    return atmospheric_vars, surface_vars


def _build_dataset_for_samples(
    forecast,
    init_times,
    cfg,
    dataset,
):
    """
    Build an xarray.Dataset for a batch of samples.

    Args:
        forecast: numpy array [B, T_forecast, F, Lat, Lon]
        init_times: numpy array [B] of datetimes, already sorted in ascending time order
    """
    atmospheric_vars, surface_vars = _replace_variable_names(cfg)
    pressure_levels = list(cfg.features.pressure_levels)

    data_vars = {}
    num_levels = len(pressure_levels)

    input_data = dataset.ds_loader.sel(time=init_times).sortby("time")["data"].values

    # Atmospheric variables
    atm_dims = ["time", "prediction_timedelta", "level", "latitude", "longitude"]
    for i, feature in enumerate(atmospheric_vars):
        beg_ind = i * num_levels
        end_ind = (i + 1) * num_levels

        data_vars[feature] = (
            atm_dims,
            numpy.concatenate(
                (
                    input_data[..., beg_ind:end_ind].transpose(0, 3, 1, 2)[:, None],
                    forecast[:, :, beg_ind:end_ind],
                ),
                axis=1,
            ),
        )

    # Surface variables
    sur_dims = ["time", "prediction_timedelta", "latitude", "longitude"]
    for i, feature in enumerate(surface_vars):
        if feature == "wind_z_10m":
            continue

        data_vars[feature] = (
            sur_dims,
            numpy.concatenate(
                (
                    input_data[..., len(atmospheric_vars) * num_levels + i][:, None],
                    forecast[:, :, len(atmospheric_vars) * num_levels + i],
                ),
                axis=1,
            ),
        )

    ds = xarray.Dataset(data_vars=data_vars)

    # Cast time-varying vars to float32
    for v in list(ds.data_vars):
        if "time" in ds[v].dims:
            ds[v] = ds[v].astype("float32")
        elif ds[v].dtype.kind == "f":
            ds[v] = ds[v].astype("float32")

    # Derived variable
    hu = ds.specific_humidity
    tt = ds.temperature
    ps = xarray.DataArray(
        numpy.asarray(pressure_levels) * 100,
        dims=["level"],
    )

    ds = ds.assign(dewpoint_depression=mhuaes3(hu, tt, ps).astype("float32"))

    return ds


def _build_template_dataset(cfg, dataset, filename):
    """
    Build the full output template with all coordinates and variables preallocated.
    No actual forecast values are written here; this only creates the store layout.
    """
    atmospheric_vars, surface_vars = _replace_variable_names(cfg)
    pressure_levels = list(cfg.features.pressure_levels)

    sorted_times, _ = _get_sorted_time_info(dataset)

    num_levels = len(pressure_levels)

    # Number of stored forecast steps, matching step % output_frequency == 0
    num_forecast_steps = cfg.model.forecast_steps
    output_frequency = cfg.forecast.output_frequency
    output_num_forecast_steps = (num_forecast_steps - 1) // output_frequency + 1

    total_pred_steps = output_num_forecast_steps + 1  # include init state
    n_time = len(sorted_times)
    n_lat = dataset.lat_size
    n_lon = dataset.lon_size

    coords = {
        "latitude": dataset.lat,
        "longitude": dataset.lon,
        "time": sorted_times,
        "level": pressure_levels,
        "prediction_timedelta": numpy.arange(output_num_forecast_steps + 1)
        * numpy.timedelta64(6 * 3600 * 10**9, "ns"),
    }

    data_vars = {}

    # Atmospheric variables
    atm_dims = ["time", "prediction_timedelta", "level", "latitude", "longitude"]
    atm_shape = (n_time, total_pred_steps, num_levels, n_lat, n_lon)
    atm_chunks = (1, min(10, total_pred_steps), num_levels, n_lat, n_lon)

    for feature in atmospheric_vars:
        arr = da.empty(atm_shape, chunks=atm_chunks, dtype=numpy.float32)
        data_vars[feature] = (atm_dims, arr)

    # Surface variables
    sur_dims = ["time", "prediction_timedelta", "latitude", "longitude"]
    sur_shape = (n_time, total_pred_steps, n_lat, n_lon)
    sur_chunks = (1, min(10, total_pred_steps), n_lat, n_lon)

    for feature in surface_vars:
        if feature == "wind_z_10m":
            continue
        arr = da.empty(sur_shape, chunks=sur_chunks, dtype=numpy.float32)
        data_vars[feature] = (sur_dims, arr)

    # Derived field
    data_vars["dewpoint_depression"] = (
        atm_dims,
        da.empty(atm_shape, chunks=atm_chunks, dtype=numpy.float32),
    )

    # Constants only once
    con_dims = ["latitude", "longitude"]
    for feature in dataset.ds_constants.data_vars:
        if feature in con_dims:
            continue

        values = dataset.ds_constants[feature].data
        if getattr(values, "dtype", None) is not None and values.dtype.kind == "f":
            values = values.astype("float32")
        data_vars[feature] = (con_dims, values)

    ds = xarray.Dataset(data_vars=data_vars, coords=coords)

    encoding = {
        "time": {"dtype": "float64"},
    }

    for var in ds.data_vars:
        if "time" in ds[var].dims:
            var_shape = ds[var].shape
            encoding[var] = {
                "chunks": (1, min(10, var_shape[1]), *var_shape[2:]),
            }

            if ds[var].dtype.kind == "f":
                encoding[var].update(_enc_f32(16))
            elif ds[var].dtype == numpy.uint8:
                encoding[var].update(_enc_uint8())
            else:
                encoding[var].update({"compressor": compressor, "dtype": ds[var].dtype})
        else:
            if ds[var].dtype.kind == "f":
                encoding[var] = _enc_f32(16)
            elif ds[var].dtype == numpy.uint8:
                encoding[var] = _enc_uint8()
            else:
                encoding[var] = {
                    "compressor": compressor,
                    "dtype": ds[var].dtype,
                }

    return ds, encoding


def init_forecast_store(cfg, dataset, filename):
    """Create an empty Zarr store with the full sorted time axis and final schema."""
    if os.path.exists(filename):
        shutil.rmtree(filename)

    ds, encoding = _build_template_dataset(cfg, dataset, filename)

    with dask.config.set(scheduler="threads"):
        ds.to_zarr(
            filename,
            mode="w",
            consolidated=True,
            zarr_format=2,
            encoding=encoding,
        )


def write_forecast_region(forecast, sample_indices, cfg, dataset, filename):
    """
    Write one prediction batch into the preinitialized Zarr store.

    Args:
        forecast: numpy array [B, T_forecast, F, Lat, Lon]
        sample_indices: original dataset sample indices for this batch
    """
    sorted_times, sample_index_to_sorted_pos = _get_sorted_time_info(dataset)

    sample_indices = numpy.asarray(sample_indices)
    order = numpy.argsort(sample_indices)
    sample_indices = sample_indices[order]
    forecast = forecast[order]

    sorted_positions = sample_index_to_sorted_pos[sample_indices]
    init_times = sorted_times[sorted_positions]

    # Usually contiguous for a non-shuffled loader, but do not assume it.
    # Group contiguous sorted_positions so region writes stay efficient.
    breaks = numpy.where(numpy.diff(sorted_positions) != 1)[0] + 1
    groups = numpy.split(numpy.arange(len(sorted_positions)), breaks)

    with dask.config.set(scheduler="threads"):
        for g in groups:
            group_positions = sorted_positions[g]
            group_times = init_times[g]
            group_forecast = forecast[g]

            ds = _build_dataset_for_samples(
                forecast=group_forecast,
                init_times=group_times,
                cfg=cfg,
                dataset=dataset,
            )

            start = int(group_positions[0])
            stop = int(group_positions[-1]) + 1

            ds.to_zarr(
                filename,
                region={"time": slice(start, stop)},
                consolidated=False,
                zarr_format=2,
            )


def write_forecast_region_chunked(
    forecast, sample_indices, start_idx, cfg, dataset, filename
):
    sorted_times, sample_index_to_sorted_pos = _get_sorted_time_info(dataset)

    sample_indices = numpy.asarray(sample_indices)
    order = numpy.argsort(sample_indices)
    sample_indices = sample_indices[order]
    forecast = forecast[order]

    sorted_positions = sample_index_to_sorted_pos[sample_indices]
    init_times = sorted_times[sorted_positions]

    breaks = numpy.where(numpy.diff(sorted_positions) != 1)[0] + 1
    groups = numpy.split(numpy.arange(len(sorted_positions)), breaks)

    with dask.config.set(scheduler="threads"):
        for g in groups:
            group_positions = sorted_positions[g]
            group_times = init_times[g]
            group_forecast = forecast[g]

            ds = _build_dataset_for_samples(
                forecast=group_forecast,
                init_times=group_times,
                cfg=cfg,
                dataset=dataset,
            )

            # _build_dataset_for_samples always includes init state at td=0,
            # then chunk forecast at td=1..chunk_len
            if start_idx > 0:
                ds = ds.isel(prediction_timedelta=slice(1, None))
                pred_slice = slice(
                    1 + start_idx, 1 + start_idx + group_forecast.shape[1]
                )
            else:
                pred_slice = slice(0, 1 + group_forecast.shape[1])

            start = int(group_positions[0])
            stop = int(group_positions[-1]) + 1

            ds.to_zarr(
                filename,
                region={
                    "time": slice(start, stop),
                    "prediction_timedelta": pred_slice,
                },
                consolidated=False,
                zarr_format=2,
            )


class ZarrForecastWriter:
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.store_initialized = False

    def init_store(self, dataset):

        if not self.store_initialized:
            init_forecast_store(
                cfg=self.cfg,
                dataset=dataset,
                filename=self.cfg.forecast.output_file,
            )

        self.store_initialized = True

    def write_forecast_chunk(self, forecast, sample_indices, start_idx, dataset):
        """Writes a portion of the forecast batch"""
        write_forecast_region_chunked(
            forecast=forecast,
            sample_indices=sample_indices,
            start_idx=start_idx,
            cfg=self.cfg,
            dataset=dataset,
            filename=self.cfg.forecast.output_file,
        )
