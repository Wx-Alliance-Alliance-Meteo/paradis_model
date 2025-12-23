import os
import shutil
import dask
import numpy
import xarray

from utils.mhuaes import mhuaes3


def save_results_to_zarr(
    data,
    ds_input_data,
    atmospheric_vars,
    surface_vars,
    constant_vars,
    dataset,
    pressure_levels,
    filename,
    ind,
    init_times,
):
    """Save results to a Zarr file."""
    data_vars = {}
    num_levels = len(pressure_levels)

    input_data = ds_input_data.sel(time=init_times).sortby("time")["data"].values

    # Prepare atmospheric variables
    atm_dims = ["time", "prediction_timedelta", "level", "latitude", "longitude"]
    for i, feature in enumerate(atmospheric_vars):
        beg_ind = i * num_levels
        end_ind = (i + 1) * num_levels

        data_vars[feature] = (
            atm_dims,
            numpy.concatenate(
                (
                    input_data[..., beg_ind:end_ind].transpose(0, 3, 1, 2)[:, None],
                    data[:, :, beg_ind:end_ind],
                ),
                axis=1,
            ),
        )

    # Prepare surface variables
    sur_dims = ["time", "prediction_timedelta", "latitude", "longitude"]
    for i, feature in enumerate(surface_vars):
        if feature == "wind_z_10m":
            continue
        data_vars[feature] = (
            sur_dims,
            numpy.concatenate(
                (
                    input_data[..., len(atmospheric_vars) * num_levels + i][:, None],
                    data[:, :, len(atmospheric_vars) * num_levels + i],
                ),
                axis=1,
            ),
        )

    if ind == 0:
        # Prepare constant variables
        con_dims = ["latitude", "longitude"]
        for i, feature in enumerate(dataset.ds_constants.data_vars):
            if feature in con_dims:
                continue
            data_vars[feature] = (con_dims, dataset.ds_constants[feature].data)

    # Define coordinates
    coords = {
        "latitude": dataset.lat,
        "longitude": dataset.lon,
        "time": init_times,
        "level": pressure_levels,
        "prediction_timedelta": (numpy.arange(data.shape[1] + 1))
        * numpy.timedelta64(6 * 3600 * 10**9, "ns"),
    }

    # If this is the first write, remove any existing Zarr store
    if ind == 0 and os.path.exists(filename):
        shutil.rmtree(filename)

    # Create dataset
    ds = xarray.Dataset(data_vars=data_vars, coords=coords)

    # Add dewpoint depression to files
    hu = ds.specific_humidity
    tt = ds.temperature
    ps = ds.level * 100
    ds = ds.assign(dewpoint_depression=mhuaes3(hu, tt, ps))

    with dask.config.set(scheduler="threads"):

        # Save to Zarr
        if ind == 0:

            # Set encoding with chunking for efficient storage
            encoding = {
                "time": {"dtype": "float64"},
            }

            # Set reasonable chunk sizes for all variables
            for var in ds.data_vars:
                if "time" in ds[var].dims:
                    var_shape = ds[var].shape

                    # Set the chunks to time=1, full size for other dims
                    encoding[var] = {
                        "chunks": (
                            1,
                            *var_shape[1:],
                        ),
                    }

            ds.to_zarr(
                filename,
                consolidated=True,
                zarr_format=2,
                encoding=encoding,
            )
        else:

            ds.to_zarr(
                filename,
                consolidated=True,
                append_dim="time",
                zarr_format=2,
            )
