import os
import shutil
import dask
import numpy
import xarray


def save_results_to_zarr(
    data,
    atmospheric_vars,
    surface_vars,
    constant_vars,
    dataset,
    pressure_levels,
    filename,
    ind,
    time_ind_start,
    time_ind_end,
):
    """Save results to a Zarr file."""
    data_vars = {}
    num_levels = len(pressure_levels)

    # Prepare atmospheric variables
    atm_dims = ["time", "prediction_timedelta", "level", "latitude", "longitude"]
    for i, feature in enumerate(atmospheric_vars):
        data_vars[feature] = (
            atm_dims,
            data[:, :, i * num_levels : (i + 1) * num_levels],
        )

    # Prepare surface variables
    sur_dims = ["time", "prediction_timedelta", "latitude", "longitude"]
    for i, feature in enumerate(surface_vars):
        data_vars[feature] = (
            sur_dims,
            data[:, :, len(atmospheric_vars) * num_levels + i],
        )

    if ind == 0:
        # Prepare constant variables
        con_dims = ["latitude", "longitude"]
        for i, feature in enumerate(constant_vars):
            if feature in con_dims:
                continue
            data_vars[feature] = (con_dims, dataset.ds_constants[feature].data)

    # Define coordinates
    coords = {
        "latitude": dataset.lat,
        "longitude": dataset.lon,
        "time": dataset.time[time_ind_start:time_ind_end],
        "level": pressure_levels,
        "prediction_timedelta": (numpy.arange(data.shape[1]) + 1)
        * numpy.timedelta64(6 * 3600 * 10**9, "ns"),
    }

    # If this is the first write, remove any existing Zarr store
    if ind == 0 and os.path.exists(filename):
        shutil.rmtree(filename)

    with dask.config.set(scheduler="threads"):

        # Save to Zarr
        if ind == 0:
            xarray.Dataset(data_vars=data_vars, coords=coords).to_zarr(
                filename,
                consolidated=True,
            )
        else:
            xarray.Dataset(data_vars=data_vars, coords=coords).to_zarr(
                filename, consolidated=True, append_dim="time"
            )
