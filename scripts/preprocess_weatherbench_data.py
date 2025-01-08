import argparse
import xarray as xr
import numpy as np
import dask
from dask.diagnostics import ProgressBar
import os
import time


def main():
    """
    Main function to process WeatherBench data by stacking data,
    precomputing static data, and computing statistics.
    """
    # Setup command line arguments
    parser = argparse.ArgumentParser(description="Preprocess WeatherBench data.")
    parser.add_argument(
        "-i",
        "--input_dir",
        required=True,
        help="Input directory containing WeatherBench data in Zarr format",
    )
    parser.add_argument(
        "-o", "--output_dir", required=True, help="Output directory for processed data"
    )
    args = parser.parse_args()

    # Open the dataset from the input Zarr directory
    ds = xr.open_zarr(args.input_dir)

    # Ensure the dataset dimensions are ordered as time, latitude, longitude, level
    ds = ds.transpose("time", "latitude", "longitude", "level")

    # Remove variables that don't have corresponding directories in the input data
    # These variables are likely placeholders or contain only NaN values
    drop_variables = set(ds.data_vars) - set(os.listdir(args.input_dir))
    ds = ds.drop_vars(drop_variables)

    # Step 1: Stack data for efficient storage and processing
    stack_data(ds, args.output_dir)

    # Step 2: Precompute static data (e.g., geographic variables)
    precompute_static_data(ds, args.output_dir)

    # Step 3: Compute mean and standard deviation for atmospheric and surface variables
    compute_statistics(args.output_dir)


def stack_data(ds, output_base_dir):
    """
    Processes and stacks data for each year, storing it in a Zarr format with a unit chunk size
    along the time dimension.

    Parameters:
        ds (xarray.Dataset): The input dataset to process.
        output_base_dir (str): Directory to store the processed yearly data.
    """

    # Determine the minimum and maximum years in the dataset
    min_year = np.min(ds["time.year"].values)
    max_year = np.max(ds["time.year"].values)

    # Keep only variables with a time dimension (e.g., atmospheric and surface variables)
    ds = ds.drop_vars([var for var in ds.data_vars if "time" not in ds[var].dims])

    # Progress bar for visualization during processing
    pbar = ProgressBar()
    pbar.register()

    # Variables to retain dimensions for stacking
    keep_dims = ["time", "latitude", "longitude"]

    # Process data year by year
    for year in range(min_year, max_year + 1):
        t0 = time.time()  # Track processing time for each year

        # Select data for the current year
        ds_year = ds.sel(time=ds["time.year"] == year)

        # Stack variables along a new "features" dimension
        ds_year = ds_year.to_stacked_array(new_dim="features", sample_dims=keep_dims)

        # Rename features to include pressure levels (if applicable)
        new_names = [
            val[0] + "_h" + str(int(val[1])) if str(val[1]) != "nan" else val[0]
            for val in ds_year.features.values
        ]

        # Ensure atmospheric variables precede surface variables in the stacked array
        num_atmospheric_variables = sum(
            1 if str(val[1]) != "nan" else 0 for val in ds_year.features.values
        )

        counter_atmospheric = 0
        counter_surface = num_atmospheric_variables
        ordered_indices = []
        for val in ds_year.features.values:
            if str(val[1]) != "nan":
                ordered_indices.append(counter_atmospheric)
                counter_atmospheric += 1
            else:
                ordered_indices.append(counter_surface)
                counter_surface += 1

        # Set up the output directory for the current year
        output_dir = os.path.join(output_base_dir, str(year))
        os.makedirs(output_dir, exist_ok=True)

        # Drop unnecessary variables and rename coordinates
        ds_year = ds_year.drop_vars(["features", "variable", "level"])
        ds_year = ds_year.assign_coords(features=new_names)

        # Add descriptive attributes to the dataset
        ds_year.attrs["description"] = "Stacked dataset per lat/lon grid point"
        ds_year.attrs["note"] = (
            "Variables have been renamed based on their original names and levels."
        )

        # Remove specific unwanted attributes
        attrs_to_remove = ["long_name", "short_name", "units"]
        for attr in attrs_to_remove:
            ds_year.attrs.pop(attr, None)

        # Define chunk sizes for optimized Zarr storage
        chunk_sizes = {
            "time": 1,
            "latitude": ds_year.latitude.size,
            "longitude": ds_year.longitude.size,
            "features": ds_year.features.size,
        }

        # Rechunk the data
        ds_year = ds_year.chunk(chunk_sizes)

        # Ensure the dataset has a name and wrap it in an xarray.Dataset if it's a DataArray
        ds_year.name = "data"
        if isinstance(ds_year, xr.DataArray):
            ds_year = xr.Dataset({"data": ds_year})

        # Write the processed dataset to a Zarr file
        output_file_path = os.path.join(output_dir)
        with dask.config.set(scheduler="threads"):
            ds_year.to_zarr(output_file_path, mode="w", consolidated=True)

        print(
            f"Successfully processed {year} -> {output_file_path} in {time.time() - t0:.2f} seconds"
        )


def precompute_static_data(ds, output_base_dir):
    pbar = ProgressBar()
    pbar.register()

    # Keep only the static data
    ds = ds.drop_vars([var for var in ds.data_vars if "time" in ds[var].dims])

    static_vars = ds.data_vars

    latitude, longitude = np.meshgrid(ds.latitude, ds.longitude, indexing="ij")

    # Convert variables
    latitude_rad = np.deg2rad(latitude)
    longitude_rad = np.deg2rad(longitude)

    coords = {"latitude": ds.latitude, "longitude": ds.longitude}
    dims = ["latitude", "longitude"]

    # Compute cosine/sine of latitude/longitude and store
    cos_latitude = xr.DataArray(np.cos(latitude_rad), dims=dims, coords=coords)
    cos_longitude = xr.DataArray(np.cos(longitude_rad), dims=dims, coords=coords)
    sin_longitude = xr.DataArray(np.sin(longitude_rad), dims=dims, coords=coords)

    data_vars = {
        "cos_latitude": cos_latitude,
        "cos_longitude": cos_longitude,
        "sin_longitude": sin_longitude,
    }

    # Add existing static variables if numeric
    for var in static_vars:
        has_nans = np.isnan(ds[var].values).any()

        if not has_nans:
            data_vars[var] = xr.DataArray(ds[var].values, dims=dims, coords=coords)

    # Convert to a dataset
    ds_result = xr.Dataset(data_vars=data_vars, coords=coords)

    # Store mean and standard deviation for these variables
    for var in ds_result.data_vars:
        mean = ds_result[var].mean().values
        std = ds_result[var].std().values
        ds_result[var] = ds_result[var].assign_attrs(mean=mean, std=std)

    with pbar, dask.config.set(scheduler="threads"):
        ds_result.to_zarr(
            os.path.join(output_base_dir, "constants"),
            mode="w",
            consolidated=True,
        )


def compute_statistics(output_base_dir):
    """Compute mean and standard deviation of data variables"""
    pbar = ProgressBar()
    pbar.register()

    years = [int(item) for item in os.listdir(output_base_dir) if item.isdigit()]

    min_year = np.min(years)
    max_year = np.max(years)

    # Create list of files to open
    files = [
        os.path.join(output_base_dir, f"{year}")
        for year in range(min_year, max_year + 1)
    ]

    # Open with a larger chunk as this will accumulate data
    ds = xr.open_mfdataset(files, chunks={"time": 200}, engine="zarr")

    # Compute time-mean and time-standard deviation (per-level)
    # This skips nan values, which may appear for certain quantities
    # in early datasets
    mean_ds = ds.mean(dim=["time", "latitude", "longitude"], skipna=True)
    std_ds = ds.std(dim=["time", "latitude", "longitude"], skipna=True)

    # Combine the mean and std into a single dataset
    result_ds = xr.Dataset({"mean": mean_ds["data"], "std": std_ds["data"]})

    with dask.config.set(scheduler="threads"):
        result_ds.to_zarr(
            os.path.join(output_base_dir, "stats"), mode="w", consolidated=True
        )


if __name__ == "__main__":
    main()
