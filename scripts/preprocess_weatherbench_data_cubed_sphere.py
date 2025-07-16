import argparse
import xarray
import numpy
import dask
from dask.diagnostics.progress import ProgressBar
import os
import time
import sys
import torch
import dask.config

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from data.forcings.toa_radiation import toa_radiation
from utils.cubed_sphere.cubed_sphere import CubedSphere
from utils.cubed_sphere.torch_interpolator import (
    TorchCubedSphereInterpolator,
)

# Constants
R_earth = 6371000.0  # Earth radius in meters
g = 9.80616  # gravitational acceleration m/s^2
R = 287.05  # Gas constant for dry air J/(kg·K)


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

    parser.add_argument(
        "-n", "--num_elem_cs", required=True, help="Number of elements in cubed sphere"
    )

    parser.add_argument(
        "--interpolation_mode",
        type=str,
        default='bilinear',
        choices=['bilinear', 'bicubic', 'nearest'],
        help="Interpolation mode for regridding."
    )

    args = parser.parse_args()

    # Open the dataset from the input Zarr directory
    ds = xarray.open_zarr(args.input_dir)
    
    ds = ds.transpose("time", "level", "latitude", "longitude")
    
    # Remove variables that don't have corresponding directories in the input data
    # These variables are likely placeholders or contain only NaN values

    keep_variables = [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "mean_sea_level_pressure",
        "surface_pressure",
        "temperature",
        "land_sea_mask",
        "time",
        "u_component_of_wind",
        "v_component_of_wind",
        "vertical_velocity",
        "level",
        "specific_humidity",
        "geopotential",
        "latitude",
        "longitude",
        "geopotential_at_surface",
        "total_precipitation_6hr",
        "total_column_water",
        "standard_deviation_of_orography",
        "slope_of_sub_gridscale_orography",
    ]

    # Determine variables to drop
    drop_variables = [var for var in ds.data_vars if var not in keep_variables]

    # Drop the unwanted variables
    ds = ds.drop_vars(drop_variables)

    # Cube-sphere grid
    num_elem_cs = int(args.num_elem_cs)
    cubed_sphere = CubedSphere(num_elem=num_elem_cs, radius = R_earth)
    lon = ds.longitude.values
    lat = ds.latitude.values
    cs_interpolator = TorchCubedSphereInterpolator(lat, lon, cubed_sphere, mode=args.interpolation_mode)

    # Step 1: Stack data for efficient storage and processing
    stack_data(ds, args.output_dir, cubed_sphere, cs_interpolator)

    # Step 2: Precompute static data (e.g., geographic variables)
    precompute_static_data(ds, args.output_dir, cubed_sphere, cs_interpolator)

    # Step 3: Compute mean and standard deviation for atmospheric and surface variables
    compute_statistics(args.output_dir)


def convert_to_cubed_sphere(ds, interpolator, cubed_sphere):
    for var in ds.data_vars:
        dims = ds[var].dims
        if dims[-2:] == ('latitude', 'longitude'):
            new_data = interpolator.interpolate(ds[var].values).numpy()
            new_dims = dims[:-2] + ("panel_id", "xi", "eta")
            ds[var] = xarray.DataArray(new_data, dims=new_dims)

    ds = ds.drop_vars(['latitude', 'longitude'])
    ds = ds.assign_coords({
        'panel_id': cubed_sphere.panel_id,
        'xi': cubed_sphere.xi,
        'eta': cubed_sphere.eta,
    })
    
    return ds
            
def convert_wind(ds, from_coord, to_coord, cubed_sphere, input_vars, output_vars):
    if len(input_vars) == 3:
        # Convert pressure velocity (Pa/s) to geometric velocity (m/s)
        # w = -ω * R * T / (p * g) where ω = dp/dt (Pa/s)
        vertical_wind = -ds[input_vars[-1]] * R * ds.temperature / (ds.level * 100 * g)
        physical_winds = numpy.stack((ds[input_vars[0]], ds[input_vars[1]], vertical_wind), axis=-1)
        vertical = True
    else:
        physical_winds = numpy.stack((ds[input_vars[0]], ds[input_vars[1]]), axis=-1)
        vertical = False
    
    J = cubed_sphere.get_jacobian(from_coord, to_coord, vertical)
    new_winds = (J @ physical_winds[..., None]).squeeze(-1)
    dims = ds[input_vars[0]].dims

    ds = ds.assign({v: (dims, new_winds[...,i]) for i, v in enumerate(output_vars)})

    return ds
    
    

def stack_data(ds, output_base_dir, cubed_sphere, cs_interpolator):
    """
    Processes and stacks data for each year, storing it in a Zarr format with a unit chunk size
    along the time dimension.

    Parameters:
        ds (xarray.Dataset): The input dataset to process.
        output_base_dir (str): Directory to store the processed yearly data.
    """
    # Determine the minimum and maximum years in the dataset
    min_year = 1979
    max_year = numpy.max(ds["time.year"].values)

    # Keep only variables with a time dimension (e.g., atmospheric and surface variables)
    ds = ds.drop_vars([var for var in ds.data_vars if "time" not in ds[var].dims])

    # Progress bar for visualization during processing
    pbar = ProgressBar()
    pbar.register()

    # Variables to retain dimensions for stacking
    keep_dims = ["time", "panel_id", "xi", "eta"]

    # Process data year by year
    for year in range(min_year, max_year + 1):
        print(f"Processing year {year}")
        t0 = time.time()  # Track processing time for each year

        # Select data for the current year
        ds_year = ds.sel(time=ds["time.year"] == year)
        
        # Convert variables to cubed-sphere and convert winds
        ds_year = convert_to_cubed_sphere(ds_year, cs_interpolator, cubed_sphere)

        ds_year = convert_wind(ds_year, 'physical', 'local', cubed_sphere, 
                                ('10m_u_component_of_wind', '10m_v_component_of_wind'),
                                ('wind_xi_10m', 'wind_eta_10m'))

        ds_year = convert_wind(ds_year, 'physical', 'local', cubed_sphere, 
                            ('u_component_of_wind', 'v_component_of_wind', 'vertical_velocity'),
                            ('wind_xi', 'wind_eta', 'wind_zeta'))

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
        ds_year.attrs["description"] = "Stacked dataset on cubed-sphere grid"
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
            "panel_id": ds_year.panel_id.size,
            "xi": ds_year.xi.size,
            "eta": ds_year.eta.size,
            "features": ds_year.features.size,
        }

        # Rechunk the data
        ds_year = ds_year.chunk(chunk_sizes)

        # Ensure the dataset has a name and wrap it in an xarray.Dataset if it's a DataArray
        ds_year.name = "data"
        if isinstance(ds_year, xarray.DataArray):
            ds_year = xarray.Dataset({"data": ds_year})

        # Write the processed dataset to a Zarr file
        output_file_path = os.path.join(output_dir)
        with dask.config.set(scheduler="threads"):
            ds_year.to_zarr(
                output_file_path, mode="w", consolidated=True, zarr_format=2
            )

        print(
            f"Successfully processed {year} -> {output_file_path} in {time.time() - t0:.2f} seconds"
        )


def precompute_static_data(ds, output_base_dir, cubed_sphere, cs_interpolator):
    pbar = ProgressBar()
    pbar.register()

    # Keep only the static data from the original lat-lon dataset
    ds_static = ds.drop_vars(
        [var for var in ds.data_vars if "time" in ds[var].dims or numpy.isnan(ds[var].values).any()]
    )
    static_dims = ("panel_id", "xi", "eta")
    
    ds_static = convert_to_cubed_sphere(ds_static, cs_interpolator, cubed_sphere)

    # Add lat/lon
    ds_static["latitude"] = xarray.DataArray(numpy.rad2deg(cubed_sphere.lat), dims=static_dims)
    ds_static["longitude"] = xarray.DataArray(numpy.rad2deg(cubed_sphere.lon), dims=static_dims)
    ds_static["cos_latitude"] = xarray.DataArray(cubed_sphere.cos_lat, dims=static_dims)
    ds_static["cos_longitude"] = xarray.DataArray(cubed_sphere.cos_lon, dims=static_dims)
    ds_static["sin_latitude"] = xarray.DataArray(cubed_sphere.sin_lat, dims=static_dims)
    ds_static["sin_longitude"] = xarray.DataArray(cubed_sphere.sin_lon, dims=static_dims)
    
    for var in ds_static.data_vars:
        mean = ds_static[var].mean().values
        std = ds_static[var].std().values
        ds_static[var] = ds_static[var].assign_attrs(mean=mean, std=std)

    with pbar, dask.config.set(scheduler="threads"):
        ds_static.to_zarr(
            os.path.join(output_base_dir, "constants"),
            mode="w",
            consolidated=True,
            zarr_format=2,
        )


def compute_statistics(output_base_dir):
    """Compute mean and standard deviation of data variables"""
    pbar = ProgressBar()
    pbar.register()

    years = [int(item) for item in os.listdir(output_base_dir) if item.isdigit()]

    min_year = 1979
    max_year = numpy.max(years)

    # Create list of files to open
    files = [
        os.path.join(output_base_dir, str(year))
        for year in range(min_year, max_year + 1)
    ]

    # Open with a larger chunk as this will accumulate data
    ds = xarray.open_mfdataset(
        files, chunks={"time": 1}, engine="zarr"
    )

    # Compute time-mean and time-standard deviation (per-level)
    mean_ds = ds.mean(dim=["time", "panel_id", "xi", "eta"], skipna=True)
    std_ds = ds.std(dim=["time", "panel_id", "xi", "eta"], skipna=True)
    max_ds = ds.max(dim=["time", "panel_id", "xi", "eta"], skipna=True)
    min_ds = ds.min(dim=["time", "panel_id", "xi", "eta"], skipna=True)

    # Load cubed sphere grid to get lat/lon for TOA radiation
    constants_ds = xarray.open_zarr(os.path.join(output_base_dir, "constants"))
    lat_values = constants_ds.latitude.values
    lon_values = constants_ds.longitude.values

    # Compute toa_solar radiation
    toa_rad = toa_radiation(ds.time.values, lat_values, lon_values)
    toa_rad_mean = numpy.mean(toa_rad)
    toa_rad_std = numpy.std(toa_rad)

    # Combine the mean and std into a single dataset
    result_ds = xarray.Dataset(
        {
            "mean": mean_ds["data"],
            "std": std_ds["data"],
            "max": max_ds["data"],
            "min": min_ds["data"],
        },
    )

    result_ds.attrs["toa_radiation_mean"] = toa_rad_mean
    result_ds.attrs["toa_radiation_std"] = toa_rad_std

    with dask.config.set(scheduler="threads"):
        result_ds.to_zarr(
            os.path.join(output_base_dir, "stats"),
            mode="w",
            consolidated=True,
            zarr_format=2,
        )


if __name__ == "__main__":
    main()
