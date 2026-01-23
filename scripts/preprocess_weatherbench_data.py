import argparse
import xarray
import numpy
import dask
from dask.diagnostics import ProgressBar
import os
import time
import sys
from numcodecs import Blosc, BitRound

# Conservative, relatively fast compressor
compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.forcings.toa_radiation import toa_radiation


def compute_cartesian_wind(ds):
    """
    Compute 3D Cartesian wind components from spherical components.
    """
    g = 9.80616
    R = 287.05
    R_earth = 6371000.0

    # Convert coordinates to radians
    lat_rad = numpy.deg2rad(ds.latitude)
    lon_rad = numpy.deg2rad(ds.longitude)

    sin_lat = numpy.sin(lat_rad)
    cos_lat = numpy.cos(lat_rad)
    sin_lon = numpy.sin(lon_rad)
    cos_lon = numpy.cos(lon_rad)

    # w = -ω * R * T / (p * g)
    dr_dt = -ds.vertical_velocity * R * ds.temperature / (ds.level * 100 * g)

    dlon_dt = ds.u_component_of_wind / (R_earth * cos_lat)
    dlat_dt = ds.v_component_of_wind / R_earth

    wind_x = (
        dr_dt * cos_lat * cos_lon
        - R_earth * sin_lat * cos_lon * dlat_dt
        - R_earth * cos_lat * sin_lon * dlon_dt
    )
    wind_y = (
        dr_dt * cos_lat * sin_lon
        - R_earth * sin_lat * sin_lon * dlat_dt
        + R_earth * cos_lat * cos_lon * dlon_dt
    )
    wind_z = dr_dt * sin_lat + R_earth * cos_lat * dlat_dt

    # Surface: dr/dt = 0
    dlon_dt_10m = ds["10m_u_component_of_wind"] / (R_earth * cos_lat)
    dlat_dt_10m = ds["10m_v_component_of_wind"] / R_earth
    wind_x_10m = (
        -R_earth * sin_lat * cos_lon * dlat_dt_10m
        - R_earth * cos_lat * sin_lon * dlon_dt_10m
    )
    wind_y_10m = (
        -R_earth * sin_lat * sin_lon * dlat_dt_10m
        + R_earth * cos_lat * cos_lon * dlon_dt_10m
    )
    wind_z_10m = R_earth * cos_lat * dlat_dt_10m

    ds = ds.assign(
        wind_x=wind_x,
        wind_y=wind_y,
        wind_z=wind_z,
        wind_x_10m=wind_x_10m,
        wind_y_10m=wind_y_10m,
        wind_z_10m=wind_z_10m,
    )

    for var in ["wind_x", "wind_y", "wind_z"]:
        ds[var].attrs["long_name"] = f'{var.split("_")[1]}_component_of_wind_cartesian'
        ds[var].attrs["units"] = "m s-1"
    for var in ["wind_x_10m", "wind_y_10m", "wind_z_10m"]:
        ds[var].attrs[
            "long_name"
        ] = f'{var.split("_")[1]}_component_of_10m_wind_cartesian'
        ds[var].attrs["units"] = "m s-1"

    return ds


def main():
    parser = argparse.ArgumentParser(description="Preprocess WeatherBench data.")
    parser.add_argument("-i", "--input_dir", required=True, help="Input Zarr dir")
    parser.add_argument("-o", "--output_dir", required=True, help="Output dir")
    parser.add_argument(
        "--remove-poles",
        action="store_true",
        default=False,
        help="Removes latitudes -90,90",
    )
    parser.add_argument(
        "--interp_deg",
        type=float,
        default=0.0,
        help="Interpolates dataset to this degree resolution",
    )
    args = parser.parse_args()

    ds = xarray.open_zarr(args.input_dir)
    ds = ds.transpose("time", "latitude", "longitude", "level")

    # Variables that will be extracted from the dataset
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
        "wind_x",
        "wind_y",
        "wind_z",
        "wind_x_10m",
        "wind_y_10m",
        "wind_z_10m",
    ]
    drop_variables = [var for var in ds.data_vars if var not in keep_variables]
    ds = ds.drop_vars(drop_variables)

    if args.remove_poles and args.interp_deg == 0:
        lat_to_drop = [v for v in (-90, 90) if v in ds.latitude.values]
        if lat_to_drop:
            ds = ds.sel(latitude=~ds.latitude.isin(lat_to_drop))

    if args.interp_deg > 0:
        # Interpolate data. For this, dataset must contain the poles and 0 longitude
        # Then, the dataset is padded with longitude=360 to avoid
        # Interpolating outside of range (gives nan)
        latitude = numpy.arange(-90, 90, args.interp_deg) + args.interp_deg / 2
        longitude = numpy.arange(0, 360, args.interp_deg) + args.interp_deg / 2
        ds_360 = ds.sel(longitude=0).assign_coords(longitude=360)
        ds_padded = xarray.concat([ds, ds_360], dim="longitude")
        ds = ds_padded.interp(latitude=latitude, longitude=longitude)

    # Step 1: Stack data for efficient storage and processing
    stack_data(ds, args.output_dir)

    # Step 2: Precompute static data (e.g., geographic variables)
    precompute_static_data(ds, args.output_dir)

    # Step 3: Compute mean and standard deviation for atmospheric and surface variables
    compute_statistics(args.output_dir)


def stack_data(ds, output_base_dir):
    ds = compute_cartesian_wind(ds)

    # Cast variables to float32
    for v in list(ds.data_vars):
        if "time" in ds[v].dims:
            ds[v] = ds[v].astype("float32")

    min_year = 1979
    max_year = numpy.max(ds["time.year"].values)

    # Keep only time-varying vars
    ds = ds.drop_vars([var for var in ds.data_vars if "time" not in ds[var].dims])

    pbar = ProgressBar()
    pbar.register()
    keep_dims = ["time", "latitude", "longitude"]

    # Conservative BitRound for f32
    def enc_bits(bits):  # helper
        return {
            "compressor": compressor,
            "filters": [BitRound(keepbits=bits)],
            "dtype": "f4",
        }

    for year in range(min_year, max_year + 1):
        t0 = time.time()
        ds_year = ds.sel(time=ds["time.year"] == year)

        ds_year = ds_year.to_stacked_array(new_dim="features", sample_dims=keep_dims)

        new_names = [
            val[0] + "_h" + str(int(val[1])) if str(val[1]) != "nan" else val[0]
            for val in ds_year.features.values
        ]

        output_dir = os.path.join(output_base_dir, str(year))
        os.makedirs(output_dir, exist_ok=True)

        ds_year = ds_year.drop_vars(["features", "variable", "level"])
        ds_year = ds_year.assign_coords(features=new_names)

        ds_year.attrs["description"] = "Stacked dataset per lat/lon grid point"
        ds_year.attrs["note"] = "Variables renamed by original names and levels."

        for attr in ["long_name", "short_name", "units"]:
            ds_year.attrs.pop(attr, None)

        # Chunk per time-step; whole spatial tile + all features
        chunk_sizes = {
            "time": 1,
            "latitude": ds_year.latitude.size,
            "longitude": ds_year.longitude.size,
            "features": ds_year.features.size,
        }
        ds_year = ds_year.chunk(chunk_sizes)

        # Name + wrap
        ds_year.name = "data"
        if isinstance(ds_year, xarray.DataArray):
            ds_year = xarray.Dataset({"data": ds_year})

        # Encoding: float32 + BitRound(16)
        encoding = {"data": enc_bits(16)}

        output_file_path = os.path.join(output_dir)
        with dask.config.set(scheduler="threads"):
            ds_year.to_zarr(
                output_file_path,
                mode="w",
                consolidated=True,
                zarr_format=2,
                encoding=encoding,
            )
        print(
            f"Successfully processed {year} -> {output_file_path} in {time.time() - t0:.2f}s"
        )


def precompute_static_data(ds, output_base_dir):
    pbar = ProgressBar()
    pbar.register()

    # Keep only static (no time) vars
    ds = ds.drop_vars([var for var in ds.data_vars if "time" in ds[var].dims])
    static_vars = ds.data_vars

    latitude, longitude = numpy.meshgrid(ds.latitude, ds.longitude, indexing="ij")
    latitude_rad = numpy.deg2rad(latitude)
    longitude_rad = numpy.deg2rad(longitude)

    coords = {"latitude": ds.latitude, "longitude": ds.longitude}
    dims = ["latitude", "longitude"]

    cos_latitude = xarray.DataArray(numpy.cos(latitude_rad), dims=dims, coords=coords)
    cos_longitude = xarray.DataArray(numpy.cos(longitude_rad), dims=dims, coords=coords)
    sin_longitude = xarray.DataArray(numpy.sin(longitude_rad), dims=dims, coords=coords)

    data_vars = {
        "cos_latitude": cos_latitude.astype("float32"),
        "cos_longitude": cos_longitude.astype("float32"),
        "sin_longitude": sin_longitude.astype("float32"),
    }

    for var in static_vars:
        has_nans = numpy.isnan(ds[var].values).any()
        if not has_nans:
            arr = xarray.DataArray(ds[var].values, dims=dims, coords=coords)
            # Store land_sea_mask as uint8, rest as float32
            if var == "land_sea_mask":
                data_vars[var] = arr.astype("uint8")
            else:
                data_vars[var] = arr.astype("float32")

    ds_result = xarray.Dataset(data_vars=data_vars, coords=coords)

    for var in ds_result.data_vars:
        mean = ds_result[var].mean().values
        std = ds_result[var].std().values
        ds_result[var] = ds_result[var].assign_attrs(mean=mean, std=std)

    # Encoding: float32 + light BitRound(18) for trigs; masks no filters
    encoding_constants = {}
    for var in ds_result.data_vars:
        if var == "land_sea_mask":
            encoding_constants[var] = {"compressor": compressor, "dtype": "uint8"}
        elif var in ("cos_latitude", "cos_longitude", "sin_longitude"):
            encoding_constants[var] = {
                "compressor": compressor,
                "filters": [BitRound(keepbits=18)],
                "dtype": "f4",
            }
        else:
            encoding_constants[var] = {
                "compressor": compressor,
                "filters": [BitRound(keepbits=16)],
                "dtype": "f4",
            }

    with dask.config.set(scheduler="threads"):
        ds_result.to_zarr(
            os.path.join(output_base_dir, "constants"),
            mode="w",
            consolidated=True,
            zarr_format=2,
            encoding=encoding_constants,
        )


def compute_statistics(output_base_dir):
    """Compute mean/std/min/max of stacked 'data'"""
    pbar = ProgressBar()
    pbar.register()

    years = [int(item) for item in os.listdir(output_base_dir) if item.isdigit()]
    min_year = 1979
    max_year = numpy.max(years)

    files = [
        os.path.join(output_base_dir, f"{year}")
        for year in range(min_year, max_year + 1)
    ]
    ds = xarray.open_mfdataset(files, chunks={"time": 1}, engine="zarr")

    mean_ds = ds.mean(dim=["time", "latitude", "longitude"], skipna=True)
    std_ds = ds.std(dim=["time", "latitude", "longitude"], skipna=True)
    max_ds = ds.max(dim=["time", "latitude", "longitude"], skipna=True)
    min_ds = ds.min(dim=["time", "latitude", "longitude"], skipna=True)

    toa_rad = toa_radiation(ds.time.values, ds.latitude.values, ds.longitude.values)
    toa_rad_mean = numpy.mean(toa_rad)
    toa_rad_std = numpy.std(toa_rad)

    result_ds = xarray.Dataset(
        {
            "mean": mean_ds["data"].astype("float32"),
            "std": std_ds["data"].astype("float32"),
            "max": max_ds["data"].astype("float32"),
            "min": min_ds["data"].astype("float32"),
        },
    )
    result_ds.attrs["toa_radiation_mean"] = float(toa_rad_mean)
    result_ds.attrs["toa_radiation_std"] = float(toa_rad_std)

    # Encoding for stats: f32 + BitRound(15)
    encoding_stats = {
        "mean": {
            "compressor": compressor,
            "filters": [BitRound(keepbits=15)],
            "dtype": "f4",
        },
        "std": {
            "compressor": compressor,
            "filters": [BitRound(keepbits=15)],
            "dtype": "f4",
        },
        "max": {
            "compressor": compressor,
            "filters": [BitRound(keepbits=15)],
            "dtype": "f4",
        },
        "min": {
            "compressor": compressor,
            "filters": [BitRound(keepbits=15)],
            "dtype": "f4",
        },
    }

    with dask.config.set(scheduler="threads"):
        result_ds.to_zarr(
            os.path.join(output_base_dir, "stats"),
            mode="w",
            consolidated=True,
            zarr_format=2,
            encoding=encoding_stats,
        )


if __name__ == "__main__":
    main()
