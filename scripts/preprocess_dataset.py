import argparse
import xarray
import xarray as xr
import numpy
import dask
from dask.diagnostics import ProgressBar
import os
import time
import sys
from numcodecs import Blosc, BitRound
from layerquantizer import LayerQuantizer

# Main compressor for stacked float32 meteorological tensors
compressor = LayerQuantizer()
fallback_compressor = Blosc(cname="zstd", clevel=5, shuffle=Blosc.SHUFFLE)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data.forcings.toa_radiation import toa_radiation, toa_radiation_stats

def clean_zarr_v2_encoding(ds):
    for v in ds.variables:
        for key in ["serializer", "filters", "compressors", "shards"]:
            ds[v].encoding.pop(key, None)
    return ds

def drop_unused_noncore_coords(ds, core_dims):
    """
    Drop auxiliary coordinates such as lat/lon bounds before transposing.

    Some inputs include coordinates like ``lat_b`` and ``lon_b`` that are not
    used by any data variable. Xarray still includes their dimensions in the
    dataset-wide dimension set, which makes ``Dataset.transpose`` fail unless
    those coordinates are removed first.
    """
    coord_names_to_drop = [
        name for name in ds.coords if name not in core_dims and name not in ds.data_vars
    ]
    if coord_names_to_drop:
        ds = ds.drop_vars(coord_names_to_drop, errors="ignore")
    return ds

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

    wind_x = (
        dr_dt * cos_lat * cos_lon
        - sin_lat * cos_lon * ds.v_component_of_wind
        - sin_lon * ds.u_component_of_wind
    )
    wind_y = (
        dr_dt * cos_lat * sin_lon
        - sin_lat * sin_lon * ds.v_component_of_wind
        + cos_lon * ds.u_component_of_wind
    )
    wind_z = dr_dt * sin_lat + cos_lat * ds.v_component_of_wind

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


def compute_scaled_angular_winds(ds):
    R_earth = 6371000.0

    lat_rad = numpy.deg2rad(ds.latitude)
    coslat = numpy.cos(lat_rad)

    safe = numpy.abs(coslat) > 1e-3

    scaled_u = xarray.where(safe, ds.u_component_of_wind / (R_earth * coslat), 0.0)
    scaled_v = xarray.where(safe, ds.v_component_of_wind / R_earth, 0.0)

    return ds.assign(
        u_component_of_wind_scaled=scaled_u,
        v_component_of_wind_scaled=scaled_v,
    )


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
    parser.add_argument("--begin_year", type=int, default=1979, help="Initial year")
    parser.add_argument("--end_year", type=int, default=2023, help="Final year")
    parser.add_argument("--levels", type=int, choices=[37, 13], default=13)

    parser.add_argument(
        "--skip-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Do not compute stats",
    )

    parser.add_argument(
        "--only-stats",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Only compute stats",
    )

    args = parser.parse_args()

    if args.skip_stats and args.only_stats:
        parser.error("--skip-stats and --only-stats cannot both be used")

    core_dims = ("time", "latitude", "longitude", "level")
    ds = xarray.open_mfdataset(args.input_dir, engine="zarr")
    ds = drop_unused_noncore_coords(ds, core_dims)
    ds = ds.transpose(*core_dims)


    ds = ds.transpose("time", "latitude", "longitude", "level")

    if args.levels == 13:
        levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]
    elif args.levels == 37:
        levels = [1, 2, 3, 5, 7, 10, 20, 30, 50,  70, 100, 125, 150, 175, 200,
                  225, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750,
                  775, 800, 825, 850, 875, 900, 925, 950, 975, 1000]
    else:
        raise("Invalid number of levels for preprocessing")

    ds = ds.sel(level=levels)

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
        "total_precipitation",
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
        print("Removing poles....")
        lat_to_drop = [v for v in (-90, 90) if v in ds.latitude.values]
        if lat_to_drop:
            ds = ds.sel(latitude=~ds.latitude.isin(lat_to_drop))

    if args.interp_deg > 0:
        # Interpolate data. For this, dataset must contain the poles and 0 longitude
        # Then, the dataset is padded with longitude=360 to avoid
        # Interpolating outside of range (gives nan)
        latitude = numpy.arange(-90, 90 + args.interp_deg, args.interp_deg)
        longitude = numpy.arange(0, 360, args.interp_deg)
        ds = ds.sel(latitude=latitude, longitude=longitude)

    # Set a small tolerance to avoid log(0)
    tolerance = 1e-10

    # Compute log_humidity safely
    # ds["log_humidity"] = numpy.log(ds["specific_humidity"].clip(min=0) + tolerance)

    # Store the tolerance as a dataset attribute
    # ds.attrs["log_humidity_tolerance"] = tolerance

    # Step 1: Stack data for efficient storage and processing
    if not args.only_stats:
        stack_data(ds, args.output_dir, args.begin_year, args.end_year)

    if not args.skip_stats:
        # Step 2: Precompute static data (e.g., geographic variables)
        precompute_static_data(ds, args.output_dir)

        # Step 3: Compute mean and standard deviation for atmospheric and surface variables
        compute_statistics(args.output_dir, args.begin_year, args.end_year)

        # Step 4: Compute the 6h standard deviation
        compute_tendency_statistics(args.output_dir, args.begin_year, args.end_year)


def stack_data(ds, output_base_dir, begin_year, end_year):
    ds = compute_cartesian_wind(ds)

    # Cast variables to float32
    ds = clean_zarr_v2_encoding(ds)

    min_year = begin_year
    max_year = end_year

    # Keep only time-varying vars
    ds = ds.drop_vars([var for var in ds.data_vars if "time" not in ds[var].dims])

    pbar = ProgressBar()
    pbar.register()
    keep_dims = ["time", "latitude", "longitude"]

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

        # IMPORTANT: move spatial dims to the end for layerquantizer
        ds_year = ds_year.transpose("time", "features", "latitude", "longitude")

        ds_year.name = "data"
        if isinstance(ds_year, xarray.DataArray):
            ds_year = xarray.Dataset({"data": ds_year})

        encoding = {
            "data": {
                "compressor": compressor,
                "dtype": "f4",
            }
        }

        ds_year = clean_zarr_v2_encoding(ds_year)

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
            encoding_constants[var] = {
                "compressor": fallback_compressor,
                "dtype": "uint8",
            }
        elif var in ("cos_latitude", "cos_longitude", "sin_longitude"):
            encoding_constants[var] = {
                "compressor": fallback_compressor,
                "filters": [BitRound(keepbits=18)],
                "dtype": "f4",
            }
        else:
            encoding_constants[var] = {
                "compressor": fallback_compressor,
                "filters": [BitRound(keepbits=16)],
                "dtype": "f4",
            }

    ds_result = clean_zarr_v2_encoding(ds_result)

    with dask.config.set(scheduler="threads"):
        ds_result.to_zarr(
            os.path.join(output_base_dir, "constants"),
            mode="w",
            consolidated=True,
            zarr_format=2,
            encoding=encoding_constants,
        )


def compute_statistics(output_base_dir, begin_year, end_year):
    """Compute mean/std/min/max of stacked 'data'"""
    pbar = ProgressBar()
    pbar.register()

    years = [int(item) for item in os.listdir(output_base_dir) if item.isdigit()]
    min_year = begin_year
    max_year = end_year

    files = [
        os.path.join(output_base_dir, f"{year}")
        for year in range(min_year, max_year + 1)
    ]
    ds = xarray.open_mfdataset(files, chunks={"time": 1}, engine="zarr")

    mean_ds = ds.mean(dim=["time", "latitude", "longitude"], skipna=True)
    std_ds = ds.std(dim=["time", "latitude", "longitude"], skipna=True)
    max_ds = ds.max(dim=["time", "latitude", "longitude"], skipna=True)
    min_ds = ds.min(dim=["time", "latitude", "longitude"], skipna=True)

    toa_rad_mean, toa_rad_std = toa_radiation_stats(
        ds.indexes["time"].values,
        ds.latitude.values,
        ds.longitude.values,
    )

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
            "compressor": fallback_compressor,
            "filters": [BitRound(keepbits=15)],
            "dtype": "f4",
        },
        "std": {
            "compressor": fallback_compressor,
            "filters": [BitRound(keepbits=15)],
            "dtype": "f4",
        },
        "max": {
            "compressor": fallback_compressor,
            "filters": [BitRound(keepbits=15)],
            "dtype": "f4",
        },
        "min": {
            "compressor": fallback_compressor,
            "filters": [BitRound(keepbits=15)],
            "dtype": "f4",
        },
    }

    result_ds = clean_zarr_v2_encoding(result_ds)

    with dask.config.set(scheduler="threads"):
        result_ds.to_zarr(
            os.path.join(output_base_dir, "stats"),
            mode="w",
            consolidated=True,
            zarr_format=2,
            encoding=encoding_stats,
        )


def compute_tendency_statistics(
    output_base_dir, begin_year, end_year, delta_hours=(6,)
):
    """Compute mean/std/min/max of N-hour tendencies of stacked 'data'.

    For each delta in `delta_hours`, computes per-feature statistics of
    the tendency  y(t + delta) - y(t)  over the time range, and writes
    them to a separate zarr group at `<output_base_dir>/tendency_stats_<delta>h`.

    The output structure mirrors `compute_statistics` so it can be loaded
    the same way by the dataset.

    Args:
        output_base_dir: base directory containing per-year stacked zarrs
        begin_year: first year to include (inclusive)
        end_year: last year to include (inclusive)
        delta_hours: iterable of tendency horizons in hours, e.g. (6,) or (6, 12, 24)
    """
    pbar = ProgressBar()
    pbar.register()

    files = [
        os.path.join(output_base_dir, f"{year}")
        for year in range(begin_year, end_year + 1)
    ]
    ds = xarray.open_mfdataset(files, chunks={"time": 1}, engine="zarr")

    # Infer the native time resolution from the data
    time_values = ds.indexes["time"].values
    if len(time_values) < 2:
        raise ValueError("Need at least two time steps to compute tendencies.")
    native_dt = time_values[1] - time_values[0]
    native_dt_hours = native_dt.astype("timedelta64[h]").astype(int)
    print(f"Native time resolution: {native_dt_hours}h")

    for delta_h in delta_hours:
        if delta_h % native_dt_hours != 0:
            raise ValueError(
                f"Requested tendency delta ({delta_h}h) is not a multiple of "
                f"the native time resolution ({native_dt_hours}h)."
            )
        stride = delta_h // native_dt_hours
        print(f"\nComputing {delta_h}h tendency statistics (stride={stride})...")

        t0 = time.time()

        # Lazy tendency: shifted difference along time
        # data[stride:] - data[:-stride]   in xarray idiom:
        data = ds["data"]
        tendency = data.isel(time=slice(stride, None)) - data.isel(
            time=slice(None, -stride)
        ).assign_coords(time=data.isel(time=slice(stride, None)).time)

        # Compute statistics over (time, latitude, longitude), per feature
        mean_t = tendency.mean(dim=["time", "latitude", "longitude"], skipna=True)
        std_t = tendency.std(dim=["time", "latitude", "longitude"], skipna=True)
        max_t = tendency.max(dim=["time", "latitude", "longitude"], skipna=True)
        min_t = tendency.min(dim=["time", "latitude", "longitude"], skipna=True)

        result_ds = xarray.Dataset(
            {
                "tendency_mean": mean_t.astype("float32"),
                "tendency_std": std_t.astype("float32"),
                "tendency_max": max_t.astype("float32"),
                "tendency_min": min_t.astype("float32"),
            },
        )
        result_ds.attrs["delta_hours"] = int(delta_h)
        result_ds.attrs["native_dt_hours"] = int(native_dt_hours)
        result_ds.attrs["stride"] = int(stride)
        result_ds.attrs["begin_year"] = int(begin_year)
        result_ds.attrs["end_year"] = int(end_year)
        result_ds.attrs["n_samples"] = int(len(time_values) - stride)

        encoding_stats = {
            "tendency_mean": {
                "compressor": fallback_compressor,
                "filters": [BitRound(keepbits=15)],
                "dtype": "f4",
            },
            "tendency_std": {
                "compressor": fallback_compressor,
                "filters": [BitRound(keepbits=15)],
                "dtype": "f4",
            },
            "tendency_max": {
                "compressor": fallback_compressor,
                "filters": [BitRound(keepbits=15)],
                "dtype": "f4",
            },
            "tendency_min": {
                "compressor": fallback_compressor,
                "filters": [BitRound(keepbits=15)],
                "dtype": "f4",
            },
        }

        out_path = os.path.join(output_base_dir, f"tendency_stats_{delta_h}h")

        result_ds = clean_zarr_v2_encoding(result_ds)

        with dask.config.set(scheduler="threads"):
            result_ds.to_zarr(
                out_path,
                mode="w",
                consolidated=True,
                zarr_format=2,
                encoding=encoding_stats,
            )

        print(
            f"Wrote {delta_h}h tendency stats to {out_path} "
            f"in {time.time() - t0:.2f}s"
        )


if __name__ == "__main__":
    main()
