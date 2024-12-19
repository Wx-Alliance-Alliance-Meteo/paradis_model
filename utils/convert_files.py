import os
import sys
import time
from glob import glob
from multiprocessing.pool import ThreadPool

from mpi4py import MPI
import numpy as np
import xarray as xr
import dask
from dask.diagnostics import ProgressBar

# Limit threads per rank (e.g., 4 threads per MPI rank)
dask.config.set(scheduler="threads", pool=ThreadPool(10))

required = MPI.THREAD_MULTIPLE
provided = MPI.Query_thread()

if provided < required:
    raise RuntimeError(
        f"Insufficient threading support: required={required}, provided={provided}"
    )

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Store the original sys.path
original_sys_path = sys.path.copy()
# Temporarily add the codec location to sys.path
sys.path.append("/fs/site5/eccc/mrd/rpnatm/csu001/ppp5/graphcast_dev")
# Import the codec
import forecast.encabulator

# Reset sys.path to its original state
sys.path = original_sys_path


def process_data(year, month, dpath, new_lat, new_lon, new_levels, output_base_dir):
    t0 = time.time()
    month_str = f"{month:02d}"
    input_dataset_path = f"{dpath}/{year}/{month_str}"

    try:

        print(f"Rank {rank}: Opening dataset {input_dataset_path}")
        dset = xr.open_dataset(input_dataset_path, engine="zarr", chunks="auto")

        # Re-chunk to larger sizes for better performance
        dset = dset.chunk({"latitude": 180, "longitude": 360})
        dset = dset.sel(level=new_levels)

        # Interpolate to the new grid
        regridded_dset = dset.interp(latitude=new_lat, longitude=new_lon)

        # Re-chunk again for saving
        regridded_dset = regridded_dset.chunk(
            {
                "time": dset.time.size,
                "latitude": len(new_lat),
                "longitude": len(new_lon),
                "level": len(new_levels),
            }
        )

        # Prepare the output directory
        output_dir = os.path.join(output_base_dir, str(year), month_str)
        os.makedirs(output_dir, exist_ok=True)

        # Save to Zarr format
        output_file_path = os.path.join(output_dir)
        with dask.config.set(scheduler="threads"):
            regridded_dset.to_zarr(output_file_path, mode="w", consolidated=True)

        print(
            f"Rank {rank}: Successfully processed {input_dataset_path} -> {output_file_path} in {time.time() - t0:.2f} seconds"
        )
    except Exception as e:
        print(f"Rank {rank}: Failed to process {input_dataset_path}: {e}")


def restack_data(year, files, output_base_dir):
    t0 = time.time()
    keep_dims = ["time", "latitude", "longitude"]

    try:
        print(f"Rank {rank}: Opening datasets for year {year}")

        # Open dataset
        ds = xr.open_mfdataset(files, engine="zarr")

        # Stack all variables at each pressure level, keeping the dimensions aggr_dims
        ds = ds.to_stacked_array(new_dim="stacked", sample_dims=keep_dims)

        # Rename variables to contain the pressure level if existing
        new_names = [
            val[0] + "_h" + str(val[1]) if str(val[1]) != "nan" else val[0]
            for val in ds.stacked.values
        ]
        ds = ds.drop_vars(["stacked", "variable", "level"]).assign_coords(
            stacked=new_names
        )

        # Set the chunk sizes to store the files
        chunk_sizes = {
            "time": 1,
            "latitude": ds.latitude.size,
            "longitude": ds.longitude.size,
            "stacked": ds.stacked.size,
        }

        # Prepare the output directory
        output_dir = os.path.join(output_base_dir, str(year))
        os.makedirs(output_dir, exist_ok=True)

        # Clean up metadata
        ds.attrs["description"] = "Stacked dataset with renamed variables"
        ds.attrs["note"] = (
            "Variables have been renamed based on their original names and levels."
        )

        del ds.attrs["long_name"]
        try:
            del ds.attrs["short_name"]
        except:
            pass
        del ds.attrs["units"]

        if "toa_incident_solar_radiation" in ds.stacked:
            ds = ds.drop_vars("toa_incident_solar_radiation")

        # Rechunk data
        ds = ds.chunk(chunk_sizes)

        # Rename dataset
        ds.name = "data"

        # Write or append to existing zarr file
        output_file_path = os.path.join(output_dir)
        with dask.config.set(scheduler="threads"):
            ds.to_zarr(output_file_path, mode="w", consolidated=True)

        print(
            f"Rank {rank}: Successfully processed {year} -> {output_file_path} in {time.time() - t0:.2f} seconds"
        )
    except Exception as e:
        print(f"Rank {rank}: Failed to process {year}: {e}")


def main():

    # Enable Dask progress bar for monitoring
    pbar = ProgressBar()
    pbar.register()

    # Specify the input data path
    dpath = "/fs/site6/eccc/mrd/rpnatm/csu001/ppp6/era5_025deg"

    # Create the new latitude and longitude grid for 4-degree resolution
    new_lat = np.arange(-90, 91, 4)
    new_lon = np.arange(0, 360, 4)

    new_levels = [50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000]

    # Specify the output base directory
    output_base_dir = "/home/cap003/hall5/gatmosphere_data"
    year_range = range(2016, 2024)
    file_lists = [
        os.path.join(output_base_dir, str(year), f"{month:02d}")
        for year in year_range
        for month in range(1, 13)
    ]
    file_years = [year for year in year_range for month in range(1, 13)]
    file_months = [month for year in year_range for month in range(1, 13)]

    for i in range(0, len(file_lists), size):
        ind = i + rank
        file = file_lists[ind]
        if os.path.exists(os.path.join(file, "10m_u_component_of_wind/0.0.0")):
            continue

        year = file_years[ind]
        month = file_months[ind]

        process_data(year, month, dpath, new_lat, new_lon, new_levels, output_base_dir)


def convert_to_gatmosphere():
    # Enable Dask progress bar for monitoring
    pbar = ProgressBar()
    pbar.register()

    # Specify the input data path
    dpath = "/home/cap003/hall5/gatmosphere_data/ERA5_4deg/"

    # Specify the output base directory
    output_base_dir = "/home/cap003/hall5/gatmosphere_data/ERA5_4deg_unified/"

    start_year = 1970
    end_year = 2023
    num_years = end_year - start_year

    # Create list of files to open
    files = [dpath + str(year) + "/*" for year in range(start_year, end_year + 1)]
    year_files = [[glob_file for glob_file in glob.glob(file)] for file in files]

    # Loop per year

    for i in range(0, num_years, size):
        year = start_year + i + rank

        if year <= end_year:

            file_group = year_files[i + rank]
            print(f"Rank {rank}: processing file")
            restack_data(year, file_group, output_base_dir)


def compute_global_means():

    pbar = ProgressBar()
    pbar.register()
    # Specify the input data path
    dpath = "/home/cap003/hall5/gatmosphere_data/ERA5_4deg_unified/"

    # Specify the output base directory
    output_base_dir = "/home/cap003/hall5/gatmosphere_data/ERA5_4deg_unified/"

    start_year = 1970
    end_year = 2022
    num_years = end_year - start_year

    # Create list of files to open
    files = [dpath + str(year) + "/" for year in range(start_year, end_year + 1)]

    ds = xr.open_mfdataset(files, chunks={"time": 200}, engine="zarr")

    # Compute time-mean and time-standard deviation (per-level)
    mean_ds = ds.mean(dim=["time", "latitude", "longitude"])
    std_ds = ds.std(dim=["time", "latitude", "longitude"])

    # Combine the mean and std into a single dataset
    result_ds = xr.Dataset({"mean": mean_ds["data"], "std": std_ds["data"]})

    with dask.config.set(scheduler="threads"):
        result_ds.to_zarr(
            "/home/cap003/hall5/gatmosphere_data/ERA5_4deg_unified/stats", mode="w"
        )


if __name__ == "__main__":
    # Interpolate data
    main()

    # Rechunk data and stack variables at each coordinate
    convert_to_gatmosphere()

    # Compute and store per-level global mean and standard deviation
    compute_global_means()
