import numpy as np
import xarray as xr


def time_forcings(ds: xr.Dataset):
    """
    Compute sine and cosine of local time of day and year progress.
    Handles high precision datetime64 objects from the 'time' coordinate.

    Args:
        ds (xr.Dataset): Dataset with 'time' coordinate.

    Returns:
        xr.Dataset: Dataset with new forcing variables added.
    """
    # Extract time information
    time = ds["time"].values  # 'datetime64[ns]' array
    hours_in_day = 24
    days_in_year = 365.25  # Average year length considering leap years

    # Handle high-precision datetime and compute local time of day
    time_as_datetime = np.array(time, dtype="datetime64[h]")  # Convert to hours
    hour_of_day = (
        time_as_datetime - time_as_datetime.astype("datetime64[D]")
    ) / np.timedelta64(1, "h")
    local_time_norm = hour_of_day / hours_in_day  # Normalize to [0, 1)

    # Compute sine and cosine for local time of day
    sine_local_time = np.sin(2 * np.pi * local_time_norm)
    cosine_local_time = np.cos(2 * np.pi * local_time_norm)

    # Compute day of year normalized to [0, 1)
    year_start = time_as_datetime.astype("datetime64[Y]")  # Year start
    day_of_year = (time_as_datetime - year_start) / np.timedelta64(1, "D")
    year_progress_norm = day_of_year / days_in_year  # Normalize to [0, 1)

    # Compute sine and cosine for year progress
    sine_year_progress = np.sin(2 * np.pi * year_progress_norm)
    cosine_year_progress = np.cos(2 * np.pi * year_progress_norm)

    # Add new variables to the dataset
    # Create a new dataset with computed variables
    time_forcings_ds = xr.Dataset(
        {
            "sin_time_of_day": (("time"), sine_local_time),
            "cos_time_of_day": (("time"), cosine_local_time),
            "sin_year_progress": (("time"), sine_year_progress),
            "cos_year_progress": (("time"), cosine_year_progress),
        },
        coords={"time": ds["time"]}  # Add the same 'time' coordinate
    )

    return time_forcings_ds