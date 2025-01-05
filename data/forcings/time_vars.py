"""Informations for temporal forcings."""

import numpy


def time_forcings(times: numpy.ndarray):
    """Compute sine and cosine of local time of day and year progress.
    Args:
        times: Array of datetime64 timestamps
    Returns:
        dict: Dictionary with time forcing variables
    """
    hours_in_day = 24
    days_in_year = 365.25  # Average year length considering leap years

    # Handle high-precision datetime and compute local time of day
    time_as_datetime = times.astype("datetime64[h]")  # Convert to hours
    hour_of_day = (
        time_as_datetime - time_as_datetime.astype("datetime64[D]")
    ) / numpy.timedelta64(1, "h")
    local_time_norm = hour_of_day / hours_in_day  # Normalize to [0, 1)

    # Compute sine and cosine for local time of day
    sine_local_time = numpy.sin(2 * numpy.pi * local_time_norm)
    cosine_local_time = numpy.cos(2 * numpy.pi * local_time_norm)

    # Compute day of year normalized to [0, 1)
    year_start = time_as_datetime.astype("datetime64[Y]")  # Year start
    day_of_year = (time_as_datetime - year_start) / numpy.timedelta64(1, "D")
    year_progress_norm = day_of_year / days_in_year  # Normalize to [0, 1)

    # Compute sine and cosine for year progress
    sine_year_progress = numpy.sin(2 * numpy.pi * year_progress_norm)
    cosine_year_progress = numpy.cos(2 * numpy.pi * year_progress_norm)

    return {
        "sin_time_of_day": sine_local_time,
        "cos_time_of_day": cosine_local_time,
        "sin_year_progress": sine_year_progress,
        "cos_year_progress": cosine_year_progress,
    }
