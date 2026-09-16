# Segregated computation of top-of-atmosphere radiation

# This follows solar approximations from various sources:
# * https://aa.usno.navy.mil/faq/sun_approx
# * https://hal.science/hal-02175988/document
# * https://squarewidget.com/solar-coordinates/
# * https://en.wikipedia.org/wiki/Equation_of_time#Right_ascension_calculation

# The functions here ultimately calculate total top-of-atmosphere solar radiation, integrated over
# the 1h period before the specified time.  This calculation integrates the solar irradiance
# (instantaneous radiation) using polynomial quadrature, loosely following the approach at
# https://github.com/ecmwf/earthkit-meteo/blob/develop/earthkit/meteo/solar/__init__.py (used
# ECMWF's ai-models, but note that the code there seemingly incorrectly provides _centered_
# solar radiation values)

# Rather than rely on the calculation in earthkit, this code directly calculates the solar angles
# (declination, right asecension) and solar distance.  Corrected approximations to any of these should
# be easier to adapt later, with more comprehensible meanings than seemingly arbitrary updates to Fourier
# coefficients.

## copy of forecast/toa_radiation.py
import numpy

# Per https://en.wikipedia.org/wiki/Julian_day, Julian 0
# is at 4713BCE 1 January, noon UTC.  This is inconventient
# because calendars change.  Instead, pick a more modern, non-
# zero reference date with a known Julian day
# See https://aa.usno.navy.mil/calculated/juliandate?ID=AA&date=2000-01-01&era=AD&time=12%3A00%3A00.000&submit=Get+Date
julian_refdatetime = numpy.datetime64("2000-01-01T12:00", "us")
julian_refdatetime_float = julian_refdatetime.astype(numpy.float64)
julian_refday = 2451545


# Solar angles and distances, per https://aa.usno.navy.mil/faq/sun_approx
# see also https://hal.science/hal-02175988/document
# and https://squarewidget.com/solar-coordinates/
def solar_parameters(datetime_float):
    # modified_julian_day = ((datetime_us - julian_refdatetime) / numpy.timedelta64(1,'D')).astype(numpy.float64)
    modified_julian_day = (datetime_float - julian_refdatetime_float) / 86400e6
    mean_anomaly_rad = (
        numpy.mod(357.529 + 0.98560028 * modified_julian_day, 360) * numpy.pi / 180
    )
    mean_longitude_rad = (
        numpy.mod(280.459 + 0.98564736 * modified_julian_day, 360) * numpy.pi / 180
    )
    geocentric_apparent_longitude_rad = (
        mean_longitude_rad
        + (
            1.915 * numpy.sin(mean_anomaly_rad)
            + 0.020 * numpy.sin(2 * mean_anomaly_rad)
        )
        * numpy.pi
        / 180
    )

    solar_distance_au = (
        1.00014
        - 0.01671 * numpy.cos(mean_anomaly_rad)
        - 0.00014 * numpy.cos(2 * mean_anomaly_rad)
    )

    mean_obliquity_rad = (23.439 - 0.00000036 * modified_julian_day) * numpy.pi / 180
    right_ascension_rad = numpy.arctan2(
        numpy.cos(mean_obliquity_rad) * numpy.sin(geocentric_apparent_longitude_rad),
        numpy.cos(geocentric_apparent_longitude_rad),
    )
    solar_declination_rad = numpy.arcsin(
        numpy.sin(mean_obliquity_rad) * numpy.sin(geocentric_apparent_longitude_rad)
    )

    return (
        right_ascension_rad,
        solar_declination_rad,
        solar_distance_au,
        mean_longitude_rad,
        mean_anomaly_rad,
        geocentric_apparent_longitude_rad,
    )


# Equation of time, per https://en.wikipedia.org/wiki/Equation_of_time#Right_ascension_calculation and navy.mil link
def equation_of_time(mean_longitude, ascension):
    # Note input values are in radians, and output is in radian angle
    return numpy.mod(mean_longitude - ascension + numpy.pi, 2 * numpy.pi) - numpy.pi


# Local solar time, based on longitude and adjusted day (mean time + EOT, in days)
def local_solar_time_rad(longitude_deg, julian_day):
    # return (longitude_deg*numpy.pi/180 + numpy.mod(julian_day,1)*2*numpy.pi)
    out = numpy.empty_like(longitude_deg, dtype=numpy.float32)
    out_ravel = out.ravel()
    lon_ravel = longitude_deg.ravel()
    mod_day = numpy.float32(numpy.mod(julian_day, 1) * 2 * numpy.pi)
    out_ravel[:] = lon_ravel * numpy.pi / 180 + mod_day
    return out


# Cosine of the zenith angle, given latitude and local true solar time
def cos_zenith_angle(latitude, declination, true_local_solar_time_rad, weight):
    # See eqn 3-15 of hal.science document; latitude, declination, and true local solar times are all
    # in radians
    assert latitude.shape[1] == 1
    assert true_local_solar_time_rad.shape[0] == 1
    ni = latitude.shape[0]
    nj = true_local_solar_time_rad.shape[1]
    out = numpy.empty((ni, nj), dtype=numpy.float32)

    sdec = numpy.sin(declination)
    cdec = numpy.cos(declination)
    clst = numpy.cos(true_local_solar_time_rad[0, :])

    slat = numpy.sin(latitude[:, 0])
    clat = numpy.cos(latitude[:, 0])

    out = (
        numpy.maximum(
            0, slat[:, numpy.newaxis] * sdec + clat[:, numpy.newaxis] * cdec * clst
        )
        * weight
    )
    return out


# Integrate solar irradiance given particular times and weights
def toa_radiation_integrated(latitude, longitude, times, weights):
    # Initialize with zero radiation
    toa_radiation = numpy.zeros(
        numpy.broadcast_shapes(longitude.shape, latitude.shape), dtype=numpy.float32
    )

    # slat = numpy.sin(latitude*numpy.pi/180)
    # clat = numpy.cos(latitude*numpy.pi/180)
    lat_deg = (latitude * numpy.pi / 180).astype(numpy.float32)
    longitude = longitude.astype(numpy.float32)

    for tt, (time, weight) in enumerate(zip(times, weights)):
        julian_day = (time - julian_refdatetime_float) / 86400e6
        (
            ascension,
            declination,
            distance,
            mean_longitude,
            mean_anomaly_rad,
            apparent_longitude_rad,
        ) = solar_parameters(time)

        eot = equation_of_time(mean_longitude, ascension) / (
            2 * numpy.pi
        )  # convert radians to days
        true_local_solar_time = local_solar_time_rad(longitude, julian_day + eot)
        declination = numpy.float32(declination)
        toa_radiation += cos_zenith_angle(
            lat_deg,
            declination,
            true_local_solar_time,
            (1360.56 / distance**2) * weight,
        )
    return toa_radiation


# Specialize to integrate 1h radiance at fixed order
(qnodes, qweights) = numpy.polynomial.legendre.leggauss(15)  # High-order quadrature


# Trapezoidal rule
def toa_radiation_1h(latitude, longitude, final_time_us):
    quad_times_us = final_time_us - 3600e6 * 1 * (1 + qnodes) / 2
    quad_weights = 3600 * 1 * qweights / 2
    return toa_radiation_integrated(latitude, longitude, quad_times_us, quad_weights)


def toa_radiation(times: numpy.ndarray, lat: numpy.ndarray, lon: numpy.ndarray):
    """Calculate top of atmosphere radiation, integrated over the 1h period ending at the specified times.
    Args:
        times: Array of datetime64 timestamps
        lat: Latitude array
        lon: Longitude array
    Returns:
        numpy.ndarray: TOA radiation values
    """
    # Convert inputs to numpy arrays if needed
    lat = numpy.asarray(lat)
    lon = numpy.asarray(lon)
    time_data = numpy.asarray(times).astype("datetime64[us]")

    # If latitude/longitude have dimension 1, assume that they really
    # specify two different axes of a 2D grid
    lat_data = lat.reshape((-1, 1))
    lon_data = lon.reshape((1, -1))
    output_shape = (lat.size, lon.size)

    output_rad = numpy.empty((time_data.size,) + output_shape, dtype=numpy.float32)
    for idx, otime in enumerate(time_data):
        output_rad[idx, ...] = toa_radiation_1h(
            lat_data, lon_data, otime.astype(numpy.float64)
        )

    return output_rad
