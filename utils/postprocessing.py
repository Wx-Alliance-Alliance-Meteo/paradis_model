import re

import numpy
import torch

from utils.normalization import (
    denormalize_precipitation,
    denormalize_humidity,
    denormalize_standard,
)


def compute_cartesian_wind(
    latitude,
    longitude,
    pressure_levels,
    temperature,
    u,
    v,
    w,
    u_10m,
    v_10m,
):
    """
    Compute spherical wind components (u, v, w) from 3D Cartesian wind components.
    """

    # Constants
    g = 9.80616  # Gravitational acceleration m/s^2
    R = 287.05  # Gas constant for dry air J/(kg·K)
    R_earth = 6371000.0  # Earth radius in meters

    # Convert coordinates to radians
    lon_rad = numpy.deg2rad(longitude)
    lat_rad = numpy.deg2rad(latitude)

    # Precompute trigonometric functions
    sin_lat = numpy.sin(lat_rad)
    cos_lat = numpy.cos(lat_rad)
    sin_lon = numpy.sin(lon_rad)
    cos_lon = numpy.cos(lon_rad)

    # Convert pressure velocity (Pa/s) to geometric velocity (m/s)
    dr_dt = (
        -w
        * R
        * temperature
        / (pressure_levels[:, numpy.newaxis, numpy.newaxis] * 100 * g)
    )

    # Convert horizontal wind speeds to angular velocities
    dlon_dt = u / (R_earth * cos_lat)
    dlat_dt = v / R_earth

    # Transform to Cartesian
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

    # Surface components (no vertical velocity)
    dlon_dt_10m = u_10m / (R_earth * cos_lat)
    dlat_dt_10m = v_10m / R_earth

    wind_x_10m = (
        -R_earth * sin_lat * cos_lon * dlat_dt_10m
        - R_earth * cos_lat * sin_lon * dlon_dt_10m
    )

    wind_y_10m = (
        -R_earth * sin_lat * sin_lon * dlat_dt_10m
        + R_earth * cos_lat * cos_lon * dlon_dt_10m
    )

    wind_z_10m = R_earth * cos_lat * dlat_dt_10m

    return wind_x, wind_y, wind_z, wind_x_10m, wind_y_10m, wind_z_10m


def compute_spherical_wind(
    latitude,
    longitude,
    pressure_levels,
    temperature,
    wind_x,
    wind_y,
    wind_z,
    wind_x_10m,
    wind_y_10m,
    wind_z_10m,
):
    """
    Compute spherical wind components (u, v, w) from 3D Cartesian wind components.

    Note: Returns w in Pa/s (pressure coordinates) to match original data format.
    """

    # Constants
    g = 9.80616  # Gravitational acceleration m/s^2
    R = 287.05  # Gas constant for dry air J/(kg·K)
    R_earth = 6371000.0  # Earth radius in meters

    # Convert coordinates to radians
    lon_rad = numpy.deg2rad(longitude)
    lat_rad = numpy.deg2rad(latitude)

    # Precompute trigonometric functions
    sin_lat = numpy.sin(lat_rad)
    cos_lat = numpy.cos(lat_rad)
    sin_lon = numpy.sin(lon_rad)
    cos_lon = numpy.cos(lon_rad)

    # Inverse transformation from Cartesian to angular velocities and geometric velocity
    # From the orthogonality of the transformation matrix:
    dlon_dt = (-wind_x * sin_lon + wind_y * cos_lon) / (R_earth * cos_lat)
    dlat_dt = (
        -wind_x * sin_lat * cos_lon - wind_y * sin_lat * sin_lon + wind_z * cos_lat
    ) / R_earth
    dr_dt = wind_x * cos_lat * cos_lon + wind_y * cos_lat * sin_lon + wind_z * sin_lat

    # Convert angular velocities back to horizontal wind speeds in m/s
    u = dlon_dt * R_earth * cos_lat  # eastward component
    v = dlat_dt * R_earth  # northward component

    # Convert geometric velocity (m/s) back to pressure velocity (Pa/s)
    # ω = -w * p * g / (R * T) where w = dr/dt (m/s)
    w = (
        -dr_dt
        * pressure_levels[:, numpy.newaxis, numpy.newaxis]
        * 100
        * g
        / (R * temperature)
    )

    # Surface wind components (same calculation for horizontal components)
    dlon_dt_10m = (-wind_x_10m * sin_lon + wind_y_10m * cos_lon) / (R_earth * cos_lat)
    dlat_dt_10m = (
        -wind_x_10m * sin_lat * cos_lon
        - wind_y_10m * sin_lat * sin_lon
        + wind_z_10m * cos_lat
    ) / R_earth

    u_10m = dlon_dt_10m * R_earth * cos_lat
    v_10m = dlat_dt_10m * R_earth

    return u, v, w, u_10m, v_10m


def get_var_indices(variable_name, variable_list):
    indices = []
    for i, var in enumerate(variable_list):
        var_name = re.sub(r"_h\d+$", "", var)  # Remove height suffix (e.g., "_h10")
        if variable_name == var_name:
            indices.append(i)
    return numpy.array(indices)


def preprocess_variable_names(atmospheric_vars, surface_vars):
    # Rename variables that require post-processing in dataset
    atmospheric_vars = replace_variable_name(
        "wind_x", "u_component_of_wind", atmospheric_vars
    )
    atmospheric_vars = replace_variable_name(
        "wind_y", "v_component_of_wind", atmospheric_vars
    )

    if "vertical velocity" not in atmospheric_vars:
        atmospheric_vars = replace_variable_name(
            "wind_z", "vertical_velocity", atmospheric_vars
        )
    else:
        atmospheric_vars = replace_variable_name(
            "wind_z", "computed_vertical_velocity", atmospheric_vars
        )

    surface_vars = replace_variable_name(
        "wind_x_10m", "10m_u_component_of_wind", surface_vars
    )
    surface_vars = replace_variable_name(
        "wind_y_10m", "10m_v_component_of_wind", surface_vars
    )


def replace_variable_name(variable_old, variable_new, variable_list):
    for i, var in enumerate(variable_list):
        var_name = re.sub(r"_h\d+$", "", var)  # Remove height suffix (e.g., "_h10")
        if variable_old == var_name:
            new_var_name = re.sub(variable_old, variable_new, var)
            variable_list[i] = new_var_name
    return variable_list


def convert_cartesian_to_spherical_winds(latitude, longitude, cfg, array, features):

    # Convert wind velocities to spherical coordinates
    longitude, latitude = numpy.meshgrid(longitude, latitude)
    pressure_levels = numpy.array([float(val) for val in cfg.features.pressure_levels])

    # Extract the variables from the results
    temperature = array[:, :, get_var_indices("temperature", features)]

    # Get the indices for the variables to transform
    u_ind = get_var_indices("wind_x", features)
    v_ind = get_var_indices("wind_y", features)
    w_ind = get_var_indices("wind_z", features)
    u10m_ind = get_var_indices("wind_x_10m", features)
    v10m_ind = get_var_indices("wind_y_10m", features)
    w10m_ind = get_var_indices("wind_z_10m", features)

    wind_x = array[:, :, u_ind]
    wind_y = array[:, :, v_ind]
    wind_z = array[:, :, w_ind]
    wind_x_10m = array[:, :, u10m_ind]
    wind_y_10m = array[:, :, v10m_ind]
    wind_z_10m = array[:, :, w10m_ind]

    # PARADIS output includes wind speeds in cartesian coordinates.
    # Here, we transform back to spherical
    u, v, w, u_10m, v_10m = compute_spherical_wind(
        latitude,
        longitude,
        pressure_levels,
        temperature,
        wind_x,
        wind_y,
        wind_z,
        wind_x_10m,
        wind_y_10m,
        wind_z_10m,
    )

    # Replace variables in dataset
    array[:, :, u_ind] = u
    array[:, :, v_ind] = v
    array[:, :, w_ind] = w
    array[:, :, u10m_ind] = u_10m
    array[:, :, v10m_ind] = v_10m


def denormalize_datasets(ground_truth, output_forecast, dataset):
    """Denormalize both ground truth and forecast datasets."""
    _denormalize_dataset(ground_truth, dataset)
    _denormalize_dataset(output_forecast, dataset)


def _denormalize_dataset(data: torch.tensor, dataset_obj):
    """Denormalize the ground truth data."""
    if dataset_obj.custom_normalization:
        data[:, :, dataset_obj.norm_precip_out] = denormalize_precipitation(
            data[:, :, dataset_obj.norm_precip_out]
        )

        data[:, :, dataset_obj.norm_humidity_out] = denormalize_humidity(
            data[:, :, dataset_obj.norm_humidity_out],
            dataset_obj.q_min,
            dataset_obj.q_max,
        )

    data[:, :, dataset_obj.norm_zscore_out] = denormalize_standard(
        data[:, :, dataset_obj.norm_zscore_out],
        dataset_obj.output_mean.view(-1, 1, 1),
        dataset_obj.output_std.view(-1, 1, 1),
    )
