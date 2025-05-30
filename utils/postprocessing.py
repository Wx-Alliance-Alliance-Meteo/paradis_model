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

    # Extract necessary data
    lon_rad = numpy.deg2rad(longitude)
    lat_rad = numpy.deg2rad(latitude)

    wind_x = (
        -u * numpy.sin(lon_rad)
        - v * numpy.sin(lat_rad) * numpy.cos(lon_rad)
        - w
        * R
        * temperature
        / (pressure_levels[:, None, None] * 100 * g)
        * numpy.cos(lat_rad)
        * numpy.cos(lon_rad)
    )

    wind_y = (
        u * numpy.cos(lon_rad)
        - v * numpy.sin(lat_rad) * numpy.sin(lon_rad)
        - w
        * R
        * temperature
        / (pressure_levels[:, None, None] * 100 * g)
        * numpy.cos(lat_rad)
        * numpy.sin(lon_rad)
    )

    wind_z = v * numpy.cos(lat_rad) - w * R * temperature / (
        pressure_levels[:, None, None] * 100 * g
    ) * numpy.sin(lat_rad)

    # Surface wind components (Assume w=0 at surface)
    wind_x_10m = -u_10m * numpy.sin(lon_rad) - v_10m * numpy.sin(lat_rad) * numpy.cos(
        lon_rad
    )
    wind_y_10m = u_10m * numpy.cos(lon_rad) - v_10m * numpy.sin(lat_rad) * numpy.sin(
        lon_rad
    )
    wind_z_10m = v_10m * numpy.cos(lat_rad)

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
    """

    # Constants
    g = 9.80616  # Gravitational acceleration m/s^2
    R = 287.05  # Gas constant for dry air J/(kg·K)

    # Extract necessary data
    lon_rad = numpy.deg2rad(longitude)
    lat_rad = numpy.deg2rad(latitude)

    # Compute spherical components
    u = -wind_x * numpy.sin(lon_rad) + wind_y * numpy.cos(lon_rad)

    v = (
        -wind_x * numpy.sin(lat_rad) * numpy.cos(lon_rad)
        - wind_y * numpy.sin(lat_rad) * numpy.sin(lon_rad)
        + wind_z * numpy.cos(lat_rad)
    )

    w = (
        -wind_x * numpy.cos(lat_rad) * numpy.cos(lon_rad)
        - wind_y * numpy.cos(lat_rad) * numpy.sin(lon_rad)
        - wind_z * numpy.sin(lat_rad)
    ) * (pressure_levels[:, None, None] * 100 * g / (R * temperature))

    # At 10m, w is considered 0
    u_10m = -wind_x_10m * numpy.sin(lon_rad) + wind_y_10m * numpy.cos(lon_rad)
    v_10m = (
        -wind_x_10m * numpy.sin(lat_rad) * numpy.cos(lon_rad)
        - wind_y_10m * numpy.sin(lat_rad) * numpy.sin(lon_rad)
        + wind_z_10m * numpy.cos(lat_rad)
    )

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
    atmospheric_vars = replace_variable_name(
        "wind_z", "vertical_velocity", atmospheric_vars
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
