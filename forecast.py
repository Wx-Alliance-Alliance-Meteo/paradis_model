"""Forecast script for the model."""

from datetime import datetime
import logging

import hydra
import matplotlib.pyplot as plt
import numpy
from omegaconf import DictConfig
import torch
import xarray

from trainer import LitParadis
from data.datamodule import Era5DataModule


def plot_forecast_map(
    date_in,
    date_out,
    output_data,
    true_data,
    datamodule,
    feature,
    cfg,
    level=None,
    temp_offset=0,
):
    """Plot comparison maps for model output and ground truth."""
    dataset = datamodule.dataset

    # Determine the feature index and name
    if level is not None:
        # For atmospheric variables with pressure levels
        base_features = cfg.features.output.atmospheric
        level_index = cfg.features.pressure_levels.index(level)
        num_levels = len(cfg.features.pressure_levels)
        base_feature_index = base_features.index(feature)
        feature_index = base_feature_index * num_levels + level_index
        feature_name = f"{feature}_h{level}"
    else:
        # For surface variables
        feature_index = dataset.dyn_output_features.index(feature)
        feature_name = feature

    latitude = dataset.lat
    longitude = dataset.lon
    longitude, latitude = numpy.meshgrid(longitude, latitude)

    # Get the forecast data
    output_plot = output_data[feature_index]
    true_plot = true_data[feature_index]

    # Configure plot settings based on variable type
    if feature == "geopotential":
        g = 9.80665  # gravitational acceleration
        output_plot = output_plot / g
        true_plot = true_plot / g
        cmap = "viridis"
        vmax = numpy.max([numpy.max(output_plot), numpy.max(true_plot)])
        vmin = numpy.min([numpy.min(output_plot), numpy.min(true_plot)])
        levels = numpy.linspace(vmin, vmax, 50)
        clabel = "Geopotential Height [m]"

    elif feature == "2m_temperature":
        cmap = "RdYlBu_r"
        vmax = numpy.max([numpy.max(output_plot), numpy.max(true_plot)])
        vmin = numpy.min([numpy.min(output_plot), numpy.min(true_plot)])
        levels = numpy.linspace(vmin, vmax, 100)
        clabel = "Temperature [°C]"

    elif feature == "total_precipitation_6hr":
        cmap = "Blues"
        max_precip = max(numpy.max(output_plot), numpy.max(true_plot))

        # Create exponentially spaced levels for precipitation
        # This ensures levels are strictly increasing and capture the range of values
        if max_precip > 0:
            # Use exponential spacing to focus on smaller values
            levels = (
                numpy.exp(
                    numpy.linspace(numpy.log(0.1), numpy.log(max_precip + 0.1), 50)
                )
                - 0.1
            )
            # Remove any negative values that might occur due to floating point arithmetic
            levels = levels[levels >= 0]
        else:
            levels = numpy.linspace(0, 0.1, 10)  # Fallback for no precipitation
        clabel = "Precipitation [mm/6h]"

    else:
        cmap = "RdYlBu_r"
        vmax = numpy.max([numpy.max(output_plot), numpy.max(true_plot)])
        vmin = numpy.min([numpy.min(output_plot), numpy.min(true_plot)])
        levels = numpy.linspace(vmin, vmax, 100)
        clabel = feature.replace("_", " ").title()

    # Create figure and axes
    fig, ax = plt.subplots(ncols=2, figsize=(12, 5))

    # Plot contours
    for i, data in enumerate([output_plot, true_plot]):
        if feature == "total_precipitation_6hr":
            contours = ax[i].contourf(
                longitude, latitude, data, levels=levels, cmap=cmap, extend="max"
            )
        else:
            contours = ax[i].contourf(
                longitude, latitude, data, levels=levels, cmap=cmap
            )

        if feature == "geopotential":
            # Add contour lines for geopotential
            contour_levels = levels[::5]  # Take every 5th level
            ax[i].contour(
                longitude,
                latitude,
                data,
                levels=contour_levels,
                colors="k",
                linewidths=0.5,
            )

    # Set titles
    title = f"{feature} at {level} hPa" if level else feature.replace("_", " ").title()
    plt.suptitle(f"{title}\nForecast date: {date_out}\nInput date: {date_in}")
    ax[0].set_title(f"PARADIS")
    ax[1].set_title(f"ERA5")

    # Adjust layout and add colorbar
    plt.tight_layout()
    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(contours, cax=cbar_ax)
    cbar.ax.set_ylabel(clabel, rotation=90)

    # Save figure
    filename = (
        f"{feature}_{level}hPa_prediction.png" if level else f"{feature}_prediction.png"
    )
    plt.savefig(filename, bbox_inches="tight", dpi=300)
    plt.close(plt.gcf())


def denormalize_ground_truth(ground_truth, dataset):
    """Denormalize the ground truth data."""
    ground_truth[:, :, dataset.norm_precip_in] = dataset._denormalize_precipitation(
        torch.from_numpy(ground_truth[:, :, dataset.norm_precip_in])
    ).numpy()

    ground_truth[:, :, dataset.norm_humidity_in] = dataset._denormalize_humidity(
        torch.from_numpy(ground_truth[:, :, dataset.norm_humidity_in])
    ).numpy()
    ground_truth[:, :, dataset.norm_zscore_in] = dataset._denormalize_standard(
        torch.from_numpy(ground_truth[:, :, dataset.norm_zscore_in]),
        dataset.input_mean.view(-1, 1, 1),
        dataset.input_std.view(-1, 1, 1),
    ).numpy()


def denormalize_forecast(output_forecast, dataset):
    """Denormalize the forecast data."""
    output_forecast[:, :, dataset.norm_precip_out] = dataset._denormalize_precipitation(
        torch.from_numpy(output_forecast[:, :, dataset.norm_precip_out])
    ).numpy()
    output_forecast[:, :, dataset.norm_humidity_out] = dataset._denormalize_humidity(
        torch.from_numpy(output_forecast[:, :, dataset.norm_humidity_out])
    ).numpy()
    output_forecast[:, :, dataset.norm_zscore_out] = dataset._denormalize_standard(
        torch.from_numpy(output_forecast[:, :, dataset.norm_zscore_out]),
        dataset.output_mean.view(-1, 1, 1),
        dataset.output_std.view(-1, 1, 1),
    ).numpy()


def save_results_to_zarr(
    data,
    atmospheric_vars,
    surface_vars,
    constant_vars,
    datamodule,
    pressure_levels,
    filename,
):
    """Save results to a Zarr file."""
    data_vars = {}
    dataset = datamodule.dataset
    num_levels = len(pressure_levels)

    # Prepare atmospheric variables
    atm_dims = ["time", "prediction_timedelta", "level", "latitude", "longitude"]
    for i, feature in enumerate(atmospheric_vars):
        data_vars[feature] = (
            atm_dims,
            data[:, :, i * num_levels : (i + 1) * num_levels],
        )

    # Prepare surface variables
    sur_dims = ["time", "prediction_timedelta", "latitude", "longitude"]
    for i, feature in enumerate(surface_vars):
        data_vars[feature] = (
            sur_dims,
            data[:, :, len(atmospheric_vars) * num_levels + i],
        )

    # Prepare constant variables
    con_dims = ["latitude", "longitude"]
    for i, feature in enumerate(constant_vars):
        if feature in con_dims:
            continue
        data_vars[feature] = (con_dims, dataset.ds_constants[feature].data)

    # Define coordinates
    coords = {
        "latitude": dataset.lat,
        "longitude": dataset.lon,
        "time": dataset.time[: data.shape[0]],
        "level": pressure_levels,
        "prediction_timedelta": (numpy.arange(data.shape[1]) + 1)
        * numpy.timedelta64(6 * 3600 * 10**9, "ns"),
    }

    # Save to Zarr
    xarray.Dataset(data_vars=data_vars, coords=coords).to_zarr(
        filename, consolidated=True, mode="w"
    )


@hydra.main(version_base=None, config_path="config/", config_name="forecast_settings")
def main(cfg: DictConfig):
    """Generate forecasts using a trained model."""

    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator == "gpu"
        else "cpu"
    )

    # Decide whether to save results to file
    save_results_to_file = False

    # Initialize data module
    datamodule = Era5DataModule(cfg)
    datamodule.setup(stage="test")
    dataset = datamodule.dataset

    # Extract features and dimensions
    atmospheric_vars = cfg.features.output.atmospheric
    surface_vars = cfg.features.output.surface
    constant_vars = cfg.features.input.constants
    pressure_levels = cfg.features.pressure_levels

    num_levels = len(pressure_levels)
    num_atm_features = len(atmospheric_vars) * num_levels
    num_sur_features = len(surface_vars)
    num_features = num_atm_features + num_sur_features
    num_time_instances = len(dataset)
    num_forecast_steps = cfg.model.forecast_steps

    # Load model
    litmodel = LitParadis(datamodule, cfg)
    if cfg.model.checkpoint_path:
        checkpoint = torch.load(cfg.model.checkpoint_path, weights_only=True)
        litmodel.load_state_dict(checkpoint["state_dict"])
    else:
        raise ValueError(
            "checkpoint_path must be specified in the config for forecasting"
        )

    litmodel.to(device).eval()

    # Initialize forecast and ground truth arrays
    output_forecast = numpy.empty(
        (
            num_time_instances,
            num_forecast_steps,
            num_features,
            dataset.lat_size,
            dataset.lon_size,
        )
    )
    ground_truth = numpy.empty_like(output_forecast)

    # Run forecast
    logging.info("Generating forecast...")
    with torch.no_grad():
        start_ind = 0
        for input_data, true_data in datamodule.test_dataloader():
            batch_size = input_data.shape[0]
            input_data_step = input_data[:, 0].to(device)

            for step in range(num_forecast_steps):
                output_data = litmodel(
                    input_data_step, torch.tensor(step, device=device)
                )

                if step + 1 < num_forecast_steps:
                    input_data_step = litmodel._autoregression_input_from_output(
                        input_data[:, step + 1], output_data
                    ).to(device)

                # Copy model output into global array results
                output_forecast[start_ind : start_ind + batch_size, step] = (
                    output_data.cpu().numpy()
                )
                ground_truth[start_ind : start_ind + batch_size, step] = (
                    true_data[:, step].cpu().numpy()
                )

            start_ind += batch_size

    logging.info("Undoing normalizations...")

    # Denormalize data
    denormalize_ground_truth(ground_truth, dataset)
    denormalize_forecast(output_forecast, dataset)

    # # Save results

    if save_results_to_file:
        save_results_to_zarr(
            output_forecast,
            atmospheric_vars,
            surface_vars,
            constant_vars,
            datamodule,
            pressure_levels,
            "forecast_result.zarr",
        )

        # Save ground truth
        save_results_to_zarr(
            ground_truth,
            atmospheric_vars,
            surface_vars,
            constant_vars,
            datamodule,
            pressure_levels,
            "forecast_observation.zarr",
        )

        logging.info("Saved forecast files successfuly")

    # Also plot results
    time_ind = 0  # First time instance
    forecast_ind = num_forecast_steps - 1  # Last in config file

    time_in = dataset.ds_input.time.values[time_ind]
    time_out = dataset.ds_input.time.values[time_ind + num_forecast_steps]

    dt_in = time_in.astype("datetime64[s]").astype(datetime)
    dt_out = time_out.astype("datetime64[s]").astype(datetime)
    date_in = dt_in.strftime("%Y-%m-%d %H:%M")
    date_out = dt_out.strftime("%Y-%m-%d %H:%M")

    output_data = output_forecast[time_ind, forecast_ind]
    true_data = ground_truth[time_ind, forecast_ind]

    # Generate plots for different variables
    logging.info("Generating forecast plots...")

    # Plot geopotential at 500 hPa
    plot_forecast_map(
        date_in,
        date_out,
        output_data,
        true_data,
        datamodule,
        "geopotential",
        cfg,
        level=500,
    )

    # Plot 2m temperature with Celsius conversion
    plot_forecast_map(
        date_in,
        date_out,
        output_data,
        true_data,
        datamodule,
        "2m_temperature",
        cfg,
        temp_offset=273.15,
    )

    # Plot precipitation
    plot_forecast_map(
        date_in,
        date_out,
        output_data,
        true_data,
        datamodule,
        "total_precipitation_6hr",
        cfg,
    )

    logging.info("Forecast plots generated successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
