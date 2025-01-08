"""Forecast script for the model."""

import logging
import hydra
import torch
import matplotlib.pyplot as plt
import numpy
from omegaconf import DictConfig
from trainer import LitParadis
from data.datamodule import Era5DataModule


def plot_forecast_map(
    output_data, true_data, datamodule, feature, cfg, level=None, temp_offset=0
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

    # Get the raw data
    output_plot = output_data[feature_index]
    true_plot = true_data[feature_index]

    # Apply inverse transformations
    if feature == "specific_humidity":
        output_plot = dataset._denormalize_humidity(torch.tensor(output_plot)).numpy()
        true_plot = dataset._denormalize_humidity(torch.tensor(true_plot)).numpy()
    elif feature == "total_precipitation_6hr":
        output_plot = dataset._denormalize_precipitation(
            torch.tensor(output_plot)
        ).numpy()
        true_plot = dataset._denormalize_precipitation(torch.tensor(true_plot)).numpy()
    elif feature_name in dataset.var_stats:
        # Apply z-score denormalization using stored statistics
        stats = dataset.var_stats[feature_name]
        output_plot = output_plot * stats["std"] + stats["mean"] - temp_offset
        true_plot = true_plot * stats["std"] + stats["mean"] - temp_offset
    else:
        raise ValueError("Unkown feature")

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
    plt.suptitle(title)
    ax[0].set_title("PARADIS")
    ax[1].set_title("ERA5")

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


@hydra.main(version_base=None, config_path="config/", config_name="paradis_settings")
def main(cfg: DictConfig):
    """Generate forecasts using a trained model."""
    # Set device based on configuration
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.trainer.accelerator == "gpu"
        else "cpu"
    )

    # Initialize data module
    datamodule = Era5DataModule(cfg)
    datamodule.setup(stage="fit")

    # Initialize and load model
    litmodel = LitParadis(datamodule, cfg)
    if cfg.model.checkpoint_path:
        checkpoint = torch.load(cfg.model.checkpoint_path, weights_only=True)
        litmodel.load_state_dict(checkpoint["state_dict"])
    else:
        raise ValueError(
            "checkpoint_path must be specified in the config for forecasting"
        )

    litmodel = litmodel.to(device)

    # Get a sample from the dataset
    input_data, true_data = datamodule.train_dataset[0]
    input_data = input_data.unsqueeze(0).to(device)

    # Run forecast
    with torch.no_grad():
        input_data_step = input_data[:, 0]
        for step in range(cfg.model.forecast_steps):
            output_data = litmodel(input_data_step, torch.tensor(step, device=device))
            if step + 1 < cfg.model.forecast_steps:
                input_data_step = litmodel._autoregression_input_from_output(
                    input_data[:, step + 1], output_data
                )

    # Move data to CPU for plotting
    output_data = output_data.squeeze(0).cpu().numpy()
    true_data = true_data[cfg.model.forecast_steps - 1].cpu().numpy()

    # Generate plots for different variables
    logging.info("Generating forecast plots...")

    # Plot geopotential at 500 hPa
    plot_forecast_map(
        output_data, true_data, datamodule, "geopotential", cfg, level=500
    )

    # Plot 2m temperature with Celsius conversion
    plot_forecast_map(
        output_data, true_data, datamodule, "2m_temperature", cfg, temp_offset=273.15
    )

    # Plot precipitation
    plot_forecast_map(
        output_data, true_data, datamodule, "total_precipitation_6hr", cfg
    )

    logging.info("Forecast plots generated successfully")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
