"""Forecast script for the Paradis model."""

import logging

import hydra
import torch
import lightning as L
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig
import xarray as xr

from trainer import LitParadis
from data.datamodule import Era5DataModule


def plot_forecast(output_data, true_data, datamodule, feature):
    scaling_factors = datamodule.dataset.scaling_factors

    feature_index = list(datamodule.dataset.dyn_output_features).index(feature)

    latitude = datamodule.dataset.lat
    longitude = datamodule.dataset.lon

    longitude, latitude = np.meshgrid(longitude, latitude)

    output_plot = output_data[feature_index] * scaling_factors[feature] - 273.15
    true_plot = true_data[feature_index] * scaling_factors[feature] - 273.15

    # Determine the colormap ranges
    vmax = np.max([np.max(true_plot), np.max(true_plot)])
    vmin = np.min([np.min(true_plot), np.min(true_plot)])

    fig, ax = plt.subplots(ncols=2, figsize=(10, 4))

    levels = ax[0].contourf(
        longitude, latitude, output_plot, 100, vmin=vmin, vmax=vmax, cmap="RdYlBu_r"
    )
    levels = ax[1].contourf(
        longitude, latitude, true_plot, 100, vmin=vmin, vmax=vmax, cmap="RdYlBu_r"
    )

    plt.suptitle("2m Temperature")
    ax[0].set_title("GATmosphere")
    ax[1].set_title("ERA5")

    plt.tight_layout()

    fig.subplots_adjust(right=0.8)

    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    cbar = fig.colorbar(levels, cax=cbar_ax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("Temperature [°C]", rotation=90)

    plt.savefig("2m_temperature_prediction.png")
    plt.close(plt.gcf())


# pylint: disable=E1120
@hydra.main(version_base=None, config_path="config/", config_name="forecast")
def main(cfg: DictConfig):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate data module
    datamodule = Era5DataModule(cfg)
    datamodule.setup(stage="fit")  # Ensure data module is set up

    # Initialize the model
    litmodel = LitParadis(datamodule, cfg)

    # Load the model weights
    checkpoint = torch.load(cfg.model.checkpoint_path, weights_only=True)
    litmodel.load_state_dict(checkpoint["state_dict"])

    input_data, true_data = datamodule.train_dataset[0]

    # Make dimensions compatible
    input_data = input_data.unsqueeze(0)

    with torch.no_grad():
        input_data_step = input_data[:, 0]
        for step in range(cfg.model.forecast_steps):
            # Call the model
            output_data = litmodel(input_data_step, torch.tensor(step, device=device))

            # Use output dataset as input for next forecasting step
            if step + 1 < cfg.model.forecast_steps:
                input_data_step = litmodel._autoregression_input_from_output(
                    input_data[:, step + 1], output_data
                )

    output_data = output_data.squeeze(0).detach().numpy()
    true_data = true_data.detach().numpy()[cfg.model.forecast_steps - 1]

    # Generate plots
    feature = "2m_temperature"

    plot_forecast(output_data, true_data, datamodule, feature)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
