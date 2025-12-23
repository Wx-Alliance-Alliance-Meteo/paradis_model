"""Forecast script for the model."""

import sys
from datetime import datetime
import logging

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
from tqdm import tqdm
import numpy

from trainer import LitParadis
from data.datamodule import Era5DataModule
from utils.file_output import save_results_to_zarr
from utils.postprocessing import (
    denormalize_datasets,
    convert_cartesian_to_spherical_winds,
    replace_variable_name,
)
from utils.visualization import plot_forecast_map


def main():
    """Generate forecasts using a trained model.
    Usage: python forecast.py path/to/config_file.yaml
    """

    cfg = OmegaConf.load(sys.argv[1])

    """
    Core forecast execution logic.

    Args:
        cfg: Fully configured DictConfig with all necessary parameters set
    """
    # Set device
    device = torch.device(
        "cuda"
        if torch.cuda.is_available() and cfg.compute.accelerator == "gpu"
        else "cpu"
    )

    # Initialize data module
    datamodule = Era5DataModule(cfg)
    datamodule.setup(stage="predict")
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
    num_forecast_steps = cfg.model.forecast_steps

    # Get the output number of forecast steps based on the output frequency
    output_frequency = cfg.forecast.output_frequency
    output_num_forecast_steps = max(1, num_forecast_steps // output_frequency)

    output_features = list(dataset.dyn_output_features)

    # Load model
    litmodel = LitParadis(datamodule, cfg)
    if not cfg.init.checkpoint_path:
        raise ValueError(
            "checkpoint_path must be specified in the config for forecasting"
        )

    litmodel.to(device).eval()

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

    # Compute initialization times from dataset
    init_times = dataset.time

    logging.info(f"Number of forecasts to generate: {len(init_times)}")

    # Run forecast
    logging.info("Generating forecast...")
    ind = 0
    with torch.inference_mode(), torch.no_grad():
        time_start_ind = 0
        for input_data, ground_truth in tqdm(
            datamodule.predict_dataloader()
        ):

            batch_size = input_data.shape[0]

            output_forecast = torch.empty(
                (
                    batch_size,
                    output_num_forecast_steps,
                    num_features,
                    dataset.lat_size,
                    dataset.lon_size,
                ),
                device=device,
            )

            frequency_counter = 0

            for step in range(num_forecast_steps):
                output_data = litmodel(
                    input_data[:, step].to(device),
                )

                input_data = litmodel._autoregression_input_from_output(
                    input_data, output_data, step, num_forecast_steps
                )

                # Store only at required frequency
                if step % cfg.forecast.output_frequency == 0:
                    output_forecast[:, frequency_counter] = output_data
                    frequency_counter += 1

            # Transfer output to cpu
            output_forecast = output_forecast.cpu()

            # Remove normalizations
            denormalize_datasets(ground_truth, output_forecast, dataset)

            # Convert from pytorch tensor to numpy array
            output_forecast = output_forecast.numpy()

            # Post-process cartesian winds to spherical
            convert_cartesian_to_spherical_winds(
                dataset.lat, dataset.lon, cfg, ground_truth, output_features
            )

            convert_cartesian_to_spherical_winds(
                dataset.lat, dataset.lon, cfg, output_forecast, output_features
            )

            # Save results
            if cfg.forecast.output_file is not None:
                save_results_to_zarr(
                    output_forecast,
                    dataset.ds_loader,
                    atmospheric_vars,
                    surface_vars,
                    constant_vars,
                    dataset,
                    pressure_levels,
                    cfg.forecast.output_file,
                    ind,
                    init_times[time_start_ind : time_start_ind + batch_size],
                )

            ind += 1
            time_start_ind += batch_size

    logging.info("Saved output files successfuly")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
