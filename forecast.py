"""Forecast script for the model."""

from datetime import datetime
import logging

import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm

from trainer import LitParadis
from data.datamodule import Era5DataModule
from utils.file_output import save_results_to_zarr
from utils.postprocessing import (
    denormalize_datasets,
    convert_cartesian_to_spherical_winds,
    replace_variable_name,
)
from utils.visualization import plot_error_map, plot_forecast_map


@hydra.main(version_base=None, config_path="config/", config_name="paradis_settings")
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
    dataset = datamodule.test_dataset

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
    output_features = list(dataset.dyn_output_features)

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

    # Run forecast
    logging.info("Generating forecast...")
    ind = 0
    with torch.no_grad():
        for input_data, ground_truth in tqdm(datamodule.test_dataloader()):
            batch_size = input_data.shape[0]
            output_forecast = torch.empty(
                (
                    batch_size,
                    num_forecast_steps,
                    num_features,
                    dataset.lat_size,
                    dataset.lon_size,
                ),
                device=device,
            )

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
                output_forecast[:, step] = output_data

            # Transfer output to cpu
            output_forecast = output_forecast.cpu()

            # Remove normalizations
            denormalize_datasets(ground_truth, output_forecast, dataset)

            # Convert to numpy arrays
            ground_truth = ground_truth.numpy()
            output_forecast = output_forecast.numpy()

            # Post-process cartesian winds to spherical
            convert_cartesian_to_spherical_winds(
                dataset.lat, dataset.lon, cfg, output_forecast, output_features
            )
            convert_cartesian_to_spherical_winds(
                dataset.lat, dataset.lon, cfg, ground_truth, output_features
            )

            # Save results
            if save_results_to_file:
                save_results_to_zarr(
                    output_forecast,
                    atmospheric_vars,
                    surface_vars,
                    constant_vars,
                    dataset,
                    pressure_levels,
                    "results/forecast_result.zarr",
                    ind,
                )

                # Save ground truth
                save_results_to_zarr(
                    ground_truth,
                    atmospheric_vars,
                    surface_vars,
                    constant_vars,
                    dataset,
                    pressure_levels,
                    "results/forecast_observation.zarr",
                    ind,
                )

            ind += 1

            # Plot results for the first time instance only
            time_ind = 0
            forecast_ind = num_forecast_steps - 1

            if time_ind == ind - 1:
                time_in = dataset.ds_input.time.values[time_ind]
                time_out = dataset.ds_input.time.values[time_ind + forecast_ind]

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
                    dataset,
                    "geopotential",
                    cfg,
                    level=500,
                    ind=forecast_ind,
                )

                # Plot 2m temperature with Celsius conversion
                plot_forecast_map(
                    date_in,
                    date_out,
                    output_data,
                    true_data,
                    dataset,
                    "2m_temperature",
                    cfg,
                    temp_offset=273.15,
                    ind=forecast_ind,
                )

                # Plot precipitations
                plot_forecast_map(
                    date_in,
                    date_out,
                    output_data,
                    true_data,
                    dataset,
                    "total_precipitation_6hr",
                    cfg,
                    ind=forecast_ind,
                )

                logging.info("Forecast plots generated successfully")

    logging.info("Saved output files successfuly")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
