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
    preprocess_variable_names,
)
from utils.visualization import plot_forecast_map


@hydra.main(version_base=None, config_path="config/", config_name="paradis_settings")
def main(cfg: DictConfig):
    """Generate forecasts using a trained model."""

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
    if cfg.init.checkpoint_path:
        checkpoint = torch.load(
            cfg.init.checkpoint_path, weights_only=True, map_location="cpu"
        )
        litmodel.load_state_dict(checkpoint["state_dict"])
    else:
        raise ValueError(
            "checkpoint_path must be specified in the config for forecasting"
        )

    litmodel.to(device).eval()

    # Modify cartesian feature names to their spherical counterparts
    preprocess_variable_names(atmospheric_vars, surface_vars)

    # Run forecast
    logging.info("Generating forecast...")
    ind = 0
    with torch.inference_mode(), torch.no_grad():
        time_start_ind = cfg.dataset.n_time_inputs - 1
        for input_data, ground_truth in tqdm(datamodule.predict_dataloader()):

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

            input_data_step = input_data[:, 0].to(device)

            frequency_counter = 0
            for step in range(num_forecast_steps):
                output_data = litmodel(input_data_step)

                if step + 1 < num_forecast_steps:
                    input_data_step = litmodel._autoregression_input_from_output(
                        input_data[:, step + 1], output_data
                    ).to(device)

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
                dataset.lat, dataset.lon, cfg, output_forecast, output_features
            )

            # Save results
            if cfg.forecast.output_file is not None:
                save_results_to_zarr(
                    output_forecast,
                    atmospheric_vars,
                    surface_vars,
                    constant_vars,
                    dataset,
                    pressure_levels,
                    cfg.forecast.output_file,
                    ind,
                    time_start_ind,
                    time_start_ind + batch_size,
                )

            ind += 1
            time_start_ind += batch_size

            # Generate plots
            if not cfg.forecast.generate_plots:
                continue

            # Plot results for the first time instance only
            time_ind = 0

            if time_ind != ind - 1:
                continue

            # Prepare ground truth for plots
            ground_truth = ground_truth.numpy()

            # Make sure ground truth has the same frequency as output_forecast
            ground_truth = ground_truth[:, :: cfg.forecast.output_frequency]

            convert_cartesian_to_spherical_winds(
                dataset.lat, dataset.lon, cfg, ground_truth, output_features
            )

            # Generate a plot for each forecast step
            for forecast_ind in range(output_num_forecast_steps):

                # Generate a string for the input and output times
                time_in = dataset.ds_input.time.values[time_ind]
                time_out = dataset.ds_input.time.values[
                    time_ind + forecast_ind * output_frequency + 1
                ]
                dt_in = time_in.astype("datetime64[s]").astype(datetime)
                dt_out = time_out.astype("datetime64[s]").astype(datetime)
                date_in = dt_in.strftime("%Y-%m-%d %H:%M")
                date_out = dt_out.strftime("%Y-%m-%d %H:%M")

                # Extract data for this forecast step and time index
                output_data = output_forecast[time_ind, forecast_ind]
                true_data = ground_truth[time_ind, forecast_ind]

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
