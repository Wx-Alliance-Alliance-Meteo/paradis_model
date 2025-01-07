"""ERA5 dataset handling"""

import dask
import logging
from omegaconf import DictConfig
import os

import numpy
import torch
import xarray as xr

from data.forcings import time_forcings, toa_radiation

TSI = 1361  # Baseline solar irradiance in W/m^2


class ERA5Dataset(torch.utils.data.Dataset):
    """Prepare and process ERA5 dataset for Pytorch."""

    def __init__(
        self,
        root_dir: str,
        start_date: str,
        end_date: str,
        forecast_steps: int = 1,
        dtype=torch.float32,
        features_cfg: DictConfig = {},
    ) -> None:

        self.root_dir = root_dir
        self.forecast_steps = forecast_steps
        self.dtype = dtype
        self.num_common_features = 0
        self.forcing_inputs = features_cfg.input.forcings

        # Dictionary mapping variables to their physical scaling
        self.scaling_factors = {
            "toa_incident_solar_radiation": TSI * 3600,  # in W s/m^2
        }

        # Extract from the years in the range
        start_year = int(start_date.split("-")[0])
        end_year = int(end_date.split("-")[0])

        # Create list of files to open (avoids loading more than necessary)
        files = [
            os.path.join(root_dir, str(year))
            for year in range(start_year, end_year + 1)
        ]

        # Lazy open this dataset
        ds = xr.open_mfdataset(
            files, chunks={"time": self.forecast_steps + 1}, engine="zarr"
        )

        # Add stats to data array
        ds_stats = xr.open_dataset(os.path.join(self.root_dir, "stats"), engine="zarr")

        ds["mean"] = ds_stats["mean"]
        ds["std"] = ds_stats["std"]

        # Lazily pre-normalize atmospheric variables
        ds = (ds["data"] - ds["mean"]) / ds["std"]

        # Filter data to time frame requested
        ds = ds.sel(time=slice(start_date, end_date))

        # Extract latitude and longitude to build the graph
        self.lat = ds.latitude.values
        self.lon = ds.longitude.values
        self.lat_size = len(self.lat)
        self.lon_size = len(self.lon)
        # The number of time instances in the dataset represents its length
        self.length = ds.time.size

        # Store the size of the grid (lat * lon)
        self.grid_size = ds.latitude.size * ds.longitude.size

        # Setup input and output features based on config
        input_atmospheric = [
            variable + f"_h{level}"
            for variable in features_cfg.input.atmospheric
            for level in features_cfg.pressure_levels
        ]

        output_atmospheric = [
            variable + f"_h{level}"
            for variable in features_cfg.output.atmospheric
            for level in features_cfg.pressure_levels
        ]

        # Update feature counts
        common_features = list(
            filter(
                lambda x: x in input_atmospheric + features_cfg["input"]["surface"],
                output_atmospheric + features_cfg["output"]["surface"],
            )
        )

        # Constant input variables
        ds_constants = xr.open_dataset(
            os.path.join(root_dir, "constants"), engine="zarr"
        )

        # Normalize all static variables if desired
        for var in features_cfg.input.constants:
            ds_constants[var] = (
                ds_constants[var] - ds_constants[var].attrs["mean"]
            ) / ds_constants[var].attrs["std"]

        # Constant data will live in CPU memory
        stacked_data = torch.stack(
            [
                torch.from_numpy(ds_constants[var].data)
                for var in features_cfg.input.constants
            ],
            dim=0,
        ).to(self.dtype)

        # Make sure constant data has the right shape for multiple forecast steps
        self.constant_data = (
            stacked_data.permute(1, 2, 0)
            .reshape(self.lat_size, self.lon_size, -1)
            .unsqueeze(0)
            .expand(self.forecast_steps, -1, -1, -1)
        )

        # Order them so that common features are placed first
        self.dyn_input_features = common_features + list(
            set(input_atmospheric) - set(output_atmospheric)
        )

        self.dyn_output_features = common_features + list(
            set(output_atmospheric) - set(input_atmospheric)
        )

        # Pre-select the features in the right order
        self.ds_input = ds.sel(features=self.dyn_input_features)
        self.ds_output = ds.sel(features=self.dyn_output_features)

        # Calculate the final number of input and output features after preparation
        self.num_in_features = (
            len(self.dyn_input_features)
            + self.constant_data.shape[-1]
            + len(self.forcing_inputs)
        )
        self.num_out_features = len(self.dyn_output_features)

        logging.info(
            "Dataset contains: %d input features, %d output features.",
            self.num_in_features,
            self.num_out_features,
        )

    def __len__(self):
        # Do not yield a value for the last time in the dataset since there
        # is no future data
        return self.length - self.forecast_steps

    def __getitem__(self, ind: int):
        # Extract values from the requested indices
        input_data = self.ds_input.isel(time=slice(ind, ind + self.forecast_steps))

        true_data = self.ds_output.isel(
            time=slice(ind + 1, ind + self.forecast_steps + 1)
        )

        # Load arrays into CPU memory
        input_data, true_data = dask.compute(input_data, true_data)

        # Convert to tensors - data comes in [time, lat, lon, features]
        x = torch.tensor(input_data.data, dtype=self.dtype)
        y = torch.tensor(true_data.data, dtype=self.dtype)

        # Compute forcings
        forcings = self._compute_forcings(input_data)

        if forcings is not None:
            x = torch.cat([x, forcings], dim=-1)

        # Add constant data to input
        x = torch.cat([x, self.constant_data], dim=-1)

        # Permute to [time, channels, latitude, longitude] format
        x_grid = x.permute(0, 3, 1, 2)
        y_grid = y.permute(0, 3, 1, 2)

        return x_grid, y_grid

    def _compute_forcings(self, input_data):
        """Computes forcing paramters based in input_data array"""

        forcings_time_ds = time_forcings(input_data["time"].values)

        forcings = []
        for var in self.forcing_inputs:
            if var == "toa_incident_solar_radiation":
                toa_rad = toa_radiation(input_data["time"].values, self.lat, self.lon)
                toa_rad = torch.tensor(
                    toa_rad / self.scaling_factors["toa_incident_solar_radiation"],
                    dtype=self.dtype,
                )
                if len(toa_rad.shape) == 3:  # If time dimension is present
                    toa_rad = toa_rad.permute(1, 2, 0)
                toa_rad = toa_rad.reshape(
                    self.forecast_steps, self.lat_size, self.lon_size, 1
                )
                forcings.append(toa_rad)
            else:
                # Get the time forcings
                if var in forcings_time_ds:
                    var_ds = forcings_time_ds[var]
                    value = (
                        torch.tensor(var_ds.data, dtype=self.dtype)
                        .view(-1, 1, 1, 1)
                        .expand(self.forecast_steps, self.lat_size, self.lon_size, 1)
                    )
                    forcings.append(value)

        if len(forcings) > 0:
            return torch.cat(forcings, dim=-1)
        return

    def _normalize_humidity(self, data: numpy.ndarray) -> numpy.ndarray:
        """Normalize specific humidity using physically-motivated logarithmic transform.

        This normalization accounts for the exponential variation of specific humidity
        with altitude, mapping values from ~10^-5 (upper atmosphere) to ~10^-2 (surface)
        onto a normalized range while preserving relative variations at all scales.

        Args:
            data: Specific humidity data in kg/kg
        Returns:
            Normalized specific humidity data
        """
        q_min = 1e-6  # Minimum specific humidity in kg/kg (stratospheric value)
        q_max = 0.035  # Maximum specific humidity in kg/kg (tropical surface maximum)

        # Add small epsilon to prevent log(0)
        epsilon = 1e-12

        # Apply normalization
        q_norm = (numpy.log(data + epsilon) - numpy.log(q_min)) / (
            numpy.log(q_max) - numpy.log(q_min)
        )

        return q_norm

    def _denormalize_humidity(self, data: numpy.ndarray) -> numpy.ndarray:
        """Denormalize specific humidity data from normalized space back to kg/kg.

        Args:
            data: Normalized specific humidity data
        Returns:
            Specific humidity data in kg/kg
        """
        q_min = 1e-6
        q_max = 0.035
        epsilon = 1e-12

        # Invert the normalization
        q = (
            numpy.exp(data * (numpy.log(q_max) - numpy.log(q_min)) + numpy.log(q_min))
            - epsilon
        )
        return q

    def _normalize_precipitation(self, data: numpy.ndarray) -> numpy.ndarray:
        """Normalize precipitation using logarithmic transform.

        Args:
            data: Precipitation data
        Returns:
            Normalized precipitation data
        """
        shift = 10
        return numpy.log(data + 1e-6) + shift

    def _denormalize_precipitation(self, data: numpy.ndarray) -> numpy.ndarray:
        """Denormalize precipitation data.

        Args:
            data: Normalized precipitation data
        Returns:
            Precipitation data in original scale
        """
        shift = 10
        return numpy.clip(numpy.exp(data) - 1e-6, a_min=0, a_max=None)


def split_dataset(dataset, train_ratio=0.8):
    """Split dataset into training and validation sets.

    Args:
        dataset: Dataset to split
        train_ratio: Fraction of data to use for training

    Returns:
        tuple: (train_dataset, val_dataset)
    """
    split_idx = int(len(dataset) * train_ratio)
    train_dataset = torch.utils.data.Subset(dataset, range(0, split_idx))
    val_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))
    return train_dataset, val_dataset
