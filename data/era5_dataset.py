"""ERA5 dataset handling"""

from datetime import datetime, timedelta
import os
import re

import dask
import numpy
from omegaconf import DictConfig
import torch
import xarray


from data.forcings import time_forcings, toa_radiation


class ERA5Dataset(torch.utils.data.Dataset):
    """Prepare and process ERA5 dataset for Pytorch."""

    def __init__(
        self,
        root_dir: str,
        start_date: str,
        end_date: str,
        forecast_steps: int = 1,
        dtype=torch.float32,
        cfg: DictConfig = {},
    ) -> None:

        features_cfg = cfg.features
        self.eps = 1e-12
        self.root_dir = root_dir
        self.forecast_steps = forecast_steps
        self.dtype = dtype
        self.forcing_inputs = features_cfg.input.forcings

        # Lazy open this dataset
        ds = xarray.open_mfdataset(
            os.path.join(root_dir, "*"),
            chunks={"time": self.forecast_steps + 1},
            engine="zarr",
        )

        # Add stats to data array
        ds_stats = xarray.open_dataset(
            os.path.join(self.root_dir, "stats"), engine="zarr"
        )

        # Store them in main dataset for easier processing
        ds["mean"] = ds_stats["mean"]
        ds["std"] = ds_stats["std"]
        ds["max"] = ds_stats["max"]
        # Store statistics for each variable (for use in forecast.py)
        self.var_stats = {}
        for i, feature in enumerate(ds_stats.features.values):
            self.var_stats[feature] = {
                "mean": float(ds_stats["mean"].values[i]),
                "std": float(ds_stats["std"].values[i]),
            }

        ds["min"] = ds_stats["min"]
        ds.attrs["toa_radiation_std"] = ds_stats.attrs["toa_radiation_std"]
        ds.attrs["toa_radiation_mean"] = ds_stats.attrs["toa_radiation_mean"]

        # Add the number of forecast steps to the range of dates
        time_resolution = int(cfg.dataset.time_resolution[:-1])

        # Number time instances per day (ignoring 24:00)
        num_times_day = 24 / time_resolution - 1

        # Get the number of additional hours needed in data for autoregression
        hours = time_resolution * (self.forecast_steps + num_times_day)
        time_delta = timedelta(hours=hours)

        # Convert end_date to a datetime object and adjust end date
        end_date_dt = datetime.strptime(end_date, "%Y-%m-%d")
        adjusted_end_date = end_date_dt + time_delta

        # Select the time range needed to process this dataset
        ds = ds.sel(time=slice(start_date, adjusted_end_date))

        # Extract latitude and longitude to build the graph
        self.lat = ds.latitude.values
        self.lon = ds.longitude.values
        self.lat_size = len(self.lat)
        self.lon_size = len(self.lon)

        # The number of time instances in the dataset represents its length
        self.time = ds.time.values
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

        self.num_common_features = len(common_features)

        # Constant input variables
        ds_constants = xarray.open_dataset(
            os.path.join(root_dir, "constants"), engine="zarr"
        )

        # Convert lat/lon to radians
        lat_rad = torch.from_numpy(numpy.deg2rad(self.lat)).to(self.dtype)
        lon_rad = torch.from_numpy(numpy.deg2rad(self.lon)).to(self.dtype)
        lat_rad_grid, lon_rad_grid = torch.meshgrid(lat_rad, lon_rad, indexing="ij")

        # Use zscore to normalize the following variables
        normalize_const_vars = {
            "geopotential_at_surface",
            "slope_of_sub_gridscale_orography",
            "standard_deviation_of_orography",
        }

        normalized_constants = []
        for var in features_cfg.input.constants:
            # Skip latitude and longitude, as they are always added in radians
            if var == "latitude" or var == "longitude":
                continue

            # Normalize constants and keep in memory
            if var in normalize_const_vars:
                array = (
                    torch.from_numpy(ds_constants[var].data)
                    - ds_constants[var].attrs["mean"]
                ) / ds_constants[var].attrs["std"]

                normalized_constants.append(array)

        # Get land-sea mask (no normalization needed)
        land_sea_mask = torch.from_numpy(ds_constants["land_sea_mask"].data).to(
            self.dtype
        )

        # Stack all constant features together
        self.constant_data = (
            torch.stack(
                [*normalized_constants, land_sea_mask, lat_rad_grid, lon_rad_grid]
            )
            .permute(1, 2, 0)
            .reshape(self.lat_size, self.lon_size, -1)
            .unsqueeze(0)
            .expand(self.forecast_steps, -1, -1, -1)
        )

        # Store these for access in forecaster
        self.ds_constants = ds_constants

        # Order them so that common features are placed first
        self.dyn_input_features = common_features + list(
            set(input_atmospheric) - set(output_atmospheric)
        )

        self.dyn_output_features = common_features + list(
            set(output_atmospheric) - set(input_atmospheric)
        )

        # Pre-select the features in the right order
        ds_input = ds.sel(features=self.dyn_input_features)
        ds_output = ds.sel(features=self.dyn_output_features)

        # Fetch data
        self.ds_input = ds_input["data"]
        self.ds_output = ds_output["data"]

        # Get the indices to apply custom normalizations
        self._prepare_normalization(ds_input, ds_output)

        # Calculate the final number of input and output features after preparation
        self.num_in_features = (
            len(self.dyn_input_features)
            + self.constant_data.shape[-1]
            + len(self.forcing_inputs)
        )
        self.num_out_features = len(self.dyn_output_features)

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

        # # Add checks for invalid values
        if numpy.isnan(input_data.data).any() or numpy.isnan(true_data.data).any():
            raise ValueError("NaN values detected in input/output data")

        # Convert to tensors - data comes in [time, lat, lon, features]
        x = torch.tensor(input_data.data, dtype=self.dtype)
        y = torch.tensor(true_data.data, dtype=self.dtype)

        # Apply normalizations
        self._apply_normalization(x, y)

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

    def _prepare_normalization(self, ds_input, ds_output):
        """
        Prepare indices and statistics for normalization in a vectorized fashion.

        This method identifies indices for specific types of features
        (e.g., precipitation, humidity, and others) for both input and output
        datasets, converts them into PyTorch tensors, and retrieves
        mean and standard deviation values for z-score normalization.

        Parameters:
            ds_input: xarray.Dataset
                Input dataset containing mean and standard deviation values.
            ds_output: xarray.Dataset
                Output dataset containing mean and standard deviation values.
        """

        # Initialize lists to store indices for each feature type
        self.norm_precip_in = []
        self.norm_humidity_in = []
        self.norm_zscore_in = []

        self.norm_precip_out = []
        self.norm_humidity_out = []
        self.norm_zscore_out = []

        # Process dynamic input features
        for i, feature in enumerate(self.dyn_input_features):
            feature_name = re.sub(
                r"_h\d+$", "", feature
            )  # Remove height suffix (e.g., "_h10")
            if feature_name == "total_precipitation_6hr":
                self.norm_precip_in.append(i)
            elif feature_name == "specific_humidity":
                self.norm_humidity_in.append(i)
            else:
                self.norm_zscore_in.append(i)

        # Process dynamic output features
        for i, feature in enumerate(self.dyn_output_features):
            feature_name = re.sub(
                r"_h\d+$", "", feature
            )  # Remove height suffix (e.g., "_h10")
            if feature_name == "total_precipitation_6hr":
                self.norm_precip_out.append(i)
            elif feature_name == "specific_humidity":
                self.norm_humidity_out.append(i)
            else:
                self.norm_zscore_out.append(i)

        # Convert lists of indices to PyTorch tensors for efficient indexing
        self.norm_precip_in = torch.tensor(self.norm_precip_in, dtype=torch.long)
        self.norm_precip_out = torch.tensor(self.norm_precip_out, dtype=torch.long)
        self.norm_humidity_in = torch.tensor(self.norm_humidity_in, dtype=torch.long)
        self.norm_humidity_out = torch.tensor(self.norm_humidity_out, dtype=torch.long)
        self.norm_zscore_in = torch.tensor(self.norm_zscore_in, dtype=torch.long)
        self.norm_zscore_out = torch.tensor(self.norm_zscore_out, dtype=torch.long)

        # Retrieve mean and standard deviation values for z-score normalization
        self.input_mean = torch.tensor(ds_input["mean"].data, dtype=self.dtype)
        self.input_std = torch.tensor(ds_input["std"].data, dtype=self.dtype)
        self.input_max = torch.tensor(ds_input["max"].data, dtype=self.dtype)
        self.input_min = torch.tensor(ds_input["min"].data, dtype=self.dtype)

        self.output_mean = torch.tensor(ds_output["mean"].data, dtype=self.dtype)
        self.output_std = torch.tensor(ds_output["std"].data, dtype=self.dtype)
        self.output_max = torch.tensor(ds_output["max"].data, dtype=self.dtype)
        self.output_min = torch.tensor(ds_output["min"].data, dtype=self.dtype)

        # Keep only statistics of variables that require standard normalization
        self.input_mean = self.input_mean[self.norm_zscore_in]
        self.input_std = self.input_std[self.norm_zscore_in]
        self.output_mean = self.output_mean[self.norm_zscore_out]
        self.output_std = self.output_std[self.norm_zscore_out]

        # Prepare variables required in custom normalization

        # Maximum and minimum specific humidity in dataset
        self.q_max = torch.max(self.input_max[self.norm_humidity_in]).detach()
        self.q_min = torch.min(self.input_min[self.norm_humidity_in]).detach()

        if self.q_min < self.eps:
            self.q_min = torch.tensor(self.eps).detach()

        # Extract the toa_radiation mean and std
        self.toa_rad_std = ds_input.attrs["toa_radiation_std"]
        self.toa_rad_mean = ds_input.attrs["toa_radiation_mean"]

    def _apply_normalization(self, input_data, output_data):

        # Apply custom normalizations to input
        input_data[..., self.norm_precip_in] = self._normalize_precipitation(
            input_data[..., self.norm_precip_in]
        )
        input_data[..., self.norm_humidity_in] = self._normalize_humidity(
            input_data[..., self.norm_humidity_in]
        )

        # Apply custom normalizations to output
        output_data[..., self.norm_precip_out] = self._normalize_precipitation(
            output_data[..., self.norm_precip_out]
        )
        output_data[..., self.norm_humidity_out] = self._normalize_humidity(
            output_data[..., self.norm_humidity_out]
        )

        # Apply standard normalizations to input and output
        input_data[..., self.norm_zscore_in] = self._normalize_standard(
            input_data[..., self.norm_zscore_in],
            self.input_mean,
            self.input_std,
        )

        output_data[..., self.norm_zscore_out] = self._normalize_standard(
            output_data[..., self.norm_zscore_out], self.output_mean, self.output_std
        )

    def _compute_forcings(self, input_data):
        """Computes forcing paramters based in input_data array"""

        forcings_time_ds = time_forcings(input_data["time"].values)

        forcings = []
        for var in self.forcing_inputs:
            if var == "toa_incident_solar_radiation":
                toa_rad = toa_radiation(input_data["time"].values, self.lat, self.lon)
                toa_rad = torch.tensor(
                    (toa_rad - self.toa_rad_mean) / self.toa_rad_std,
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

    def _normalize_standard(self, input_data, mean, std):
        return (input_data - mean) / std

    def _denormalize_standard(self, norm_data, mean, std):
        return norm_data * std + mean

    def _normalize_humidity(self, data: torch.tensor) -> torch.tensor:
        """Normalize specific humidity using physically-motivated logarithmic transform.

        This normalization accounts for the exponential variation of specific humidity
        with altitude, mapping values from ~10^-5 (upper atmosphere) to ~10^-2 (surface)
        onto a normalized range while preserving relative variations at all scales.

        Args:
            data: Specific humidity data in kg/kg
        Returns:
            Normalized specific humidity data
        """
        # Apply normalization
        q_norm = (
            torch.log(torch.clip(data, 0, self.q_max) + self.eps)
            - torch.log(self.q_min)
        ) / (torch.log(self.q_max) - torch.log(self.q_min))

        return q_norm

    def _denormalize_humidity(self, data: torch.tensor) -> torch.tensor:
        """Denormalize specific humidity data from normalized space back to kg/kg.

        Args:
            data: Normalized specific humidity data
        Returns:
            Specific humidity data in kg/kg
        """

        # Invert the normalization
        q = (
            torch.exp(
                data * (torch.log(self.q_max) - torch.log(self.q_min))
                + torch.log(self.q_min)
            )
            - self.eps
        )
        return torch.clip(q, 0, self.q_max)

    def _normalize_precipitation(self, data: torch.tensor) -> torch.tensor:
        """Normalize precipitation using logarithmic transform.

        Args:
            data: Precipitation data
        Returns:
            Normalized precipitation data
        """
        shift = 10
        return torch.log(data + 1e-6) + shift

    def _denormalize_precipitation(self, data: torch.tensor) -> torch.tensor:
        """Denormalize precipitation data.

        Args:
            data: Normalized precipitation data
        Returns:
            Precipitation data in original scale
        """
        shift = 10
        return torch.clip(torch.exp(data - shift) - 1e-6, min=0, max=None)
