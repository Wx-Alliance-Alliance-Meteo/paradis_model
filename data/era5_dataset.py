"""ERA5 dataset handling"""

from datetime import timedelta
import os
import re

import dask
import numpy
from omegaconf import DictConfig
import torch
import xarray


from data.forcings import time_forcings, toa_radiation
from utils.normalization import (
    normalize_standard,
    normalize_humidity,
    normalize_precipitation,
)


class ERA5Dataset(torch.utils.data.Dataset):
    """Prepare and process ERA5 dataset for Pytorch."""

    def __init__(
        self,
        root_dir: str,
        start_date: str,
        end_date: str,
        forecast_steps: int,
        dtype: torch.dtype = torch.float32,
        preload: bool = False,  # Whether to preload the dataset
        cfg: DictConfig = DictConfig({}),
        time_interval: str = None,
    ) -> None:

        self.cfg = cfg
        features_cfg = cfg.features
        self.preload = preload
        self.eps = 1e-12
        self.root_dir = root_dir
        self.forecast_steps = forecast_steps
        self.dtype = dtype
        self.forcing_inputs = features_cfg.input.forcings
        self.concat_input = cfg.dataset.n_time_inputs > 1
        self.n_time_inputs = cfg.dataset.n_time_inputs if self.concat_input else 1
        self.custom_normalization = not cfg.normalization.standard

        # Lazy open this dataset
        ds = xarray.open_mfdataset(
            os.path.join(root_dir, "*"),
            chunks={"time": 1},
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

        # Make sure start date and end_date provide the time, othersize asume 0Z and 24Z respectively
        if "T" not in start_date:
            start_date += "T00:00:00"

        # Add the number of forecast steps to the range of dates
        time_resolution = int(cfg.dataset.time_resolution[:-1])

        # Apply an initialization time interval if necessary
        if time_interval is None:
            time_interval = time_resolution
        else:
            time_interval = int(time_interval[:-1])

        self.interval_steps = time_interval // time_resolution

        # Get the number of additional time instances needed in data for autoregression
        hours = time_resolution * self.forecast_steps
        time_delta = timedelta(hours=hours)
        time_delta = numpy.timedelta64(int(time_delta.total_seconds()), "s")

        start_date_dt = numpy.datetime64(start_date, "s")
        step = numpy.timedelta64(6, "h")
        adjusted_start_date = start_date_dt - (self.n_time_inputs - 1) * step

        # Convert end_date to a datetime object and adjust end date
        if end_date is not None:

            if "T" not in end_date:
                end_date += "T23:59:59"

            end_date_dt = numpy.datetime64(end_date)
            adjusted_end_date = end_date_dt + time_delta * self.interval_steps
        else:
            start_date_dt = numpy.datetime64(start_date)
            adjusted_end_date = start_date_dt + time_delta * self.interval_steps

        # Select the time range needed to process this dataset
        ds = ds.sel(time=slice(adjusted_start_date, adjusted_end_date))

        # Extract latitude and longitude to build the graph
        self.lat = torch.from_numpy(ds.latitude.values)
        self.lon = torch.from_numpy(ds.longitude.values)
        self.lat_size = len(self.lat)
        self.lon_size = len(self.lon)
        self.pressure_levels = features_cfg.pressure_levels

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
        ).compute()  # Definitely preload constants

        # Convert lat/lon to radians
        lat_rad = torch.deg2rad(self.lat).to(self.dtype)
        lon_rad = torch.deg2rad(self.lon).to(self.dtype)
        self.lat_rad_grid, self.lon_rad_grid = torch.meshgrid(
            lat_rad, lon_rad, indexing="ij"
        )

        # Use zscore to normalize the following variables
        normalize_const_vars = {
            "geopotential_at_surface",
            "slope_of_sub_gridscale_orography",
            "standard_deviation_of_orography",
        }

        # Extract pre-processed constants that require normalization
        pre_constants = []
        for var in features_cfg.input.constants:

            # Normalize constants and keep in memory
            if var in normalize_const_vars:
                array = (
                    torch.from_numpy(ds_constants[var].data)
                    - ds_constants[var].attrs["mean"]
                ) / ds_constants[var].attrs["std"]

                pre_constants.append(array)

        # Get land-sea mask (no normalization needed)
        pre_constants.append(
            torch.from_numpy(ds_constants["land_sea_mask"].data).to(self.dtype)
        )

        # Include the distance variation of two longitude points along the latitude direction
        self._compute_geometric_constants()

        post_constants = []
        for feature in ["lon_spacing", "latitude", "longitude"]:
            if feature in features_cfg.input.constants:
                if feature == "lon_spacing":
                    post_constants.append(self.d_lon_inv)
                if feature == "latitude":
                    post_constants.append(self.lat_rad_grid)
                if feature == "longitude":
                    post_constants.append(self.lon_rad_grid)

        # Stack all constant features together
        self.constant_data = (
            torch.stack([*pre_constants, *post_constants])
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

        # Store the number of dynamic features without concatenation
        self.num_dyn_inputs_single = len(self.dyn_input_features)

        ds_input = ds.sel(features=self.dyn_input_features)
        ds_output = ds.sel(features=self.dyn_output_features)

        # Pre-select the features in the right order
        if self.preload:
            ds_input = ds_input.compute()
            ds_output = ds_output.compute()

        # Concatenate dynamic input features as many times as needed
        if self.concat_input:
            self.dyn_input_features *= self.n_time_inputs

        # Fetch data
        self.ds_input = ds_input["data"]
        self.ds_output = ds_output["data"]

        # Get the indices to apply custom normalizations
        self._prepare_normalization(ds_input, ds_output)

        # Calculate the final number of input and output features after preparation
        # Number of dynamic inputs
        self.num_in_dyn_features = (
            len(self.dyn_input_features) + len(self.forcing_inputs) * self.n_time_inputs
        )

        # Number of static features
        self.num_in_static_features = self.constant_data.shape[-1]

        # Number of total inputs
        self.num_in_features = self.num_in_dyn_features + self.num_in_static_features

        # Number of total outputs
        self.num_out_features = len(self.dyn_output_features)

        # Ensure dataset configuration is well-aligned with requirements
        self._run_dataset_checks()

        # Store the mean and standard deviation of quantities to be reported
        if not cfg.forecast.enable and cfg.training.reports.enable:
            report_features = ds_output.sel(features=cfg.training.reports.features)
            self.report_stats = {
                "mean": report_features["mean"].values,
                "std": report_features["std"].values,
            }

    def __len__(self):
        # Do not yield a value for the last time in the dataset since there
        # is no future data
        return (
            self.length - self.forecast_steps - (self.n_time_inputs - 1)
        ) // self.interval_steps

    def __getitem__(self, ind: int):
        ind = ind * self.interval_steps

        # Retrieve the current value of forecast steps
        steps = self.forecast_steps

        # Extract values from the requested indices
        input_ini = ind
        input_end = input_ini + steps + self.n_time_inputs - 1
        input_data = self.ds_input.isel(time=slice(input_ini, input_end))

        output_ini = input_ini + self.n_time_inputs
        output_end = ind + steps + self.n_time_inputs
        true_data = self.ds_output.isel(time=slice(output_ini, output_end))

        # Load arrays into CPU memory
        input_data, true_data = dask.compute(input_data, true_data)  # type: ignore -- dask.compute is really dask.base.compute

        # Convert to tensors - data comes in [time, lat, lon, features]
        x = torch.tensor(input_data.data, dtype=self.dtype)

        # Concatenate n_time_inputs if requested
        if self.concat_input:
            x = torch.stack(
                [
                    torch.cat([x[j] for j in range(i, i + self.n_time_inputs)], dim=-1)
                    for i in range(steps)
                ]
            )

        y = torch.tensor(true_data.data, dtype=self.dtype)

        # Apply normalizations
        self._apply_normalization(x, y)

        # Compute forcings
        forcings = self._compute_forcings(input_data, steps)

        if forcings is not None:
            x = torch.cat([x, forcings], dim=-1)

        # Add constant data to input
        x = torch.cat([x, self.constant_data[:steps]], dim=-1)

        # Permute to [time, channels, latitude, longitude] format
        x_grid = x.permute(0, 3, 1, 2)
        y_grid = y.permute(0, 3, 1, 2)

        return x_grid.float(), y_grid.float()

    def _run_dataset_checks(self):
        # Check if grid includes poles
        has_poles = torch.any(
            torch.isclose(
                torch.abs(self.lat_rad_grid),
                torch.tensor(torch.pi, dtype=self.lat_rad_grid.dtype),
            )
        )
        assert not has_poles, "Grid with poles unsupported!"

        # Make sure latitude and longitude are in input file
        assert (
            self.cfg.features.input.constants[-2] == "latitude"
        ), "Latitude must be the second-to-last feature in constants!"

        assert (
            self.cfg.features.input.constants[-1] == "longitude"
        ), "Latitude must be the last feature in constants!"

    def _prepare_normalization(
        self, ds_input: xarray.Dataset, ds_output: xarray.Dataset
    ) -> None:
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
            if feature_name == "total_precipitation_6hr" and self.custom_normalization:
                self.norm_precip_in.append(i)
            elif feature_name == "specific_humidity" and self.custom_normalization:
                self.norm_humidity_in.append(i)
            else:
                self.norm_zscore_in.append(i)

        # Process dynamic output features
        for i, feature in enumerate(self.dyn_output_features):
            feature_name = re.sub(
                r"_h\d+$", "", feature
            )  # Remove height suffix (e.g., "_h10")
            if feature_name == "total_precipitation_6hr" and self.custom_normalization:
                self.norm_precip_out.append(i)
            elif feature_name == "specific_humidity" and self.custom_normalization:
                self.norm_humidity_out.append(i)
            else:
                self.norm_zscore_out.append(i)

        # Convert lists of indices to PyTorch tensors for efficient indexing
        if self.custom_normalization:
            self.norm_precip_in = torch.tensor(self.norm_precip_in, dtype=torch.long)
            self.norm_precip_out = torch.tensor(self.norm_precip_out, dtype=torch.long)
            self.norm_humidity_in = torch.tensor(
                self.norm_humidity_in, dtype=torch.long
            )
            self.norm_humidity_out = torch.tensor(
                self.norm_humidity_out, dtype=torch.long
            )
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
        self.input_mean = self.input_mean[
            self.norm_zscore_in % self.num_dyn_inputs_single
        ]
        self.input_std = self.input_std[
            self.norm_zscore_in % self.num_dyn_inputs_single
        ]
        self.output_mean = self.output_mean[self.norm_zscore_out]
        self.output_std = self.output_std[self.norm_zscore_out]

        # Prepare variables required in custom normalization
        if self.custom_normalization:

            if len(self.norm_humidity_in) > 0:
                # Maximum and minimum specific humidity in dataset
                self.q_max = torch.max(
                    self.input_max[self.norm_humidity_in % self.num_dyn_inputs_single]
                ).detach()
                self.q_min = torch.min(
                    self.input_min[self.norm_humidity_in % self.num_dyn_inputs_single]
                ).detach()
                if self.q_min < self.eps:
                    self.q_min = torch.tensor(self.eps).detach()
            else:
                self.q_max = torch.tensor(0.0).detach()
                self.q_min = torch.tensor(self.eps).detach()

        # Extract the toa_radiation mean and std
        self.toa_rad_std = ds_input.attrs["toa_radiation_std"]
        self.toa_rad_mean = ds_input.attrs["toa_radiation_mean"]

    def _apply_normalization(
        self, input_data: torch.Tensor, output_data: torch.Tensor
    ) -> None:

        # Apply custom normalizations to input
        if self.custom_normalization:
            input_data[..., self.norm_precip_in] = normalize_precipitation(
                input_data[..., self.norm_precip_in]
            )
            input_data[..., self.norm_humidity_in] = normalize_humidity(
                input_data[..., self.norm_humidity_in], self.q_min, self.q_max, self.eps
            )

            # Apply custom normalizations to output
            output_data[..., self.norm_precip_out] = normalize_precipitation(
                output_data[..., self.norm_precip_out]
            )
            output_data[..., self.norm_humidity_out] = normalize_humidity(
                output_data[..., self.norm_humidity_out],
                self.q_min,
                self.q_max,
                self.eps,
            )

        # Apply standard normalizations to input and output
        input_data[..., self.norm_zscore_in] = normalize_standard(
            input_data[..., self.norm_zscore_in],
            self.input_mean,
            self.input_std,
        )

        output_data[..., self.norm_zscore_out] = normalize_standard(
            output_data[..., self.norm_zscore_out], self.output_mean, self.output_std
        )

    def _compute_forcings(self, input_data: xarray.Dataset, steps: int) -> torch.Tensor:
        """Computes forcing paramters based in input_data array"""

        forcings_time_ds = time_forcings(input_data["time"].values)

        forcings = []
        for var in self.forcing_inputs:
            if var == "toa_incident_solar_radiation":
                toa_rad = toa_radiation(
                    input_data["time"].values,
                    self.lat.cpu().numpy(),
                    self.lon.cpu().numpy(),
                )

                toa_rad = torch.tensor(
                    (toa_rad - self.toa_rad_mean) / self.toa_rad_std,
                    dtype=self.dtype,
                )
                toa_rad = toa_rad.unfold(0, self.n_time_inputs, 1)

                forcings.append(toa_rad)
            else:
                # Get the time forcings
                if var in forcings_time_ds:
                    var_ds = forcings_time_ds[var]
                    var_forcing = torch.tensor(var_ds.data, dtype=self.dtype)
                    var_forcing = (
                        var_forcing.unfold(0, self.n_time_inputs, 1)
                        .view(steps, 1, 1, self.n_time_inputs)
                        .expand(steps, self.lat_size, self.lon_size, self.n_time_inputs)
                    )

                    forcings.append(var_forcing)

        if len(forcings) > 0:
            return torch.cat(forcings, dim=-1)
        return

    def _compute_geometric_constants(self):
        # Approximate distances using Haversine formula
        # This can be later improved for better approximation close to the poles
        R = 6371  # Average earth radius in km

        # Compute distance between latitude points
        dlon = torch.diff(torch.deg2rad(self.lon))[0]
        distance_lon_inv = 1.0 / (
            2
            * torch.arcsin(torch.cos(self.lat_rad_grid) ** 2 * torch.sin(dlon / 2))
            * R
        )

        # Normalize using z-score
        distance_lon_mean = torch.mean(distance_lon_inv)
        distance_lon_std = torch.std(distance_lon_inv)
        self.d_lon_inv = (distance_lon_inv - distance_lon_mean) / distance_lon_std
