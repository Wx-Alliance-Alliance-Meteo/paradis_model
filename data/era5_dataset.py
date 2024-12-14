"""ERA5 dataset handling for the model."""

import logging
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import xarray as xr


class ERA5Dataset(Dataset):
    """Prepare and process ERA5 dataset."""

    def __init__(
        self,
        root_dir: str,
        start_date: str,
        end_date: str,
        forecast_steps: int = 1,
        dtype=torch.float32,
        features_cfg: dict = {},
    ) -> None:
        self.dtype = dtype
        self.root_dir = root_dir

        # Physical constants
        self.EARTH_RADIUS = 6371220.0  # Earth's radius in meters
        self.OMEGA = 7.29212e-5  # Earth's rotation rate in rad/s
        self.G = 9.80616  # Gravitational acceleration in m/s²
        self.R = 287.05  # Gas constant for dry air in J/(kg·K)
        self.P0 = 1.0e5  # Reference pressure in Pa (1000 hPa)
        self.T0 = 288.0  # Reference temperature in K

        # Derived characteristic scales
        self.L = self.EARTH_RADIUS  # Length scale
        self.T = 1 / self.OMEGA  # Time scale
        self.U = self.OMEGA * self.L  # Velocity scale
        self.PHI0 = self.U * self.U  # Geopotential scale

        # Dictionary mapping variables to their physical scaling
        self.scaling_factors = {
            "u_component_of_wind": self.U,  # Scale by characteristic velocity
            "v_component_of_wind": self.U,  # Scale by characteristic velocity
            "vertical_velocity": self.P0 * self.OMEGA,  # Scale omega by p₀Ω
            "temperature": self.T0,  # Scale by reference temperature
            "geopotential": self.PHI0,  # Scale by characteristic geopotential
            "specific_humidity": 1.0,  # TODO
            "relative_humidity": 100,
            "vorticity": self.OMEGA,  # Scale by Earth's rotation rate
            "divergence": self.OMEGA,  # Scale by Earth's rotation rate
            "potential_temperature": self.T0,  # Scale by reference temperature
            "10m_u_component_of_wind": self.U,
            "10m_v_component_of_wind": self.U,
            "2m_temperature": self.T0,
            "surface_pressure": self.P0,
            "mean_sea_level_pressure": self.P0,
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
            files, chunks={"time": forecast_steps + 1}, engine="zarr"
        )

        # Filter data to time frame requested
        ds = ds.sel(time=slice(start_date, end_date))

        # The number of time instances in the dataset represents its length
        self.length = ds.time.size
        self.lat_size = ds.latitude.size
        self.lon_size = ds.longitude.size

        # Extract the name of the variables after stacking
        names = [value for value in ds.stacked.values]

        # Determine input and output variables
        input_names = []
        for variable in features_cfg.input.atmospheric:
            for level in features_cfg.pressure_levels:
                input_names.append(f"{variable}_h{level}")
        input_names.extend(features_cfg.input.surface)

        output_names = []
        for variable in features_cfg.output.atmospheric:
            for level in features_cfg.pressure_levels:
                output_names.append(f"{variable}_h{level}")
        output_names.extend(features_cfg.output.surface)

        # Get the indices to drop from dataset
        drop_vars_in = [name for name in names if name not in input_names]
        drop_vars_out = [name for name in names if name not in output_names]
        drop_vars = [var for var in drop_vars_out if var in drop_vars_in]
        drop_inds = [names.index(drop_var) for drop_var in drop_vars]

        # Store cleaned up dataset
        if len(drop_inds) > 0:
            self.ds = ds.drop_isel(stacked=drop_inds)
        else:
            self.ds = ds

        # Get the remaining names
        self.names = [value for value in self.ds.stacked.values]

        # Create input/output masks
        self.input_mask = torch.tensor(~np.isin(names, drop_vars_in), dtype=torch.bool)
        self.output_mask = torch.tensor(
            ~np.isin(names, drop_vars_out), dtype=torch.bool
        )

        # Check that the number of features is consistent for inputs and outputs
        self.num_in_features = len(features_cfg.pressure_levels) * len(
            features_cfg.input.atmospheric
        ) + len(features_cfg.input.surface)
        self.num_out_features = len(features_cfg.pressure_levels) * len(
            features_cfg.output.atmospheric
        ) + len(features_cfg.output.surface)

        assert sum(self.input_mask) == self.num_in_features
        assert sum(self.output_mask) == self.num_out_features

        # Create scaling tensors based on variable names
        self._create_scaling_tensors()

        # Extract latitude and longitude for the graph using the first month of this data
        self.lat = ds.latitude.values
        self.lon = ds.longitude.values
        self.forecast_steps = forecast_steps

        # Create masks for autoregression
        self._create_autoreg_maps()

        logging.info(
            f"Dataset contains: {self.num_in_features} input features, "
            f"{self.num_out_features} output features"
        )

    def _create_scaling_tensors(self):
        """Create scaling tensors for normalization based on physical variables."""
        # Initialize scaling factors for all variables
        scaling_values = []
        for name in self.names:
            # Extract base variable name (remove pressure level suffix if present)
            base_name = name.split("_h")[0]
            scaling_values.append(self.scaling_factors.get(base_name, 1.0))

        # Convert to tensor
        scaling_tensor = torch.tensor(scaling_values, dtype=self.dtype)

        # Create input and output scaling tensors
        self.input_scaling = scaling_tensor[self.input_mask]
        self.output_scaling = scaling_tensor[self.output_mask]

    def _create_autoreg_maps(self):
        """Create masks for autoregressive prediction."""
        names = np.array(self.names)
        names_in = names[self.input_mask]
        names_out = names[self.output_mask]
        names_indep = names[~self.output_mask]

        self.autoreg_maps = {
            "in": torch.tensor(np.isin(names_in, names_out), dtype=torch.bool),
            "out": torch.tensor(np.isin(names_out, names_in), dtype=torch.bool),
            "indep_in": torch.tensor(np.isin(names_in, names_indep), dtype=torch.bool),
            "indep_out": torch.tensor(np.isin(names_indep, names_in), dtype=torch.bool),
        }

    def __len__(self):
        """Return the length of the dataset."""
        return self.length - self.forecast_steps

    def __getitem__(self, idx: int):
        """Extract values from the requested indices and load array into CPU memory."""
        stack = (
            self.ds["data"]
            .isel(time=slice(idx, idx + self.forecast_steps + 1))
            .compute()
            .data
        )

        # Convert to tensors - data comes in [time, lat, lon, features]
        x = torch.tensor(stack[0], dtype=self.dtype)  # [lat, lon, features]
        y = torch.tensor(stack[1:], dtype=self.dtype)  # [time, lat, lon, features]

        # Apply feature masks
        x_features = x[..., self.input_mask]  # [lat, lon, input_features]
        y_features = y[..., self.output_mask]  # [time, lat, lon, output_features]
        indep_features = y[..., ~self.output_mask]  # [time, lat, lon, static_features]

        # Apply physical scaling
        x_normalized = x_features / self.input_scaling
        y_normalized = y_features / self.output_scaling
        indep_normalized = indep_features  # Static features remain unnormalized

        # Permute to [channels, height, width] format
        x_grid = x_normalized.permute(2, 0, 1)  # [input_features, lat, lon]
        y_grid = y_normalized.permute(0, 3, 1, 2)  # [time, output_features, lat, lon]
        indep_grid = indep_normalized.permute(
            0, 3, 1, 2
        )  # [time, static_features, lat, lon]

        return x_grid, y_grid, indep_grid


def input_from_output(input_shape, output_data, indep_output, autoreg_maps, device):
    """Process the next input in autoregression"""
    # Add features needed from the output
    mask_in = autoreg_maps["in"]
    mask_out = autoreg_maps["out"]

    new_input_data = torch.empty(input_shape, dtype=output_data.dtype).to(device)
    new_input_data[..., mask_in] = output_data[..., mask_out]

    # Add features that do not exist in the output
    mask_in = autoreg_maps["indep_in"]
    mask_out = autoreg_maps["indep_out"]

    new_input_data[..., mask_in] = indep_output[..., mask_out].to(output_data.dtype)
    return new_input_data


def split_dataset(dataset, train_ratio=0.8):
    """
    Splits a time-series dataset into training and validation sequentially.

    Args:
        dataset (ERA5Dataset): The full dataset.
        train_ratio (float): Fraction of data to use for training.

    Returns:
        Tuple[Dataset, Dataset]: Training and validation datasets.
    """
    split_idx = int(len(dataset) * train_ratio)

    train_dataset = torch.utils.data.Subset(dataset, range(0, split_idx))
    val_dataset = torch.utils.data.Subset(dataset, range(split_idx, len(dataset)))

    return train_dataset, val_dataset
