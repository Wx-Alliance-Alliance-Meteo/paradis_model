"""ERA5 dataset handling for the model."""

import logging
from omegaconf import DictConfig
import os
import re

import dask
import torch
from torch.utils.data import Dataset
import xarray as xr


class ERA5Dataset(Dataset):
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

        ds = xr.open_mfdataset(
            files, chunks={"time": self.forecast_steps + 1}, engine="zarr"
        )

        # Lazy open this dataset
        standardized_data = False
        if standardized_data:

            # Add stats to data array
            ds_stats = xr.open_dataset(
                os.path.join(self.root_dir, "stats"), engine="zarr"
            )

            ds["mean"] = ds_stats["mean"]
            ds["std"] = ds_stats["std"]

            # Normalize atmospheric variables
            ds = (ds["data"] - ds["mean"]) / ds["std"]
        else:
            ds = ds["data"]

        # Inspect the stacked variable names
        variable_names = ds.coords["stacked"].values

        # Extract and clean variable names, ensuring unique names are preserved
        output_name_order = []
        for name in variable_names:
            # Remove the height suffix (_h50, _h100, etc.)
            clean_name = re.sub(r"_h\d+$", "", name)

            # Ensure unique names
            if clean_name not in output_name_order:
                output_name_order.append(clean_name)

        # Store the cleaned variable names in self.output_name_order
        self.output_name_order = output_name_order

        # Filter data to time frame requested
        ds = ds.sel(time=slice(start_date, end_date))

        # Extract latitude and longitude to build the graph
        self.lat = ds.latitude.values
        self.lon = ds.longitude.values

        # The number of time instances in the dataset represents its length
        self.length = ds.time.size
        self.lat_size = ds.latitude.size
        self.lon_size = ds.longitude.size

        # Store the size of the grid (lat * lon)
        self.grid_size = ds.latitude.size * ds.longitude.size

        # Extract the name of the stacked variable
        names = [value for value in ds.stacked.values]

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

        # remove masks from self.output_name_order
        output_names_nolevel = []
        output_names_nolevel.extend(features_cfg.output.atmospheric)
        output_names_nolevel.extend(features_cfg.output.surface)

        # Filter self.output_name_order to include only variables present in output_names
        self.output_name_order = [
            name for name in self.output_name_order if name in output_names_nolevel
        ]

        # Determine which variables to drop (not in config files)
        drop_vars = []
        for name in names:
            if name not in input_names and name not in output_names:
                drop_vars.append(str(name))

        if len(drop_vars) > 0:
            self.ds = ds.drop_sel(stacked=drop_vars)
        else:
            self.ds = ds

        # Align arrays with common features first
        common_features = []
        input_only = []
        output_only = []

        for name in input_names:
            if name in output_names:
                common_features.append(name)
            else:
                input_only.append(name)

        for name in output_names:
            if not (name in input_names):
                output_only.append(name)

        self.num_common_features = len(common_features)
        self.dyn_input_features = common_features + input_only
        self.dyn_output_features = common_features + output_only

        # Ensure order is consistent with config file
        self.ds_input = self.ds.sel(stacked=self.dyn_input_features)
        self.ds_output = self.ds.sel(stacked=self.dyn_output_features)

        # Constant input variables
        self._load_constants(features_cfg)

        # Create scaling tensors based on variable names
        self._create_scaling_tensors()

        self.num_in_features = (
            len(self.dyn_input_features) + self.constant_data.shape[-1]
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
        x = torch.tensor(input_data.data, dtype=self.dtype) / self.input_scaling

        # Get rid of variables that are not part of the output as well
        y = torch.tensor(true_data.data, dtype=self.dtype) / self.output_scaling

        # Add constant and forcing data to input
        x = torch.cat([x, self.constant_data], dim=-1)

        # Permute to [channels, height, width] format
        x_grid = x.permute(0, 3, 1, 2)  # [time, input_features, lat, lon]
        y_grid = y.permute(0, 3, 1, 2)  # [time, output_features, lat, lon]

        return x_grid, y_grid

    def _create_scaling_tensors(self):
        """Create scaling tensors for normalization based on physical variables."""
        # Initialize scaling factors for all variables
        input_scaling = []
        for name in self.dyn_input_features:
            base_name = re.sub(r"_h\d+$", "", name)
            input_scaling.append(self.scaling_factors.get(base_name, 1.0))

        output_scaling = []
        for name in self.dyn_output_features:
            base_name = re.sub(r"_h\d+$", "", name)
            output_scaling.append(self.scaling_factors.get(base_name, 1.00))

        # Convert to tensor
        self.input_scaling = torch.tensor(input_scaling, dtype=self.dtype)
        self.output_scaling = torch.tensor(output_scaling, dtype=self.dtype)

    def _load_constants(self, features_cfg):
        ds_constants = xr.open_dataset(
            os.path.join(self.root_dir, "constants"), engine="zarr"
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
            .unsqueeze(0)
            .expand(self.forecast_steps, -1, -1, -1)
        )


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
