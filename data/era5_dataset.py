"""ERA5 dataset handling for WeatherBench2."""

import logging
from datetime import datetime
import numpy
import torch
import zarr

from data.forcings import time_forcings, toa_radiation

TSI = 1361  # Baseline solar irradiance in W/m^2


class ERA5Dataset(torch.utils.data.Dataset):
    """Prepare and process ERA5 dataset from WeatherBench2 format."""

    def __init__(
        self,
        root_dir: str,
        start_date: str,
        end_date: str,
        forecast_steps: int = 1,
        dtype=torch.float32,
        features_cfg: dict = {},
    ) -> None:
        """Initialize ERA5Dataset.

        Args:
            root_dir: Root directory containing the data
            start_date: Start date for the dataset
            end_date: End date for the dataset
            forecast_steps: Number of timesteps to forecast
            dtype: Data type for tensors
            features_cfg: Configuration for input/output features
        """
        self.root_dir = root_dir
        self.forecast_steps = forecast_steps
        self.dtype = dtype
        self.num_common_features = 0
        self.forcing_inputs = features_cfg.input.forcings

        # Variable statistics for z-score normalization
        self.var_stats = {
            "geopotential": {"mean": 199776, "std": 3777.9},
            "u_component_of_wind": {"mean": 3.81107, "std": 10.0795},
            "v_component_of_wind": {"mean": 0.0966389, "std": 6.63847},
            "temperature": {"mean": 213.016, "std": 12.8394},
            "vertical_velocity": {"mean": 0.000139974, "std": 0.00573747},
            "10m_u_component_of_wind": {"mean": -0.239831, "std": 5.16198},
            "10m_v_component_of_wind": {"mean": -0.0354346, "std": 4.33157},
            "2m_temperature": {"mean": 277.038, "std": 19.6462},
            "mean_sea_level_pressure": {"mean": 101152, "std": 1224.07},
        }

        # Dictionary mapping variables to their physical scaling
        self.scaling_factors = {
            "toa_incident_solar_radiation": TSI * 3600,  # in W s/m^2
        }

        # Variables that need custom normalization, used in forecast script
        self.custom_norm_vars = {"specific_humidity", "total_precipitation_6hr"}

        # Convert string dates to datetime64
        start_datetime = numpy.datetime64(start_date)
        end_datetime = numpy.datetime64(end_date)

        # Load coordinates using zarr
        logging.info(f"Loading coordinates from {root_dir}")
        self.store = zarr.open(root_dir, mode="r")

        self.lat = self.store["latitude"][:]
        self.lon = self.store["longitude"][:]

        # Convert hours since 1959-01-01 to datetime64
        self.ref_date = numpy.datetime64("1959-01-01")
        times = self.ref_date + numpy.timedelta64(1, "h") * self.store["time"][:]

        # Create time mask
        mask = (times >= numpy.datetime64(start_date)) & (
            times <= numpy.datetime64(end_date)
        )
        self.time_indices = numpy.where(mask)[0]

        if len(self.time_indices) == 0:
            raise ValueError(f"No data found between {start_date} and {end_date}")

        # Set dimensions
        self.length = len(self.time_indices)
        self.lat_size = len(self.lat)
        self.lon_size = len(self.lon)
        self.grid_size = self.lat_size * self.lon_size

        # Load constants
        self._load_constants(features_cfg)

        # Setup input and output features based on config
        # Resolve base features if they are referenced in input/output
        input_atmospheric = (
            features_cfg["base"]["atmospheric"]
            if features_cfg["input"].get("atmospheric")
            == "${features.base.atmospheric}"
            else features_cfg["input"]["atmospheric"]
        )
        output_atmospheric = (
            features_cfg["base"]["atmospheric"]
            if features_cfg["output"].get("atmospheric")
            == "${features.base.atmospheric}"
            else features_cfg["output"]["atmospheric"]
        )

        # Set input features directly from config
        self.dyn_input_features = input_atmospheric + features_cfg["input"]["surface"]

        # Set output features directly from config
        self.output_name_order = output_atmospheric + features_cfg["output"]["surface"]

        # Get constant features
        self.constant_features = features_cfg["input"].get("constants", [])

        # Update feature counts
        self.num_common_features = len(
            set(self.dyn_input_features) & set(self.output_name_order)
        )
        self.num_in_features = (
            len(self.dyn_input_features)
            + len(self.constant_features)
            + len(self.forcing_inputs)
        )
        self.num_out_features = len(self.output_name_order)

        logging.info(
            f"Dataset initialized with {self.num_in_features} input features "
            f"({len(self.dyn_input_features)} dynamic, {len(self.constant_features)} constant, "
            f"{len(self.forcing_inputs)} forcing), {self.num_out_features} output features, "
            f"and {self.length} timesteps"
        )

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

    def __len__(self):
        """Return the length of the dataset."""
        return self.length - self.forecast_steps

    def __getitem__(self, ind: int):
        """Extract values from the requested indices."""
        if ind < 0:
            ind += len(self)

        # Initialize arrays
        x = numpy.zeros(
            (
                self.forecast_steps,
                self.lat_size,
                self.lon_size,
                len(self.dyn_input_features),
            ),
            dtype=numpy.float32,
        )
        y = numpy.zeros(
            (
                self.forecast_steps,
                self.lat_size,
                self.lon_size,
                len(self.output_name_order),
            ),
            dtype=numpy.float32,
        )

        # Load data for each timestep
        for t in range(self.forecast_steps):
            time_idx = self.time_indices[ind + t]

            # Apply transformations
            for i, feature in enumerate(self.dyn_input_features):
                data = self._load_variable(feature, time_idx)
                # if ind == 0 and t == 0:
                #     self.print_variable_stats(feature, data)

                if feature == "specific_humidity":
                    x[t, ..., i] = self._normalize_humidity(data)
                elif feature == "total_precipitation_6hr":
                    x[t, ..., i] = self._normalize_precipitation(data)
                elif feature == "toa_incident_solar_radiation":
                    x[t, ..., i] = data / self.scaling_factors[feature]
                else:
                    # Apply z-score normalization
                    stats = self.var_stats[feature]
                    x[t, ..., i] = (data - stats["mean"]) / stats["std"]

                # if ind == 0 and t == 0:
                #     self.print_variable_stats(feature, x[t, ..., i], "normalized")

            # Load output features
            next_time_idx = self.time_indices[ind + t + 1]
            for i, feature in enumerate(self.output_name_order):
                data = self._load_variable(feature, next_time_idx)
                if feature == "specific_humidity":
                    y[t, ..., i] = self._normalize_humidity(data)
                elif feature == "total_precipitation_6hr":
                    y[t, ..., i] = self._normalize_precipitation(data)
                elif feature == "toa_incident_solar_radiation":
                    y[t, ..., i] = data / self.scaling_factors[feature]
                else:
                    # Apply z-score normalization
                    stats = self.var_stats[feature]
                    y[t, ..., i] = (data - stats["mean"]) / stats["std"]

        # Convert to tensors
        x = torch.from_numpy(x).to(self.dtype)
        y = torch.from_numpy(y).to(self.dtype)

        # Get time indices for this sample
        time_indices = [self.time_indices[ind + t] for t in range(self.forecast_steps)]

        # Compute forcings
        forcings = self._compute_forcings(time_indices)

        if forcings is not None:
            x = torch.cat([x, forcings], dim=-1)

        # Add constants to input
        x = torch.cat([x, self.constant_data], dim=-1)

        # Permute to [time, channels, latitude, longitude] format
        x_grid = x.permute(0, 3, 1, 2)
        y_grid = y.permute(0, 3, 1, 2)

        return x_grid, y_grid

    def _load_constants(self, features_cfg):
        """Load and process constant features."""
        constants = []
        for var in features_cfg.input.constants:
            if var == "land_sea_mask":
                data = self.store["land_sea_mask"][:]
                data = self._check_array_ordering(data, "land_sea_mask")
            else:
                data = self.store[var][:]
                data = self._check_array_ordering(data, var)
                mean, std = data.mean(), data.std()
                data = (data - mean) / std
            constants.append(data)

        stacked_data = torch.from_numpy(numpy.array(constants)).to(dtype=torch.float32)

        self.constant_data = (
            stacked_data.permute(1, 2, 0)
            .unsqueeze(0)
            .expand(self.forecast_steps, -1, -1, -1)
            .contiguous()
        )

    def _load_variable(self, var_name, time_idx, level=None):
        """Load a variable from zarr store for a specific time index."""
        data = self.store[var_name]

        # Handle different variable structures
        if len(data.shape) == 3:  # Time, lat, lon or Time, lon, lat
            data_slice = data[time_idx]
            return self._check_array_ordering(data_slice, var_name)

        elif len(data.shape) == 4:  # Time, level, lat, lon or Time, level, lon, lat
            if level is not None:
                level_idx = list(self.store["level"][:]).index(level)
            else:
                level_idx = 0
            data_slice = data[time_idx, level_idx]
            return self._check_array_ordering(data_slice, var_name)
        else:
            raise ValueError(f"Unexpected shape for variable {var_name}: {data.shape}")

    def _check_array_ordering(self, data, var_name):
        """Check and adjust array orientation to ensure consistent [lat, lon] ordering.

        Some WeatherBench 2 Zarr data stores, particularly at lower resolutions,
        are stored as [lon, lat] instead of [lat, lon]

        Args:
            data: Input numpy array
            var_name: Variable name for logging

        Returns:
            Array with [lat, lon] ordering
        """
        if data.shape[-2] > data.shape[-1]:  # If dimensions are transposed
            return data.transpose(-1, -2)
        else:
            return data

    def _compute_forcings(self, time_indices):
        """Computes forcing paramters based in input_data array"""
        # Convert time indices to datetime64
        times = (
            self.ref_date + numpy.timedelta64(1, "h") * self.store["time"][time_indices]
        )

        # Get forcings from time variables
        forcings_dict = time_forcings(times)
        forcings = []

        for var in self.forcing_inputs:
            if var == "toa_incident_solar_radiation":
                toa_rad = toa_radiation(times, self.lat, self.lon)
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
            elif var in forcings_dict:
                value = torch.tensor(forcings_dict[var], dtype=self.dtype)
                value = value.view(-1, 1, 1, 1).expand(
                    self.forecast_steps, self.lat_size, self.lon_size, 1
                )
                forcings.append(value)
            else:
                logging.warning(f"Missing forcing variable: {var}")

        if not forcings:
            return None

        return torch.cat(forcings, dim=-1)

    def print_variable_stats(
        self, feature_name: str, data: numpy.ndarray, stage: str = "original"
    ) -> None:
        """Print statistics for a variable to help debug normalization.

        Args:
            feature_name: Name of the feature being analyzed
            data: Data array to analyze
            stage: Description of processing stage (e.g., "original", "normalized")
        """
        stats = {
            "mean": numpy.mean(data),
            "std": numpy.std(data),
            "min": numpy.min(data),
            "max": numpy.max(data),
            "median": numpy.median(data),
            "zeros": numpy.sum(data == 0),
            "negatives": numpy.sum(data < 0),
        }

        print(f"\n{feature_name} Statistics ({stage}):")
        print(f"{'='*50}")
        for stat_name, value in stats.items():
            if stat_name in ["zeros", "negatives"]:
                print(f"{stat_name:>10}: {value:,d} values")
            else:
                print(f"{stat_name:>10}: {value:>12.6g}")


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
