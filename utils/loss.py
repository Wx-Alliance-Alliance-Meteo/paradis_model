"""Custom loss functions for weather prediction."""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """MSE loss with enhanced weighting for mid-latitudes, jet stream levels, and area correction."""

    def __init__(
        self,
        grid_lat: torch.Tensor,
        pressure_levels: torch.Tensor,
        num_features: int,
        num_surface_vars: int,
        var_loss_weights: torch.Tensor,
    ):
        """Initialize with grid coordinates and pressure levels.

        Args:
            grid_lat: Latitude values in degrees, shape [num_lats]
            pressure_levels: Pressure levels in hPa, shape [num_levels]
            num_features: Total number of features per grid point
            num_surface_vars: Number of surface-level variables
            var_loss_weights: Custom weights for each feature (from the YAML configuration)
        """
        super().__init__()

        # Ensure inputs are float32
        grid_lat = grid_lat.to(torch.float32)
        pressure_levels = pressure_levels.to(torch.float32)

        # Store dimensions
        self.num_lats = len(grid_lat)
        self.num_levels = len(pressure_levels)
        self.num_features = num_features
        self.num_surface_vars = num_surface_vars
        self.num_level_vars = num_features - num_surface_vars
        self.var_loss_weights = var_loss_weights

        # Calculate combined latitude weights
        self._compute_latitude_weights(grid_lat)

        # Calculate pressure level weights with Gaussian jet stream emphasis
        self._compute_pressure_weights(pressure_levels)

        # Create combined feature weights
        self._create_feature_weights()

    def _compute_latitude_weights(self, grid_lat: torch.Tensor) -> None:
        """Compute latitude-dependent weights for area correction."""
        # Convert to radians
        lat_rad = torch.deg2rad(grid_lat)

        # Basic area weights (cos(lat))
        self.lat_weights = torch.cos(lat_rad)

        # Normalize weights
        self.lat_weights = self.lat_weights / self.lat_weights.mean()
        self.lat_weights = self.lat_weights.view(1, 1, -1, 1)

    def _compute_pressure_weights(self, pressure_levels: torch.Tensor) -> None:
        """Compute pressure level weights with enhanced weighting for jet stream levels."""
        # Basic pressure-proportional weights
        base_weights = pressure_levels / pressure_levels.mean()

        # Enhance weights around jet stream levels
        jet_center = 250.0  # Center of jet stream (hPa)
        jet_width = 50.0  # Width of jet stream region (hPa)
        jet_enhancement = 2.0  # Maximum enhancement factor

        gaussian = torch.exp(-0.5 * ((pressure_levels - jet_center) / jet_width) ** 2)
        enhancement = 1.0 + (jet_enhancement - 1.0) * gaussian

        self.pressure_weights = (base_weights * enhancement) / (
            base_weights * enhancement
        ).mean()

    def _create_feature_weights(self) -> None:
        """Create weights for all features, handling both pressure-level and surface variables."""
        # Repeat pressure weights for each variable at that level
        vars_per_level = self.num_level_vars // self.num_levels
        level_weights = self.pressure_weights.repeat_interleave(vars_per_level)

        # Set surface weights to one
        surface_weights = torch.ones(self.num_surface_vars, dtype=torch.float32)

        # Get atmospheric variable weights following config file
        var_atmospheric_weights = self.var_loss_weights[
            :vars_per_level
        ].repeat_interleave(self.num_levels)

        var_surface_weights = self.var_loss_weights[vars_per_level:]

        # Combine and normalize
        self.feature_weights = torch.cat(
            [
                level_weights * var_atmospheric_weights,
                surface_weights * var_surface_weights,
            ]
        )

        self.feature_weights = self.feature_weights / self.feature_weights.mean()

        # Reshape for broadcasting with 4D input
        self.feature_weights = self.feature_weights.view(1, -1, 1, 1)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate weighted MSE loss.

        Args:
            pred: Predicted values, shape [batch, channels, height, width]
            target: Target values, shape [batch, channels, height, width]

        Returns:
            Weighted MSE loss value
        """
        # Move weights to correct device if needed
        if self.lat_weights.device != pred.device:
            self.lat_weights = self.lat_weights.to(pred.device)
            self.feature_weights = self.feature_weights.to(pred.device)

        # Calculate squared errors
        squared_errors = (pred - target) ** 2

        # Apply latitude and feature weights
        weighted_errors = squared_errors * self.lat_weights * self.feature_weights

        return weighted_errors.mean()
