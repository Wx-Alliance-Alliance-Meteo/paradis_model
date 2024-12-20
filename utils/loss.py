"""Custom loss functions for the GATmosphere weather forecasting model."""

import torch


class WeightedMSELoss(torch.nn.Module):
    """MSE loss with enhanced weighting for mid-latitudes, jet stream levels, and area correction."""

    def __init__(
        self,
        grid_lat: torch.Tensor,
        pressure_levels: torch.Tensor,
        num_features: int,
        num_surface_vars: int,
        var_loss_weights: torch.Tensor,  # Added Var_loss_weights as an argument
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
        self.lat_weights = self._compute_latitude_weights(grid_lat)

        # Calculate pressure level weights with Gaussian jet stream emphasis
        self.pressure_weights = self._compute_pressure_weights(pressure_levels)

        # Create combined feature weights
        self.feature_weights = self._create_feature_weights()

    def _gaussian_weight(
        self, x: torch.Tensor, mu: float, sigma: float
    ) -> torch.Tensor:
        """Compute Gaussian weights centered at mu with standard deviation sigma."""
        return torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

    def _compute_latitude_weights(self, grid_lat: torch.Tensor) -> torch.Tensor:
        """Compute latitude weights combining area correction and mid-latitude emphasis."""
        # Basic area weights (cos(lat) weighting)
        delta_lat = torch.abs(grid_lat[1] - grid_lat[0])
        pole_threshold = torch.tensor(90.0, dtype=torch.float32)
        has_poles = torch.any(torch.isclose(torch.abs(grid_lat), pole_threshold))

        if has_poles:
            area_weights = torch.cos(torch.deg2rad(grid_lat)) * torch.sin(
                torch.deg2rad(delta_lat / 2)
            )
            pole_indices = torch.where(
                torch.isclose(torch.abs(grid_lat), pole_threshold)
            )[0]
            pole_weight = torch.sin(torch.deg2rad(delta_lat / 4)) ** 2
            area_weights[pole_indices] = pole_weight
        else:
            area_weights = torch.cos(torch.deg2rad(grid_lat))

        # Mid-latitude emphasis parameters
        mid_lat_center = 50.0  # Center of emphasis in Northern Hemisphere
        mid_lat_spread = 15.0  # Spread of the emphasis region
        mid_lat_enhancement = 2.0  # Maximum enhancement factor

        # Calculate mid-latitude enhancement
        mid_lat_weights = self._gaussian_weight(
            grid_lat, mid_lat_center, mid_lat_spread
        )
        enhancement_factor = 1.0 + (mid_lat_enhancement - 1.0) * mid_lat_weights

        # Combine area weights with mid-latitude enhancement
        combined_weights = area_weights * enhancement_factor

        return combined_weights / combined_weights.mean()

    def _compute_pressure_weights(self, pressure_levels: torch.Tensor) -> torch.Tensor:
        """Compute pressure level weights with Gaussian jet stream emphasis."""
        # Basic pressure-proportional weights
        base_weights = pressure_levels / pressure_levels.mean()

        # Parameters for jet stream Gaussian
        jet_center = 250.0  # Center of jet stream (hPa)
        jet_width = 50.0  # Width of jet stream region (hPa)
        jet_enhancement = 2.0  # Maximum enhancement factor

        # Compute Gaussian weights centered on jet stream level
        gaussian_weights = self._gaussian_weight(pressure_levels, jet_center, jet_width)
        enhancement_factor = 1.0 + (jet_enhancement - 1.0) * gaussian_weights

        # Combine base weights with jet stream enhancement
        weights = base_weights * enhancement_factor

        return weights / weights.mean()

    def _create_feature_weights(self) -> torch.Tensor:
        """Create weights for all features, including both pressure-level and surface variables."""
        vars_per_level = self.num_level_vars // self.num_levels
        level_weights = self.pressure_weights.repeat(vars_per_level)
        # Repeat the first 6 components 13 times
        var_atmospheric_weights = self.var_loss_weights[
            :vars_per_level
        ].repeat_interleave(self.num_levels)
        var_surface_weights = self.var_loss_weights[vars_per_level:]
        surface_weights = torch.ones(self.num_surface_vars, dtype=torch.float32)
        feature_weights = torch.cat(
            [
                level_weights * var_atmospheric_weights,
                surface_weights * var_surface_weights,
            ]
        )

        return feature_weights / feature_weights.mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate weighted MSE loss.

        Args:
            pred: Predicted values, shape [batch, channels, height, width]
            target: Target values, shape [batch, channels, height, width]

        Returns:
            Weighted MSE loss value
        """
        # Calculate squared errors
        squared_errors = (pred - target) ** 2

        # Prepare latitude weights
        lat_weights = self.lat_weights.view(-1, 1).to(pred.device)

        # Prepare feature weights
        feature_weights = self.feature_weights.view(1, -1, 1, 1).to(pred.device)

        # Apply all weights
        weighted_errors = squared_errors * lat_weights * feature_weights

        return weighted_errors.mean()
