"""Loss functions for the weather forecasting model."""

import re
import torch


class ParadisLoss(torch.nn.Module):
    """Loss function.

    This loss function implements a weighting scheme that accounts for key aspects
    of meteorological data:

    1. Vertical weighting: Implements pressure-level dependent weights that decrease with altitude.
    2. Variable-specific weighting: Allows different weights for various meteorological variables
       (e.g., temperature, wind, precipitation) to balance their relative importance.
    3. Spatial weighting: Applies latitude-dependent weights to account for the varying grid cell
       areas on a spherical surface, ensuring proper representation of polar and equatorial regions.

    The final loss can use a reversed Huber loss, which applies:
     - Linear penalties to small errors (|error| ≤ delta)
     - Quadratic penalties to large errors (|error| > delta)
    or a simple MSE loss function.
    """

    def __init__(
        self,
        loss_function: str,
        lat_grid: torch.Tensor,
        pressure_levels: torch.Tensor,
        num_features: int,
        num_surface_vars: int,
        var_loss_weights: torch.Tensor,
        output_name_order: list,
        delta_loss: float = 1.0,
    ) -> None:
        """Initialize the weighted reversed Huber loss function.

        Args:
            pressure_levels: Pressure levels in hPa used in the model
            num_features: Total number of features in the output
            num_surface_vars: Number of surface-level variables
            var_loss_weights: Variable-specific weights for the loss calculation
            output_name_order: List of variable names in order of output features
            delta_loss: Threshold parameter for the Huber loss
        """
        super().__init__()

        # Ensure inputs are float32
        self.pressure_levels = pressure_levels.to(torch.float32)

        self.delta = delta_loss

        # Store dimensions
        self.num_levels = len(pressure_levels)
        self.num_features = num_features
        self.num_surface_vars = num_surface_vars
        self.num_atmospheric_vars = num_features - num_surface_vars
        self.var_loss_weights = var_loss_weights
        self.output_name_order = output_name_order

        # Whether to flip geopotential weights
        self.flip_geopotential_weights = False

        # Whether to apply latitude weights in loss integration
        self.apply_latitude_weights = True
        self.lat_weights = self._compute_latitude_weights(lat_grid)

        # Create combined feature weights
        self.feature_weights = self._create_feature_weights()

        if loss_function == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="none")
        elif loss_function == "reversed_huber":
            self.loss_fn = self._pseudo_reversed_huber_loss

    def _check_uniform_spacing(self, grid: torch.Tensor) -> float:
        """Check if grid has uniform spacing and return the delta.

        Args:
            grid: Input coordinate grid tensor

        Returns:
            Grid spacing delta

        Raises:
            ValueError: If grid spacing is not uniform
        """
        diff = torch.diff(grid)
        if not torch.allclose(diff, diff[0]):
            raise ValueError(f"Grid {grid} is not uniformly spaced")
        return diff[0].item()

    def _compute_latitude_weights(self, grid_lat: torch.Tensor) -> torch.Tensor:
        """Compute latitude weights based on grid cell areas.

        For a latitude grid, this handles two cases:
        1. Grids without poles: Points represent slices between lat±Δλ/2
           Weight proportional to cos(lat)
        2. Grids with poles: Points at poles represent half-slices
           For non-pole points: weight ∝ cos(λ)⋅sin(Δλ/2)
           For pole points: weight ∝ sin(Δλ/4)²

        Args:
            grid_lat: Latitude coordinates in degrees

        Returns:
            Normalized weights with unit mean

        Raises:
            ValueError: If grid is not uniformly spaced or has invalid endpoints
        """

        # Validate uniform spacing
        delta_lat = torch.abs(torch.tensor(self._check_uniform_spacing(grid_lat)))

        # Check if grid includes poles
        has_poles = torch.any(
            torch.isclose(torch.abs(grid_lat), torch.tensor(90.0, dtype=grid_lat.dtype))
        )

        if has_poles:
            raise ValueError("Grid must not contain poles!")
        else:
            # Validate grid endpoints
            if not (
                torch.isclose(
                    torch.abs(grid_lat.max()),
                    (90.0 - delta_lat / 2) * torch.ones_like(grid_lat.max()),
                )
            ):
                raise ValueError("Grid without poles must end at ±(90° - Δλ/2)")

            # Simple cosine weights for grids without poles
            weights = torch.cos(torch.deg2rad(grid_lat))

        return weights / weights.mean()

    def _create_feature_weights(self) -> torch.Tensor:
        """Create weights for all features."""
        # Initialize weights tensor for all features
        feature_weights = torch.zeros(self.num_features, dtype=torch.float32)

        # Standard pressure weights normalized by number of levels
        pressure_weights = (self.pressure_levels).to(torch.float32)
        pressure_weights /= torch.mean(pressure_weights)

        # Process atmospheric variables (with pressure levels)
        for i in range(0, self.num_atmospheric_vars, self.num_levels):

            # Get the variable name independent of pressure level
            var_name = re.sub(r"_h\d+$", "", self.output_name_order[i])

            # Get the base weights for this variable
            base_weights = self.var_loss_weights[i : i + self.num_levels]

            # Multiply by pressure weights
            if self.flip_geopotential_weights and var_name == "geopotential":
                feature_weights[i : i + self.num_levels] = (
                    base_weights * pressure_weights.flip(0)
                )
            else:
                feature_weights[i : i + self.num_levels] = (
                    base_weights * pressure_weights
                )

        # Process surface variables
        feature_weights[self.num_atmospheric_vars :] = self.var_loss_weights[
            self.num_atmospheric_vars :
        ]

        return feature_weights

    def _pseudo_reversed_huber_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute the pseudo reversed Huber loss.

        Args:
            error: Error tensor (pred - target)

        Returns:
            Loss tensor with same shape as input
        """
        error = pred - target
        abs_error = torch.abs(error)
        small_error = self.delta * abs_error
        large_error = 0.5 * error**2
        weight = 1 / (1 + torch.exp(-2 * (abs_error - self.delta)))
        return (1 - weight) * small_error + weight * large_error

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate weighted reversed Huber loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Weighted loss value
        """
        # Prepare weights with correct shapes for broadcasting
        lat_weights = self.lat_weights.view(1, 1, -1, 1).to(pred.device)
        feature_weights = self.feature_weights.view(1, -1, 1, 1).to(pred.device)

        # Get the loss using the appropriate function
        loss = self.loss_fn(pred, target)

        # Apply weights to loss components
        weighted_loss = loss * feature_weights

        if self.apply_latitude_weights:
            weighted_loss *= lat_weights

        return weighted_loss.mean()
