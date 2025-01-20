"""Loss functions for the weather forecasting model."""

import re

import torch


class WeightedHybridLoss(torch.nn.Module):
    """Loss function that combines MAE and RMSE metrics.

    This loss function implements a weighting scheme that accounts for three key aspects
    of meteorological data:

    1. Spatial weighting: Applies latitude-dependent weights to account for the varying grid cell
       areas on a spherical surface, ensuring proper representation of polar and equatorial regions.

    2. Vertical weighting: Implements pressure-level dependent weights that decrease with altitude.

    3. Variable-specific weighting: Allows different weights for various meteorological variables
       (e.g., temperature, wind, precipitation) to balance their relative importance in the total
       loss calculation.

    The final loss is computed as a weighted combination of MAE and RMSE:
        loss = α * MAE + (1-α) * RMSE
    where α is a user-provided parameter that balances the contribution of each metric.
    """

    def __init__(
        self,
        grid_lat: torch.Tensor,
        pressure_levels: torch.Tensor,
        num_features: int,
        num_surface_vars: int,
        var_loss_weights: torch.Tensor,
        output_name_order: list,
        alpha: float,
    ) -> None:
        """Initialize the weighted hybrid loss function.

        Args:
            grid_lat (torch.Tensor): Latitude values in degrees for each grid point
            pressure_levels (torch.Tensor): Pressure levels in hPa used in the model
            num_features (int): Total number of features in the output
            num_surface_vars (int): Number of surface-level variables
            var_loss_weights (torch.Tensor): Variable-specific weights for the loss calculation
            output_name_order (list): List of variable names in order of output features
            alpha (float, optional): Balance factor between MAE (alpha) and RMSE (1-alpha).
        """
        super().__init__()

        # Ensure inputs are float32
        grid_lat = grid_lat.to(torch.float32)
        self.pressure_levels = pressure_levels.to(torch.float32)

        # Store dimensions
        self.num_lats = len(grid_lat)
        self.num_levels = len(pressure_levels)
        self.num_features = num_features
        self.num_surface_vars = num_surface_vars
        self.num_atmospheric_vars = num_features - num_surface_vars
        self.var_loss_weights = var_loss_weights
        self.output_name_order = output_name_order
        self.alpha = alpha

        # Create combined feature weights
        self.feature_weights = self._create_feature_weights()

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

    def _create_feature_weights(self) -> torch.Tensor:
        """Create weights for all features."""
        # Initialize weights tensor for all features
        feature_weights = torch.zeros(self.num_features, dtype=torch.float32)

        # Standard pressure weights normalized by number of levels
        pressure_weights = (self.pressure_levels / self.pressure_levels[-1]).to(
            torch.float32
        ) / self.num_levels

        # Process atmospheric variables (with pressure levels)
        for i in range(0, self.num_atmospheric_vars, self.num_levels):

            # Get the variable name independent of pressure level
            var_name = re.sub(r"_h\d+$", "", self.output_name_order[i])

            # Get the base weights for this variable
            base_weights = self.var_loss_weights[i : i + self.num_levels]

            # Multiply by pressure weights
            if var_name == "geopotential":
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

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate weighted hybrid loss combining MAE and RMSE components.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Combined weighted loss value
        """
        # Prepare weights with correct shapes for broadcasting
        feature_weights = self.feature_weights.view(1, -1, 1, 1).to(pred.device)

        # Compute errors
        error = pred - target
        abs_error = torch.abs(error)
        squared_error = error**2

        # Apply weights to both MAE and MSE components
        weighted_abs_error = abs_error * feature_weights
        weighted_squared_error = squared_error * feature_weights

        # Compute final hybrid loss
        mae_component = weighted_abs_error.mean()
        mse_component = weighted_squared_error.mean()
        rmse_component = torch.sqrt(mse_component)

        return self.alpha * mae_component + (1 - self.alpha) * rmse_component
