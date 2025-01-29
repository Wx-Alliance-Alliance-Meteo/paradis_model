"""Loss functions for the weather forecasting model."""

import torch
import re


class ReversedHuberLoss(torch.nn.Module):
    """Loss function that combines reversed Huber loss with custom weighting.

    This loss function implements a weighting scheme that accounts for key aspects
    of meteorological data:

    1. Vertical weighting: Implements pressure-level dependent weights that decrease with altitude.
    2. Variable-specific weighting: Allows different weights for various meteorological variables
       (e.g., temperature, wind, precipitation) to balance their relative importance.

    The final loss uses an reversed Huber loss which applies:
    - Linear penalties to small errors (|error| ≤ delta)
    - Quadratic penalties to large errors (|error| > delta)
    """

    def __init__(
        self,
        pressure_levels: torch.Tensor,
        num_features: int,
        num_surface_vars: int,
        var_loss_weights: torch.Tensor,
        output_name_order: list,
        delta: float = 1.0,
    ) -> None:
        """Initialize the weighted reversed Huber loss function.

        Args:
            pressure_levels: Pressure levels in hPa used in the model
            num_features: Total number of features in the output
            num_surface_vars: Number of surface-level variables
            var_loss_weights: Variable-specific weights for the loss calculation
            output_name_order: List of variable names in order of output features
            delta: Threshold parameter for the reversed Huber loss
        """
        super().__init__()

        # Ensure inputs are float32
        self.pressure_levels = pressure_levels.to(torch.float32)
        self.delta = delta

        # Store dimensions
        self.num_levels = len(pressure_levels)
        self.num_features = num_features
        self.num_surface_vars = num_surface_vars
        self.num_atmospheric_vars = num_features - num_surface_vars
        self.var_loss_weights = var_loss_weights
        self.output_name_order = output_name_order

        # Create combined feature weights
        self.feature_weights = self._create_feature_weights()

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

    def _pseudo_reversed_huber_loss(self, error: torch.Tensor) -> torch.Tensor:
        """Compute the psudo reversed Huber loss.

        Args:
            error: Error tensor (pred - target)

        Returns:
            Loss tensor with same shape as input
        """
        abs_error = torch.abs(error)
        small_error = self.delta * abs_error
        large_error = 0.5 * error**2
        weight = 1 / (1 + torch.exp(-2 * (abs_error - self.delta)))
        return (1 - weight) * small_error + weight * large_error

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate weighted inverted Huber loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Weighted loss value
        """
        # Prepare weights with correct shapes for broadcasting
        feature_weights = self.feature_weights.view(1, -1, 1, 1).to(pred.device)

        # Compute errors
        error = pred - target

        # Pseudo-reversed-Huber loss
        loss = self._pseudo_reversed_huber_loss(error)

        # Apply weights to loss components
        weighted_loss = loss * feature_weights

        return weighted_loss.mean()
