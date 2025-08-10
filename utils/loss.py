"""Loss functions for the weather forecasting model."""

import re
import torch


class ParadisLoss(torch.nn.Module):
    """BerHu loss function.

    This loss function implements a weighting scheme that accounts for key aspects
    of meteorological data:

    1. Vertical weighting: Implements pressure-level dependent weights that decrease with altitude.
    2. Variable-specific weighting: Allows different weights for various meteorological variables
       (e.g., temperature, wind, precipitation) to balance their relative importance.

    The loss uses a BerHu function:
    - L1 penalty for small errors (|error| ≤ δ)
    - L2-like penalty for large errors (|error| > δ)
    """

    def __init__(
        self,
        lat_grid: torch.Tensor,
        pressure_levels: torch.Tensor,
        num_features: int,
        num_surface_vars: int,
        var_loss_weights: torch.Tensor,
        output_name_order: list,
        delta: float = 1.0,
    ) -> None:
        """Initialize the weighted BerHu loss function.

        Args:
            lat_grid: Latitude grid
            pressure_levels: Pressure levels in hPa used in the model
            num_features: Total number of features in the output
            num_surface_vars: Number of surface-level variables
            var_loss_weights: Variable-specific weights for the loss calculation
            output_name_order: List of variable names in order of output features
            delta: Threshold value (δ)
        """
        super().__init__()

        self.delta = delta

        # Ensure inputs are float32
        self.pressure_levels = pressure_levels.to(torch.float32)

        # Store dimensions
        self.num_levels = len(pressure_levels)
        self.num_features = num_features
        self.num_surface_vars = num_surface_vars
        self.num_atmospheric_vars = num_features - num_surface_vars
        self.var_loss_weights = var_loss_weights
        self.output_name_order = output_name_order

        # Whether to flip geopotential weights
        self.flip_geopotential_weights = False

        # Whether to apply pressure weights
        self.apply_pressure_weights = True

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
        if self.apply_pressure_weights:
            # Compute proper integration weights along pressure coordinate
            pressure_weights = torch.where(
                self.pressure_levels / 1000 > 0.2, 0.2, self.pressure_levels / 1000
            )
        else:
            pressure_weights = torch.ones(
                len(self.pressure_levels), dtype=torch.float32
            )

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

    def _berhu_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the BerHu loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Loss tensor with same shape as input
        """
        delta = self.delta
        diff = torch.abs(pred - target)

        loss = torch.where(
            diff <= delta,
            diff,  # L1 for small errors
            (diff**2 + delta**2) / (2 * delta),  # L2-like for large errors
        )
        return loss

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate weighted BerHu loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Weighted loss value
        """
        # Prepare weights with correct shapes for broadcasting
        feature_weights = self.feature_weights.view(1, -1, 1, 1).to(pred.device)

        # Get the BerHu loss and apply weights
        weighted_loss = self._berhu_loss(pred, target) * feature_weights

        return weighted_loss.mean()
