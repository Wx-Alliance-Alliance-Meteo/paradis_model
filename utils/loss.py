"""Loss functions for the weather forecasting model."""

import re
import torch

from model.padding import GeoCyclicPadding
from utils.amse_loss import AMSELoss


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
        apply_latitude_weights: bool = False,
    ) -> None:
        """Initialize the weighted reversed Huber loss function.

        Args:
            loss_function: A choice between reversed_huber or mse loss functions
            lat_grid: Latitude grid
            pressure_levels: Pressure levels in hPa used in the model
            num_features: Total number of features in the output
            num_surface_vars: Number of surface-level variables
            var_loss_weights: Variable-specific weights for the loss calculation
            output_name_order: List of variable names in order of output features
            delta_loss: Threshold parameter for the Huber loss
            apply_latitude_weights: Whether to integrate the loss using geometric weights along latitude
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

        # Whether to apply pressure weights
        self.apply_pressure_weights = True

        # Whether to apply latitude weights in loss integration
        self.apply_latitude_weights = apply_latitude_weights
        self.lat_weights = self._compute_latitude_weights(lat_grid)

        # Create combined feature weights
        self.feature_weights = self._create_feature_weights()

        if loss_function == "mse":
            self.loss_fn = torch.nn.MSELoss(reduction="none")
        elif loss_function == "reversed_huber":
            self.loss_fn = self._call_reversed_huber_loss
        elif loss_function == "amse":
            self.loss_fn = AMSELoss(
                nlat=lat_grid.shape[0], nlon=lat_grid.shape[1], grid="equiangular"
            )
            # Latitude weight application needs to be deactivated
            self.apply_latitude_weights = False
        else:
            raise Exception(
                f"{loss_function} not supported, choose between [reversed_huber, mse, amse]"
            )

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

    def _compute_latitude_weights(self, grid_lat_deg: torch.Tensor) -> torch.Tensor:
        """
        GraphCast-consistent latitude weights (unit-mean).

        Supports uniform latitude vectors that either:
        A) include poles:  [-90, ..., 90]
        B) exclude poles:  [-90 + d/2, ..., 90 - d/2]

        grid_lat_deg: 1D tensor [H] in degrees, uniformly spaced, monotone.
        """
        lat = grid_lat_deg.to(dtype=torch.float64)

        # --- checks: 1D, uniform spacing ---
        if lat.ndim != 1:
            raise ValueError(f"grid_lat_deg must be 1D [H], got {lat.shape}")

        d = lat[1:] - lat[:-1]
        d0 = d[0]
        if not torch.allclose(d, d0.expand_as(d), rtol=0.0, atol=1e-6):
            raise ValueError("Latitude grid is not uniformly spaced.")

        delta = torch.abs(d0)
        lat_min = torch.min(lat)
        lat_max = torch.max(lat)

        has_poles = torch.isclose(
            lat_min, lat.new_tensor(-90.0), atol=1e-6
        ) and torch.isclose(lat_max, lat.new_tensor(90.0), atol=1e-6)

        if has_poles:
            # Interior weights proportional to cos(lat) * sin(d/2),
            # pole weights proportional to sin(d/4)^2
            lat_rad = torch.deg2rad(lat)
            delta_rad = torch.deg2rad(delta)

            weights = torch.cos(lat_rad) * torch.sin(delta_rad / 2.0)
            pole_w = torch.sin(delta_rad / 4.0) ** 2

            # poles are the extrema
            idx_min = torch.argmin(lat)
            idx_max = torch.argmax(lat)
            weights[idx_min] = pole_w
            weights[idx_max] = pole_w

        else:
            expected_max = 90.0 - float(delta) / 2.0
            expected_min = -90.0 + float(delta) / 2.0
            if not (
                torch.isclose(lat_max, lat.new_tensor(expected_max), atol=1e-6)
                and torch.isclose(lat_min, lat.new_tensor(expected_min), atol=1e-6)
            ):
                raise ValueError(
                    f"Latitude vector must end at ±(90 - Δ/2). "
                    f"Got min={lat_min.item()}, max={lat_max.item()}, Δ={delta.item()}."
                )
            weights = torch.cos(torch.deg2rad(lat))

        # Unit-mean normalization (GraphCast style)
        weights = weights / weights.mean()

        # Return in original dtype (float32 typically) for cheap broadcasting
        return weights.to(dtype=grid_lat_deg.dtype)

    def _create_feature_weights(self) -> torch.Tensor:
        """Create weights for all features."""
        # Initialize weights tensor for all features
        feature_weights = torch.zeros(self.num_features, dtype=torch.float32)

        # Standard pressure weights normalized by number of levels
        if self.apply_pressure_weights:
            # Compute proper integration weights along pressure coordinate
            pressure_weights = torch.where(
                self.pressure_levels / 1000 > 0.2, self.pressure_levels / 1000, 0.2
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

    def _huber_quad(self, error, delta):
        return (error**2 + delta**2) / (2 * delta)

    def _reversed_huber_loss(
        self, pred: torch.Tensor, target: torch.Tensor, delta
    ) -> torch.Tensor:
        """Compute the pseudo reversed Huber loss.

        Args:
            error: Error tensor (pred - target)

        Returns:
            Loss tensor with same shape as input
        """
        delta = torch.as_tensor(
            delta, device=pred.device, dtype=pred.dtype
        )
        error = pred - target
        abs_error = torch.abs(error)
        small_error = delta * abs_error
        large_error = self._huber_quad(error, delta)
        weight = 1 / (1 + torch.exp(-2 * (abs_error - delta)))
        return (1 - weight) * small_error + weight * large_error

    def _call_reversed_huber_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        return self._reversed_huber_loss(pred, target, self.delta)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Calculate weighted reversed Huber loss.

        Args:
            pred: Predicted values
            target: Target values

        Returns:
            Weighted loss value
        """
        # Prepare weights with correct shapes for broadcasting
        feature_weights = self.feature_weights.view(1, -1, 1, 1).to(pred.device)

        # Get the loss using the appropriate function
        loss = self.loss_fn(pred, target)

        # Apply weights to loss components
        weighted_loss = loss * feature_weights

        if self.apply_latitude_weights:
            lat_weights = self.lat_weights.view(1, 1, -1, 1).to(pred.device)
            weighted_loss *= lat_weights

        return weighted_loss.mean()
