"""Cyclic padding

The implementation assumes:
- The grid has an even number of longitude points
- Longitude points are evenly spaced
- The grid does not include poles
- The sign must be flipped for channels corresponding to vector components across polar boundaries. This is a learned transformation.
"""

import torch


class GeoCyclicPadding(torch.nn.Module):
    """Cyclic padding layer for regular lat-lon grids."""

    def __init__(self, pad_width, channels):
        super().__init__()
        self.pad_width = pad_width

        self.register_parameter(
            "sign_mask_north", torch.nn.Parameter(torch.ones(1, channels, 1, 1))
        )
        self.register_parameter(
            "sign_mask_south", torch.nn.Parameter(torch.ones(1, channels, 1, 1))
        )

    def _constrain_mask(self):
        """Force mask to be either -1 or 1."""
        with torch.no_grad():
            for mask in [self.sign_mask_north, self.sign_mask_south]:
                mask.data.copy_(torch.sign(mask))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cyclic padding to the input tensor."""

        # Validate input dimensions
        assert (
            len(x.shape) == 4
        ), "Input must be 4-dimensional [batch, channels, lat, lon]"
        batch_size, channels, height, width = x.shape
        assert width % 2 == 0, "Number of longitude points must be even"

        # Move masks to input device if needed
        if self.sign_mask_north.device != x.device:
            self.sign_mask_north.data = self.sign_mask_north.data.to(x.device)
            self.sign_mask_south.data = self.sign_mask_south.data.to(x.device)

        self._constrain_mask()

        # Longitude periodic padding
        x_padded = torch.cat(
            [x[:, :, :, -self.pad_width :], x, x[:, :, :, : self.pad_width]], dim=3
        )

        # Extract and transform boundary regions
        top_rows = x_padded[:, :, : self.pad_width, :] * self.sign_mask_north
        bottom_rows = x_padded[:, :, -self.pad_width :, :] * self.sign_mask_south

        # For latitude padding, we need to rotate by 180° and account for longitude padding
        middle_index = width // 2 + self.pad_width

        # Apply 180° shift
        top_padding = torch.roll(top_rows, shifts=middle_index, dims=3)
        bottom_padding = torch.roll(bottom_rows, shifts=middle_index, dims=3)

        # Combine padded regions
        return torch.cat([top_padding.flip(2), x_padded, bottom_padding.flip(2)], dim=2)
