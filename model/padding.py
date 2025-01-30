"""Cyclic padding

The implementation assumes:
- The grid has an even number of longitude points
- Longitude points are evenly spaced
- The grid does not include poles
"""

import torch


class GeoCyclicPadding(torch.nn.Module):
    """Cyclic padding layer for regular lat-lon grids."""

    def __init__(self, pad_width, channels):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cyclic padding to the input tensor."""
        # Validate input dimensions
        assert (
            len(x.shape) == 4
        ), "Input must be 4-dimensional [batch, channels, lat, lon]"
        batch_size, channels, height, width = x.shape
        assert width % 2 == 0, "Number of longitude points must be even"

        # Longitude periodic padding
        x_padded = torch.cat(
            [x[:, :, :, -self.pad_width :], x, x[:, :, :, : self.pad_width]], dim=3
        )

        # For latitude padding, we need to rotate by 180° and account for longitude padding
        middle_index = width // 2 + self.pad_width

        # Apply 180° shift
        top_padding = torch.roll(
            x_padded[:, :, : self.pad_width, :], shifts=middle_index, dims=3
        )
        bottom_padding = torch.roll(
            x_padded[:, :, -self.pad_width :, :], shifts=middle_index, dims=3
        )

        # Combine padded regions
        return torch.cat([top_padding.flip(2), x_padded, bottom_padding.flip(2)], dim=2)
