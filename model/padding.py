import torch


class GeoCyclicPadding(torch.nn.Module):
    """Cyclic padding layer for equiangular grids with poles."""

    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cyclic padding to the input tensor."""
        if self.pad_width == 0:
            return x

        assert (
            len(x.shape) == 4
        ), "Input must be 4-dimensional [batch, channels, lat, lon]"
        batch_size, channels, height, width = x.shape

        assert width % 2 == 0, "Number of longitude points must be even"

        middle_index = width // 2

        # Apply 180 degree roll to top and bottom rows
        top_source = x[:, :, 1 : self.pad_width + 1, :]
        bottom_source = x[:, :, -(self.pad_width + 1) : -1, :]

        top_padding = torch.roll(top_source, shifts=middle_index, dims=3)
        bottom_padding = torch.roll(bottom_source, shifts=middle_index, dims=3)

        x = torch.cat([top_padding.flip(2), x, bottom_padding.flip(2)], dim=2)

        # Apply periodic padding
        x_padded = torch.cat(
            [x[:, :, :, -self.pad_width :], x, x[:, :, :, : self.pad_width]], dim=3
        )

        return x_padded
