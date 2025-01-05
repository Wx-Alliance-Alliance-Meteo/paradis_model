"""Cyclic padding implementation for spherical geometry."""

import torch


class GeoCyclicPadding(torch.nn.Module):
    """Cyclic padding for geographical data (for a grid without poles)."""

    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x):
        # Circular padding for longitude (periodic boundary)
        x = torch.cat(
            [x[:, :, :, -self.pad_width :], x, x[:, :, :, : self.pad_width]], dim=3
        )

        # Reflection padding for latitude (mirroring boundaries)
        x = torch.cat(
            [
                x[:, :, : self.pad_width, :].flip(dims=[2]),
                x,
                x[:, :, -self.pad_width :, :].flip(dims=[2]),
            ],
            dim=2,
        )
        return x
