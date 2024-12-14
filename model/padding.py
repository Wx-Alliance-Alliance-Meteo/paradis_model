"""Cyclic padding implementation for spherical geometry."""

import torch
import torch.nn as nn

class GeoCyclicPadding(nn.Module):
    """Cyclic padding for geographical data."""

    def __init__(self, pad_width):
        super().__init__()
        self.pad_width = pad_width

    def forward(self, x):
        batch_size, channels, height, width = x.shape

        # Circular padding for longitude (periodic boundary)
        circular_padded = torch.cat(
            [x[:, :, :, -self.pad_width :], x, x[:, :, :, : self.pad_width]], dim=3
        )

        # Zero padding for latitude
        top_bottom_padded = torch.zeros(
            batch_size,
            channels,
            height + 2 * self.pad_width,
            circular_padded.shape[3],
            device=x.device,
        )

        # Place circular padded tensor in center
        top_bottom_padded[:, :, self.pad_width : height + self.pad_width, :] = (
            circular_padded
        )

        # Custom padding logic for poles
        middle_index = width // 2
        for i in range(self.pad_width):
            # North Pole
            top_row = (self.pad_width - i - 1) % height
            top_padding = torch.cat(
                (
                    circular_padded[:, :, top_row, middle_index:],
                    circular_padded[:, :, top_row, :middle_index],
                ),
                dim=-1,
            )
            top_padding = top_padding.reshape(batch_size, channels, 1, -1)
            top_bottom_padded[:, :, i, :] = top_padding[:, :, 0, :]

            # South Pole
            bottom_row = (height - i - 1) % height
            bottom_padding = torch.cat(
                (
                    circular_padded[:, :, bottom_row, middle_index:],
                    circular_padded[:, :, bottom_row, :middle_index],
                ),
                dim=-1,
            )
            bottom_padding = bottom_padding.reshape(batch_size, channels, 1, -1)
            top_bottom_padded[:, :, height + self.pad_width + i, :] = bottom_padding[
                :, :, 0, :
            ]

        return top_bottom_padded
