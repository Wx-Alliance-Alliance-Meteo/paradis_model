import torch


class GeoCyclicPadding(torch.nn.Module):
    """Cyclic padding layer for equiangular grids with poles.

    When a SpatialParallelContext is supplied, latitude is distributed across
    ranks and only the halo rows are exchanged. Longitude remains local and is
    padded periodically on every rank.
    """

    def __init__(self, pad_width, spatial_context=None, global_height=None):
        super().__init__()
        self.pad_width = pad_width
        self.spatial_context = spatial_context
        self.global_height = global_height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply pole-aware latitude padding and periodic longitude padding."""
        if self.pad_width == 0:
            return x

        if len(x.shape) != 4:
            raise ValueError("Input must be 4-dimensional [batch, channels, lat, lon]")
        _, _, height, width = x.shape

        if width % 2 != 0:
            raise ValueError("Number of longitude points must be even")

        if self.spatial_context is not None and self.spatial_context.enabled:
            if self.global_height is None:
                raise ValueError(
                    "global_height is required for spatially distributed GeoCyclicPadding"
                )
            expected = self.spatial_context.local_height(self.global_height)
            if height != expected:
                raise ValueError(
                    f"Expected local latitude height {expected} for global height "
                    f"{self.global_height}, got {height}"
                )
            x = self.spatial_context.latitude_halo(x, self.pad_width)
        else:
            middle_index = width // 2

            # Apply 180 degree roll to top and bottom rows.
            top_source = x[:, :, 1 : self.pad_width + 1, :]
            bottom_source = x[:, :, -(self.pad_width + 1) : -1, :]

            top_padding = torch.roll(top_source, shifts=middle_index, dims=3)
            bottom_padding = torch.roll(bottom_source, shifts=middle_index, dims=3)

            x = torch.cat(
                [top_padding.flip(2), x, bottom_padding.flip(2)], dim=2
            )

        # Longitude is never sharded, so periodic padding stays entirely local.
        return torch.cat(
            [x[:, :, :, -self.pad_width :], x, x[:, :, :, : self.pad_width]],
            dim=3,
        )
