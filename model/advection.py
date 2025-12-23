import torch

from model.blocks import GMBlock
from model.padding import GeoCyclicPadding


class NeuralSemiLagrangian(torch.nn.Module):
    """Neural semi-Lagrangian advection operator."""

    def __init__(
        self,
        hidden_dim: int,
        mesh_size: tuple,
        num_vels: int,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        interpolation: str = "bicubic",
        project_advection=True,
    ):
        super().__init__()

        self.padding = 1
        if interpolation == "bicubic":
            self.padding = 2

        self.padding_interp = GeoCyclicPadding(self.padding)
        self.hidden_dim = hidden_dim
        self.num_vels = num_vels
        self.mesh_size = mesh_size

        if project_advection:
            self.down_projection = GMBlock(
                layers=["CLinear"],
                input_dim=hidden_dim,
                output_dim=num_vels,
                mesh_size=mesh_size,
                kernel_size=1,
            )

            self.up_projection = GMBlock(
                layers=["SepConv"],
                input_dim=num_vels,
                output_dim=hidden_dim,
                mesh_size=mesh_size,
                kernel_size=1,
            )
        else:
            self.num_vels = hidden_dim
            self.down_projection = lambda x: x
            self.up_projection = lambda x: x

        self.interpolation = interpolation

        H, W = mesh_size

        self.register_buffer(
            "lat_grid", lat_grid.unsqueeze(0).unsqueeze(0).contiguous().clone()
        )
        self.register_buffer(
            "lon_grid", lon_grid.unsqueeze(0).unsqueeze(0).contiguous().clone()
        )

        self.register_buffer("Hf", torch.tensor(float(H)))
        self.register_buffer("Wf", torch.tensor(float(W)))
        self.register_buffer("min_lat", torch.min(lat_grid))
        self.register_buffer("max_lat", torch.max(lat_grid))
        self.register_buffer("min_lon", torch.min(lon_grid))
        self.register_buffer("max_lon", torch.max(lon_grid))
        self.register_buffer("d_lon", self.max_lon - self.min_lon)
        self.register_buffer("d_lat", self.max_lat - self.min_lat)

    def _transform_to_latlon(
        self,
        lat_prime: torch.Tensor,
        lon_prime: torch.Tensor,
        lat_p: torch.Tensor,
        lon_p: torch.Tensor,
    ) -> tuple:
        """Transform from local rotated coordinates back to standard latlon coordinates."""
        sin_lat_prime = torch.sin(lat_prime)
        cos_lat_prime = torch.cos(lat_prime)
        sin_lon_prime = torch.sin(lon_prime)
        cos_lon_prime = torch.cos(lon_prime)
        sin_lat_p = torch.sin(lat_p)
        cos_lat_p = torch.cos(lat_p)

        sin_lat = sin_lat_prime * cos_lat_p + cos_lat_prime * cos_lon_prime * sin_lat_p
        lat = torch.arcsin(torch.clamp(sin_lat, -1 + 1e-7, 1 - 1e-7))

        num = cos_lat_prime * sin_lon_prime
        den = cos_lat_prime * cos_lon_prime * cos_lat_p - sin_lat_prime * sin_lat_p
        lon = lon_p + torch.atan2(num, den)

        lon = torch.remainder(lon + 2 * torch.pi, 2 * torch.pi)

        return lat, lon

    def enforce_pole_continuity(self, x):
        """
        Forces the South Pole (row 0) and North Pole (row -1) to have
        a single scalar value (mean of the row).
        """
        # Calculate global mean for South Pole (Row 0)
        # x: [B, C, H, W]
        south_mean = x[:, :, 0:1, :].mean(dim=3, keepdim=True)
        north_mean = x[:, :, -1:, :].mean(dim=3, keepdim=True)

        # Overwrite the pole rows with the broadcasted mean
        x_fixed = x.clone()
        x_fixed[:, :, 0, :] = south_mean.squeeze(-1)
        x_fixed[:, :, -1, :] = north_mean.squeeze(-1)
        return x_fixed

    def forward(
        self,
        hidden_features: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute advection using rotated coordinate system."""
        batch_size = hidden_features.shape[0]
        H, W = self.mesh_size

        hidden_features = self.enforce_pole_continuity(hidden_features)

        projected_inputs = self.down_projection(hidden_features)

        lon_prime = -u * dt
        lat_prime = -v * dt

        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, self.lat_grid, self.lon_grid
        )

        pix_x = (lon_dep - self.min_lon) / self.d_lon * (self.Wf - 1.0)
        pix_y = (lat_dep - self.min_lat) / self.d_lat * (self.Hf - 1.0)

        projected_padded = self.padding_interp(projected_inputs)

        pix_x_pad = pix_x + self.padding
        pix_y_pad = pix_y + self.padding

        H_pad = H + 2 * self.padding
        W_pad = W + 2 * self.padding

        grid_x = 2.0 * (pix_x_pad / float(W_pad - 1)) - 1.0
        grid_y = 2.0 * (pix_y_pad / float(H_pad - 1)) - 1.0

        grid_x = grid_x.reshape(batch_size * self.num_vels, H, W)
        grid_y = grid_y.reshape(batch_size * self.num_vels, H, W)

        grid = torch.stack([grid_x, grid_y], dim=-1)

        projected_padded = projected_padded.reshape(
            batch_size * self.num_vels, 1, H_pad, W_pad
        )

        interpolated = torch.nn.functional.grid_sample(
            projected_padded,
            grid,
            align_corners=True,
            mode=self.interpolation,
            padding_mode="zeros",
        )

        interpolated = self.up_projection(
            interpolated.reshape(batch_size, self.num_vels, H, W)
        )

        interpolated = self.enforce_pole_continuity(interpolated)
        return interpolated
