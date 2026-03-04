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
        south_mean = x[:, :, 0:1, :].mean(dim=3, keepdim=True)
        north_mean = x[:, :, -1:, :].mean(dim=3, keepdim=True)

        # Overwrite the pole rows with the broadcasted mean
        x_fixed = x.clone()
        x_fixed[:, :, 0, :] = south_mean.squeeze(-1)
        x_fixed[:, :, -1, :] = north_mean.squeeze(-1)
        return x_fixed

    def forward(
        self,
        hidden_n: torch.Tensor,
        hidden_nm1: torch.Tensor,
        u_vel: torch.Tensor,
        v_vel: torch.Tensor,
        u_acc: torch.Tensor,
        v_acc: torch.Tensor,
    ) -> tuple:
        """BDF2 semi-Lagrangian advection step."""
        batch_size = hidden_n.shape[0]

        hidden_n = self.enforce_pole_continuity(hidden_n)
        hidden_nm1 = self.enforce_pole_continuity(hidden_nm1)

        proj_n = self.down_projection(hidden_n)
        proj_nm1 = self.down_projection(hidden_nm1)

        # alpha^(1): displacement evaluated at xi=1 (back-trace by dt)
        alpha1_lon = u_vel + u_acc
        alpha1_lat = v_vel + v_acc

        # alpha^(2): displacement evaluated at xi=2 (back-trace by 2*dt)
        alpha2_lon = 2.0 * u_vel + 4.0 * u_acc
        alpha2_lat = 2.0 * v_vel + 4.0 * v_acc

        # Interpolate z^n at x - alpha^(1)
        grid1 = self._compute_grid(-alpha1_lat, -alpha1_lon, batch_size)
        z_tilde_n = self._sample(proj_n, grid1, batch_size)
        z_tilde_n = self.up_projection(z_tilde_n)
        z_tilde_n = self.enforce_pole_continuity(z_tilde_n)

        # Interpolate z^{n-1} at x - alpha^(2)
        grid2 = self._compute_grid(-alpha2_lat, -alpha2_lon, batch_size)
        z_tilde_nm1 = self._sample(proj_nm1, grid2, batch_size)
        z_tilde_nm1 = self.up_projection(z_tilde_nm1)
        z_tilde_nm1 = self.enforce_pole_continuity(z_tilde_nm1)

        return z_tilde_n, z_tilde_nm1

    def _compute_grid(
        self,
        lat_prime: torch.Tensor,
        lon_prime: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Compute normalised grid_sample coordinates from local angular displacements."""
        H, W = self.mesh_size

        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, self.lat_grid, self.lon_grid
        )

        pix_x = (lon_dep - self.min_lon) / self.d_lon * (self.Wf - 1.0)
        pix_y = (lat_dep - self.min_lat) / self.d_lat * (self.Hf - 1.0)

        H_pad = H + 2 * self.padding
        W_pad = W + 2 * self.padding

        pix_x_pad = pix_x + self.padding
        pix_y_pad = pix_y + self.padding

        grid_x = 2.0 * (pix_x_pad / float(W_pad - 1)) - 1.0
        grid_y = 2.0 * (pix_y_pad / float(H_pad - 1)) - 1.0

        grid_x = grid_x.reshape(batch_size * self.num_vels, H, W)
        grid_y = grid_y.reshape(batch_size * self.num_vels, H, W)

        return torch.stack([grid_x, grid_y], dim=-1)

    def _sample(
        self,
        projected: torch.Tensor,
        grid: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Apply geocyclic padding and grid_sample, then reshape."""
        H, W = self.mesh_size
        H_pad = H + 2 * self.padding
        W_pad = W + 2 * self.padding

        padded = self.padding_interp(projected)
        padded = padded.reshape(batch_size * self.num_vels, 1, H_pad, W_pad)

        interpolated = torch.nn.functional.grid_sample(
            padded,
            grid,
            align_corners=True,
            mode=self.interpolation,
            padding_mode="zeros",
        )

        return interpolated.reshape(batch_size, self.num_vels, H, W)
