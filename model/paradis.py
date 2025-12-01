"""Physically inspired neural architecture for the weather forecasting model."""

import torch
from torch import nn

from torch_harmonics import DiscreteContinuousConvS2

from model.padding import GeoCyclicPadding
from model.gmblock import GMBlock


class NeuralSemiLagrangian(nn.Module):
    """Implements the semi-Lagrangian advection."""

    def __init__(
        self,
        hidden_dim: int,
        mesh_size: tuple,
        num_vels: int,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        interpolation: str = "bicubic",
        bias_channels: int = 4,
        grid_type: str = "equiangular",
        basis_type: str = "morlet",
    ):
        super().__init__()

        # For cubic interpolation
        self.padding = 1
        if interpolation == "bicubic":
            self.padding = 2

        self.padding_interp = GeoCyclicPadding(self.padding)
        self.hidden_dim = hidden_dim

        self.num_vels = num_vels
        self.mesh_size = mesh_size

        self.down_projection = GMBlock(
            input_dim=hidden_dim,
            output_dim=num_vels,
            mesh_size=mesh_size,
            layers=["CLinear"],
            grid_type=grid_type,
            basis_type=basis_type,
        )

        self.up_projection = GMBlock(
            input_dim=num_vels,
            output_dim=hidden_dim,
            mesh_size=mesh_size,
            layers=["CLinear"],
            grid_type=grid_type,
            basis_type=basis_type,
        )

        self.interpolation = interpolation

        # Neural network that will learn an effective velocity along the trajectory
        # Output 2 channels per hidden dimension for u and v
        self.velocity_net = GMBlock(
            input_dim=hidden_dim,
            output_dim=2 * num_vels,
            hidden_dim=hidden_dim,
            kernel_size=3,
            mesh_size=mesh_size,
            layers=["SepConv"],
            bias_channels=bias_channels,
            activation=False,
            pre_normalize=True,
            grid_type=grid_type,
            basis_type=basis_type,
        )

        H, W = mesh_size

        # Store for later use
        self.register_buffer(
            "lat_grid", lat_grid.unsqueeze(0).unsqueeze(0).contiguous().clone()
        )
        self.register_buffer(
            "lon_grid", lon_grid.unsqueeze(0).unsqueeze(0).contiguous().clone()
        )

        # Buffers: normalization constants
        self.register_buffer("Hf", torch.tensor(float(H)))
        self.register_buffer("Wf", torch.tensor(float(W)))
        self.register_buffer("pad", torch.tensor(float(self.padding)))

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
        # Pre-compute trigonometric functions
        sin_lat_prime = torch.sin(lat_prime)
        cos_lat_prime = torch.cos(lat_prime)
        sin_lon_prime = torch.sin(lon_prime)
        cos_lon_prime = torch.cos(lon_prime)
        sin_lat_p = torch.sin(lat_p)
        cos_lat_p = torch.cos(lat_p)

        # Compute standard latitude
        sin_lat = sin_lat_prime * cos_lat_p + cos_lat_prime * cos_lon_prime * sin_lat_p
        lat = torch.arcsin(torch.clamp(sin_lat, -1 + 1e-7, 1 - 1e-7))

        # Compute standard longitude
        num = cos_lat_prime * sin_lon_prime
        den = cos_lat_prime * cos_lon_prime * cos_lat_p - sin_lat_prime * sin_lat_p

        lon = lon_p + torch.atan2(num, den)

        # Normalize longitude to [0, 2π]
        lon = torch.remainder(lon + 2 * torch.pi, 2 * torch.pi)

        return lat, lon

    def forward(
        self,
        hidden_features: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute advection using rotated coordinate system."""
        batch_size = hidden_features.shape[0]
        H, W = self.mesh_size

        # Get learned velocities for each channel
        velocities = self.velocity_net(hidden_features)

        # Reshape velocities to separate u,v components per channel
        # [batch, 2*num_vels, lat, lon] -> [batch, 2, num_vels, 2, lat, lon]
        velocities = velocities.reshape(batch_size, 2, self.num_vels, H, W)

        # Extract learned u,v components
        u = velocities[:, 0]
        v = velocities[:, 1]

        # Down-project latent features to num_vel channels
        projected_inputs = self.down_projection(hidden_features)

        # Compute departure points in a local rotated coordinate system in which the origin
        # of latitude and longitude is moved to the arrival point
        lon_prime = -u * dt
        lat_prime = -v * dt

        # Transform from rotated coordinates back to standard coordinates
        # Expand lat/lon grid for broadcasting with per-channel coordinates
        lat_grid = self.lat_grid.expand(-1, self.num_vels, -1, -1)
        lon_grid = self.lon_grid.expand(-1, self.num_vels, -1, -1)

        # Compute the departure lat/lon grid
        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, lat_grid, lon_grid
        )

        # Convert departure points to pixel locations
        # For example, pixel_x now in [0 .. W-1], pixel_y in [0 .. H-1]
        pix_x = (lon_dep - self.min_lon) / self.d_lon * (self.Wf - 1.0)
        pix_y = (lat_dep - self.min_lat) / self.d_lat * (self.Hf - 1.0)

        # Add padding
        dynamic_padded = self.padding_interp(projected_inputs)

        # Shift pixels by the padding width
        # [0, 1, 2, 3...] -> [2, 3, 4, 5...] (if pad_width=2)
        pix_x_pad = pix_x + self.pad
        pix_y_pad = pix_y + self.pad

        # Normalize into [-1, 1]
        H_pad = H + 2 * self.padding
        W_pad = W + 2 * self.padding

        grid_x = 2.0 * (pix_x_pad / float(W_pad - 1)) - 1.0
        grid_y = 2.0 * (pix_y_pad / float(H_pad - 1)) - 1.0

        grid_x = grid_x.reshape(batch_size * self.num_vels, H, W)
        grid_y = grid_y.reshape(batch_size * self.num_vels, H, W)

        # Create interpolation grid
        grid = torch.stack([grid_x, grid_y], dim=-1)

        # Apply padding and reshape features
        dynamic_padded = dynamic_padded.reshape(
            batch_size * self.num_vels, 1, H_pad, W_pad
        )

        # Interpolate
        interpolated = torch.nn.functional.grid_sample(
            dynamic_padded,
            grid,
            align_corners=True,
            mode=self.interpolation,
            padding_mode="zeros",
        )

        # Reshape back to original dimensions and project back up to latent space
        interpolated = self.up_projection(
            interpolated.reshape(batch_size, self.num_vels, H, W)
        )

        return interpolated


class Paradis(nn.Module):
    """Weather forecasting model main class."""

    # Synoptic time scale (~1/Ω) in seconds
    SYNOPTIC_TIME_SCALE = 7.29212e5

    def __init__(self, datamodule, cfg, lat_grid, lon_grid):
        super().__init__()

        grid_type = "equiangular"
        basis_type = "morlet"

        # Extract dimensions from config
        output_dim = datamodule.num_out_features
        mesh_size = (datamodule.lat_size, datamodule.lon_size)

        self.num_common_features = datamodule.num_common_features

        # Get channel sizes
        self.dynamic_channels = datamodule.dataset.num_in_dyn_features
        self.static_channels = datamodule.dataset.num_in_static_features

        # Get the number of time inputs
        self.n_inputs = datamodule.dataset.n_time_inputs

        # Specify hidden dimension based on multiplier or fixed size,
        # following configuration file
        if cfg.model.latent_multiplier > 0:
            hidden_dim = (
                cfg.model.latent_multiplier * self.dynamic_channels
                + self.static_channels
            )
            num_vels = hidden_dim
            diffusion_size = hidden_dim
            reaction_size = hidden_dim
        else:
            hidden_dim = cfg.model.latent_size
            num_vels = cfg.model.velocity_vectors
            diffusion_size = cfg.model.diffusion_size
            reaction_size = cfg.model.reaction_size

        # Get the interpolation type
        adv_interpolation = cfg.model.adv_interpolation
        bias_channels = cfg.model.get("bias_channels", 4)

        # Input projection for combined dynamic and static features
        self.input_proj = GMBlock(
            input_dim=self.dynamic_channels + self.static_channels,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            layers=["SepConv", "SepConv"],
            bias_channels=bias_channels,
            mesh_size=mesh_size,
            activation=False,
            pre_normalize=True,
            grid_type=grid_type,
            basis_type=basis_type,
        )

        # Rescale the time step to a fraction of a synoptic time scale
        self.num_layers = max(1, cfg.model.num_layers)
        self.dt = cfg.model.base_dt / self.SYNOPTIC_TIME_SCALE / self.num_layers

        # Advection layer
        self.advection = nn.ModuleList(
            [
                NeuralSemiLagrangian(
                    hidden_dim,
                    mesh_size,
                    num_vels=num_vels,
                    lat_grid=lat_grid,
                    lon_grid=lon_grid,
                    interpolation=adv_interpolation,
                    bias_channels=bias_channels,
                    grid_type=grid_type,
                    basis_type=basis_type,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Diffusion-reaction layer
        self.diffusion = nn.ModuleList(
            [
                GMBlock(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=diffusion_size,
                    layers=["SepConv"],
                    mesh_size=mesh_size,
                    activation=False,
                    pre_normalize=True,
                    bias_channels=bias_channels,
                    grid_type=grid_type,
                    basis_type=basis_type,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.reaction = nn.ModuleList(
            [
                GMBlock(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=reaction_size,
                    layers=["CLinear", "CLinear"],
                    mesh_size=mesh_size,
                    activation=False,
                    pre_normalize=True,
                    bias_channels=bias_channels,
                    grid_type=grid_type,
                    basis_type=basis_type,
                )
                for _ in range(self.num_layers)
            ]
        )

        # Output projection
        self.output_proj = GMBlock(
            input_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            layers=["SepConv", "CLinear"],
            mesh_size=mesh_size,
            activation=False,
            bias_channels=bias_channels,
            grid_type=grid_type,
            basis_type=basis_type,
        )

    def _DR(self, z: torch.Tensor, i: int) -> torch.Tensor:
        return self.diffusion[i](z) + self.reaction[i](z)

    def _step(self, z: torch.Tensor, i: int) -> torch.Tensor:
        # Lie-Trotter splitting with RK2 on diffusion and reaction layers
        zadv = self.advection[i](z, self.dt)
        k1 = self._DR(zadv, i)
        zmid = zadv + 0.5 * self.dt * k1
        k2 = self._DR(zmid, i)
        return zadv + self.dt * k2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project features to latent space
        z = self.input_proj(x)
        z0 = z.clone()

        # Compute advection and diffusion-reaction
        for i in range(self.num_layers):
            z = self._step(z, i)

        return x[
            :,
            (self.n_inputs - 1)
            * self.num_common_features : self.n_inputs
            * self.num_common_features,
        ] + self.output_proj(z - z0)
