"""Physically inspired neural architecture for the weather forecasting model."""

import torch
from torch import nn

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
    ):
        super().__init__()

        # For cubic interpolation
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
        )

        self.up_projection = GMBlock(
            input_dim=num_vels,
            output_dim=hidden_dim,
            mesh_size=mesh_size,
            layers=["CLinear"],
        )

        self.interpolation = interpolation

        # Neural network that will learn an effective velocity along the trajectory
        # Output 2 channels per hidden dimension for u and v
        self.velocity_net = GMBlock(
            input_dim=hidden_dim,
            output_dim=3 * num_vels,
            hidden_dim=hidden_dim,
            kernel_size=3,
            mesh_size=mesh_size,
            layers=["SepConv"],
            bias_channels=bias_channels,
            activation=False,
            pre_normalize=True,
        )

        H, W = mesh_size

        # Store for later use
        self.register_buffer("lat_grid", lat_grid.unsqueeze(0).unsqueeze(0))
        self.register_buffer("lon_grid", lon_grid.unsqueeze(0).unsqueeze(0))

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
    
    def _angular_rotation(
        self,
        lat: torch.Tensor, 
        lon: torch.Tensor, 
        omega: torch.Tensor, 
        dt: float):
        sin_lat = torch.sin(lat)
        cos_lat = torch.cos(lat)
        sin_lon = torch.sin(lon)
        cos_lon = torch.cos(lon)
        
        # Coordinates in cartesian
        x0 = torch.stack([
            cos_lat * cos_lon, 
            cos_lat * sin_lon, 
            sin_lat
        ], dim=1)
        
        # Rotation angle
        omega_norm = torch.norm(omega, dim=1)
        theta = -omega_norm * dt
        
        u = omega / omega_norm.unsqueeze(1)
        
        sin_theta = torch.sin(theta).unsqueeze(1)
        cos_theta = torch.cos(theta).unsqueeze(1)
        
        u_cross_x0 = torch.cross(u, x0, dim=1)
        u_dot_x0 = (u * x0).sum(dim=1, keepdim=True)
        
        # Compute new position
        x_new = x0 * cos_theta + u_cross_x0 * sin_theta + u * (u_dot_x0 * (1 - cos_theta))
        x_new = x_new / torch.norm(x_new, dim=1, keepdim=True)
        
        # Back to (lat, lon)
        x_comp = x_new[:, 0, :, :, :]
        y_comp = x_new[:, 1, :, :, :]
        z_comp = x_new[:, 2, :, :, :]

        lat_dep = torch.asin(z_comp)
        lon_dep = torch.atan2(y_comp, x_comp)

        return lat_dep, lon_dep

    def forward(
        self,
        hidden_features: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute advection using rotated coordinate system."""
        batch_size = hidden_features.shape[0]

        # Get learned velocities for each channel
        velocities = self.velocity_net(hidden_features)

        # Reshape velocities to separate u,v components per channel
        # [batch, 2*hidden_dim, lat, lon] -> [batch, hidden_dim, 3, lat, lon]
        velocities = velocities.view(batch_size, 3, self.num_vels, *self.mesh_size)

        lat_dep, lon_dep = self._angular_rotation(self.lat_grid, self.lon_grid, velocities, dt)
        
        # Convert departure points to pixel locations
        # For example, pixel_x now in [0 .. W-1], pixel_y in [0 .. H-1]
        pix_x = (lon_dep - self.min_lon) / self.d_lon * (self.Wf - 1.0)
        pix_y = (lat_dep - self.min_lat) / self.d_lat * (self.Hf - 1.0)

        # Down-project latent features to num_vel channels
        projected_inputs = self.down_projection(hidden_features)
        
        # Add padding
        dynamic_padded = self.padding_interp(projected_inputs)

        # Shift pixels by the padding width
        # [0, 1, 2, 3...] -> [2, 3, 4, 5...] (if pad_width=2)
        pix_x_pad = pix_x + self.pad
        pix_y_pad = pix_y + self.pad

        # Normalize into [-1, 1]
        _, _, H_pad, W_pad = dynamic_padded.shape
        grid_x = 2.0 * (pix_x_pad / float(W_pad - 1)) - 1.0
        grid_y = 2.0 * (pix_y_pad / float(H_pad - 1)) - 1.0

        grid_x = grid_x.view(batch_size * self.num_vels, *grid_x.shape[-2:])
        grid_y = grid_y.view(batch_size * self.num_vels, *grid_y.shape[-2:])

        # Create interpolation grid
        grid = torch.stack([grid_x, grid_y], dim=-1)

        # Apply padding and reshape features
        dynamic_padded = dynamic_padded.reshape(
            batch_size * self.num_vels, 1, *dynamic_padded.shape[-2:]
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
            interpolated.view(batch_size, self.num_vels, *self.mesh_size)
        )

        return interpolated


class Paradis(nn.Module):
    """Weather forecasting model main class."""

    # Synoptic time scale (~1/Ω) in seconds
    SYNOPTIC_TIME_SCALE = 7.29212e5

    def __init__(self, datamodule, cfg, lat_grid, lon_grid):
        super().__init__()

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
        else:
            hidden_dim = cfg.model.latent_size
            num_vels = cfg.model.velocity_vectors
            diffusion_size = cfg.model.diffusion_size

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
        )

        # Rescale the time step to a fraction of a synoptic time scale
        self.num_layers = cfg.model.num_layers
        self.dt = (
            cfg.model.base_dt / self.SYNOPTIC_TIME_SCALE  # / max(1, self.num_layers)
        )

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
                )
                for _ in range(self.num_layers)
            ]
        )

        # Diffusion-reaction layer
        self.diffusion_reaction = nn.ModuleList(
            [
                GMBlock(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=diffusion_size,
                    layers=["SepConv", "CLinear", "SepConv"],
                    mesh_size=mesh_size,
                    activation=False,
                    pre_normalize=True,
                    bias_channels=bias_channels,
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
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # No gradients on lat/lon, ever
        x_static = x[:, self.dynamic_channels :].detach()

        # Project features to latent space
        z = self.input_proj(x)

        # Compute advection and diffusion-reaction
        for i in range(self.num_layers):
            # Advect the features in latent space using a Semi-Lagrangian step
            z_adv = self.advection[i](z, self.dt)

            # Compute the diffusion residual
            dz = self.diffusion_reaction[i](z_adv)

            # Update the latent space features
            z = z + dz * self.dt

        # Return a scaled residual formulation
        return x[
            :,
            (self.n_inputs - 1)
            * self.num_common_features : self.n_inputs
            * self.num_common_features,
        ] + self.output_proj(z)
