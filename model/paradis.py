"""Physically inspired neural architecture for the weather forecasting model."""

import torch
from torch import nn

from model.padding import GeoCyclicPadding
from model.gmblock import GMBlock
from model.simple_blocks import ChannelNorm


class NeuralSemiLagrangian(nn.Module):
    """Implements the semi-Lagrangian advection."""

    def __init__(
        self,
        hidden_dim: int,
        mesh_size: tuple,
        num_vels: int,
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
            output_dim=2 * num_vels,
            hidden_dim=hidden_dim,
            kernel_size=3,
            mesh_size=mesh_size,
            layers=["SepConv"],
            bias_channels=bias_channels,
            activation=False,
            pre_normalize=True,
        )

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
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute advection using rotated coordinate system."""
        batch_size = hidden_features.shape[0]

        # Get learned velocities for each channel
        velocities = self.velocity_net(hidden_features)

        # Reshape velocities to separate u,v components per channel
        # [batch, 2*hidden_dim, lat, lon] -> [batch, hidden_dim, 2, lat, lon]
        velocities = velocities.view(batch_size, 2, self.num_vels, *self.mesh_size)

        # Extract learned u,v components
        u = velocities[:, 0]
        v = velocities[:, 1]

        # Compute departure points in a local rotated coordinate system in which the origin
        # of latitude and longitude is moved to the arrival point
        lon_prime = -u * dt
        lat_prime = -v * dt

        # Transform from rotated coordinates back to standard coordinates
        # Expand lat/lon grid for broadcasting with per-channel coordinates
        lat_grid = lat_grid.unsqueeze(1).expand(-1, self.num_vels, -1, -1)
        lon_grid = lon_grid.unsqueeze(1).expand(-1, self.num_vels, -1, -1)

        # Get the max and min values for normalization
        min_lat = torch.min(lat_grid)
        max_lat = torch.max(lat_grid)

        min_lon = torch.min(lon_grid)
        max_lon = torch.max(lon_grid)

        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, lat_grid, lon_grid
        )

        # Normalize grid to ensure consistency in interpolation
        grid_x = 2 * (lon_dep - min_lon) / (max_lon - min_lon) - 1
        grid_y = 2 * (lat_dep - min_lat) / (max_lat - min_lat) - 1

        # Apply periodicity for outside values along longitude set to [-1, 1]
        grid_x = torch.remainder(grid_x + 1, 2) - 1

        # Apply geocyclic longitude roll for values beyond +/-90 degrees latitude
        geo_mask_left = grid_x <= 0
        geo_mask_right = grid_x > 0
        lat_mask_outer = torch.abs(grid_y) > 1
        grid_x = torch.where(lat_mask_outer & geo_mask_left, grid_x + 1, grid_x)
        grid_x = torch.where(lat_mask_outer & geo_mask_right, grid_x - 1, grid_x)

        # Mirror values outside of the range [-1, 1] in the latitude direction
        grid_y = torch.where(grid_y < -1, -(2 + grid_y), grid_y)
        grid_y = torch.where(grid_y > 1, 2 - grid_y, grid_y)

        # Reshape grid coordinates for interpolation
        # [batch, dynamic_channels, lat, lon] -> [batch*dynamic_channels, lat, lon]
        grid_x = grid_x.view(batch_size * self.num_vels, *grid_x.shape[-2:])
        grid_y = grid_y.view(batch_size * self.num_vels, *grid_y.shape[-2:])

        # Down-project to num_vel channels
        projected_inputs = self.down_projection(hidden_features)

        # Apply padding and reshape hidden features
        dynamic_padded = self.padding_interp(projected_inputs)

        # Make sure interpolation remains in right range after padding
        grid_x = grid_x * hidden_features.size(-1) / dynamic_padded.size(-1)
        grid_y = grid_y * hidden_features.size(-2) / dynamic_padded.size(-2)

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
            padding_mode="border",
        )

        # Reshape back to original dimensions
        interpolated = interpolated.view(batch_size, self.num_vels, *self.mesh_size)

        # Project back up to latent space
        interpolated = self.up_projection(interpolated)

        return interpolated


class Paradis(nn.Module):
    """Weather forecasting model main class."""

    def __init__(self, datamodule, cfg):
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
            diffusion_size = cfg.model.get("diffusion_size", hidden_dim)

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

        num_layers = cfg.model.num_layers

        # Timestep is set to 1 in the latent space
        self.dt = 1

        # Advection layer
        self.advection = NeuralSemiLagrangian(
            hidden_dim,
            mesh_size,
            num_vels=num_vels,
            interpolation=adv_interpolation,
            bias_channels=bias_channels,
        )

        # Shared normalization for diffusion-reaction layers
        self.shared_rhs_norm = ChannelNorm(
            input_dim=hidden_dim, output_dim=hidden_dim, bias=True
        )

        # Diffusion-reaction layers
        self.diffusion_reactions_layers = nn.ModuleList(
            [
                GMBlock(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=diffusion_size,
                    layers=["SepConv"],
                    mesh_size=mesh_size,
                    activation=True,
                    pre_normalize=False,
                    bias_channels=bias_channels,
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = GMBlock(
            input_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            layers=["SepConv", "SepConv"],
            mesh_size=mesh_size,
            activation=False,
            bias_channels=bias_channels,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model time step."""

        # No gradients on lat/lon, ever
        x_static = x[:, self.dynamic_channels :].detach()

        # Extract lat/lon from static features (last 2 channels)
        lat_grid = x_static[:, -2, :, :]
        lon_grid = x_static[:, -1, :, :]

        # Extract relevant input features that correspond to output (common features from last time step)
        skip = x[
            :,
            (self.n_inputs - 1)
            * self.num_common_features : self.n_inputs
            * self.num_common_features,
        ]

        # Project features to latent space
        z0 = self.input_proj(x)

        # Advection
        z_adv = self.advection(z0, lat_grid, lon_grid, self.dt)

        # Shared normalization for all RHS computations
        z_adv_normalized = self.shared_rhs_norm(z_adv)

        # Diffusion-reaction - sum contributions from all layers
        rhs = torch.zeros_like(z0)
        for layer in self.diffusion_reactions_layers:
            rhs = rhs + layer(z_adv_normalized)

        # Final state
        z_final = z_adv + self.dt * rhs

        # Input-output skip connections: Model learns incremental changes rather than full state predictions
        return skip + self.output_proj(z_final)
