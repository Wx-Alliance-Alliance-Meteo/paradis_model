"""Physically-inspired neural architecture for the weather forecasting model."""

import typing
import torch
from torch import nn

from model.padding import GeoCyclicPadding


def CLP(dim_in, dim_out, mesh_size, kernel_size=3, activation=nn.SiLU):
    """Convolutional layer processor."""
    return nn.Sequential(
        GeoCyclicPadding(kernel_size // 2),
        nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size),
        nn.LayerNorm([dim_in, mesh_size[0], mesh_size[1]]),
        activation(),
        GeoCyclicPadding(kernel_size // 2),
        nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size),
    )


class NeuralSemiLagrangian(nn.Module):
    """Implements the semi-Lagrangian advection."""

    def __init__(self, dynamic_channels: int, static_channels: int, mesh_size: tuple):
        super().__init__()

        # For cubic interpolation
        self.padding = 1

        # Neural network that will learn an effective velocity along the trajectory
        # Output 2 channels, for u and v
        self.velocity_net = CLP(dynamic_channels + static_channels, 2, mesh_size)

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
        lon = torch.where(lon < 0, lon + 2 * torch.pi, lon)
        lon = torch.where(lon > 2 * torch.pi, lon - 2 * torch.pi, lon)

        return lat, lon

    def forward(
        self, dynamic: torch.Tensor, static: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """Compute advection using rotated coordinate system."""
        batch_size = dynamic.shape[0]

        # Extract lat/lon from static features (last 2 channels)
        lat_grid = static[:, -2, :, :]
        lon_grid = static[:, -1, :, :]

        combined = torch.cat([dynamic, static], dim=1)

        # Get learned velocities
        velocities = self.velocity_net(combined)
        u = velocities[:, 0]
        v = velocities[:, 1]

        # Compute departure points in a local rotated coordinate system in which the origin
        # of latitude and longitude is moved to the arrival point
        lon_prime = -u * dt
        lat_prime = -v * dt

        # Transform from rotated coordinates back to standard coordinates
        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, lat_grid, lon_grid
        )

        # Compute the dimensions of the padded input
        padded_width = dynamic.size(-1) + 2 * self.padding
        padded_height = dynamic.size(-2) + 2 * self.padding

        # Convert to normalized grid coordinates [-1, 1] adjusted for padding
        grid_x = ((lon_dep / (2 * torch.pi)) * 2 - 1) * (
            dynamic.size(-1) / padded_width
        )
        grid_y = ((lat_dep / torch.pi) * 2 - 1) * (dynamic.size(-2) / padded_height)

        # Create interpolation grid
        grid = torch.stack(
            [grid_x.expand(batch_size, -1, -1), grid_y.expand(batch_size, -1, -1)],
            dim=-1,
        )

        # Apply padding
        dynamic_padded = GeoCyclicPadding(self.padding)(dynamic)

        # Interpolate
        return torch.nn.functional.grid_sample(
            dynamic_padded,
            grid,
            align_corners=True,
            mode="bicubic",
            padding_mode="border",
        )


class ForcingsIntegrator(nn.Module):
    """Implements the time integration of the forcings along the Lagrangian trajectories."""

    def __init__(self, dynamic_channels: int, static_channels: int, mesh_size: tuple):
        super().__init__()

        self.diffusion_reaction_net = CLP(
            dynamic_channels + static_channels, dynamic_channels, mesh_size
        )

    def forward(
        self, dynamic: torch.Tensor, static: torch.Tensor, dt: float
    ) -> torch.Tensor:
        """Integrate over a time step of size dt."""

        # Network processes both feature types but only outputs updattorch.cat([dynamic, static]es for dynamic features
        combined = torch.cat([dynamic, static], dim=1)
        return dynamic + dt * self.diffusion_reaction_net(combined)


class Paradis(nn.Module):
    """Weather forecasting model main class."""

    def __init__(self, datamodule, cfg):
        super().__init__()

        # Extract dimensions from config
        output_dim = datamodule.num_out_features

        mesh_size = [datamodule.lat_size, datamodule.lon_size]

        num_levels = len(cfg.features.pressure_levels)

        # Get channel sizes
        self.dynamic_channels = len(
            cfg.features.input.get("atmospheric", [])
        ) * num_levels + len(cfg.features.input.get("surface", []))

        self.static_channels = len(cfg.features.input.get("constants", [])) + len(
            cfg.features.input.get("forcings", [])
        )

        hidden_dim = cfg.model.hidden_multiplier * self.dynamic_channels

        # Input projection for dynamic features
        self.dynamic_proj = CLP(self.dynamic_channels, hidden_dim, mesh_size)

        # Rescale the time step to a fraction of a synoptic time scale (~1/Ω)
        time_scale = 7.29212e5
        self.dt = cfg.model.base_dt / time_scale

        # Physics operators
        self.advection = NeuralSemiLagrangian(
            hidden_dim, self.static_channels, mesh_size
        )
        self.solve_along_trajectories = ForcingsIntegrator(
            hidden_dim, self.static_channels, mesh_size
        )

        # Output projection
        self.output_proj = CLP(hidden_dim, output_dim, mesh_size)

    def forward(
        self, x: torch.Tensor, t: typing.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the model."""
        # Setup time input if not provided
        if t is None:
            t = torch.zeros(x.shape[0], device=x.device)

        # Split features
        x_dynamic = x[:, : self.dynamic_channels]
        x_static = x[:, self.dynamic_channels :]

        # Project dynamic features
        z = self.dynamic_proj(x_dynamic)

        # Apply physics operators, analogous to the method of characteristics.
        # The advection first discovers the characteristic curves where the PDE simplifies into
        # an ODE. This ODE resembles a diffusion-reaction problem, which can then be solved along
        # the characteristic curves by a neural network.
        z = self.advection(z, x_static, self.dt)
        z = self.solve_along_trajectories(z, x_static, self.dt)

        # Project to output space
        return self.output_proj(z)
