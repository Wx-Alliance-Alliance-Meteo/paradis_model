"""Physically-informed neural architecture for the weather forecasting model."""

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


# CLP processor with structured latent
class VariationalCLP(nn.Module):
    """Convolutional layer processor with variational latent space."""
    def __init__(self, dim_in, dim_out, mesh_size, kernel_size=3, 
                 latent_dim=8, activation=nn.SiLU):
        super().__init__()
        
        self.latent_dim = latent_dim
        
        # Encoder that produces pre latent
        self.encoder = nn.Sequential(
            GeoCyclicPadding(kernel_size // 2),
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size),
            nn.LayerNorm([dim_in, mesh_size[0], mesh_size[1]]),
            activation(),
            nn.Conv2d(dim_in, 2 * latent_dim, kernel_size=1)  # project down
        )
        
        # Small projection up and down in latent
        self.mu = nn.Sequential(
            nn.Conv2d(latent_dim, 4 * latent_dim, kernel_size=1),
            nn.LayerNorm([4 * latent_dim, mesh_size[0], mesh_size[1]]),
            activation(),
            nn.Conv2d(4 * latent_dim, latent_dim, kernel_size=1)
        )
        self.logvar = nn.Sequential(
            nn.Conv2d(latent_dim, 4 * latent_dim, kernel_size=1),
            nn.LayerNorm([4 * latent_dim, mesh_size[0], mesh_size[1]]),
            activation(),
            nn.Conv2d(4 * latent_dim, latent_dim, kernel_size=1)
        )
        
        # Decoder that takes the concat of the logvar and mu
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, dim_in, kernel_size=1), # project up
            GeoCyclicPadding(kernel_size // 2),
            nn.Conv2d(dim_in, dim_in, kernel_size=kernel_size),
            nn.LayerNorm([dim_in, mesh_size[0], mesh_size[1]]),
            activation(),
            GeoCyclicPadding(kernel_size // 2),
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size),
        )

    def reparameterize(self, mean, log_var):
        """Reparameterization trick to sample from N(mean, var) while remaining differentiable."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        # TODO
        pass


class NeuralSemiLagrangian(nn.Module):
    """Implements the advection operator."""

    def __init__(self, dynamic_channels: int, static_channels: int, mesh_size: tuple):
        super().__init__()

        # For cubic interpolation
        self.padding = 1

        # Grid spacing in radians
        self.d_lat = torch.pi / (mesh_size[0] - 1)
        self.d_lon = 2 * torch.pi / mesh_size[1]

        # Neural network that will learn an effective velocity along the trajectory
        # Output 2 channels, for u and v
        self.velocity_net = CLP(dynamic_channels + static_channels, 2, mesh_size)

        # Create coordinate grids
        lat = torch.linspace(-torch.pi / 2, torch.pi / 2, mesh_size[0])
        lon = torch.linspace(0, 2 * torch.pi, mesh_size[1])
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")

        # Register buffers for coordinates
        self.register_buffer("lat_grid", lat_grid.clone())
        self.register_buffer("lon_grid", lon_grid.clone())

    def _transform_to_latlon(
        self,
        lat_prime: torch.Tensor,
        lon_prime: torch.Tensor,
        lat_p: torch.Tensor,
        lon_p: torch.Tensor,
    ) -> tuple:
        """Transform from local rotated coordinates back to standard latlon coordinates."""
        # Compute standard latitude
        sin_lat = torch.sin(lat_prime) * torch.cos(lat_p) + torch.cos(
            lat_prime
        ) * torch.cos(lon_prime) * torch.sin(lat_p)
        lat = torch.arcsin(torch.clamp(sin_lat, -1 + 1e-7, 1 - 1e-7))

        # Compute standard longitude
        num = torch.cos(lat_prime) * torch.sin(lon_prime)
        den = torch.cos(lat_prime) * torch.cos(lon_prime) * torch.cos(
            lat_p
        ) - torch.sin(lat_prime) * torch.sin(lat_p)

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

        combined = torch.cat([dynamic, static], dim=1)

        # Get learned velocities
        velocities = self.velocity_net(combined)
        u = velocities[:, 0]
        v = velocities[:, 1]
        
        # TODO | learn a distribution of velocities
        # via the changing CLP to a have a small latent space with KL divergence loss normal distriubtion
        # then sample a few times (not entirely sure what to do with the samples)

        # Compute departure points in a local rotated coordinate system in which the origin
        # of latitude and longitude is moved to the arrival point
        lon_prime = -u * dt
        lat_prime = -v * dt

        # Transform from rotated coordinates back to standard coordinates
        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, self.lat_grid, self.lon_grid
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
        self.advection = NeuralSemiLagrangian(hidden_dim, self.static_channels, mesh_size)
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
