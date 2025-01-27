"""Physically inspired neural architecture for the weather forecasting model."""

import typing
import torch
from torch import nn

from model.padding import GeoCyclicPadding


class CLPBlock(nn.Module):
    """Convolutional Layer Processor block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int = 3,
        activation: nn.Module = nn.SiLU,
        double_conv: bool = False,
    ):
        super().__init__()

        # First convolution block
        layers = [
            GeoCyclicPadding(kernel_size // 2, input_dim),
            nn.Conv2d(
                input_dim,
                input_dim if double_conv else output_dim,
                kernel_size=kernel_size,
            ),
            nn.LayerNorm(
                [input_dim if double_conv else output_dim, mesh_size[0], mesh_size[1]]
            ),
            activation(),
        ]

        # Optional second convolution block
        if double_conv:
            layers.extend(
                [
                    GeoCyclicPadding(kernel_size // 2, input_dim),
                    nn.Conv2d(input_dim, output_dim, kernel_size=kernel_size),
                ]
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# Helper function
def CLP(
    dim_in: int,
    dim_out: int,
    mesh_size: tuple,
    kernel_size: int = 3,
    activation: nn.Module = nn.SiLU,
):
    """Create a double-convolution CLP block."""
    return CLPBlock(
        dim_in, dim_out, mesh_size, kernel_size, activation, double_conv=True
    )


# CLP processor with structured latent
class VariationalCLP(nn.Module):
    """Convolutional layer processor with variational latent space."""

    def __init__(
        self,
        dim_in,
        dim_out,
        mesh_size,
        kernel_size=3,
        latent_dim=8,
        activation=nn.SiLU,
        expansion_factor=8,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.expansion_factor = expansion_factor

        # Encoder that produces pre latent
        self.encoder = nn.Sequential(
            CLPBlock(dim_in, dim_in, mesh_size),
            nn.Conv2d(dim_in, 2 * latent_dim, kernel_size=1),  # project down
        )

        # Small projection up and down in latent
        self.mu = nn.Sequential(
            nn.Conv2d(latent_dim, self.expansion_factor * latent_dim, kernel_size=1),
            nn.LayerNorm(
                [self.expansion_factor * latent_dim, mesh_size[0], mesh_size[1]]
            ),
            activation(),
            nn.Conv2d(self.expansion_factor * latent_dim, latent_dim, kernel_size=1),
        )
        self.logvar = nn.Sequential(
            nn.Conv2d(latent_dim, self.expansion_factor * latent_dim, kernel_size=1),
            nn.LayerNorm(
                [self.expansion_factor * latent_dim, mesh_size[0], mesh_size[1]]
            ),
            activation(),
            nn.Conv2d(self.expansion_factor * latent_dim, latent_dim, kernel_size=1),
        )

        # Decoder that takes the concat of the logvar and mu
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, dim_in, kernel_size=1),  # project up
            CLPBlock(dim_in, dim_in, mesh_size),
            GeoCyclicPadding(kernel_size // 2, dim_in),
            nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size),
        )

    def reparameterize(self, mean, log_var):
        """Reparameterization trick to sample from N(mean, var) while remaining differentiable."""
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x, num_samples=1):
        batch_size = x.shape[0]

        pre_latent = self.encoder(x)

        # Split into two groups for mu and logvar networks
        pre_mu, pre_logvar = torch.chunk(pre_latent, 2, dim=1)

        # Get distribution parameters from separate networks
        mean = self.mu(pre_mu)
        log_var = self.logvar(pre_logvar)

        # Calculate KL divergence loss against normal
        kl_loss = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp(), dim=1)
        kl_loss = kl_loss.mean()  # Average over batch

        # Sample from latent and decode to velocity
        z = self.reparameterize(mean, log_var)
        output = self.decoder(z)

        return output, kl_loss


class NeuralSemiLagrangian(nn.Module):
    """Implements the semi-Lagrangian advection."""

    def __init__(self, hidden_dim: int, mesh_size: tuple, variational: bool):
        super().__init__()

        # For cubic interpolation
        self.padding = 1
        self.padding_interp = GeoCyclicPadding(self.padding, hidden_dim)
        self.hidden_dim = hidden_dim

        # Flag for variational variant to be used in forward
        self.variational = variational

        # Neural network that will learn an effective velocity along the trajectory
        # Output 2 channels per hidden dimension for u and v
        if not self.variational:
            self.velocity_net = CLP(hidden_dim, 2 * hidden_dim, mesh_size)
        else:
            self.velocity_net = VariationalCLP(hidden_dim, 2 * hidden_dim, mesh_size)

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
        if self.variational:
            velocities, kl_loss = self.velocity_net(hidden_features)
        else:
            velocities = self.velocity_net(hidden_features)

        # Reshape velocities to separate u,v components per channel
        # [batch, 2*hidden_dim, lat, lon] -> [batch, hidden_dim, 2, lat, lon]
        velocities = velocities.view(
            batch_size, 2, self.hidden_dim, *velocities.shape[-2:]
        ).transpose(1, 2)

        # Extract u,v components
        u = velocities[:, :, 0]  # [batch, hidden_dim, lat, lon]
        v = velocities[:, :, 1]

        # Compute departure points in a local rotated coordinate system in which the origin
        # of latitude and longitude is moved to the arrival point
        lon_prime = -u * dt
        lat_prime = -v * dt

        # Transform from rotated coordinates back to standard coordinates
        # Expand lat/lon grid for broadcasting with per-channel coordinates
        lat_grid = lat_grid.unsqueeze(1).expand(-1, self.hidden_dim, -1, -1)
        lon_grid = lon_grid.unsqueeze(1).expand(-1, self.hidden_dim, -1, -1)

        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, lat_grid, lon_grid
        )

        # Compute the dimensions of the padded input
        padded_width = hidden_features.size(-1) + 2 * self.padding
        padded_height = hidden_features.size(-2) + 2 * self.padding

        # Convert to normalized grid coordinates [-1, 1] adjusted for padding
        grid_x = ((lon_dep / (2 * torch.pi)) * 2 - 1) * (
            hidden_features.size(-1) / padded_width
        )
        grid_y = ((lat_dep / torch.pi) * 2 - 1) * (
            hidden_features.size(-2) / padded_height
        )

        # Reshape grid coordinates for interpolation
        # [batch, hidden_dim, lat, lon] -> [batch*hidden_dim, lat, lon]
        grid_x = grid_x.view(batch_size * self.hidden_dim, *grid_x.shape[-2:])
        grid_y = grid_y.view(batch_size * self.hidden_dim, *grid_y.shape[-2:])

        # Create interpolation grid
        grid = torch.stack([grid_x, grid_y], dim=-1)

        # Apply padding and reshape hidden features
        dynamic_padded = self.padding_interp(hidden_features)
        dynamic_padded = dynamic_padded.view(
            batch_size * self.hidden_dim, 1, *dynamic_padded.shape[-2:]
        )

        # Interpolate
        interpolated = torch.nn.functional.grid_sample(
            dynamic_padded,
            grid,
            align_corners=True,
            mode="bicubic",
            padding_mode="border",
        )

        # Reshape back to original dimensions
        interpolated = interpolated.view(
            batch_size, self.hidden_dim, *interpolated.shape[-2:]
        )

        if self.variational:
            return interpolated, kl_loss

        return interpolated


class ForcingsIntegrator(nn.Module):
    """Implements the time integration of the forcings along the Lagrangian trajectories."""

    def __init__(self, hidden_dim: int, mesh_size: tuple):
        super().__init__()

        self.diffusion_reaction_net = CLP(hidden_dim, hidden_dim, mesh_size)

    def forward(self, hidden_features: torch.Tensor, dt: float) -> torch.Tensor:
        """Integrate over a time step of size dt."""

        return hidden_features + dt * self.diffusion_reaction_net(hidden_features)


class Paradis(nn.Module):
    """Weather forecasting model main class."""

    # Synoptic time scale (~1/Ω) in seconds
    SYNOPTIC_TIME_SCALE = 7.29212e5

    def __init__(self, datamodule, cfg):
        super().__init__()

        # Extract dimensions from config
        output_dim = datamodule.num_out_features

        mesh_size = [datamodule.lat_size, datamodule.lon_size]

        num_levels = len(cfg.features.pressure_levels)

        # Flag for variational
        self.variational = cfg.model.variational

        # Get channel sizes
        self.dynamic_channels = len(
            cfg.features.input.get("atmospheric", [])
        ) * num_levels + len(cfg.features.input.get("surface", []))

        self.static_channels = len(cfg.features.input.get("constants", [])) + len(
            cfg.features.input.get("forcings", [])
        )

        hidden_dim = (
            cfg.model.hidden_multiplier * self.dynamic_channels
        ) + self.static_channels

        # Input projection for combined dynamic and static features
        self.input_proj = CLP(
            self.dynamic_channels + self.static_channels, hidden_dim, mesh_size
        )

        # Rescale the time step to a fraction of a synoptic time scale
        self.dt = cfg.model.base_dt / self.SYNOPTIC_TIME_SCALE

        # Physics operators
        self.advection = NeuralSemiLagrangian(hidden_dim, mesh_size, self.variational)
        self.solve_along_trajectories = ForcingsIntegrator(hidden_dim, mesh_size)

        # Output projection
        self.output_proj = CLP(hidden_dim, output_dim, mesh_size)

    def forward(
        self, x: torch.Tensor, t: typing.Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through the model."""

        # Extract lat/lon from static features (last 2 channels)
        x_static = x[:, self.dynamic_channels :]
        lat_grid = x_static[:, -2, :, :]
        lon_grid = x_static[:, -1, :, :]

        # Project combined features to latent space
        z = self.input_proj(x)

        # Apply the neural semi-Lagrangian operators
        if self.variational:
            # Propogates up the KL
            z, kl_loss = self.advection(z, lat_grid, lon_grid, self.dt)
        else:
            z = self.advection(z, lat_grid, lon_grid, self.dt)

        z = self.solve_along_trajectories(z, self.dt)

        # Project to output space
        if self.variational:
            # Propogates up the KL
            return (self.output_proj(z), kl_loss)

        return self.output_proj(z)
