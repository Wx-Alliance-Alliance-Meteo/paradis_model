"""Weather forecasting model with operator splitting."""

import torch
from torch import nn
import torch.nn.functional as F

from model.padding import GeoCyclicPadding


class AdvectionOperator(nn.Module):
    """Implements the advection operator."""

    def __init__(self, channels, mesh_size):
        super().__init__()
        self.earth_radius = 6371220.0
        self.mesh_size = mesh_size

        self.u_net = nn.Sequential(
            GeoCyclicPadding(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

        self.v_net = nn.Sequential(
            GeoCyclicPadding(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.InstanceNorm2d(channels, affine=True),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )

        lat = torch.linspace(-torch.pi / 2, torch.pi / 2, mesh_size[0])
        lon = torch.linspace(0, 2 * torch.pi, mesh_size[1])

        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")

        self.register_buffer("cos_lat", torch.cos(lat_grid))
        self.register_buffer("sin_lat", torch.sin(lat_grid))
        self.register_buffer("cos_lon", torch.cos(lon_grid))
        self.register_buffer("sin_lon", torch.sin(lon_grid))

        self.register_buffer("grid_y", None, persistent=False)
        self.register_buffer("grid_x", None, persistent=False)

    def initialize_grid(self, height: int, width: int, device: torch.device):
        """Initialize the sampling grid."""
        if self.grid_y is None or self.grid_x is None:
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1, 1, height, device=device),
                torch.linspace(-1, 1, width, device=device),
                indexing="ij",
            )
            self.register_buffer("grid_y", grid_y.clone(), persistent=False)
            self.register_buffer("grid_x", grid_x.clone(), persistent=False)

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Computes the forward Semi-Lagrangian advection using great circles.

        Inspired by Ritchie, H. (1987). "Semi-Lagrangian advection on a Gaussian grid."
        *Monthly Weather Review*, 115(2), 608-619.
        https://doi.org/10.1175/1520-0493(1987)115<0608:SLAOAG>2.0.CO;2

        Args:
            x (torch.Tensor): The input tensor representing the initial state.
            dt (float, optional): The time step for advection. Default is 1.0.

        Returns:
            torch.Tensor: The tensor representing the state after advection.
        """
        batch_size, _, height, width = x.shape
        self.initialize_grid(height, width, x.device)

        # Get velocities
        u = self.u_net(x)
        v = self.v_net(x)

        # Convert to Cartesian velocity components
        dx = -u[:, 0] * self.sin_lon - v[:, 0] * self.cos_lon * self.sin_lat
        dy = u[:, 0] * self.cos_lon - v[:, 0] * self.sin_lon * self.sin_lat
        dz = v[:, 0] * self.cos_lat

        # Initial Cartesian coordinates
        x0 = self.cos_lon * self.cos_lat
        y0 = self.sin_lon * self.cos_lat
        z0 = self.sin_lat.clone()

        # Compute correction factor
        velocity_squared = dx * dx + dy * dy + dz * dz
        position_dot_velocity = dx * x0 + dy * y0 + dz * z0
        denominator = (
            1.0 + dt * dt * velocity_squared - 2.0 * dt * position_dot_velocity
        )
        b = 1.0 / torch.sqrt(denominator)

        # Calculate new positions
        x1 = b * (x0 + dt * dx)
        y1 = b * (y0 + dt * dy)
        z1 = b * (z0 + dt * dz)

        # Ensure points stay on unit sphere
        norm = torch.sqrt(x1 * x1 + y1 * y1 + z1 * z1)
        x1 = x1 / norm
        y1 = y1 / norm
        z1 = z1 / norm

        # Convert back to spherical coordinates
        eps = 1e-6
        lat_new = torch.arcsin(torch.clamp(z1, min=-1 + eps, max=1 - eps))
        lon_new = torch.atan2(y1, x1)
        lon_new = torch.where(lon_new < 0, lon_new + 2 * torch.pi, lon_new)

        # Convert to grid coordinates
        grid_x = (lon_new / (2 * torch.pi)) * 2 - 1
        grid_y = (lat_new / torch.pi) * 2

        grid = torch.stack(
            [grid_x.expand(batch_size, -1, -1), grid_y.expand(batch_size, -1, -1)],
            dim=-1,
        )

        # Interpolation
        return F.grid_sample(x, grid, align_corners=True, padding_mode="border")


class DiffusionOperator(nn.Module):
    """Implements the diffusion operator."""

    def __init__(self, channels, mesh_size):
        super().__init__()
        self.channels = channels
        self.mesh_size = mesh_size
        self.padding = GeoCyclicPadding(2)
        self.kappa = nn.Parameter(torch.ones(channels, 1, 1))

        # Initialize diffusion kernel
        self.conv = nn.Conv2d(channels, channels, kernel_size=5, padding=0, bias=False)
        self.norm = nn.InstanceNorm2d(channels, affine=True)

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Apply diffusion operator for time step dt with proper padding."""
        # Apply padding with size check
        padded = self.padding(x)

        # Compute diffusion
        diffusion = self.conv(padded)
        diffusion = self.norm(diffusion)

        return x + dt * self.kappa * diffusion


class ReactionOperator(nn.Module):
    """Implements the reaction operator."""

    def __init__(self, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.InstanceNorm2d(channels * 2, affine=True),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Apply reaction operator for time step dt."""
        return x + dt * self.net(x)


class WeatherADRBlock(nn.Module):
    """Enhanced ADR block with configurable operator splitting schemes."""

    def __init__(self, channels, mesh_size, is_static=False, splitting_scheme="lie"):
        super().__init__()
        self.is_static = is_static
        self.splitting_scheme = splitting_scheme

        if not is_static:
            # Initialize operators
            self.advection = AdvectionOperator(channels, mesh_size)
            self.diffusion = DiffusionOperator(channels, mesh_size)
            self.reaction = ReactionOperator(channels)

            # Time embedding
            self.time_embed = nn.Parameter(torch.randn(1, channels, 1, 1) * 1e-4)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply operators following the specified splitting scheme.

        For Lie splitting: A -> D -> R
        For Strang splitting: A -> D -> R -> R -> D -> A
        """
        if self.is_static:
            return x

        dt = 1.0  # Base (scaled) timestep
        t_embed = t.reshape(-1, 1, 1, 1) * self.time_embed

        if self.splitting_scheme == "lie":
            # Lie splitting (first order)
            x = self.advection(x + t_embed, dt)
            x = self.diffusion(x, dt)
            x = self.reaction(x, dt)

        elif self.splitting_scheme == "strang":
            # Strang splitting (second order)
            # First half-step
            x = self.advection(x + t_embed, dt / 2)
            x = self.diffusion(x, dt / 2)

            # Full step for reaction
            x = self.reaction(x, dt)

            # Second half-step in reverse order
            x = self.diffusion(x, dt / 2)
            x = self.advection(x, dt / 2)

        else:
            raise ValueError(f"Unknown splitting scheme: {self.splitting_scheme}")

        return x


class Paradis(nn.Module):
    """Weather forecasting model with operator splitting."""

    def __init__(self, datamodule, cfg):
        super().__init__()

        # Extract dimensions from config
        self.input_dim = datamodule.num_in_features
        self.output_dim = datamodule.num_out_features
        self.hidden_dim = cfg.model.hidden_dim
        self.mesh_size = [datamodule.lat_size, datamodule.lon_size]
        self.splitting_scheme = cfg.model.get("splitting_scheme", "lie")

        # Setup static and dynamic channels
        static_vars = ["geopotential_at_surface", "land_sea_mask"]
        self.static_channels = len(
            [var for var in cfg.features.input.surface if var in static_vars]
        )
        self.dynamic_channels = self.input_dim - self.static_channels

        # Input projections
        self.static_proj = nn.Conv2d(self.static_channels, self.hidden_dim // 4, 1)
        self.dynamic_proj = nn.Conv2d(self.dynamic_channels, self.hidden_dim, 1)

        # ADR processing layers
        self.adr_layers = nn.ModuleList()
        for _ in range(cfg.model.num_layers):
            static_block = WeatherADRBlock(
                self.hidden_dim // 4,
                self.mesh_size,
                is_static=True,
                splitting_scheme=self.splitting_scheme,
            )
            dynamic_block = WeatherADRBlock(
                self.hidden_dim,
                self.mesh_size,
                is_static=False,
                splitting_scheme=self.splitting_scheme,
            )
            self.adr_layers.append(
                nn.ModuleDict({"static": static_block, "dynamic": dynamic_block})
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim + self.hidden_dim // 4, self.hidden_dim, 1),
            nn.InstanceNorm2d(self.hidden_dim, affine=True),
            nn.SiLU(),
            nn.Conv2d(self.hidden_dim, self.output_dim, 1),
        )

    def forward(self, x, t=None):
        """Forward pass processing both static and dynamic features."""
        # Setup time input
        batch_size = x.shape[0]
        if t is None:
            t = torch.zeros(batch_size, device=x.device)

        # Split into static and dynamic components
        x_static = x[:, : self.static_channels]
        x_dynamic = x[:, self.static_channels :]

        # Initial projections
        z_static = self.static_proj(x_static)
        z_dynamic = self.dynamic_proj(x_dynamic)

        # Process through ADR layers
        for layer in self.adr_layers:
            z_static = layer["static"](z_static, t)
            z_dynamic = layer["dynamic"](z_dynamic, t)

        # Combine and project to output space
        z_combined = torch.cat([z_static, z_dynamic], dim=1)
        return self.output_proj(z_combined)
