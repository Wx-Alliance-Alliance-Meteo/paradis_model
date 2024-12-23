"""Weather forecasting model with operator splitting."""

import torch
from torch import nn
import torch.nn.functional as F

from model.padding import GeoCyclicPadding


class AdvectionOperator(nn.Module):
    """Implements the advection operator."""

    def __init__(self, channels: int, mesh_size: tuple):
        """Initialize the advection operator.

        Args:
            channels: Number of input channels
            mesh_size: Tuple of (height, width) for the spatial dimensions
        """
        super().__init__()

        self.mesh_size = mesh_size

        # Grid spacing in radians
        self.d_lat = torch.pi / (mesh_size[0] - 1)
        self.d_lon = 2 * torch.pi / mesh_size[1]

        # Set maximum displacement and ensure padding is sufficient
        self.max_displacement = 6
        # Add two points for cubic interpolation
        self.pad_size = self.max_displacement + 2

        # Initialize velocity networks with small weights
        # this prevents generating supersonic motions at the start of training ...
        def init_weights(m):
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, mean=0.0, std=2e-5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Neural networks that will learn an effective velocity along the trajectory
        self.u_net = nn.Sequential(
            GeoCyclicPadding(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.LayerNorm(
                [channels, mesh_size[0], mesh_size[1]], elementwise_affine=True
            ),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.u_net.apply(init_weights)

        self.v_net = nn.Sequential(
            GeoCyclicPadding(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0),
            nn.LayerNorm(
                [channels, mesh_size[0], mesh_size[1]], elementwise_affine=True
            ),
            nn.SiLU(),
            nn.Conv2d(channels, 1, kernel_size=1),
        )
        self.v_net.apply(init_weights)

        # Create coordinate grids
        lat = torch.linspace(-torch.pi / 2, torch.pi / 2, mesh_size[0])
        lon = torch.linspace(0, 2 * torch.pi, mesh_size[1])
        lat_grid, lon_grid = torch.meshgrid(lat, lon, indexing="ij")

        # Register buffers for coordinates (without prior assignment)
        self.register_buffer("lat_grid", lat_grid.clone())
        self.register_buffer("lon_grid", lon_grid.clone())

    def _transform_to_latlon(
        self,
        lat_prime: torch.Tensor,
        lon_prime: torch.Tensor,
        lat_p: torch.Tensor,
        lon_p: torch.Tensor,
    ) -> tuple:
        """Transform from local rotated coordinates back to standard latlon coordinates.

        Args:
            lat_prime, lon_prime: Coordinates in rotated system
            lat_p, lon_p: Coordinates of the rotation point P

        Returns:
            tuple: (latitude, longitude) in standard coordinates
        """
        # Compute standard latitude
        sin_lat = torch.sin(lat_prime) * torch.cos(lat_p) + torch.cos(
            lat_prime
        ) * torch.cos(lon_prime) * torch.sin(lat_p)
        lat = torch.arcsin(torch.clamp(sin_lat, -1 + 1e-6, 1 - 1e-6))

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

    def _clip_velocities(self, u: torch.Tensor, v: torch.Tensor, dt: float) -> tuple:
        """Clip velocities to ensure trajectories stay within padding.

        Args:
            u: Zonal velocity
            v: Meridional velocity
            dt: Time step

        Returns:
            Tuple of clipped (u, v)
        """
        max_u = self.max_displacement * self.d_lon / dt
        max_v = self.max_displacement * self.d_lat / dt

        u_clipped = torch.clamp(u, min=-max_u, max=max_u)
        v_clipped = torch.clamp(v, min=-max_v, max=max_v)

        return u_clipped, v_clipped

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Compute advection using rotated coordinate system.

        Args:
            x: Input tensor [batch, channels, lat, lon]
            dt: Time step

        Returns:
            Advected tensor
        """
        batch_size = x.shape[0]

        # Get learned velocities
        u = self.u_net(x)
        v = self.v_net(x)
        u, v = self._clip_velocities(u[:, 0], v[:, 0], dt)

        # For each grid point (lat_p, lon_p), compute departure point
        # in a local rotated coordinate system in which the origin
        # of latitude and longitude is moved to the arrival point
        lon_prime = -u * dt
        lat_prime = -v * dt

        # Transform from rotated coordinates back to standard coordinates
        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, self.lat_grid, self.lon_grid
        )

        # Convert to normalized grid coordinates [-1, 1]
        grid_x = (lon_dep / (2 * torch.pi)) * 2 - 1
        grid_y = (lat_dep / torch.pi) * 2

        # Create interpolation grid
        grid = torch.stack(
            [grid_x.expand(batch_size, -1, -1), grid_y.expand(batch_size, -1, -1)],
            dim=-1,
        )

        # Apply padding
        x_padded = GeoCyclicPadding(self.pad_size)(x)

        # Interpolate using grid sampling
        return F.grid_sample(
            x_padded, grid, align_corners=True, mode="bicubic", padding_mode="border"
        )


class DiffusionOperator(nn.Module):
    """Implements the diffusion operator."""

    def __init__(self, channels, mesh_size):
        super().__init__()
        self.channels = channels
        self.mesh_size = mesh_size
        self.padding = GeoCyclicPadding(2)

        # One learnable coefficient per channel
        self.kappa = nn.Parameter(torch.ones(channels, 1, 1))

        # Initialize diffusion kernel
        # Channel-wise convolution (groups=channels means each channel has its own kernel)
        self.conv = nn.Conv2d(
            channels, channels, kernel_size=5, padding=0, bias=False, groups=channels
        )

        self.norm = nn.LayerNorm(
            [channels, mesh_size[0], mesh_size[1]], elementwise_affine=True
        )

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Apply diffusion operator for time step dt."""
        # Apply padding with size check
        padded = self.padding(x)

        # Compute diffusion
        diffusion = self.conv(padded)
        diffusion = self.norm(diffusion)

        return x + dt * self.kappa * diffusion


class ReactionOperator(nn.Module):
    """Implements the reaction operator."""

    def __init__(self, channels, mesh_size):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=1),
            nn.LayerNorm(
                [channels * 2, mesh_size[0], mesh_size[1]], elementwise_affine=True
            ),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor, dt: float = 1.0) -> torch.Tensor:
        """Apply reaction operator for time step dt."""
        return x + dt * self.net(x)


class WeatherADRBlock(nn.Module):
    """Enhanced ADR block with configurable operator splitting schemes."""

    def __init__(
        self, channels, mesh_size, is_static=False, splitting_scheme="lie", dt=1.0
    ):
        super().__init__()
        self.is_static = is_static
        self.splitting_scheme = splitting_scheme
        self.dt = dt

        if not is_static:
            # Initialize operators
            self.advection = AdvectionOperator(channels, mesh_size)
            self.diffusion = DiffusionOperator(channels, mesh_size)
            self.reaction = ReactionOperator(channels, mesh_size)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Apply operators following the specified splitting scheme.

        For Lie splitting: A -> R -> D
        For Strang splitting: A -> D -> R -> R -> D -> A
        """
        if self.is_static:
            return x

        if self.splitting_scheme == "lie":
            # Lie splitting (first order)
            x = self.advection(x, self.dt)
            x = self.reaction(x, self.dt)
            x = self.diffusion(x, self.dt)

        elif self.splitting_scheme == "strang":
            # Strang splitting (second order)
            # First half-step
            x = self.advection(x, self.dt / 2)
            x = self.diffusion(x, self.dt / 2)

            # Full step for reaction
            x = self.reaction(x, self.dt)

            # Second half-step in reverse order
            x = self.diffusion(x, self.dt / 2)
            x = self.advection(x, self.dt / 2)

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
        self.static_channels = len(cfg.features.input.constants)
        self.dynamic_channels = self.input_dim - self.static_channels

        # Input projections
        self.static_proj = nn.Conv2d(self.static_channels, self.hidden_dim // 4, 1)
        self.dynamic_proj = nn.Conv2d(self.dynamic_channels, self.hidden_dim, 1)

        # Rescale the time step size for consistency with the normalization scheme
        OMEGA = 7.29212e-5  # Earth's rotation rate in rad/s.
        self.base_dt = cfg.model.base_dt * OMEGA  # The time scale is 1/Ω

        # ADR processing layers
        self.adr_layers = nn.ModuleList()
        for _ in range(cfg.model.num_layers):
            static_block = WeatherADRBlock(
                self.hidden_dim // 4,
                self.mesh_size,
                is_static=True,
                splitting_scheme=self.splitting_scheme,
                dt=self.base_dt,
            )
            dynamic_block = WeatherADRBlock(
                self.hidden_dim,
                self.mesh_size,
                is_static=False,
                splitting_scheme=self.splitting_scheme,
                dt=self.base_dt,
            )
            self.adr_layers.append(
                nn.ModuleDict({"static": static_block, "dynamic": dynamic_block})
            )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim + self.hidden_dim // 4, self.hidden_dim, 1),
            nn.LayerNorm(
                [self.hidden_dim, self.mesh_size[0], self.mesh_size[1]],
                elementwise_affine=True,
            ),
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
        x_static = x[:, self.dynamic_channels :]
        x_dynamic = x[:, : self.dynamic_channels]

        # Initial projections
        z_static = self.static_proj(x_static)
        z_dynamic = self.dynamic_proj(x_dynamic)

        # Process through ADR layers
        for layer in self.adr_layers:
            z_static = layer["static"](z_static, t)
            z_dynamic = layer["dynamic"](z_dynamic, t)

        # Combine and project to output space
        z_combined = torch.cat([z_dynamic, z_static], dim=1)
        return self.output_proj(z_combined)
