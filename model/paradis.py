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
        interpolation: str = "bicubic",
        bias_channels: int = 4,
    ):
        super().__init__()

        # For cubic interpolation
        self.padding = 1
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

    def forward(
        self,
        hidden_features: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:

        batch_size = hidden_features.shape[0]

        # Get learned velocities for each channel
        velocities = self.velocity_net(hidden_features)

        # Reshape velocities to separate u,v components per channel
        # [batch, 2*hidden_dim, lat, lon] -> [batch, hidden_dim, 2, lat, lon]
        velocities = velocities.view(batch_size, 2, self.num_vels, *self.mesh_size)

        # Extract u,v compone
        # u = velocities[:, 0] # [batch, hidden_dim, lat*lon]
        v = velocities[:, 1]
        u = velocities[:, 0] / torch.cos(
            lat_grid[:, None, :, :]
        )  # [batch, num_vels, lat*lon]
        projected_inputs = self.down_projection(hidden_features)

        z_p = self.padding_interp(projected_inputs)

        dx = torch.pi / 180
        dy = torch.pi / 180

        flux_x = u * (z_p[..., 1:-1, 2:] - z_p[..., 1:-1, :-2]) / (2.0 * dx)
        flux_y = v * (z_p[..., 2:, 1:-1] - z_p[..., :-2, 1:-1]) / (2.0 * dy)


        # Reshape back to original dimensions
        interpolated = (flux_x + flux_y).view(batch_size, self.num_vels, *self.mesh_size)

        # Project back up to latent space
        interpolated = self.up_projection(interpolated)

        return interpolated


class Paradis(nn.Module):
    """Weather forecasting model main class."""

    # Synoptic time scale (~1/Ω) in seconds
    SYNOPTIC_TIME_SCALE = 7.29212e5

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

        # Extract lat/lon from static features (last 2 channels)
        lat_grid = x_static[:, -2, :, :]
        lon_grid = x_static[:, -1, :, :]

        # Project features to latent space
        z = self.input_proj(x)

        # Compute advection and diffusion-reaction
        for i in range(self.num_layers):
            # Advect the features in latent space using a Semi-Lagrangian step
            z_adv = self.advection[i](z, lat_grid, lon_grid, self.dt)

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
