"""Paradis neural architecture."""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from model.advection import NeuralSemiLagrangian
from model.blocks import GMBlock


def get_scaled_timestep(original_timestep_seconds: float) -> float:
    return original_timestep_seconds * 7.29212e-5


class Paradis(nn.Module):
    """Paradis model adapted for shallow water equations."""

    def __init__(self, datamodule, cfg, lat_grid, lon_grid):
        super().__init__()

        self.nlat = lat_grid.shape[0]
        self.nlon = lat_grid.shape[1]

        self.grid = "equiangular"

        if self.grid != "equiangular":
            raise ValueError(
                f"Paradis model only supports 'equiangular' grid, got '{self.grid}'. "
                "Please set data.grid='equiangular' in your config."
            )

        mesh_size = (self.nlat, self.nlon)

        hidden_dim = cfg.model.get("latent_size")

        self.num_vels = cfg.model.get("velocity_vectors")
        diffusion_size = cfg.model.get("diffusion_size")
        reaction_size = cfg.model.get("reaction_size")

        adv_interpolation = cfg.model.get("adv_interpolation")
        bias_channels = cfg.model.get("bias_channels", 4)
        num_encoder_layers = cfg.model.get("num_encoder_layers", 1)

        self.num_layers = max(1, cfg.model.num_layers)
        self.dt = get_scaled_timestep(cfg.model.get("base_dt")) / self.num_layers

        # Input projection
        self.activation_function = nn.SiLU

        input_dim = (
            datamodule.dataset.num_in_dyn_features
            + datamodule.dataset.num_in_static_features
        )

        self.gradient_checkpoint = cfg.compute.get("gradient_checkpointing", False)

        current_dim = input_dim
        encoder_layers = []
        bias = False
        for l in range(num_encoder_layers - 1):
            fc = nn.Conv2d(current_dim, hidden_dim, 1, bias=True)

            # Initialize the weights correctly
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)

            if fc.bias is not None:
                nn.init.constant_(fc.bias, 0.0)

            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())

            current_dim = hidden_dim

        fc = nn.Conv2d(current_dim, hidden_dim, 1, bias=bias)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        if fc.bias is not None:
            nn.init.constant_(fc.bias, 0.0)
        encoder_layers.append(fc)

        self.input_proj = nn.Sequential(*encoder_layers)

        self.velocity_nets = nn.ModuleList(
            [
                GMBlock(
                    layers=["SepConv"],
                    input_dim=hidden_dim,
                    output_dim=2 * self.num_vels,
                    hidden_dim=hidden_dim,
                    kernel_size=3,
                    mesh_size=mesh_size,
                    bias_channels=bias_channels,
                    pre_normalize=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.advection = nn.ModuleList(
            [
                NeuralSemiLagrangian(
                    hidden_dim,
                    mesh_size,
                    num_vels=self.num_vels,
                    lat_grid=lat_grid,
                    lon_grid=lon_grid,
                    interpolation=adv_interpolation,
                    project_advection=cfg.model.get("projected_advection", True),
                )
                for _ in range(self.num_layers)
            ]
        )

        self.diffusion = nn.ModuleList(
            [
                GMBlock(
                    layers=["SepConv"],
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=diffusion_size,
                    mesh_size=mesh_size,
                    pre_normalize=True,
                    bias_channels=bias_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.reaction = nn.ModuleList(
            [
                GMBlock(
                    layers=["CLinear"] * 2,
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=reaction_size,
                    kernel_size=1,
                    mesh_size=mesh_size,
                    pre_normalize=True,
                    bias_channels=bias_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_proj = GMBlock(
            layers=["SepConv", "CLinear"],
            input_dim=hidden_dim,
            output_dim=datamodule.num_out_features,
            hidden_dim=hidden_dim,
            mesh_size=mesh_size,
            kernel_size=3,
            activation=False,
            bias_channels=bias_channels,
        )

    def _apply_checkpoint(self, func, *args):
        if self.gradient_checkpoint:
            return checkpoint(func, *args, use_reentrant=False)
        else:
            return func(*args)

    def _layer_step(self, i: int, hidden: torch.Tensor) -> torch.Tensor:
        """Execute a single ADR layer (advection, diffusion, reaction)."""
        batch_size = hidden.shape[0]

        velocities_raw = self.velocity_nets[i](hidden)

        # Obtain velocities in latent space
        velocities = velocities_raw.reshape(
            batch_size, 2, self.num_vels, self.nlat, self.nlon
        )
        u = velocities[:, 0]
        v = velocities[:, 1]

        # Apply SL advection, reaction and diffusion blocks
        advected = self.advection[i](hidden, u, v, self.dt)
        hidden = hidden + advected

        diffused = self.diffusion[i](hidden)
        hidden = hidden + diffused

        reacted = self.reaction[i](hidden)
        hidden = hidden + reacted

        return hidden

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fields and winds.

        Parameters
        ----------
        fields : torch.Tensor
            Input fields of shape (batch, in_channels, nlat, nlon)
        Returns
        -------
        torch.Tensor
            Output fields of shape (batch, out_channels, nlat, nlon)
        """

        # Project features to latent space
        hidden = self._apply_checkpoint(self.input_proj, fields)

        for i in range(self.num_layers):

            if self.gradient_checkpoint:

                def checkpoint_wrapper(h, idx=i):
                    return self._layer_step(idx, h)

                hidden = checkpoint(checkpoint_wrapper, hidden, use_reentrant=False)
            else:
                hidden = self._layer_step(i, hidden)

        # Project back to physical space
        return self._apply_checkpoint(self.output_proj, hidden)
