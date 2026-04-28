"""Paradis neural architecture adapted for shallow water equations."""

from torch.utils.checkpoint import checkpoint

import torch
from torch import nn
import torch.nn.functional as F


from model.advection import NeuralSemiLagrangian
from model.blocks import GMBlock, PhysicalDownsample, SepConv
from model.padding import GeoCyclicPadding


def get_scaled_timestep(original_timestep_seconds: float) -> float:
    return original_timestep_seconds * 7.29212e-5


_ACTIVATIONS = {
    "SiLU": nn.SiLU,
    "GELU": nn.GELU,
}


def _get_activation_cls(name: str) -> type[nn.Module]:
    if name not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation_fn '{name}'. Allowed: {list(_ACTIVATIONS.keys())}"
        )
    return _ACTIVATIONS[name]


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

        adv_interpolation = cfg.model.get("adv_interpolation")
        bias_channels = cfg.model.get("bias_channels", 4)

        self.num_layers = max(1, cfg.model.num_layers)
        self.dt = get_scaled_timestep(cfg.model.get("base_dt")) / self.num_layers

        # Input projection
        self.activation_function = _get_activation_cls(cfg.model.activation)

        input_dim = (
            datamodule.dataset.num_in_dyn_features
            + datamodule.dataset.num_in_static_features
        )
        self.num_common_features = datamodule.num_common_features
        self.n_inputs = datamodule.dataset.n_time_inputs

        # Wrapper for gradient checkpointing
        self.step_fn = self._layer_step
        self.gradient_checkpoint = cfg.compute.get("gradient_checkpointing", False)

        # Enable downsampling automatically for
        self.downsample_diffusion = cfg.model.get("downsample_diffusion", False)

        if self.gradient_checkpoint:
            self.step_fn = lambda i, h, hs: checkpoint(
                self._layer_step, i, h, hs, use_reentrant=False
            )

        input_layers = cfg.model.physblock.input_proj.layers
        vnet_layers = cfg.model.physblock.velocity_net.layers
        diffusion_layers = cfg.model.physblock.diffusion.layers
        reaction_layers = cfg.model.physblock.reaction.layers
        output_layers = cfg.model.physblock.output_proj.layers

        input_ldim = cfg.model.physblock.input_proj.hidden_dim
        vnet_ldim = cfg.model.physblock.velocity_net.hidden_dim
        diff_ldim = cfg.model.physblock.diffusion.hidden_dim
        reac_ldim = cfg.model.physblock.reaction.hidden_dim
        output_ldim = cfg.model.physblock.output_proj.hidden_dim

        self.nlat_coarse = self.nlat
        self.nlon_coarse = self.nlon

        if self.downsample_diffusion:
            self.nlat_coarse = (self.nlat - 1) // 4 + 1
            self.nlon_coarse = self.nlon // 4

        mesh_size_coarse = (self.nlat_coarse, self.nlon_coarse)

        self.input_proj = GMBlock(
            layers=input_layers,
            input_dim=input_dim,
            output_dim=hidden_dim,
            hidden_dim=input_ldim,
            mesh_size=mesh_size,
            activation=True,
            activation_fn=self.activation_function,
            pre_normalize=False,
            bias_channels=0,
        )

        self.velocity_nets = nn.ModuleList(
            [
                GMBlock(
                    layers=vnet_layers,
                    input_dim=hidden_dim,
                    output_dim=2 * self.num_vels,
                    hidden_dim=vnet_ldim,
                    mesh_size=mesh_size,
                    bias_channels=bias_channels,
                    activation_fn=self.activation_function,
                    pre_normalize=True,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.advection = nn.ModuleList(
            [
                NeuralSemiLagrangian(
                    cfg,
                    hidden_dim,
                    mesh_size,
                    num_vels=self.num_vels,
                    lat_grid=lat_grid,
                    lon_grid=lon_grid,
                    interpolation=adv_interpolation,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.diffusion = nn.ModuleList(
            [
                GMBlock(
                    layers=diffusion_layers,
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=diff_ldim,
                    mesh_size=mesh_size_coarse,
                    pre_normalize=True,
                    activation_fn=self.activation_function,
                    bias_channels=bias_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

        static_size = cfg.model.latent_size_static
        self.reaction = nn.ModuleList(
            [
                GMBlock(
                    layers=reaction_layers,
                    input_dim=hidden_dim + static_size,
                    output_dim=hidden_dim,
                    hidden_dim=reac_ldim,
                    mesh_size=mesh_size,
                    pre_normalize=True,
                    activation_fn=self.activation_function,
                    bias_channels=bias_channels,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_proj = GMBlock(
            layers=output_layers,
            input_dim=hidden_dim,
            output_dim=datamodule.num_out_features,
            hidden_dim=output_ldim,
            mesh_size=mesh_size,
            activation=False,
            activation_fn=self.activation_function,
            bias_channels=bias_channels,
        )

        self.alpha_adv = nn.Parameter(torch.full((self.num_layers, hidden_dim), -1.0))

        if self.downsample_diffusion:
            self.upsample = self.upsampler
            self.downsample = PhysicalDownsample()
        else:
            self.downsample = lambda x: x
            self.upsample = lambda x: x

        self.n_static = n_static = len(cfg.features.input.constants)

        # Encoder block for static features
        self.static_encoder = nn.Sequential(
            SepConv(n_static, 64, mesh_size, kernel_size=7),
            nn.SiLU(),
            GeoCyclicPadding(3),
            nn.Conv2d(64, 64, groups=64, kernel_size=7),
            nn.SiLU(),
            SepConv(64, static_size, mesh_size, kernel_size=5),
        )

    def upsampler(self, x: torch.Tensor) -> torch.Tensor:
        # Make longitude explicitly periodic before interpolation
        x_ext = torch.cat([x, x[..., :1]], dim=-1)

        # Interpolate to include both latitude endpoints and both 0/360 endpoints
        y_ext = F.interpolate(
            x_ext,
            size=(self.nlat, self.nlon + 1),
            mode="bilinear",
            align_corners=True,
        )

        return y_ext[..., :-1]

    def _apply_checkpoint(self, func, *args):
        if self.gradient_checkpoint:
            return checkpoint(func, *args, use_reentrant=False)
        else:
            return func(*args)

    def _diffusion(self, i: int, z: torch.Tensor) -> torch.Tensor:
        return self.upsample(self.diffusion[i](self.downsample(z)))

    def _layer_step(
        self, i: int, hidden: torch.Tensor, hidden_static: torch.Tensor
    ) -> torch.Tensor:
        """Single physics-informed latent update."""
        B = hidden.shape[0]

        # Predict latent velocities (u, v) for advection
        velocities_raw = self.velocity_nets[i](hidden)
        velocities = velocities_raw.view(B, 2, self.num_vels, self.nlat, self.nlon)
        u, v = velocities[:, 0], velocities[:, 1]

        g_adv = torch.sigmoid(self.alpha_adv[i]).to(hidden.dtype).view(1, -1, 1, 1)

        # Transport: Semi-Lagrangian advection
        advected = self.advection[i](hidden, u, v, self.dt)
        hidden = hidden + g_adv * (advected - hidden)

        # Mixing: Learned diffusion
        hidden = hidden + self._diffusion(i, hidden)

        # Add static features
        hidden_reac = torch.cat([hidden, hidden_static], dim=1)

        # Forcing: Pointwise reaction (primary nonlinearity)
        hidden = hidden + self.reaction[i](hidden_reac)

        return hidden

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        # Encode physical variables to latent space
        hidden = self._apply_checkpoint(self.input_proj, fields)
        hidden_static = self._apply_checkpoint(
            self.static_encoder, fields[:, -self.n_static :]
        )

        # Recurrent integration through physics layers
        for i in range(self.num_layers):
            hidden = self.step_fn(i, hidden, hidden_static)

        # Decode latent state back to prognostic variables
        return fields[
            :,
            (self.n_inputs - 1)
            * self.num_common_features : self.n_inputs
            * self.num_common_features,
        ] + self._apply_checkpoint(self.output_proj, hidden)
