from torch.utils.checkpoint import checkpoint

import torch
from torch import nn
import torch.nn.functional as F

from model.advection import NeuralSemiLagrangian
from model.blocks import GMBlock, PhysicalDownsample, SepConv
from model.padding import GeoCyclicPadding
from utils.spatial_parallel import SpatialParallelContext
from torch.cuda import nvtx


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
    """PARADIS model with optional latitude-band spatial parallelism."""

    def __init__(self, datamodule, cfg, lat_grid, lon_grid):
        super().__init__()

        self.nlat = lat_grid.shape[0]
        self.nlon = lat_grid.shape[1]
        mesh_size = (self.nlat, self.nlon)

        self.spatial = SpatialParallelContext(cfg)

        hidden_dim = cfg.model.get("latent_size")
        self.num_vels = cfg.model.get("velocity_vectors")
        adv_interpolation = cfg.model.get("adv_interpolation")
        bias_channels = cfg.model.get("bias_channels", 4)

        self.num_layers = max(1, cfg.model.num_layers)
        self.dt = get_scaled_timestep(cfg.model.get("base_dt")) / self.num_layers

        self.activation_function = _get_activation_cls(cfg.model.activation)

        input_dim = (
            datamodule.dataset.num_in_dyn_features
            + datamodule.dataset.num_in_static_features
        )
        self.num_common_features = datamodule.num_common_features
        self.n_inputs = cfg.dataset.n_time_inputs

        self.step_fn = self._layer_step
        self.gradient_checkpoint = cfg.compute.get("gradient_checkpointing", False)
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
        static_dim = 128

        stride = cfg.model.get("coarsening_factor", 1)
        if stride < 1:
            raise ValueError("Coarsening factor must be >=1")
        self.stride = stride

        self.nlat_coarse = (self.nlat - 1) // stride + 1
        self.nlon_coarse = self.nlon // stride
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
            spatial_context=self.spatial,
        )

        self.velocity_nets = nn.ModuleList(
            [
                GMBlock(
                    layers=vnet_layers,
                    input_dim=hidden_dim,
                    output_dim=2 * self.num_vels,
                    hidden_dim=vnet_ldim,
                    mesh_size=mesh_size_coarse,
                    bias_channels=bias_channels,
                    activation_fn=self.activation_function,
                    pre_normalize=True,
                    spatial_context=self.spatial,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.advection = nn.ModuleList(
            [
                NeuralSemiLagrangian(
                    cfg,
                    hidden_dim,
                    mesh_size_coarse,
                    num_vels=self.num_vels,
                    lat_grid=lat_grid[::stride, ::stride],
                    lon_grid=lon_grid[::stride, ::stride],
                    interpolation=adv_interpolation,
                    spatial_context=self.spatial,
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
                    spatial_context=self.spatial,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.reaction = nn.ModuleList(
            [
                GMBlock(
                    layers=reaction_layers,
                    input_dim=hidden_dim + static_dim,
                    output_dim=hidden_dim,
                    hidden_dim=reac_ldim,
                    mesh_size=mesh_size_coarse,
                    pre_normalize=True,
                    activation_fn=self.activation_function,
                    bias_channels=bias_channels,
                    spatial_context=self.spatial,
                )
                for _ in range(self.num_layers)
            ]
        )

        self.output_proj = GMBlock(
            pre_normalize=True,
            layers=output_layers,
            input_dim=hidden_dim,
            output_dim=datamodule.num_out_features,
            hidden_dim=output_ldim,
            mesh_size=mesh_size,
            activation=False,
            activation_fn=self.activation_function,
            bias_channels=bias_channels,
            spatial_context=self.spatial,
        )

        self.alpha_adv = nn.Parameter(torch.full((self.num_layers, hidden_dim), -1.0))

        self.downsample = PhysicalDownsample(
            stride=stride,
            spatial_context=self.spatial,
            global_input_height=self.nlat,
            global_output_height=self.nlat_coarse,
        )

        self.n_static = n_static = len(cfg.features.input.constants)

        self.static_encoder = nn.Sequential(
            SepConv(
                n_static,
                64,
                mesh_size,
                kernel_size=7,
                spatial_context=self.spatial,
            ),
            nn.SiLU(),
            GeoCyclicPadding(
                3,
                spatial_context=self.spatial,
                global_height=self.nlat,
            ),
            nn.Conv2d(64, 64, groups=64, kernel_size=7),
            nn.SiLU(),
            SepConv(
                64,
                static_dim,
                mesh_size,
                kernel_size=5,
                spatial_context=self.spatial,
            ),
        )

    def _compile(self):
        # Distributed point-to-point halo exchanges are intentionally kept out
        # of torch.compile for now.  This method remains valid for non-spatial runs.
        # if self.spatial.enabled:
        #     return
        self._layer_step = torch.compile(self._layer_step)
        self.static_encoder = torch.compile(self.static_encoder)
        self.input_proj = torch.compile(self.input_proj)
        self.output_proj = torch.compile(self.output_proj)

        if self.gradient_checkpoint:
            self.step_fn = lambda i, h, hs: checkpoint(
                self._layer_step, i, h, hs, use_reentrant=False
            )
        else:
            self.step_fn = self._layer_step

    def upsample(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial.enabled:
            x = self.spatial.gather_latitude(x, self.nlat_coarse)

        # Make longitude explicitly periodic before interpolation.
        x_ext = torch.cat([x, x[..., :1]], dim=-1)
        y_ext = F.interpolate(
            x_ext,
            size=(self.nlat, self.nlon + 1),
            mode="bilinear",
            align_corners=True,
        )
        y = y_ext[..., :-1]

        if self.spatial.enabled:
            y = self.spatial.slice_latitude(y, self.nlat, dim=-2)
        return y

    def _apply_checkpoint(self, func, *args):
        if self.gradient_checkpoint:
            return checkpoint(func, *args, use_reentrant=False)
        return func(*args)

    def _layer_step(
        self, i: int, hidden: torch.Tensor, hidden_static: torch.Tensor
    ) -> torch.Tensor:
        B, _, H_local, W_local = hidden.shape

        with nvtx.range(f"L{i}/velocity"):
            velocities_raw = self.velocity_nets[i](hidden)

        velocities = velocities_raw.view(
            B, 2, self.num_vels, H_local, W_local
        )
        u, v = velocities[:, 0], velocities[:, 1]

        g_adv = (
            torch.sigmoid(self.alpha_adv[i])
            .to(hidden.dtype)
            .view(1, -1, 1, 1)
        )

        with nvtx.range(f"L{i}/advection"):
            advected = self.advection[i](hidden, u, v, self.dt)
            hidden = hidden + g_adv * (advected - hidden)

        with nvtx.range(f"L{i}/diffusion"):
            hidden = hidden + self.diffusion[i](hidden)

        with nvtx.range(f"L{i}/reaction"):
            hidden_reac = torch.cat([hidden, hidden_static], dim=1)
            hidden = hidden + self.reaction[i](hidden_reac)

        return hidden

    def forward(self, fields):
        if self.spatial.enabled:
            fields = self.spatial.slice_latitude(fields, self.nlat, dim=-2)

        with nvtx.range("input_proj"):
            hidden = self._apply_checkpoint(self.input_proj, fields)

        with nvtx.range("static_encoder"):
            hidden_static = self._apply_checkpoint(
                self.static_encoder, fields[:, -self.n_static :]
            )

        with nvtx.range("downsample"):
            skip = hidden
            hidden = self.downsample(hidden)
            hidden_static = self.downsample(hidden_static)

        with nvtx.range("step"):
            for i in range(self.num_layers):
                hidden = self.step_fn(i, hidden, hidden_static)

        with nvtx.range("output"):
            hidden = self.upsample(hidden) + skip
            return self._apply_checkpoint(self.output_proj, hidden)
