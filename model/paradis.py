from torch.utils.checkpoint import checkpoint
from torch.cuda import nvtx

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda import nvtx

from model.advection import NeuralSemiLagrangian
from model.blocks import GMBlock, PhysicalDownsample, SepConv
from model.tensor_parallel import TensorShardSpec, collect_tp_specs, contiguous_indices, register_tp_spec, split_range
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

        mesh_size = (self.nlat, self.nlon)

        hidden_dim = cfg.model.get("latent_size")
        self.hidden_dim = hidden_dim
        self.static_dim = 128
        self._tp_enabled = False
        self._tp_state_specs = {}

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
        self.n_inputs = cfg.dataset.n_time_inputs

        # Wrapper for gradient checkpointing
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
        static_dim = self.static_dim

        stride = cfg.model.get("coarsening_factor", 1)
        if stride < 1:
            raise ValueError("Coarsening factor must be >=1")

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
        )

        self.alpha_adv = nn.Parameter(torch.full((self.num_layers, hidden_dim), -1.0))

        self.downsample = PhysicalDownsample(stride=stride)

        self.n_static = n_static = len(cfg.features.input.constants)

        self.static_encoder = nn.Sequential(
            SepConv(n_static, 64, mesh_size, kernel_size=7),
            nn.SiLU(),
            GeoCyclicPadding(3),
            nn.Conv2d(64, 64, groups=64, kernel_size=7),
            nn.SiLU(),
            SepConv(64, static_dim, mesh_size, kernel_size=5),
        )

    def enable_tensor_parallel(
        self,
        *,
        group,
        rank: int,
        world_size: int,
        src_global_rank: int,
    ) -> None:
        """Convert PARADIS to persistent channel tensor parallelism in-place.

        The input weather fields and final predictions remain replicated. The
        latent, reaction, diffusion, advection-projection and velocity-vector
        channels stay sharded for the entire physics stack.
        """
        if world_size == 1 or self._tp_enabled:
            return

        required = {
            "latent_size": self.hidden_dim,
            "velocity_vectors": self.num_vels,
            "static_dim": self.static_dim,
        }

        for name, size in required.items():
            if size % world_size != 0:
                raise ValueError(
                    f"{name}={size} must be divisible by tensor_parallel_size={world_size}"
                )

        self._tp_group = group
        self._tp_rank = rank
        self._tp_size = world_size
        self._tp_src_global_rank = src_global_rank

        # Replicated meteorological input -> sharded latent state.
        self.input_proj.enable_tensor_parallel(
            group=group,
            rank=rank,
            world_size=world_size,
            input_layout="replicated",
            output_layout="sharded",
        )

        # Velocity-net output is reordered from [all u, all v] to
        # [u0,v0,u1,v1,...] before channel sharding. A contiguous shard then
        # contains both velocity components for the same vector subset.
        velocity_permutation = []
        for j in range(self.num_vels):
            velocity_permutation.extend((j, self.num_vels + j))

        for vnet in self.velocity_nets:
            vnet.enable_tensor_parallel(
                group=group,
                rank=rank,
                world_size=world_size,
                input_layout="sharded",
                output_layout="sharded",
                final_output_permutation=velocity_permutation,
            )

        for adv in self.advection:
            adv.enable_tensor_parallel(group=group, rank=rank, world_size=world_size)

        for diff in self.diffusion:
            diff.enable_tensor_parallel(
                group=group,
                rank=rank,
                world_size=world_size,
                input_layout="sharded",
                output_layout="sharded",
            )

        # The serial reaction input ordering is [hidden_all, static_all]. Each
        # TP rank physically holds [hidden_rank, static_rank]. Tell the first
        # CLinear exactly which original checkpoint columns belong to that rank.
        reaction_rank_indices = []
        for r in range(world_size):
            hs, he = split_range(self.hidden_dim, r, world_size)
            ss, se = split_range(self.static_dim, r, world_size)
            hidx = list(range(hs, he))
            sidx = list(range(ss, se))
            reaction_rank_indices.append(
                hidx + [self.hidden_dim + j for j in sidx]
            )

        for reac in self.reaction:
            reac.enable_tensor_parallel(
                group=group,
                rank=rank,
                world_size=world_size,
                input_layout="sharded",
                output_layout="sharded",
                first_input_rank_indices=reaction_rank_indices,
            )

        # Keep the output features replicated so the existing loss, reporting,
        # autoregressive loop and forecast writer need no changes.
        self.output_proj.enable_tensor_parallel(
            group=group,
            rank=rank,
            world_size=world_size,
            input_layout="sharded",
            output_layout="replicated",
        )

        # Static encoder: only the first depthwise convolution is duplicated.
        # Its pointwise projection creates the first sharded static channels;
        # the remaining depthwise/pointwise work stays local.
        self.static_encoder[0].enable_tensor_parallel(
            group=group,
            rank=rank,
            world_size=world_size,
            input_layout="replicated",
            output_layout="sharded",
        )

        depthwise = self.static_encoder[3]
        if not isinstance(depthwise, nn.Conv2d) or depthwise.groups != depthwise.in_channels:
            raise RuntimeError("Unexpected static encoder depthwise convolution")
        global_static_hidden = depthwise.in_channels
        idx = contiguous_indices(
            global_static_hidden, rank, world_size, depthwise.weight.device
        )
        depthwise.weight = nn.Parameter(
            depthwise.weight.detach().index_select(0, idx).contiguous().clone(),
            requires_grad=depthwise.weight.requires_grad,
        )
        if depthwise.bias is not None:
            depthwise.bias = nn.Parameter(
                depthwise.bias.detach().index_select(0, idx).contiguous().clone(),
                requires_grad=depthwise.bias.requires_grad,
            )
        depthwise.in_channels = idx.numel()
        depthwise.out_channels = idx.numel()
        depthwise.groups = idx.numel()
        register_tp_spec(
            depthwise,
            "weight",
            TensorShardSpec(shard_dim=0, global_size=global_static_hidden),
        )
        if depthwise.bias is not None:
            register_tp_spec(
                depthwise,
                "bias",
                TensorShardSpec(shard_dim=0, global_size=global_static_hidden),
            )

        self.static_encoder[5].enable_tensor_parallel(
            group=group,
            rank=rank,
            world_size=world_size,
            input_layout="sharded",
            output_layout="sharded",
        )

        # Gate follows the same hidden-channel partition.
        alpha_idx = contiguous_indices(
            self.hidden_dim, rank, world_size, self.alpha_adv.device
        )
        self.alpha_adv = nn.Parameter(
            self.alpha_adv.detach().index_select(1, alpha_idx).contiguous().clone(),
            requires_grad=self.alpha_adv.requires_grad,
        )
        register_tp_spec(
            self,
            "alpha_adv",
            TensorShardSpec(shard_dim=1, global_size=self.hidden_dim),
        )

        self.local_hidden_dim = self.hidden_dim // world_size
        self.local_static_dim = self.static_dim // world_size
        self.local_num_vels = self.num_vels // world_size
        self._tp_enabled = True

        # Replicated trainable tensors (GlobalBias factors, final output bias,
        # and the first static depthwise kernel) must begin identical across TP
        # ranks. Their backward paths already combine TP gradient contributions.
        state_specs = collect_tp_specs(self)
        params = dict(self.named_parameters())
        for name, spec in state_specs.items():
            if spec.replicated and name in params:
                torch.distributed.broadcast(
                    params[name].data,
                    src=src_global_rank,
                    group=group,
                )

        self._tp_state_specs = state_specs

    def tensor_parallel_state_specs(self) -> dict[str, TensorShardSpec]:
        return dict(self._tp_state_specs) if self._tp_enabled else {}

    def _compile(self):
        for i in range(self.num_layers):
            self.velocity_nets[i] = torch.compile(self.velocity_nets[i])
            self.diffusion[i] = torch.compile(self.diffusion[i])
            self.reaction[i] = torch.compile(self.reaction[i])
            self.advection[i] = torch.compile(self.advection[i])

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

    def _layer_step(
        self, i: int, hidden: torch.Tensor, hidden_static: torch.Tensor
    ) -> torch.Tensor:
        """Single physics-informed latent update."""
        B = hidden.shape[0]

        with nvtx.range("velocity_net"):
            # Predict latent velocities (u, v) for advection
            velocities_raw = self.velocity_nets[i](hidden)
            if self._tp_enabled:
                velocities = velocities_raw.view(
                    B, self.local_num_vels, 2, self.nlat_coarse, self.nlon_coarse
                )
                u, v = velocities[:, :, 0], velocities[:, :, 1]
            else:
                velocities = velocities_raw.view(
                    B, 2, self.num_vels, self.nlat_coarse, self.nlon_coarse
                )
                u, v = velocities[:, 0], velocities[:, 1]

        with nvtx.range("advection"):
            g_adv = torch.sigmoid(self.alpha_adv[i]).to(hidden.dtype).view(1, -1, 1, 1)

            # Transport: Semi-Lagrangian advection
            advected = self.advection[i](hidden, u, v, self.dt)
            hidden = hidden + g_adv * (advected - hidden)

        # Mixing: Learned diffusion
        with nvtx.range("diffusion"):
            hidden = hidden + self.diffusion[i](hidden)

        # Add static features
        with nvtx.range("reaction"):
            hidden_reac = torch.cat([hidden, hidden_static], dim=1)

            # Forcing: Pointwise reaction (primary nonlinearity)
            hidden = hidden + self.reaction[i](hidden_reac)

        return hidden

    def forward(self, fields):
        with nvtx.range("input_proj"):
            hidden = self._apply_checkpoint(self.input_proj, fields)

        with nvtx.range("static_encoder"):
            hidden_static = self._apply_checkpoint(
                self.static_encoder, fields[:, -self.n_static:]
            )

        skip = hidden
        with nvtx.range("downsample"):
            hidden = self.downsample(hidden)
            hidden_static = self.downsample(hidden_static)

        for i in range(self.num_layers):
            hidden = self.step_fn(i, hidden, hidden_static)

        with nvtx.range("upsample"):
            hidden = self.upsample(hidden) + skip

        with nvtx.range("output_proj"):
            out = self._apply_checkpoint(self.output_proj, hidden)

        return out
