"""Small channel-tensor-parallel primitives used by PARADIS.

The TP layout is deliberately simple: tensors inside the physics stack are split
along channel dimension 1. Pointwise channel mixing computes partial outputs from
local input channels and reduce-scatters the result back along channels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

import torch
import torch.distributed as dist
from torch.autograd import Function


def _require_dist(group) -> None:
    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("Tensor parallelism requires torch.distributed to be initialized")
    if group is None:
        raise RuntimeError("Tensor parallel process group is not initialized")




# Keep NCCL collectives outside TorchDynamo/Inductor.  PyTorch collectives
# require dense contiguous buffers, while Inductor is free to choose layouts
# for tensors inside a compiled graph.  These small eager boundaries make the
# TP implementation robust under torch.compile while leaving the surrounding
# model compute compilable.
@torch.compiler.disable
def _tp_all_reduce(x: torch.Tensor, group) -> torch.Tensor:
    x = x.contiguous()
    dist.all_reduce(x, op=dist.ReduceOp.SUM, group=group)
    return x


@torch.compiler.disable
def _tp_all_gather_dim0(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    x = x.contiguous()
    out = torch.empty(
        (x.shape[0] * world_size, *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    dist.all_gather_into_tensor(out, x, group=group)
    return out


@torch.compiler.disable
def _tp_reduce_scatter_dim0(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    x = x.contiguous()
    if x.shape[0] % world_size != 0:
        raise RuntimeError(
            f"Leading dimension {x.shape[0]} is not divisible by TP size {world_size}"
        )
    out = torch.empty(
        (x.shape[0] // world_size, *x.shape[1:]),
        device=x.device,
        dtype=x.dtype,
    )
    dist.reduce_scatter_tensor(out, x, op=dist.ReduceOp.SUM, group=group)
    return out


def split_range(size: int, rank: int, world_size: int) -> tuple[int, int]:
    if size % world_size != 0:
        raise ValueError(
            f"Channel dimension {size} must be divisible by tensor_parallel_size={world_size}"
        )
    width = size // world_size
    return rank * width, (rank + 1) * width


def contiguous_indices(size: int, rank: int, world_size: int, device=None) -> torch.Tensor:
    start, end = split_range(size, rank, world_size)
    return torch.arange(start, end, dtype=torch.long, device=device)


class _CopyToTensorParallel(Function):
    """Identity in forward, sum gradients across TP ranks in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad = _tp_all_reduce(grad_output, ctx.group)
        return grad, None


class _ReduceFromTensorParallel(Function):
    """Sum partial outputs in forward, identity in backward.

    This is the model-parallel reduction used when a sharded channel stack emits
    a replicated tensor (for example the final weather-variable projection).
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, group):
        return _tp_all_reduce(x.clone(), group)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


class _GatherChannels(Function):
    """Gather channel shards in forward, reduce-scatter gradients in backward."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, world_size: int):
        ctx.group = group
        ctx.world_size = world_size
        x_cfirst = x.movedim(1, 0).contiguous()
        full_c = x_cfirst.shape[0] * world_size
        gathered = _tp_all_gather_dim0(x_cfirst, group, world_size)
        return gathered.movedim(0, 1).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_cfirst = grad_output.movedim(1, 0).contiguous()
        local_c = grad_cfirst.shape[0] // ctx.world_size
        out = _tp_reduce_scatter_dim0(grad_cfirst, ctx.group, ctx.world_size)
        return out.movedim(0, 1).contiguous(), None, None


class _ReduceScatterChannels(Function):
    """Sum partial outputs and scatter channel chunks across TP ranks."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, group, world_size: int):
        ctx.group = group
        ctx.world_size = world_size

        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], got shape {tuple(x.shape)}")
        if x.shape[1] % world_size != 0:
            raise ValueError(
                f"Output channels {x.shape[1]} must be divisible by TP size {world_size}"
            )

        # reduce_scatter_tensor splits dimension 0. Move channels there so each
        # rank receives a contiguous channel shard.
        x_cfirst = x.movedim(1, 0).contiguous()
        local_c = x.shape[1] // world_size
        out_cfirst = _tp_reduce_scatter_dim0(x_cfirst, group, world_size)
        return out_cfirst.movedim(0, 1).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_cfirst = grad_output.movedim(1, 0).contiguous()
        full_c = grad_cfirst.shape[0] * ctx.world_size
        gathered = _tp_all_gather_dim0(grad_cfirst, ctx.group, ctx.world_size)
        return gathered.movedim(0, 1).contiguous(), None, None


def copy_to_tensor_parallel(x: torch.Tensor, group) -> torch.Tensor:
    return _CopyToTensorParallel.apply(x, group)


def reduce_from_tensor_parallel(x: torch.Tensor, group) -> torch.Tensor:
    return _ReduceFromTensorParallel.apply(x, group)


def gather_channels(x: torch.Tensor, group, world_size: int) -> torch.Tensor:
    return _GatherChannels.apply(x, group, world_size)


def reduce_scatter_channels(
    x: torch.Tensor, group, world_size: int
) -> torch.Tensor:
    return _ReduceScatterChannels.apply(x, group, world_size)


@dataclass
class TensorShardSpec:
    """How a local TP parameter maps to the original full-model state tensor."""

    shard_dim: Optional[int] = None
    global_size: Optional[int] = None
    # Optional rank-specific indices in the original/pre-permuted global tensor.
    # Used by the first reaction CLinear where local input is
    # [hidden_rank, static_rank].
    rank_indices: Optional[Sequence[Sequence[int]]] = None
    # Permutations applied to a full checkpoint before sharding. Mapping dim ->
    # permutation where transformed = original.index_select(dim, permutation).
    permutations: Dict[int, Sequence[int]] = field(default_factory=dict)

    @property
    def replicated(self) -> bool:
        return self.shard_dim is None

    def _permute_full(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor
        for dim, perm in self.permutations.items():
            index = torch.as_tensor(perm, dtype=torch.long, device=out.device)
            out = out.index_select(dim, index)
        return out

    def _unpermute_full(self, tensor: torch.Tensor) -> torch.Tensor:
        out = tensor
        # Invert in reverse order in case multiple dimensions are ever used.
        for dim, perm in reversed(list(self.permutations.items())):
            p = torch.as_tensor(perm, dtype=torch.long, device=out.device)
            inv = torch.empty_like(p)
            inv[p] = torch.arange(p.numel(), device=p.device)
            out = out.index_select(dim, inv)
        return out

    def localize(
        self, full: torch.Tensor, rank: int, world_size: int
    ) -> torch.Tensor:
        if self.replicated:
            return full

        transformed = self._permute_full(full)
        dim = int(self.shard_dim)
        if self.rank_indices is not None:
            indices = torch.as_tensor(
                self.rank_indices[rank], dtype=torch.long, device=transformed.device
            )
            return transformed.index_select(dim, indices).contiguous()

        start, end = split_range(int(self.global_size), rank, world_size)
        return transformed.narrow(dim, start, end - start).contiguous()

    def gather(
        self,
        local: torch.Tensor,
        group,
        rank: int,
        world_size: int,
    ) -> torch.Tensor:
        if self.replicated:
            return local

        gathered = [torch.empty_like(local) for _ in range(world_size)]
        dist.all_gather(gathered, local.contiguous(), group=group)
        dim = int(self.shard_dim)

        if self.rank_indices is None:
            transformed = torch.cat(gathered, dim=dim)
        else:
            shape = list(local.shape)
            shape[dim] = int(self.global_size)
            transformed = torch.empty(
                shape, device=local.device, dtype=local.dtype
            )
            for r, shard in enumerate(gathered):
                indices = torch.as_tensor(
                    self.rank_indices[r], dtype=torch.long, device=local.device
                )
                transformed.index_copy_(dim, indices, shard)

        return self._unpermute_full(transformed)


def register_tp_spec(module: torch.nn.Module, name: str, spec: TensorShardSpec) -> None:
    specs = getattr(module, "_tp_state_specs", None)
    if specs is None:
        specs = {}
        setattr(module, "_tp_state_specs", specs)
    specs[name] = spec


def collect_tp_specs(module: torch.nn.Module) -> dict[str, TensorShardSpec]:
    specs: dict[str, TensorShardSpec] = {}
    for module_name, child in module.named_modules():
        child_specs = getattr(child, "_tp_state_specs", None)
        if not child_specs:
            continue
        for local_name, spec in child_specs.items():
            full_name = f"{module_name}.{local_name}" if module_name else local_name
            specs[full_name] = spec
    return specs
