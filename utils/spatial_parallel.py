"""Utilities for hybrid spatial parallelism + DDP.

The spatial decomposition is one-dimensional over latitude.  Ranks are arranged
in contiguous spatial groups.  Ranks inside one spatial group process the same
batch and own different latitude bands; different spatial groups are ordinary
data-parallel replicas and process different batches.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.cuda import nvtx


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def distributed_rank_world() -> tuple[int, int]:
    """Return rank/world size before or after torch.distributed initialization."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size()

    # Lightning/DDP exports RANK/WORLD_SIZE in worker processes.  LOCAL_RANK is
    # intentionally not used because spatial groups may span nodes.
    return _env_int("RANK", 0), _env_int("WORLD_SIZE", 1)


@dataclass(frozen=True)
class HybridParallelInfo:
    global_rank: int
    world_size: int
    spatial_size: int
    spatial_rank: int
    data_parallel_rank: int
    data_parallel_size: int


def get_hybrid_parallel_info(cfg) -> HybridParallelInfo:
    rank, world_size = distributed_rank_world()
    sp_cfg = cfg.compute.get("spatial_parallel", {})
    enabled = bool(sp_cfg.get("enabled", False))
    spatial_size = int(sp_cfg.get("size", 1)) if enabled else 1

    # Lightning constructs the module once before launching DDP workers. At that
    # point torch.distributed and WORLD_SIZE may not exist yet, so use the
    # configured topology for rank 0. Worker processes will subsequently use
    # their real RANK/WORLD_SIZE values.
    if not (dist.is_available() and dist.is_initialized()) and "WORLD_SIZE" not in os.environ:
        world_size = int(cfg.compute.get("num_nodes", 1)) * int(
            cfg.compute.get("num_devices", 1)
        )
        rank = _env_int("RANK", 0)

    if spatial_size < 1:
        raise ValueError("compute.spatial_parallel.size must be >= 1")
    if world_size % spatial_size != 0:
        raise ValueError(
            f"World size ({world_size}) must be divisible by spatial parallel size "
            f"({spatial_size})."
        )

    return HybridParallelInfo(
        global_rank=rank,
        world_size=world_size,
        spatial_size=spatial_size,
        spatial_rank=rank % spatial_size,
        data_parallel_rank=rank // spatial_size,
        data_parallel_size=world_size // spatial_size,
    )




class _SpatialAllGather(torch.autograd.Function):
    """Autograd-aware equal-shape all-gather that works on subgroups.

    torch.distributed.nn.functional.all_gather has backend-dependent subgroup
    behavior in its backward path.  This implementation uses only all_gather
    and all_reduce with the explicit process group.
    """

    @staticmethod
    @torch.compiler.disable
    def forward(ctx, x, group, spatial_rank: int, spatial_size: int):
        x = x.contiguous()
        ctx.group = group
        ctx.spatial_rank = int(spatial_rank)
        ctx.spatial_size = int(spatial_size)
        outputs = [torch.empty_like(x) for _ in range(spatial_size)]
        with nvtx.range("SP/all_gather/FWD"):
            dist.all_gather(outputs, x, group=group)
        return tuple(outputs)

class _SpatialAllGather(torch.autograd.Function):
    """Autograd-aware equal-shape all-gather that works on subgroups."""

    @staticmethod
    @torch.compiler.disable
    def forward(ctx, x, group, spatial_rank: int, spatial_size: int):
        x = x.contiguous()

        ctx.group = group
        ctx.spatial_rank = int(spatial_rank)
        ctx.spatial_size = int(spatial_size)

        outputs = [torch.empty_like(x) for _ in range(spatial_size)]

        with nvtx.range("SP/all_gather/FWD"):
            dist.all_gather(outputs, x, group=group)

        return tuple(outputs)

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, *grad_outputs):
        with nvtx.range("SP/all_gather/BWD_stack"):
            stacked = torch.stack(
                [g.contiguous() for g in grad_outputs],
                dim=0,
            )

        backend = dist.get_backend(ctx.group)

        if backend == "nccl" or backend == dist.Backend.NCCL:
            grad_x = torch.empty_like(
                grad_outputs[ctx.spatial_rank]
            )

            with nvtx.range("SP/all_gather/BWD_reduce_scatter"):
                dist.reduce_scatter_tensor(
                    grad_x,
                    stacked,
                    op=dist.ReduceOp.SUM,
                    group=ctx.group,
                )

        else:
            with nvtx.range("SP/all_gather/BWD_all_reduce"):
                dist.all_reduce(
                    stacked,
                    op=dist.ReduceOp.SUM,
                    group=ctx.group,
                )

            grad_x = stacked[
                ctx.spatial_rank
            ].contiguous()

        return grad_x, None, None, None

class _LatitudeHalo(torch.autograd.Function):
    """Autograd-aware exchange of latitude halos between adjacent ranks.

    Pole halos are generated exactly like GeoCyclicPadding: mirror across the
    pole and rotate longitude by 180 degrees.  Interior halos are exchanged
    with the neighboring spatial rank.  The backward pass returns halo
    gradients to their owner rank so this operation is mathematically
    equivalent to padding a non-sharded global tensor.
    """

    @staticmethod
    @torch.compiler.disable
    def forward(
        ctx,
        x,
        pad_width: int,
        group,
        spatial_rank: int,
        spatial_size: int,
        group_start_rank: int,
    ):
        pad_width = int(pad_width)
        if pad_width == 0:
            ctx.pad_width = 0
            return x

        if x.ndim != 4:
            raise ValueError("Latitude halo exchange expects [batch, channels, lat, lon]")
        if x.shape[-2] <= pad_width:
            raise ValueError(
                f"Local latitude band ({x.shape[-2]}) must be wider than halo "
                f"({pad_width}). Reduce spatial_parallel.size."
            )
        if x.shape[-1] % 2 != 0:
            raise ValueError("Longitude size must be even for pole padding")

        ctx.pad_width = pad_width
        ctx.group = group
        ctx.spatial_rank = int(spatial_rank)
        ctx.spatial_size = int(spatial_size)
        ctx.local_height = x.shape[-2]
        ctx.width = x.shape[-1]
        ctx.group_start_rank = int(group_start_rank)

        p = pad_width
        prev_rank = spatial_rank - 1 if spatial_rank > 0 else None
        next_rank = spatial_rank + 1 if spatial_rank + 1 < spatial_size else None

        recv_before = None
        recv_after = None
        ops = []

        if prev_rank is not None:
            recv_before = torch.empty_like(x[:, :, :p, :])
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_before,
                    peer=group_start_rank + prev_rank,
                    group=group,
                )
            )
            ops.append(
                dist.P2POp(
                    dist.isend,
                    x[:, :, :p, :].contiguous(),
                    peer=group_start_rank + prev_rank,
                    group=group,
                )
            )
        else:
            source = x[:, :, 1 : p + 1, :]
            recv_before = torch.roll(source, shifts=x.shape[-1] // 2, dims=3).flip(2)

        if next_rank is not None:
            recv_after = torch.empty_like(x[:, :, -p:, :])
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_after,
                    peer=group_start_rank + next_rank,
                    group=group,
                )
            )
            ops.append(
                dist.P2POp(
                    dist.isend,
                    x[:, :, -p:, :].contiguous(),
                    peer=group_start_rank + next_rank,
                    group=group,
                )
            )
        else:
            source = x[:, :, -(p + 1) : -1, :]
            recv_after = torch.roll(source, shifts=x.shape[-1] // 2, dims=3).flip(2)

        if ops:
            with nvtx.range("SP/halo/FWD_exchange"):
                works = dist.batch_isend_irecv(ops)

                with nvtx.range("SP/halo/FWD_wait"):
                    for work in works:
                        work.wait()

        return torch.cat([recv_before, x, recv_after], dim=2)

    @staticmethod
    @torch.compiler.disable
    def backward(ctx, grad_output):
        if ctx.pad_width == 0:
            return grad_output, None, None, None, None, None

        p = ctx.pad_width
        h = ctx.local_height
        width = ctx.width
        spatial_rank = ctx.spatial_rank
        spatial_size = ctx.spatial_size
        group = ctx.group
        group_start_rank = ctx.group_start_rank

        grad_before = grad_output[:, :, :p, :].contiguous()
        grad_x = grad_output[:, :, p : p + h, :].contiguous()
        grad_after = grad_output[:, :, p + h :, :].contiguous()

        prev_rank = spatial_rank - 1 if spatial_rank > 0 else None
        next_rank = spatial_rank + 1 if spatial_rank + 1 < spatial_size else None
        ops = []

        # Gradient produced on the previous rank from our first p rows.
        recv_from_prev = None
        if prev_rank is not None:
            recv_from_prev = torch.empty_like(grad_x[:, :, :p, :])
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_from_prev,
                    peer=group_start_rank + prev_rank,
                    group=group,
                )
            )
            ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_before,
                    peer=group_start_rank + prev_rank,
                    group=group,
                )
            )
        else:
            edge_grad = torch.roll(
                grad_before.flip(2), shifts=-(width // 2), dims=3
            )
            grad_x[:, :, 1 : p + 1, :] += edge_grad

        # Gradient produced on the next rank from our last p rows.
        recv_from_next = None
        if next_rank is not None:
            recv_from_next = torch.empty_like(grad_x[:, :, -p:, :])
            ops.append(
                dist.P2POp(
                    dist.irecv,
                    recv_from_next,
                    peer=group_start_rank + next_rank,
                    group=group,
                )
            )
            ops.append(
                dist.P2POp(
                    dist.isend,
                    grad_after,
                    peer=group_start_rank + next_rank,
                    group=group,
                )
            )
        else:
            # after = roll(x[:, :, -(p+1):-1], W/2).flip(lat)
            edge_grad = torch.roll(
                grad_after.flip(2), shifts=-(width // 2), dims=3
            )
            grad_x[:, :, -(p + 1) : -1, :] += edge_grad

        if ops:
            with nvtx.range("SP/halo/BWD_exchange"):
                works = dist.batch_isend_irecv(ops)

                with nvtx.range("SP/halo/BWD_wait"):
                    for work in works:
                        work.wait()

        if recv_from_prev is not None:
            grad_x[:, :, :p, :] += recv_from_prev
        if recv_from_next is not None:
            grad_x[:, :, -p:, :] += recv_from_next

        return grad_x, None, None, None, None, None


class SpatialParallelContext:
    """Runtime description and collectives for latitude-band parallelism."""

    def __init__(self, cfg):
        self.cfg = cfg
        sp_cfg = cfg.compute.get("spatial_parallel", {})
        self.requested = bool(sp_cfg.get("enabled", False))
        self.configured_size = int(sp_cfg.get("size", 1)) if self.requested else 1
        self._group = None
        self._group_world_size: Optional[int] = None
        self._group_global_rank: Optional[int] = None

    @property
    def info(self) -> HybridParallelInfo:
        return get_hybrid_parallel_info(self.cfg)

    @property
    def enabled(self) -> bool:
        return self.info.spatial_size > 1

    @property
    def spatial_size(self) -> int:
        return self.info.spatial_size

    @property
    def spatial_rank(self) -> int:
        return self.info.spatial_rank

    @property
    def data_parallel_size(self) -> int:
        return self.info.data_parallel_size

    @property
    def data_parallel_rank(self) -> int:
        return self.info.data_parallel_rank

    @property
    def is_group_leader(self) -> bool:
        return self.spatial_rank == 0

    def initialize_group(self):
        if not self.enabled:
            return None
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError(
                "Spatial parallelism requires torch.distributed to be initialized. "
                "Launch with Lightning DDP/torchrun and use at least spatial_parallel.size GPUs."
            )

        world_size = dist.get_world_size()
        global_rank = dist.get_rank()
        if (
            self._group is not None
            and self._group_world_size == world_size
            and self._group_global_rank == global_rank
        ):
            return self._group

        size = self.spatial_size
        if world_size % size != 0:
            raise ValueError(
                f"World size ({world_size}) must be divisible by spatial size ({size})"
            )

        my_group = None
        # All ranks create all groups in the same order, as required by c10d.
        for start in range(0, world_size, size):
            ranks = list(range(start, start + size))
            group = dist.new_group(ranks=ranks)
            if start <= global_rank < start + size:
                my_group = group

        self._group = my_group
        self._group_world_size = world_size
        self._group_global_rank = global_rank
        return self._group

    @staticmethod
    def _range(height: int, parts: int, index: int) -> tuple[int, int]:
        if height < parts:
            raise ValueError(
                f"Cannot split {height} latitude rows over {parts} spatial ranks"
            )
        base, rem = divmod(int(height), int(parts))
        start = index * base + min(index, rem)
        length = base + (1 if index < rem else 0)
        return start, start + length

    def lat_range(self, global_height: int) -> tuple[int, int]:
        if not self.enabled:
            return 0, int(global_height)
        return self._range(global_height, self.spatial_size, self.spatial_rank)

    def local_height(self, global_height: int) -> int:
        start, end = self.lat_range(global_height)
        return end - start

    def slice_latitude(
        self, x: torch.Tensor, global_height: Optional[int] = None, dim: int = -2
    ) -> torch.Tensor:
        if not self.enabled:
            return x
        dim = dim if dim >= 0 else x.ndim + dim
        if global_height is None:
            global_height = x.shape[dim]
        start, end = self.lat_range(global_height)
        if x.shape[dim] == end - start:
            return x
        if x.shape[dim] != global_height:
            raise ValueError(
                f"Expected global latitude size {global_height} or local size {end-start}, "
                f"got {x.shape[dim]}"
            )
        sl = [slice(None)] * x.ndim
        sl[dim] = slice(start, end)
        return x[tuple(sl)]

    def gather_latitude(self, x: torch.Tensor, global_height: int) -> torch.Tensor:
        """Autograd-aware all-gather of uneven latitude bands."""
        if not self.enabled:
            return x
        group = self.initialize_group()
        sizes = [
            self._range(global_height, self.spatial_size, r)[1]
            - self._range(global_height, self.spatial_size, r)[0]
            for r in range(self.spatial_size)
        ]
        expected = sizes[self.spatial_rank]
        if x.shape[-2] != expected:
            raise ValueError(
                f"Rank {self.spatial_rank} owns {expected} rows for global height "
                f"{global_height}, got tensor with {x.shape[-2]} rows"
            )

        max_h = max(sizes)
        pad_h = max_h - x.shape[-2]
        x_pad = F.pad(x, (0, 0, 0, pad_h)) if pad_h else x
        chunks = _SpatialAllGather.apply(
            x_pad.contiguous(), group, self.spatial_rank, self.spatial_size
        )
        trimmed = [chunk[..., : sizes[r], :] for r, chunk in enumerate(chunks)]
        return torch.cat(trimmed, dim=-2)

    def latitude_halo(self, x: torch.Tensor, pad_width: int) -> torch.Tensor:
        if not self.enabled or pad_width == 0:
            return x
        group = self.initialize_group()
        group_start_rank = self.data_parallel_rank * self.spatial_size
        return _LatitudeHalo.apply(
            x,
            int(pad_width),
            group,
            self.spatial_rank,
            self.spatial_size,
            group_start_rank,
        )

    def loss_scale(self, global_height: int) -> float:
        """Scale a local mean loss so DDP's world average equals global-grid mean."""
        if not self.enabled:
            return 1.0
        return self.spatial_size * self.local_height(global_height) / float(global_height)
