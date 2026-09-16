"""
Simple PARADIS neural-network blocks.

The modules operate on tensors shaped [batch, channels, latitude, longitude].
They can optionally be converted in-place to channel tensor parallelism after
``torch.distributed`` is initialized. The serial path is unchanged.
"""

from collections import OrderedDict
from collections.abc import Sequence
from typing import Union, Type, Tuple, Optional, Sequence as TypingSequence

import torch
import torch.distributed as dist
import torch.distributed.nn.functional as dist_nn
from torch import nn
import torch.nn.functional as F

from model.padding import GeoCyclicPadding
from model.tensor_parallel import (
    TensorShardSpec,
    contiguous_indices,
    copy_to_tensor_parallel,
    reduce_from_tensor_parallel,
    reduce_scatter_channels,
    register_tp_spec,
    split_range,
)


def init_conv2d_default(conv: nn.Conv2d, *, scale: float = 1.0) -> None:
    nn.init.kaiming_normal_(conv.weight, mode="fan_in", nonlinearity="relu")
    if scale != 1.0:
        with torch.no_grad():
            conv.weight.mul_(scale)
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0.0)


def init_module_convs(m: nn.Module, *, last_conv_scale: float = 1.0) -> None:
    convs = []
    for module in m.modules():
        if isinstance(module, GlobalBias):
            continue
        if isinstance(module, nn.Conv2d):
            convs.append(module)

    for i, conv in enumerate(convs):
        scale = last_conv_scale if (i == len(convs) - 1) else 1.0
        init_conv2d_default(conv, scale=scale)


def _rank_indices_lists(size: int, world_size: int) -> list[list[int]]:
    return [
        list(range(*split_range(size, rank, world_size)))
        for rank in range(world_size)
    ]


def _normalize_rank_indices(
    indices: Optional[TypingSequence[TypingSequence[int]]],
    size: int,
    world_size: int,
) -> list[list[int]]:
    if indices is None:
        return _rank_indices_lists(size, world_size)
    out = [list(map(int, x)) for x in indices]
    if len(out) != world_size:
        raise ValueError("rank_indices must contain one index list per TP rank")
    local = len(out[0])
    if any(len(x) != local for x in out):
        raise ValueError("All TP ranks must own the same number of channels")
    flattened = sorted(i for x in out for i in x)
    if flattened != list(range(size)):
        raise ValueError("rank_indices must form a partition of the global channels")
    return out


def _slice_parameter(param: nn.Parameter, dim: int, indices: torch.Tensor) -> nn.Parameter:
    data = param.detach().index_select(dim, indices.to(param.device)).contiguous().clone()
    return nn.Parameter(data, requires_grad=param.requires_grad)


def _permute_parameter(
    param: nn.Parameter, dim: int, permutation: Optional[TypingSequence[int]]
) -> nn.Parameter:
    if permutation is None:
        return param
    index = torch.as_tensor(permutation, dtype=torch.long, device=param.device)
    data = param.detach().index_select(dim, index).contiguous().clone()
    return nn.Parameter(data, requires_grad=param.requires_grad)


class PhysicalDownsample(nn.Module):
    """Downsample a physical field."""

    def __init__(self, stride=4):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=stride, count_include_pad=False)
        self.padding = GeoCyclicPadding(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.padding(x)
        return self.pool(x)


class CLinear(nn.Module):
    """Pointwise channel mixing, optionally tensor-parallel across channels."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int = 1,
        bias: bool = True,
    ):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)
        self._tp_mode = None

    def enable_tensor_parallel(
        self,
        *,
        group,
        rank: int,
        world_size: int,
        input_layout: str,
        output_layout: str,
        input_rank_indices: Optional[TypingSequence[TypingSequence[int]]] = None,
        output_permutation: Optional[TypingSequence[int]] = None,
    ) -> None:
        if world_size == 1 or self._tp_mode is not None:
            return
        if input_layout not in {"replicated", "sharded"}:
            raise ValueError(f"Unknown input layout: {input_layout}")
        if output_layout not in {"replicated", "sharded"}:
            raise ValueError(f"Unknown output layout: {output_layout}")
        if input_layout == "replicated" and output_layout == "replicated":
            return

        self._tp_group = group
        self._tp_rank = rank
        self._tp_size = world_size
        global_in = self.conv.in_channels
        global_out = self.conv.out_channels
        self._tp_global_in = global_in
        self._tp_global_out = global_out

        original_weight = self.conv.weight
        original_bias = self.conv.bias
        if output_permutation is not None:
            if len(output_permutation) != global_out:
                raise ValueError("output_permutation length must equal output_dim")
            original_weight = _permute_parameter(original_weight, 0, output_permutation)
            if original_bias is not None:
                original_bias = _permute_parameter(original_bias, 0, output_permutation)

        permutations = {0: list(map(int, output_permutation))} if output_permutation is not None else {}

        if input_layout == "replicated" and output_layout == "sharded":
            out_idx = contiguous_indices(global_out, rank, world_size, original_weight.device)
            self.conv.weight = _slice_parameter(original_weight, 0, out_idx)
            if original_bias is not None:
                self.conv.bias = _slice_parameter(original_bias, 0, out_idx)
            self.conv.in_channels = global_in
            self.conv.out_channels = len(out_idx)
            self._tp_mode = "replicated_to_sharded"
            register_tp_spec(
                self,
                "conv.weight",
                TensorShardSpec(
                    shard_dim=0,
                    global_size=global_out,
                    permutations=permutations,
                ),
            )
            if self.conv.bias is not None:
                register_tp_spec(
                    self,
                    "conv.bias",
                    TensorShardSpec(
                        shard_dim=0,
                        global_size=global_out,
                        permutations=permutations,
                    ),
                )
            return

        if input_layout == "sharded":
            rank_indices = _normalize_rank_indices(input_rank_indices, global_in, world_size)
            in_idx = torch.as_tensor(rank_indices[rank], dtype=torch.long, device=original_weight.device)
            self.conv.weight = _slice_parameter(original_weight, 1, in_idx)
            self.conv.in_channels = len(in_idx)
            self.conv.out_channels = global_out
            register_tp_spec(
                self,
                "conv.weight",
                TensorShardSpec(
                    shard_dim=1,
                    global_size=global_in,
                    rank_indices=rank_indices if input_rank_indices is not None else None,
                    permutations=permutations,
                ),
            )

            if output_layout == "sharded":
                out_idx = contiguous_indices(global_out, rank, world_size, original_weight.device)
                if original_bias is not None:
                    self.conv.bias = _slice_parameter(original_bias, 0, out_idx)
                    register_tp_spec(
                        self,
                        "conv.bias",
                        TensorShardSpec(
                            shard_dim=0,
                            global_size=global_out,
                            permutations=permutations,
                        ),
                    )
                self._tp_mode = "sharded_to_sharded"
            else:
                # Bias is added after the forward all-reduce and remains replicated.
                self.conv.bias = original_bias
                if original_bias is not None:
                    register_tp_spec(self, "conv.bias", TensorShardSpec())
                self._tp_mode = "sharded_to_replicated"
            return

        raise ValueError(
            f"Unsupported CLinear TP transition {input_layout} -> {output_layout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._tp_mode is None:
            return self.conv(x)

        if self._tp_mode == "replicated_to_sharded":
            x = copy_to_tensor_parallel(x, self._tp_group)
            return F.conv2d(x, self.conv.weight, self.conv.bias)

        partial = F.conv2d(x, self.conv.weight, bias=None)
        if self._tp_mode == "sharded_to_sharded":
            out = reduce_scatter_channels(partial, self._tp_group, self._tp_size)
            if self.conv.bias is not None:
                out = out + self.conv.bias.view(1, -1, 1, 1)
            return out

        if self._tp_mode == "sharded_to_replicated":
            out = reduce_from_tensor_parallel(partial, self._tp_group)
            if self.conv.bias is not None:
                out = out + self.conv.bias.view(1, -1, 1, 1)
            return out

        raise RuntimeError(f"Unknown TP mode {self._tp_mode}")


class SepConv(nn.Module):
    """Depthwise spatial convolution followed by pointwise channel mixing."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int = 3,
        bias: bool = True,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.geo_padding = GeoCyclicPadding(self.padding)

        self.depthwise = nn.Conv2d(
            input_dim, input_dim, kernel_size, groups=input_dim, bias=False
        )
        self.pointwise = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)
        self._tp_mode = None

    def enable_tensor_parallel(
        self,
        *,
        group,
        rank: int,
        world_size: int,
        input_layout: str,
        output_layout: str,
        output_permutation: Optional[TypingSequence[int]] = None,
    ) -> None:
        if world_size == 1 or self._tp_mode is not None:
            return
        if input_layout not in {"replicated", "sharded"} or output_layout not in {
            "replicated",
            "sharded",
        }:
            raise ValueError("Invalid SepConv TP layout")

        global_in = self.depthwise.in_channels
        global_out = self.pointwise.out_channels
        self._tp_group = group
        self._tp_rank = rank
        self._tp_size = world_size

        original_pw_weight = self.pointwise.weight
        original_pw_bias = self.pointwise.bias
        if output_permutation is not None:
            if len(output_permutation) != global_out:
                raise ValueError("output_permutation length must equal output_dim")
            original_pw_weight = _permute_parameter(original_pw_weight, 0, output_permutation)
            if original_pw_bias is not None:
                original_pw_bias = _permute_parameter(original_pw_bias, 0, output_permutation)
        permutations = {0: list(map(int, output_permutation))} if output_permutation is not None else {}

        if input_layout == "sharded":
            in_idx = contiguous_indices(global_in, rank, world_size, self.depthwise.weight.device)
            self.depthwise.weight = _slice_parameter(self.depthwise.weight, 0, in_idx)
            self.depthwise.in_channels = len(in_idx)
            self.depthwise.out_channels = len(in_idx)
            self.depthwise.groups = len(in_idx)
            register_tp_spec(
                self,
                "depthwise.weight",
                TensorShardSpec(shard_dim=0, global_size=global_in),
            )

            self.pointwise.weight = _slice_parameter(original_pw_weight, 1, in_idx)
            self.pointwise.in_channels = len(in_idx)
            self.pointwise.out_channels = global_out
            register_tp_spec(
                self,
                "pointwise.weight",
                TensorShardSpec(
                    shard_dim=1,
                    global_size=global_in,
                    permutations=permutations,
                ),
            )
        else:
            # Depthwise work is small for the only replicated-input SepConv
            # (the static encoder). Keep it replicated, and make pointwise
            # channel mixing column-parallel.
            register_tp_spec(self, "depthwise.weight", TensorShardSpec())

        if input_layout == "replicated" and output_layout == "sharded":
            out_idx = contiguous_indices(global_out, rank, world_size, original_pw_weight.device)
            self.pointwise.weight = _slice_parameter(original_pw_weight, 0, out_idx)
            if original_pw_bias is not None:
                self.pointwise.bias = _slice_parameter(original_pw_bias, 0, out_idx)
            self.pointwise.in_channels = global_in
            self.pointwise.out_channels = len(out_idx)
            register_tp_spec(
                self,
                "pointwise.weight",
                TensorShardSpec(
                    shard_dim=0,
                    global_size=global_out,
                    permutations=permutations,
                ),
            )
            if self.pointwise.bias is not None:
                register_tp_spec(
                    self,
                    "pointwise.bias",
                    TensorShardSpec(
                        shard_dim=0,
                        global_size=global_out,
                        permutations=permutations,
                    ),
                )
            self._tp_mode = "replicated_to_sharded"
            return

        if input_layout == "sharded" and output_layout == "sharded":
            out_idx = contiguous_indices(global_out, rank, world_size, original_pw_weight.device)
            if original_pw_bias is not None:
                self.pointwise.bias = _slice_parameter(original_pw_bias, 0, out_idx)
                register_tp_spec(
                    self,
                    "pointwise.bias",
                    TensorShardSpec(
                        shard_dim=0,
                        global_size=global_out,
                        permutations=permutations,
                    ),
                )
            self._tp_mode = "sharded_to_sharded"
            return

        if input_layout == "sharded" and output_layout == "replicated":
            self.pointwise.bias = original_pw_bias
            if original_pw_bias is not None:
                register_tp_spec(self, "pointwise.bias", TensorShardSpec())
            self._tp_mode = "sharded_to_replicated"
            return

        raise ValueError(
            f"Unsupported SepConv TP transition {input_layout} -> {output_layout}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.geo_padding(x)
        x = self.depthwise(x)

        if self._tp_mode is None:
            return self.pointwise(x)

        if self._tp_mode == "replicated_to_sharded":
            x = copy_to_tensor_parallel(x, self._tp_group)
            return F.conv2d(x, self.pointwise.weight, self.pointwise.bias)

        partial = F.conv2d(x, self.pointwise.weight, bias=None)
        if self._tp_mode == "sharded_to_sharded":
            out = reduce_scatter_channels(partial, self._tp_group, self._tp_size)
            if self.pointwise.bias is not None:
                out = out + self.pointwise.bias.view(1, -1, 1, 1)
            return out

        if self._tp_mode == "sharded_to_replicated":
            out = reduce_from_tensor_parallel(partial, self._tp_group)
            if self.pointwise.bias is not None:
                out = out + self.pointwise.bias.view(1, -1, 1, 1)
            return out

        raise RuntimeError(f"Unknown TP mode {self._tp_mode}")


class ChannelNorm(nn.Module):
    """Normalize over the global channel dimension at every grid point."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        assert input_dim == output_dim
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        self._tp_enabled = False

    def enable_tensor_parallel(
        self,
        *,
        group,
        rank: int,
        world_size: int,
        rank_indices: Optional[TypingSequence[TypingSequence[int]]] = None,
    ) -> None:
        if world_size == 1 or self._tp_enabled:
            return
        global_channels = self.weight.numel()
        all_indices = _normalize_rank_indices(rank_indices, global_channels, world_size)
        idx = torch.as_tensor(
            all_indices[rank], dtype=torch.long, device=self.weight.device
        )
        self.weight = _slice_parameter(self.weight, 0, idx)
        self.bias = _slice_parameter(self.bias, 0, idx)
        self._tp_group = group
        self._tp_rank = rank
        self._tp_size = world_size
        self._tp_global_channels = global_channels
        self._tp_enabled = True
        spec = TensorShardSpec(
            shard_dim=0,
            global_size=global_channels,
            rank_indices=all_indices if rank_indices is not None else None,
        )
        register_tp_spec(self, "weight", spec)
        register_tp_spec(
            self,
            "bias",
            TensorShardSpec(
                shard_dim=0,
                global_size=global_channels,
                rank_indices=all_indices if rank_indices is not None else None,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._tp_enabled:
            cvar, cmean = torch.var_mean(x, dim=-3, keepdim=False)
            inv_std = (self.eps + cvar) ** -0.5
            shifted_x = x - cmean[..., None, :, :]
            x = torch.einsum("...cij,...ij,c->...cij", shifted_x, inv_std, self.weight)
            return x + self.bias[..., :, None, None]

        # Pack sum and squared-sum into one collective. Accumulate in fp32 for
        # numerical stability under bf16 mixed precision. torch.var_mean uses
        # correction=1 by default, which is reproduced below.
        xf = x.float()
        local_sum = xf.sum(dim=1, keepdim=True)
        local_sq = (xf * xf).sum(dim=1, keepdim=True)
        packed = torch.cat([local_sum, local_sq], dim=1)
        packed = dist_nn.all_reduce(packed, op=dist.ReduceOp.SUM, group=self._tp_group)
        total_sum, total_sq = packed[:, :1], packed[:, 1:]
        n = self._tp_global_channels
        mean = total_sum / n
        if n > 1:
            var = (total_sq - total_sum.square() / n) / (n - 1)
        else:
            var = torch.zeros_like(mean)
        var = torch.clamp(var, min=0.0)
        inv_std = torch.rsqrt(var + self.eps)
        y = (xf - mean) * inv_std
        y = y.to(x.dtype)
        y = y * self.weight.view(1, -1, 1, 1)
        return y + self.bias.view(1, -1, 1, 1)


class GlobalBias(nn.Module):
    """Low-rank factorized spatial bias operator."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool = True,
        kernel_size: int = 0,
        mesh_size: Tuple[int, int],
        rank: int = 128,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.height, self.width = mesh_size

        self.A = nn.Parameter(torch.zeros(input_dim, rank), requires_grad=True)
        self.U = nn.Parameter(torch.zeros(rank, self.height), requires_grad=True)
        self.V = nn.Parameter(torch.zeros(rank, self.width), requires_grad=True)

        with torch.no_grad():
            nn.init.normal_(self.A, mean=0.0, std=1e-3)
            nn.init.normal_(self.U, mean=0.0, std=1e-3)
            nn.init.normal_(self.V, mean=0.0, std=1e-3)

        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = None
        self._tp_enabled = False

    def enable_tensor_parallel(self, *, group, rank: int, world_size: int) -> None:
        if world_size == 1 or self._tp_enabled:
            return
        self._tp_group = group
        self._tp_rank = rank
        self._tp_size = world_size

        # A/U/V remain replicated. copy_to_tensor_parallel() below sums their
        # backward contributions from all output-channel shards.
        register_tp_spec(self, "A", TensorShardSpec())
        register_tp_spec(self, "U", TensorShardSpec())
        register_tp_spec(self, "V", TensorShardSpec())

        if self.projection is not None:
            global_out = self.projection.out_features
            idx = contiguous_indices(global_out, rank, world_size, self.projection.weight.device)
            self.projection.weight = _slice_parameter(self.projection.weight, 0, idx)
            self.projection.out_features = len(idx)
            register_tp_spec(
                self,
                "projection.weight",
                TensorShardSpec(shard_dim=0, global_size=global_out),
            )
            self.output_dim = len(idx)
        else:
            # Rare case: the low-rank maps themselves are the output channels.
            global_in = self.A.shape[0]
            idx = contiguous_indices(global_in, rank, world_size, self.A.device)
            self.A = _slice_parameter(self.A, 0, idx)
            register_tp_spec(
                self,
                "A",
                TensorShardSpec(shard_dim=0, global_size=global_in),
            )
            self.input_dim = len(idx)
            self.output_dim = len(idx)
        self._tp_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection is None and self._tp_enabled:
            # A is channel-sharded in this uncommon case, while U/V remain
            # replicated. Sum only the gradient of the shared spatial basis.
            basis = torch.einsum("kh,kw->khw", self.U, self.V)
            basis = copy_to_tensor_parallel(
                basis.unsqueeze(0), self._tp_group
            ).squeeze(0)
            bias_maps = torch.einsum("ck,khw->chw", self.A, basis)
        else:
            bias_maps = torch.einsum("ck,kh,kw->chw", self.A, self.U, self.V)

        if self.projection is not None:
            if self._tp_enabled:
                # Forward is identity; backward sums the contributions of every
                # output shard so replicated A/U/V receive the full gradient.
                bias_maps = copy_to_tensor_parallel(
                    bias_maps.unsqueeze(0), self._tp_group
                ).squeeze(0)
            bias_maps = torch.einsum("oc,chw->ohw", self.projection.weight, bias_maps)

        return x + bias_maps.unsqueeze(0)


BLOCK_REGISTRY = {
    "SepConv": SepConv,
    "CLinear": CLinear,
    "ChannelNorm": ChannelNorm,
    "GlobalBias": GlobalBias,
}


class GMBlock(nn.Sequential):
    """Generic multilayer block with optional channel tensor parallelism."""

    def __init__(
        self,
        layers: Sequence[Union[str, Type[nn.Module]]],
        input_dim: int,
        output_dim: int,
        mesh_size: Tuple[int, int],
        kernel_size: Union[Sequence[int], int] = 5,
        hidden_dim: Union[Sequence, int] = 0,
        activation_fn: Type[nn.Module] = nn.SiLU,
        bias_channels: int = 0,
        activation: Union[Sequence, bool] = False,
        pre_normalize: bool = False,
    ):
        num_layers = len(layers)
        if num_layers == 0:
            raise ValueError("GMBlock: must specify at least one layer")

        if isinstance(activation, Sequence):
            assert len(activation) == num_layers
        else:
            activation = (True,) * (num_layers - 1) + (activation,)

        if isinstance(hidden_dim, Sequence):
            assert len(hidden_dim) == num_layers - 1
        else:
            if hidden_dim <= 0:
                hidden_dim = max(input_dim, output_dim)
            hidden_dim = (hidden_dim,) * (num_layers - 1)

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * num_layers
        else:
            assert len(kernel_size) == num_layers

        blocks = []

        if pre_normalize:
            blocks.append(
                (
                    "0-ChannelNorm",
                    ChannelNorm(input_dim=input_dim, output_dim=input_dim),
                )
            )

        layer_in_size = input_dim

        for idx, l in enumerate(layers):
            if isinstance(l, str):
                if l not in BLOCK_REGISTRY:
                    raise ValueError(
                        f"Unknown layer type: {l}. Available: {list(BLOCK_REGISTRY.keys())}"
                    )
                ltype = BLOCK_REGISTRY[l]
            else:
                ltype = l

            if idx == num_layers - 1:
                layer_out_size = output_dim
            else:
                layer_out_size = hidden_dim[idx]

            layer_name = f"{idx}-{ltype.__name__}"
            layer_obj = ltype(
                input_dim=layer_in_size,
                output_dim=layer_out_size,
                mesh_size=mesh_size,
                kernel_size=kernel_size[idx],
            )
            blocks.append((layer_name, layer_obj))

            if idx == 0 and bias_channels > 0:
                blocks.append(
                    (
                        "0-GlobalBias",
                        GlobalBias(
                            input_dim=bias_channels,
                            output_dim=layer_out_size,
                            mesh_size=mesh_size,
                        ),
                    )
                )

            if activation[idx]:
                blocks.append((f"{idx}-{activation_fn.__name__}", activation_fn()))

            layer_in_size = layer_out_size

        super().__init__(OrderedDict(blocks))
        init_module_convs(self, last_conv_scale=0.1)

    def enable_tensor_parallel(
        self,
        *,
        group,
        rank: int,
        world_size: int,
        input_layout: str,
        output_layout: str,
        first_input_rank_indices: Optional[TypingSequence[TypingSequence[int]]] = None,
        final_output_permutation: Optional[TypingSequence[int]] = None,
    ) -> None:
        """Convert channel-mixing layers in-place.

        All intermediate neural channels remain sharded. Only the first layer may
        consume a replicated tensor, and only the final layer may emit a
        replicated tensor.
        """
        if world_size == 1:
            return

        compute_layers = [m for m in self if isinstance(m, (CLinear, SepConv))]
        if not compute_layers:
            return
        last_compute = compute_layers[-1]
        first_compute = compute_layers[0]
        current_layout = input_layout

        for module in self:
            if isinstance(module, ChannelNorm):
                if current_layout == "sharded":
                    module.enable_tensor_parallel(
                        group=group,
                        rank=rank,
                        world_size=world_size,
                        rank_indices=(
                            first_input_rank_indices
                            if module is self[0] and first_input_rank_indices is not None
                            else None
                        ),
                    )
                continue

            if isinstance(module, (CLinear, SepConv)):
                target_layout = output_layout if module is last_compute else "sharded"
                kwargs = {}
                if module is first_compute and isinstance(module, CLinear):
                    kwargs["input_rank_indices"] = first_input_rank_indices
                if module is last_compute:
                    kwargs["output_permutation"] = final_output_permutation
                module.enable_tensor_parallel(
                    group=group,
                    rank=rank,
                    world_size=world_size,
                    input_layout=current_layout,
                    output_layout=target_layout,
                    **kwargs,
                )
                current_layout = target_layout
                continue

            if isinstance(module, GlobalBias):
                if current_layout != "sharded":
                    raise RuntimeError("GlobalBias TP expects sharded channel activations")
                module.enable_tensor_parallel(
                    group=group, rank=rank, world_size=world_size
                )

        if current_layout != output_layout:
            raise RuntimeError(
                f"GMBlock TP layout ended as {current_layout}, expected {output_layout}"
            )
