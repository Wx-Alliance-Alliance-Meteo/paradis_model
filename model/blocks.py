"""
Simple neural-network blocks specialized for tensors shaped
[batch, channels, latitude, longitude].
"""

from collections import OrderedDict
from collections.abc import Sequence
from typing import Union, Type, Tuple

import torch
from torch import nn
from torch.cuda import nvtx

from model.padding import GeoCyclicPadding


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


class PhysicalDownsample(nn.Module):
    """Downsample a physical field exactly as in the non-sharded model.

    With spatial parallelism, the fine-grid latitude bands are gathered only for
    this inexpensive pooling operation.  The resulting coarse field is then
    re-sharded.  This avoids alignment errors between fine and coarse band
    boundaries for 2N-1 latitude grids.
    """

    def __init__(
        self,
        stride=4,
        spatial_context=None,
        global_input_height=None,
        global_output_height=None,
    ):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size=5, stride=stride, count_include_pad=False)
        self.padding = GeoCyclicPadding(2)
        self.spatial_context = spatial_context
        self.global_input_height = global_input_height
        self.global_output_height = global_output_height

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_context is not None and self.spatial_context.enabled:
            if self.global_input_height is None or self.global_output_height is None:
                raise ValueError("Global fine/coarse heights are required for spatial downsampling")

            with nvtx.range("downsample/SP_gather"):
                x = self.spatial_context.gather_latitude(x, self.global_input_height)

        with nvtx.range("downsample/pool"):
            x = self.padding(x)
            x = self.pool(x)

        if self.spatial_context is not None and self.spatial_context.enabled:
            if x.shape[-2] != self.global_output_height:
                raise RuntimeError(
                    f"Downsample produced {x.shape[-2]} latitude rows, expected "
                    f"{self.global_output_height}"
                )
            x = self.spatial_context.slice_latitude(
                x, self.global_output_height, dim=-2
            )
        return x


class CLinear(nn.Module):
    """Channel-wise linear transformation."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int = 1,
        bias: bool = True,
        **kwargs,
    ):
        super().__init__()
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SepConv(nn.Module):
    """Depthwise spatial convolution followed by a pointwise channel projection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int = 3,
        bias: bool = True,
        spatial_context=None,
        **kwargs,
    ):
        super().__init__()
        self.padding = (kernel_size - 1) // 2
        self.geo_padding = GeoCyclicPadding(
            self.padding,
            spatial_context=spatial_context,
            global_height=mesh_size[0],
        )
        self.depthwise = nn.Conv2d(
            input_dim, input_dim, kernel_size, groups=input_dim, bias=False
        )
        self.pointwise = nn.Conv2d(input_dim, output_dim, kernel_size=1, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.geo_padding(x)
        x = self.depthwise(x)
        return self.pointwise(x)


class ChannelNorm(nn.Module):
    """Normalize channels independently at every grid point."""

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        super().__init__()
        assert input_dim == output_dim
        self.eps = 1e-5
        self.weight = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(input_dim), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cvar, cmean = torch.var_mean(x, dim=-3, keepdim=False)
        inv_std = (self.eps + cvar) ** -0.5
        shifted_x = x - cmean[..., None, :, :]
        x = torch.einsum("...cij,...ij,c->...cij", shifted_x, inv_std, self.weight)
        return x + self.bias[..., :, None, None]


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
        spatial_context=None,
        **kwargs,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.rank = rank
        self.height, self.width = mesh_size
        self.spatial_context = spatial_context

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        U = self.U
        if self.spatial_context is not None and self.spatial_context.enabled:
            local_h = self.spatial_context.local_height(self.height)
            if x.shape[-2] == local_h:
                start, end = self.spatial_context.lat_range(self.height)
                U = U[:, start:end]
            elif x.shape[-2] != self.height:
                raise ValueError(
                    f"GlobalBias expected latitude height {self.height} or local height "
                    f"{local_h}, got {x.shape[-2]}"
                )

        bias_maps = torch.einsum("ck,kh,kw->chw", self.A, U, self.V)

        if self.projection is not None:
            bias_maps = torch.einsum(
                "oc,chw->ohw", self.projection.weight, bias_maps
            )

        return x + bias_maps.unsqueeze(0)


BLOCK_REGISTRY = {
    "SepConv": SepConv,
    "CLinear": CLinear,
    "ChannelNorm": ChannelNorm,
    "GlobalBias": GlobalBias,
}


class GMBlock(nn.Sequential):
    """Generic multilayer block."""

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
        spatial_context=None,
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

        for idx, layer in enumerate(layers):
            if isinstance(layer, str):
                if layer not in BLOCK_REGISTRY:
                    raise ValueError(
                        f"Unknown layer type: {layer}. Available: {list(BLOCK_REGISTRY.keys())}"
                    )
                ltype = BLOCK_REGISTRY[layer]
            else:
                ltype = layer

            layer_out_size = output_dim if idx == num_layers - 1 else hidden_dim[idx]
            layer_name = f"{idx}-{ltype.__name__}"
            layer_obj = ltype(
                input_dim=layer_in_size,
                output_dim=layer_out_size,
                mesh_size=mesh_size,
                kernel_size=kernel_size[idx],
                spatial_context=spatial_context,
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
                            spatial_context=spatial_context,
                        ),
                    )
                )

            if activation[idx]:
                blocks.append((f"{idx}-{activation_fn.__name__}", activation_fn()))

            layer_in_size = layer_out_size

        super().__init__(OrderedDict(blocks))
        init_module_convs(self, last_conv_scale=0.1)
