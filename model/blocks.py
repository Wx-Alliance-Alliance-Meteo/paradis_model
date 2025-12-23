from collections import OrderedDict
from collections.abc import Sequence
from typing import Tuple
from typing import Union, Type, Tuple

import torch
from torch import nn

from model.padding import GeoCyclicPadding

"""
    simple_blocks: Wrapper file to consolidate simple NN layers, defined as those that do
    not have activation functions.  This includes convolution, normalization, and bias layers.

    These modules are specialized to expect Tensors of shape (batch,channels,lat,lon).

    For consistency, all of these modules are defined to take the following parameters:
        * input_dim -- number of input channels (required)
        * output_dim -- number of output channels (required)
        * kernel_size -- width of the convolution kernel (default: varies by class)
        * bias -- whether to add a bias term (default: True)
        * mesh_size -- (lat, lon) tuple of 2D mesh size (required for some classes)
        * **kwargs -- additional parameters are accepted and ignored

    Not all parameters are relevant to each module. Each class explicitly defines only
    the parameters it uses, and accepts **kwargs for the rest. Unused parameters are
    silently ignored. Some modules impose stricter requirements and will assert/check
    parameter values (e.g., FlatConv requires input_dim == output_dim).
"""


class CLinear(nn.Module):
    """Channel-wise linear transformation."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class SepConv(nn.Module):
    """Separable convolution."""

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.geo_padding(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ChannelNorm(nn.Module):
    """Channel normalization layer."""

    def __init__(self, input_dim: int, output_dim: int):
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
        x = x + self.bias[..., :, None, None]
        return x


class GlobalBias(nn.Module):
    """Learned bias operator with geophysical features."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        bias: bool = True,
        kernel_size: int = 0,
    ):
        super().__init__()
        self.bias = nn.Parameter(
            torch.zeros(((input_dim,) + mesh_size)), requires_grad=True
        )

        if input_dim != output_dim:
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.projection is None:
            y = self.bias
        else:
            y = torch.einsum("iab,ji->jab", self.bias, self.projection.weight)

        x = x + y[..., :, :, :]
        return x


ActivationType = Type[nn.Module]

BLOCK_REGISTRY = {
    "SepConv": SepConv,
    "CLinear": CLinear,
    "ChannelNorm": ChannelNorm,
    "GlobalBias": GlobalBias,
}


class GMBlock(nn.Sequential):
    """
    Generic Multilayer Block.
    Composes several simple blocks with activation functions.
    """

    def __init__(
        self,
        layers: Sequence[Union[str, Type[nn.Module]]],
        input_dim: int,
        output_dim: int,
        mesh_size: Tuple[int, int],
        kernel_size: int = 3,
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
                kernel_size=kernel_size,
            )
            blocks.append((layer_name, layer_obj))

            if idx == 0 and bias_channels > 0:
                blocks.append(
                    (
                        f"0-GlobalBias",
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
