import torch
from torch import nn

from model.padding import GeoCyclicPadding

from typing import Callable

# For type hinting, define a type for Pytorch activation functions.  These
# are classes that take an optional boolean (inplace, ignored) and return a
# function/object that takes tensors and returns tensors
ActivationType = Callable[[], Callable[[torch.Tensor], torch.Tensor]]


class CLPBlock(nn.Module):
    """Convolutional Layer Processor block."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        mesh_size: tuple,
        kernel_size: int = 3,
        activation: ActivationType = nn.SiLU,
        double_conv: bool = False,
        pointwise_conv: bool = False,
    ):
        super().__init__()

        # First convolution block
        intermediate_dim = input_dim if double_conv else output_dim
        layers = [
            GeoCyclicPadding(kernel_size // 2),
            nn.Conv2d(input_dim, intermediate_dim, kernel_size=kernel_size),
            nn.LayerNorm([intermediate_dim, mesh_size[0], mesh_size[1]]),
            activation(),
        ]

        # Optional pointwise convolutions for additional channel mixing
        if pointwise_conv:
            expanded_dim = 2 * intermediate_dim
            layers.extend(
                [
                    nn.Conv2d(intermediate_dim, expanded_dim, kernel_size=1),
                    activation(),
                    nn.Conv2d(expanded_dim, intermediate_dim, kernel_size=1),
                ]
            )

        # Optional second convolution block
        if double_conv:
            layers.extend(
                [
                    GeoCyclicPadding(kernel_size // 2),
                    nn.Conv2d(intermediate_dim, output_dim, kernel_size=kernel_size),
                    activation(),
                ]
            )

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# Helper function
def CLP(
    dim_in: int,
    dim_out: int,
    mesh_size: tuple,
    kernel_size: int = 3,
    activation: ActivationType = nn.SiLU,
    pointwise_conv: bool = False,
):
    """Create a double-convolution CLP block."""
    return CLPBlock(
        dim_in,
        dim_out,
        mesh_size,
        kernel_size,
        activation,
        double_conv=True,
        pointwise_conv=pointwise_conv,
    )
