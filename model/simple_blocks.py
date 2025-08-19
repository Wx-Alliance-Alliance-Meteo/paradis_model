import torch
from torch import nn

from model.padding import GeoCyclicPadding

from typing import Any      # For optional/unused parameters
from typing import Tuple    # For mesh_size when used
from typing import Callable # For padding function 

"""
    simple_blocks: Wrapper file to consolidate simple NN layers, defined as those that do
    not have activation functions.  This includes convolution, normalization, and bias layers.

    These modules are specialized to expect Tensors of shape (batch,channels,lat,lon).

    For consistency, all of these modules are defined to take the following parameters:
        * input_dim -- number of input channels
        * output_dim -- number of output channels
        * kernel_size -- width of the convolution kernel
        * bias -- whether to add a bias term
        * mesh_size -- (lat, lon) tuple of 2D mesh size

    Not all parameters are relevant to each module, and they will be either ignored if
    inapplicable or tested for consistency if the module imposes stricter requirements.
    For example, the convolution layers don't care about grid size and will ignore the
    mesh_size parameter, but FlatConv (2D-only convolution) cannot change the number
    of channels and will check that input_dim == output_dim.
"""


class FullConv(nn.Conv2d):
    """Wrapper function for a full 2D convolution, acting across all channels.  This
    creates C_out×C_in×(kernel)×(kernel) weight parameters and C bias parameters.  Includes
    GeoCyclic padding.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        kernel_size: int,  # Required
        bias: bool = True,  # Optional
        padding: Callable | None = None,
        **kwargs
    ):

        self._kernel_size = kernel_size

        super().__init__(input_dim, output_dim, kernel_size=kernel_size, bias=bias)

        if self._kernel_size > 1:
            pad = kernel_size // 2
            if padding is None:
                self._padding = GeoCyclicPadding(pad)
            else:
                self._padding = padding(pad)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._kernel_size > 1:
            input = self._padding(input)
        return super().forward(input)


class FlatConv(nn.Conv2d):
    """Wrapper class for a limited 2D convolution, acting on each channel independently.
    This creates C×1x(kernel)x(kernel) paremeters and C bias parameters.  Includes
    GeoCyclic padding and cannot change the number of channels.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,  # Must equal input dimension
        *,
        kernel_size: int,  # Required
        bias: bool = True,  # Optional
        padding: Callable | None = None,
        **kwargs
    ):
        assert input_dim == output_dim
        dim = input_dim

        self._kernel_size = kernel_size

        super().__init__(dim, dim, kernel_size=kernel_size, groups=dim, bias=bias)

        if self._kernel_size > 1:
            pad = kernel_size // 2
            if padding is None:
                self._padding = GeoCyclicPadding(pad)
            else:
                self._padding = padding(pad)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if self._kernel_size > 1:
            input = self._padding(input)
        return super().forward(input)


class CLinear(nn.Conv2d):
    """Linear layer that acts across the channel dimension, having C_out × C_in weight
    parameters and C_out bias parameters."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool = True,  # Optional
        **kwargs
    ):
        super().__init__(input_dim, output_dim, kernel_size=1, bias=bias)


class SepConv(nn.Module):
    """Wrapper class for a "separated" convolution, which acts first in 2D (across channels)
    and then applies a CLinear operator.  Has C_in×1×K×K + C_out×C_in weight parameters and
    C_out bias parameters (no bias applied during the 2D convolution)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        kernel_size: int,  # Required
        bias: bool = True,  # Optional
        padding: Callable | None = None,
        **kwargs
    ):
        super().__init__()

        self.kernel_size = kernel_size

        if kernel_size > 1:
            pad = kernel_size // 2
            if padding is None:
                self.padding = GeoCyclicPadding(pad)
            else:
                self.padding = padding(pad)

        self.conv = nn.Conv2d(
            input_dim, input_dim, kernel_size, groups=input_dim, bias=False
        )
        self.linear = CLinear(input_dim, output_dim, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.kernel_size > 1:
            x = self.padding(x)
        x = self.conv(x)
        x = self.linear(x)
        return x


## GlobalNorm -- full 3D normalization
class GlobalNorm(nn.LayerNorm):
    """Performs a "global normalization" using the pytorch LayerNorm infrasturcture.

    LayerNorm is defined to act over the last several dimensions of a tensor, and in our
    ordering that means (channel,lat,lon).  A three-dimensional layer norm thus normalizes
    the model *globally*.  As a result, this module creates 2*C*lat*lon parameters, for the
    scale and bias of each point separately.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool = True,  # passed to LayerNorm
        mesh_size: Tuple[int, int],  # required
        **kwargs
    ):
        assert input_dim == output_dim
        super().__init__(list((input_dim,) + mesh_size), bias=bias)


## ChannelNorm -- layer normalization across channels only


class ChannelNorm(nn.Module):
    """Performs a "channel normalization," equivalent to LayerNorm in attention and
    graph neural networks.

    Unlike GlobalNorm above, this module is defined to normalize along the channel
    dimension (dim -3 or +1) only.  This is more consistent to standard practice in
    attention-based models, whereby each token's latent spae is normalized separately;
    it is also used throughout Graphcast in a similar way.  Unlike GlobalNorm, this
    module only creates 2*C parameters.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool = True,  # Optional
        **kwargs
    ):
        assert input_dim == output_dim
        super().__init__()

        self.eps = 1e-5  # Fudge factor for standard deviation, copied from LayerNorm
        self.dim = input_dim

        self.weight = nn.Parameter(torch.ones(input_dim), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute current channel statistics
        cvar, cmean = torch.var_mean(x, dim=-3, keepdim=False)

        # Compute the inverse standard deviation for normalization
        inv_std = (self.eps + cvar) ** -0.5

        # Subtract mean from x
        shifted_x = x - cmean[..., None, :, :]  # Broadcast mean to correct shape

        # Apply inverse standard deviation and affine weight.
        # Using einsum (especially with opt_einsum) gives the engine a chance
        # to more efficiently fuse and reorder operations
        x = torch.einsum("...cij,...ij,c->...cij", shifted_x, inv_std, self.weight)

        # If present, add bias (broadcast to the correct shape)
        if self.bias is not None:
            x = x + self.bias[..., :, None, None]

        return x


## NormedConv -- convolution with global norm (used by CLP)


class NormedConv(nn.Module):
    """
    NormedConv builds a post-normalized convolution operator that applise a FullConv followed
    by a GlobalNorm, in terms of the simple layers defined in this file.  This is a complicated
    case that should ordinarily be constructed separately, but the current formulation of CLP
    uses this as a basic quasi-linear building block.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        bias: bool = True,  # passed to GlobalNorm
        kernel_size: int,  # required
        mesh_size: Tuple[int, int],  # required
        padding: Callable | None = None,
        **kwargs
    ):
        super().__init__()

        self.conv = FullConv(input_dim, output_dim, kernel_size=kernel_size, bias=True, padding=padding)
        self.norm = GlobalNorm(output_dim, output_dim, mesh_size=mesh_size, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        return x


## GlobalBias -- learned bias operator
class GlobalBias(nn.Module):
    """
    GlobalBias -- construct a learned bias operator, consisting of a few channels that
    independently vary at each grid point.  This operator allows the model designer to
    add learned geophyiscal features in a controlled way, see AIFS's small-number-of-
    parameters per token for example.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        mesh_size: Tuple[int, int],  # required
        **kwargs
    ):
        super().__init__()

        # The bias is a Cin  * lat * lon set of parameters.  The bias parameters start at zero
        # (no bias) and are learned over the optimization
        self.bias = nn.Parameter(
            torch.zeros(((input_dim,) + mesh_size)), requires_grad=True
        )

        # If the input (number of bias channels) and output (size of latent space) dimensions
        # aren't the same, we need a projection matrix to move from one to the other.  This
        # projection matrix should be randomly initialized; if it were also zero then gradients
        # couldn't flow back to the bias term
        if input_dim != output_dim:
            # Note that we'll use the projection weights directly in forward(), but wrapping
            # it in a Linear is handy for initialization.  No projection bias term is necessary
            # or desired.
            self.projection = nn.Linear(input_dim, output_dim, bias=False)
        else:
            self.projection = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Construct the expanded bias term
        if self.projection is None:
            y = self.bias  # No projection necessary
        else:
            # Apply the projection operator to the bias along the channel dimension
            # note that no batch dimension exists yet, that's only on x and we can
            # broadcast that later, saving on memory bandwidth
            y = torch.einsum("iab,ji->jab", self.bias, self.projection.weight)

        # Apply the expanded bias to x
        x = x + y[..., :, :, :]
        return x
