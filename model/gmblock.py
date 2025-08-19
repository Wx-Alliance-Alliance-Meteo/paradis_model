import torch
from torch import nn
from collections import OrderedDict  # For nn.Sequential
from collections.abc import Sequence

from typing import Union, Type, Tuple

from model.padding import GeoCyclicPadding

ActivationType = Type[nn.Module]

# Build directory of simple blocks by inspecting the simple_blocks submodule, allowing
# configuration by string
directory = {}
import inspect
import model.simple_blocks

for k, v in inspect.getmembers(model.simple_blocks):
    if inspect.isclass(v) and issubclass(v, nn.Module):
        directory[k] = v


class GMBlock(nn.Sequential):
    """GMBlock: Generic Multilayer Block: compose several simple blocks with activation
    functions, giving a composite multilayer, nonlinear block.

    This class generalizes the previous convolutional layer processor block.  The CLPBlock
    contained many bespoke configuration options that in effect defined the implied number
    of layers, but it's simpler to just specify the layers directly.
    """

    def __init__(
        self,
        layers: Sequence[
            Union[str, Type[nn.Module]]
        ],  # sequence of modules for layers, allowing string lookup
        input_dim: int,  # Size of the input dimension
        output_dim: int,  # Size of the output dimension
        mesh_size: Tuple[int, int],  # 2D mesh size, NLat by NLon
        kernel_size: int = 3,  # Size of kernel for all 2D convolutional layers
        hidden_dim: Union[
            Sequence, int
        ] = 0,  # Size of hidden dimension; if <=0 use max(input,output); if list use per-layer
        activation_fn: ActivationType = nn.SiLU,  # Activation function to use
        bias_channels: int = 0,  # Number of learned bias channels
        activation: Union[
            Sequence, bool
        ] = False,  # Sequence: apply activation function per layer?  Bool: final layer only?
        pre_normalize: bool = False,  # Whether to apply a prenorm (ChannelNorm) to the input before other layers
        padding = GeoCyclicPadding, # Name of padding function to use
    ):
        """Perform initialization and construct layer sequence"""

        num_layers = len(layers)
        if num_layers == 0:
            raise ValueError("GMBlock: must specify at least one layer")

        if isinstance(activation, Sequence):  # Given a list of activations
            assert (
                len(activation) == num_layers
            )  # Assert activatation list is the correct size
        else:
            # Otherwise, build the activation list; activate all layers except optionally the
            # final one.  Note that two linear layers concatenated without an activation just
            # form a more complicated linear layer.
            activation = (True,) * (num_layers - 1) + (activation,)

        if isinstance(hidden_dim, Sequence):  # Given an input list
            assert len(hidden_dim) == num_layers - 1  # Assert it must be the right size
        else:
            if hidden_dim <= 0:
                # If the hidden dimension size is unspecified, use the
                # larger of input and output dimensions
                hidden_dim = max(input_dim, output_dim)
            # Replicate hidden_dim to sequence, re-using the same hidden dimension
            # for each internal layer
            hidden_dim = (hidden_dim,) * (num_layers - 1)

        # Initialize first layer and bias layer specially
        layer_in_size = input_dim

        blocks = []
        if pre_normalize:
            # Add a ChannelNorm layer before everything else
            blocks.append(
                (
                    "0-ChannelNorm",
                    model.simple_blocks.ChannelNorm(
                        input_dim=input_dim, output_dim=input_dim
                    ),
                )
            )

        for idx, l in enumerate(layers):
            # Construct the list of blocks
            if isinstance(
                l, str
            ):  # Layer specified by string, look up in SimpleBlocks directory
                ltype = directory[l]
            else:  # Otherwise, assume it's a type
                assert issubclass(l, nn.Module)
                ltype = l

            # Get the layer output size
            if idx == num_layers - 1:
                # Output layer
                layer_out_size = output_dim
            else:
                # Use hidden dimensions ize
                layer_out_size = hidden_dim[idx]

            ltypename = ltype.__name__
            layer_name = f"{idx}-{ltypename}"

            layer_obj = ltype(
                input_dim=layer_in_size,
                output_dim=layer_out_size,
                mesh_size=mesh_size,
                kernel_size=kernel_size,
                padding=padding,
            )

            blocks.append((layer_name, layer_obj))

            if idx == 0 and bias_channels > 0:  # Add optional bias after first layer
                # The GlobalBias block is constructed a bit specially, where input_dim refers to the fundamental
                # number of bias channels rather than the size of (x) given to .forward(x).  When invoked, GlobalBias
                # just adds the bias factor to (x), so #channels(x) == output_dim.  The construction used here effectively
                # acts like appending #bias channels to the input, then expanding the first (input) layer to input_dim+bias_channels.

                blocks.append(
                    (
                        f"0-GlobalBias",
                        model.simple_blocks.GlobalBias(
                            input_dim=bias_channels,
                            output_dim=layer_out_size,
                            mesh_size=mesh_size,
                        ),
                    )
                )

            # Add activation if necessary
            if activation[idx]:
                blocks.append((f"{idx}-{activation_fn.__name__}", activation_fn()))

            # input size of next layer is the output of this layer
            layer_in_size = layer_out_size

        super().__init__(OrderedDict(blocks))
