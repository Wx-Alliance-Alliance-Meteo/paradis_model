import torch
from torch import nn
from collections import OrderedDict
from collections.abc import Sequence

from typing import Union, Type, Tuple

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
    """

    def __init__(
        self,
        layers: Sequence[Union[str, Type[nn.Module]]],
        input_dim: int,
        output_dim: int,
        mesh_size: Tuple[int, int],
        kernel_size: int = 3,
        hidden_dim: Union[Sequence, int] = 0,
        activation_fn: ActivationType = nn.SiLU,
        bias_channels: int = 0,
        activation: Union[Sequence, bool] = False,
        pre_normalize: bool = False,
        grid_type: str = "equiangular",
        basis_type: str = "morlet",
    ):
        """Perform initialization and construct layer sequence"""

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

        layer_in_size = input_dim

        blocks = []
        if pre_normalize:
            blocks.append(
                (
                    "0-ChannelNorm",
                    model.simple_blocks.ChannelNorm(
                        input_dim=input_dim, output_dim=input_dim
                    ),
                )
            )

        for idx, l in enumerate(layers):
            if isinstance(l, str):
                ltype = directory[l]
            else:
                assert issubclass(l, nn.Module)
                ltype = l

            if idx == num_layers - 1:
                layer_out_size = output_dim
            else:
                layer_out_size = hidden_dim[idx]

            ltypename = ltype.__name__
            layer_name = f"{idx}-{ltypename}"

            layer_obj = ltype(
                input_dim=layer_in_size,
                output_dim=layer_out_size,
                mesh_size=mesh_size,
                kernel_size=kernel_size,
                grid_type=grid_type,
                basis_type=basis_type,
            )

            blocks.append((layer_name, layer_obj))

            if idx == 0 and bias_channels > 0:
                blocks.append(
                    (
                        "0-GlobalBias",
                        model.simple_blocks.GlobalBias(
                            input_dim=bias_channels,
                            output_dim=layer_out_size,
                            mesh_size=mesh_size,
                        ),
                    )
                )

            if activation[idx]:
                blocks.append((f"{idx}-activation", activation_fn()))

            layer_in_size = layer_out_size

        super().__init__(OrderedDict(blocks))
