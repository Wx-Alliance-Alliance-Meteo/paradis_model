"""Physically inspired neural architecture for the weather forecasting model."""

import torch
from torch import nn

from model.padding import GeoCyclicPadding
from model.gmblock import GMBlock
from model.neuralSemiLagrangian import NeuralSemiLagrangian

from utils.cubed_sphere.cubed_sphere import CubedSphere
from utils.cubed_sphere.cubed_sphere_padding import CubedSpherePadding


class Paradis(nn.Module):
    """Weather forecasting model main class."""

    def __init__(self, datamodule, cfg):
        super().__init__()

        # Extract dimensions from config
        output_dim = datamodule.num_out_features
        mesh_size = datamodule.grid_shape
        self.grid_type = datamodule.grid_type
        
        self.num_common_features = datamodule.num_common_features

        # Get channel sizes
        self.dynamic_channels = datamodule.dataset.num_in_dyn_features
        self.static_channels = datamodule.dataset.num_in_static_features

        # Get the number of time inputs
        self.n_inputs = datamodule.dataset.n_time_inputs

        # Specify hidden dimension based on multiplier or fixed size,
        # following configuration file
        if cfg.model.latent_multiplier > 0:
            hidden_dim = (
                cfg.model.latent_multiplier * self.dynamic_channels
                + self.static_channels
            )
            num_vels = hidden_dim
            diffusion_size = hidden_dim
        else:
            hidden_dim = cfg.model.latent_size
            num_vels = cfg.model.velocity_vectors
            diffusion_size = cfg.model.get("diffusion_size", hidden_dim)

        if self.grid_type == 'latlon':
            padding = GeoCyclicPadding
        elif self.grid_type == 'cubed_sphere':
            mesh_size = mesh_size[1:]
            num_elem = mesh_size[0]
            cubed_sphere = CubedSphere(num_elem=num_elem)
            padding = lambda pad: CubedSpherePadding(cubed_sphere, pad,)

        # Get the interpolation type
        adv_interpolation = cfg.model.adv_interpolation
        bias_channels = cfg.model.get("bias_channels", 4)

        # Input projection for combined dynamic and static features
        self.input_proj = GMBlock(
            input_dim=self.dynamic_channels + self.static_channels,
            hidden_dim=hidden_dim,
            output_dim=hidden_dim,
            layers=["SepConv", "SepConv"],
            bias_channels=bias_channels,
            mesh_size=mesh_size,
            activation=False,
            padding=padding
        )

        num_layers = cfg.model.num_layers

        # Timestep is set to 1 in the latent space
        self.dt = 1

        # Advection layer
        self.advection = NeuralSemiLagrangian(
            hidden_dim,
            mesh_size,
            num_vels=num_vels,
            interpolation=adv_interpolation,
            bias_channels=bias_channels,
            padding = padding 
        )

        # Diffusion-reaction layers
        self.diffusion_layers = nn.ModuleList(
            [
                GMBlock(
                    input_dim=hidden_dim,
                    output_dim=hidden_dim,
                    hidden_dim=diffusion_size,
                    layers=["SepConv", "CLinear", "SepConv"],
                    mesh_size=mesh_size,
                    activation=False,
                    pre_normalize=True,
                    bias_channels=bias_channels,
                    padding=padding
                )
                for _ in range(num_layers)
            ]
        )

        # Output projection
        self.output_proj = GMBlock(
            input_dim=hidden_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            layers=["SepConv", "CLinear"],
            mesh_size=mesh_size,
            activation=False,
            bias_channels=bias_channels,
            padding=padding,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model time step."""
        # Move panel dim with batch 
        # [batch, features, panel, xi, eta] -> [batch * panel, features, xi, eta]
        if self.grid_type == 'cubed_sphere':
            x = x.movedim(2,1).reshape(x.shape[0]*x.shape[2], x.shape[1], *x.shape[3:])

        # No gradients on lat/lon, ever
        # x_static = x[:, self.dynamic_channels :].detach()

        # Extract lat/lon from static features (last 2 channels)
        # lat_grid = x_static[:, -2, ...]
        # lon_grid = x_static[:, -1, ...]

        # Extract relevant input features that correspond to output (common features from last time step)
        skip = x[
            :,
            (self.n_inputs - 1) * self.num_common_features : self.n_inputs * self.num_common_features,
        ]

        # Project features to latent space
        z0 = self.input_proj(x)

        # Advection
        # z_adv = self.advection(z0, lat_grid, lon_grid, self.dt)
        z_adv = z0

        # Diffusion-reaction - sum contributions from all layers
        diffusion_rhs = torch.zeros_like(z0)
        for diffusion_layer in self.diffusion_layers:
            diffusion_rhs = diffusion_rhs + diffusion_layer(z_adv)

        # Final state
        z_final = z_adv + self.dt * diffusion_rhs


        # Input-output skip connections: Model learns incremental changes rather than full state predictions
        pred = skip + self.output_proj(z_final)
    
        if self.grid_type == 'cubed_sphere':
            n_panel = 6
            pred = pred.reshape(-1, n_panel, *pred.shape[1:]).movedim(1, 2)
    
        return pred
