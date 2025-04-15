"""Physically inspired neural architecture for the weather forecasting model."""

import torch
from torch import nn
import torch.nn.functional as F

from model.clp_block import CLP
from model.clp_variational import VariationalCLP
from model.padding import GeoCyclicPadding


class NeuralSemiLagrangian(nn.Module):
    """Implements the semi-Lagrangian advection."""

    def __init__(self, hidden_dim: int, mesh_size: tuple, lat_grid, lon_grid):
        super().__init__()

        # For cubic interpolation
        self.padding = 2
        self.padding_interp = GeoCyclicPadding(self.padding)
        self.hidden_dim = hidden_dim

        # Neural network that will learn an effective velocity along the trajectory
        # Output 2 channels per hidden dimension for u and v
        self.position_net = CLP(2*hidden_dim, 2*hidden_dim, mesh_size, pointwise_conv=True)

        base_grid_x = torch.linspace(-1, 1, mesh_size[0])
        base_grid_y = torch.linspace(-1, 1, mesh_size[1])

        base_grid_x, base_grid_y = torch.meshgrid(base_grid_x, base_grid_y, indexing='ij')

        self.base_grid = torch.stack([base_grid_x, base_grid_y], dim=-1)

        self.lat_grid = lat_grid.expand(self.hidden_dim, -1, -1).unsqueeze(0)
        self.lon_grid = lon_grid.expand(self.hidden_dim, -1, -1).unsqueeze(0)


    def forward(
        self,
        hidden_features_0: torch.Tensor,
        hidden_features_1: torch.Tensor,
    ) -> torch.Tensor:
        """Compute advection using rotated coordinate system."""
        batch_size = hidden_features_0.shape[0]

        # Get learned velocities for each channel
        position = self.position_net(torch.cat([hidden_features_0, hidden_features_1], dim=1))

        # position = torch.sin(position)
        # Reshape velocities to separate u,v components per channel
        # [batch, 2*hidden_dim, lat, lon] -> [batch, hidden_dim, 2, lat, lon]
        position = position.view(
            batch_size, 2, self.hidden_dim, *position.shape[-2:]
        )

        lat_grid = self.lat_grid.to(position.device)
        lon_grid = self.lon_grid.to(position.device)

        grid_x = lon_grid + position[:, 0]
        grid_y = lat_grid + position[:, 1]

        # # Normalize grid
        # max_x = torch.max(grid_x)
        # min_x = torch.min(grid_x)

        # max_y = torch.max(grid_y)
        # min_y = torch.min(grid_y)

        # # Normalize grid between -1 and 1
        # grid_x = 2 * (grid_x - min_x) / (max_x - min_x) - 1
        # grid_y = 2 * (grid_y - min_y) / (max_y - min_y) - 1

        # Get the max and min values for normalization
        min_lat = torch.min(lat_grid)
        max_lat = torch.max(lat_grid)

        min_lon = torch.min(lon_grid)
        max_lon = torch.max(lon_grid)

        grid_x = 2 * (grid_x - min_lon) / (max_lon - min_lon) - 1
        grid_y = 2 * (grid_y - min_lat) / (max_lat - min_lat) - 1

        # Apply periodicity for outside values along longitude set to [-1, 1]
        grid_x = torch.remainder(grid_x + 1, 2) - 1

        # Apply geocyclic longitude roll for values beyond +/-90 degrees latitude
        geo_mask_left = grid_x <= 0
        geo_mask_right = grid_x > 0
        lat_mask_outer = torch.abs(grid_y) > 1
        grid_x = torch.where(lat_mask_outer & geo_mask_left, grid_x + 1, grid_x)
        grid_x = torch.where(lat_mask_outer & geo_mask_right, grid_x - 1, grid_x)

        # Mirror values outside of the range [-1, 1] along the latitude direction
        grid_y = torch.where(grid_y < -1, -(2 + grid_y), grid_y)
        grid_y = torch.where(grid_y > 1, 2 - grid_y, grid_y)

        # Apply padding and reshape hidden features
        dynamic_padded = self.padding_interp(hidden_features_0)

        # Make sure interpolation remains in right range after padding
        grid_x = grid_x * hidden_features_0.size(-1) / dynamic_padded.size(-1)
        grid_y = grid_y * hidden_features_0.size(-2) / dynamic_padded.size(-2)

        # Create interpolation grid
        grid = torch.stack([grid_x, grid_y], dim=-1)

        grid = grid.view(batch_size * self.hidden_dim, *grid.shape[-3:])

        # Apply padding and reshape features
        dynamic_padded = dynamic_padded.reshape(
            batch_size * self.hidden_dim, 1, *dynamic_padded.shape[-2:]
        )

        # Interpolate
        interpolated = torch.nn.functional.grid_sample(
            dynamic_padded,
            grid,
            align_corners=True,
            mode="bicubic",
            padding_mode="border",
        )

        # Reshape back to original dimensions
        interpolated = interpolated.view(
            batch_size, self.hidden_dim, *interpolated.shape[-2:]
        )

        return interpolated


class Paradis(nn.Module):
    """Weather forecasting model main class."""

    # Synoptic time scale (~1/Ω) in seconds
    SYNOPTIC_TIME_SCALE = 7.29212e5

    def __init__(self, datamodule, cfg):
        super().__init__()

        # Extract dimensions from config
        output_dim = datamodule.num_out_features
        mesh_size = [datamodule.lat_size, datamodule.lon_size]
        num_levels = len(cfg.features.pressure_levels)
        self.num_common_features = datamodule.num_common_features
        self.variational = cfg.ensemble.enable

        # Get channel sizes
        self.dynamic_channels = len(datamodule.dataset.dyn_input_features)
        self.static_channels = len(cfg.features.input.constants)

        hidden_dim = (
            cfg.model.hidden_multiplier * self.dynamic_channels
        ) + self.static_channels

        # Input projection for combined dynamic and static features
        self.input_proj = CLP(
            self.dynamic_channels + self.static_channels, hidden_dim, mesh_size
        )

        # Rescale the time step to a fraction of a synoptic time scale
        self.num_substeps = cfg.model.num_substeps
        self.dt = cfg.model.base_dt / self.SYNOPTIC_TIME_SCALE / self.num_substeps

        # Advection layer
        self.advection = nn.ModuleList(
            [
                CLP(hidden_dim, hidden_dim, mesh_size, pointwise_conv=True)
                for i in range(self.num_substeps)
            ]
        )
        self.advection_correct = nn.ModuleList(
            [
                CLP(hidden_dim, hidden_dim, mesh_size, pointwise_conv=True)
                for i in range(self.num_substeps)
            ]
        )


        # Diffusion-reaction layer
        self.diffusion_reaction = nn.ModuleList(
            [
                CLP(hidden_dim, hidden_dim, mesh_size, pointwise_conv=True)
                for i in range(self.num_substeps)
            ]
        )

        self.diffusion_reaction_correct = nn.ModuleList(
            [
                CLP(hidden_dim, hidden_dim, mesh_size, pointwise_conv=True)
                for i in range(self.num_substeps)
            ]
        )

        self.interpolator = nn.ModuleList(
            [
                NeuralSemiLagrangian(hidden_dim, mesh_size, datamodule.dataset.lat_rad_grid, datamodule.dataset.lon_rad_grid)
                for i in range(self.num_substeps)
            ]
        )

        # Diffusion-reaction layer
        self.diffusion_reaction_2 = nn.ModuleList(
            [
                CLP(2*hidden_dim, hidden_dim, mesh_size, pointwise_conv=True)
                for i in range(self.num_substeps)
            ]
        )

        # Output projection
        self.output_proj = nn.Sequential(
            GeoCyclicPadding(1),
            nn.Conv2d(hidden_dim, output_dim, kernel_size=3),
        )

        # Integrator method 
        self.integrator = cfg.compute.integrator 
        # Operator splitting 
        self.os = cfg.compute.operator_splitting
        #corrector 
        self.corrector= cfg.compute.corrector

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Extract lat/lon from static features (last 2 channels)
        x_static = x[:, self.dynamic_channels :]
        lat_grid = x_static[:, -2, :, :]
        lon_grid = x_static[:, -1, :, :]

        # Project features to latent space
        zt = self.input_proj(x)

        # Keep a copy for the residual projection
        z0 = zt.clone()

        # Compute advection and diffusion-reaction
        for i in range(self.num_substeps):
            if self.corrector == "CLP_Adv":
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)

                zt = z_adv + self.dt*dz

            elif self.corrector == "CLP_Adv2":
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)
                elif self.integrator == "rk4":
                    # Compute the diffusion residual with RK4
                    k1 = self.diffusion_reaction[i](z_adv)
                    k1y = z_adv + self.dt/2 * k1
                    k2 = self.diffusion_reaction[i](k1y)
                    k2y =  z_adv + self.dt/2 * k2
                    k3 = self.diffusion_reaction[i](k2y)
                    k3y =  z_adv + self.dt * k3
                    k4 = self.diffusion_reaction[i](k3y)
                    dz = 1./6. *(k1 + 2*k2 + 2*k3 + k4)

                zt = zt+z_adv*self.dt + self.dt*dz

            elif self.corrector == "CLP_Adv3":  # same as Calors' inplementation
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)

                zt = zt+z_adv + self.dt*dz

            elif self.corrector == "PC1":
                # Advect the features in latent space using a Semi-Lagrangian step
                z_adv = self.advection[i](zt)
                
                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dzp = self.diffusion_reaction[i](z_adv)

                z1 = z_adv + self.dt*dzp

                z_dep = self.interpolator[i](zt, z1)

                dzc = self.diffusion_reaction_2[i](torch.cat([z_dep, zt], dim=1))

                # Update the latent space features
                zt = zt + z_dep + self.dt * dzc
            elif self.corrector == "PC2": 
                # Advect the features in latent space using a Semi-Lagrangian step
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dzp = self.diffusion_reaction[i](z_adv)

                z1 = z_adv + self.dt*dzp

                z_dep = self.interpolator[i](zt, z1)

                dzc = self.diffusion_reaction_2[i](torch.cat([z_dep, zt], dim=1))

                # Update the latent space features
                zt = z_dep + self.dt * dzc
            elif self.corrector=="PC3":
            # Advect the features in latent space using a Semi-Lagrangian step
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dzp = self.diffusion_reaction[i](z_adv)

                z1 = z_adv + self.dt*dzp

                z_dep = self.interpolator[i](zt, z1)

                dzc = self.diffusion_reaction[i](z_dep)

                # Update the latent space features
                zt = zt + z_dep + self.dt * dzc
            elif self.corrector=="PC4":
            # Advect the features in latent space using a Semi-Lagrangian step
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dzp = self.diffusion_reaction[i](z_adv)

                z1 = z_adv + self.dt*dzp

                z_dep = self.interpolator[i](zt, z1)

                dzc = self.diffusion_reaction[i](z_dep)
                                                                                                                                                     # Update the latent space features
                zt = z_dep + self.dt * dzc
            elif self.corrector=="PC5":
            # Advect the features in latent space using a Semi-Lagrangian step
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dzp = self.diffusion_reaction[i](z_adv)

                z1 = z_adv + self.dt*dzp   #z at the t_{n+1} 
                
                z_dep = self.interpolator[i](zt, z1)

                dzc = self.diffusion_reaction[i](z_dep)
                # Update the latent space features
                zt = zt + 0.5*self.dt * (dzc+dzp)
            elif self.corrector == "CLP_Adv2_PC":
                z_adv = self.advection[i](zt)
                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)

                zp = zt+z_adv*self.dt + self.dt*dz     
                z_dep = self.interpolator[i](zt, zp)  # semi-lagrangian to find departure
                
                if self.integrator == "fe":
                    dzc = self.diffusion_reaction[i](z_dep)
                zt = z_dep + self.dt*dzc 
            elif self.corrector == "CLP_Adv2_PC2":
                z_adv = self.advection[i](zt)
                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)

                zp = zt+z_adv*self.dt + self.dt*dz
                z_dep = self.interpolator[i](zt, zp)  # semi-lagrangian to find departure

                if self.integrator == "fe":
                    dzc = self.diffusion_reaction_correct[i](z_dep)
                zt = z_dep + self.dt*dzc 
            elif self.corrector == "CLP_Adv2_SL":
                z_adv = self.advection[i](zt)
                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)

                zp = zt+z_adv*self.dt + self.dt*dz
                z_dep = self.interpolator[i](zt, zp)  # semi-lagrangian to find departure

                if self.integrator == "fe":
                    dzc = self.diffusion_reaction_correct[i](z_dep)
                zt = zt+ self.dt*z_dep + self.dt*dzc
            elif self.corrector == "CLP_Adv3_PC":
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)

                zp = zt+z_adv + self.dt*dz   
                
                z_dep = self.interpolator[i](zt, zp)  # semi-lagrangian to find departure
                if self.integrator == "fe":
                    dzc = self.diffusion_reaction[i](z_dep)
                zt = z_dep + self.dt*dzc
            elif self.corrector == "CLP_Adv3_PC2":
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)

                zp = zt+z_adv + self.dt*dz

                z_dep = self.interpolator[i](zt, zp)  # semi-lagrangian to find departure
                if self.integrator == "fe":
                    dzc = self.diffusion_reaction_correct[i](z_dep)
                zt = z_dep + self.dt*dzc
            elif self.corrector == "CLP_Adv3_SL":
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)

                zp = zt+z_adv + self.dt*dz

                z_dep = self.interpolator[i](zt, zp)  # semi-lagrangian to find departure
                if self.integrator == "fe":
                    dzc = self.diffusion_reaction_correct[i](z_dep)
                zt = zt + z_dep + self.dt*dzc
            elif self.corrector == "CLP_Adv1_PC":
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)

                zp = z_adv + self.dt*dz

                z_dep = self.interpolator[i](zt, zp)  # semi-lagrangian to find departure
                if self.integrator == "fe":
                    dzc = self.diffusion_reaction[i](z_dep)
                zt = z_dep + self.dt*dzc
            elif self.corrector == "CLP_Adv1_PC2":
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)

                zp = z_adv + self.dt*dz

                z_dep = self.interpolator[i](zt, zp)  # semi-lagrangian to find departure
                if self.integrator == "fe":
                    dzc = self.diffusion_reaction_correct[i](z_dep)
                zt = z_dep + self.dt*dzc
            elif self.corrector == "PC6":
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dz = self.diffusion_reaction[i](z_adv)

                zp = z_adv*self.dt + self.dt*dz

                z_dep = self.interpolator[i](zt, zp)  # semi-lagrangian to find departure
                if self.integrator == "fe":
                    dzc = self.diffusion_reaction[i](z_dep)
                zt = z_dep + self.dt*dzc
            elif self.corrector == "CLP_Adv2_CLP_Corr": 
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dzp = self.diffusion_reaction[i](z_adv)

                zp = zt+z_adv*self.dt + self.dt*dzp

                # correct with clp 
                z_dep = self.advection_correct[i](zp) 
                dzc = self.diffusion_reaction_correct[i](z_dep) 

                zt = zt + z_dep*self.dt+dzc*self.dt 

            elif self.corrector == "CLP_Adv3_CLP_Corr":  
                z_adv = self.advection[i](zt)

                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dzp = self.diffusion_reaction[i](z_adv)

                zp = zt+z_adv + self.dt*dzp

                # correct with clp
                z_dep = self.advection_correct[i](zp)
                dzc = self.diffusion_reaction_correct[i](z_dep)

                zt = zt + z_dep+dzc*self.dt
            elif self.corrector == "CLP_Adv4_CLP_Corr":
                z_adv = self.advection[i](zt)
                z_adv = zt + z_adv*self.dt
                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dzp = self.diffusion_reaction[i](z_adv)

                zp = zt+z_adv*self.dt + self.dt*dzp

                # correct with clp
                z_adv2= self.advection_correct[i](zp)
                z_dep = zt + z_adv2*self.dt
                dzc = self.diffusion_reaction_correct[i](z_dep)

                zt = zt + z_adv2*self.dt+dzc*self.dt
            elif self.corrector == "CLP_Adv4":
                z_adv = self.advection[i](zt)
                z_adv = zt + z_adv*self.dt
                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dzp = self.diffusion_reaction[i](z_adv)

                zt = zt+z_adv*self.dt + self.dt*dzp
            elif self.corrector == "CLP_Adv4_SL":
                z_adv = self.advection[i](zt)
                z_adv = zt + z_adv*self.dt
                # integrate the diffusion/reaction using RK methods
                if self.integrator == "fe":
                    dzp = self.diffusion_reaction[i](z_adv)

                zp = zt+z_adv*self.dt + self.dt*dzp

                # correct with semi lagrangian
                z_adv2=self.interpolator[i](zt, zp)
                z_dep = zt + z_adv2*self.dt
                dzc = self.diffusion_reaction_correct[i](z_dep)

                zt = zt + z_adv2*self.dt+dzc*self.dt
            
        # Return a scaled residual formulation
        return x[:, : self.num_common_features] + self.output_proj(zt - z0)


# zt = zt + z_adv +dt * dzp
# current value: zt
# zp = z_adv + dt*diff_react
# zp = (zt + dt*CLP1) + dt*diff_react
# zp = zt + z_adv + dt*diff_react (current prediction)

# step 2: correction
# call semi-lagrangian: (zt, zp) use this to compute trajectory
# z_dep = semi_lagragian(zt,zp)
# dz = diff_react(z_dep)

# zc = z_dep + dt*dz

# variation 2:
# same predictor with dt/2
# second predictor with dt/2
# to get value at t_{n+1}
# z_1/2 = z_t + dt/2*CLP1 + dt/2*diff_react
# z_p = z_1/2 + dt/2*CLP1 + dt/2*diff_react
# z_t, z_1/2, z_p
# correction: call semi-lagrangian (z_t, z_1/2, z_p)
# use this to get z_dep
# same diff_react


# variation 3:
# sub_step variation 1, variation 2
