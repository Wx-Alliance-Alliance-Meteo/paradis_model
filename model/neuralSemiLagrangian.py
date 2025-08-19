import torch
from torch import nn
import numpy as np 

from model.padding import GeoCyclicPadding
from model.gmblock import GMBlock

from utils.cubed_sphere.cubed_sphere import CubedSphere
from utils.cubed_sphere.cubed_sphere_padding import CubedSpherePadding
from math import pi 

class NeuralSemiLagrangian(nn.Module):
    """Implements the semi-Lagrangian advection."""

    def __init__(
        self,
        hidden_dim: int,
        mesh_size: tuple,
        num_vels: int,
        interpolation: str = "bicubic",
        bias_channels: int = 4,
        padding = GeoCyclicPadding
    ):
        super().__init__()

        # For cubic interpolation
        self.padding = 2
        self.padding_interp = padding(self.padding)
        self.hidden_dim = hidden_dim

        self.num_vels = num_vels
        self.mesh_size = mesh_size

        self.down_projection = GMBlock(
            input_dim=hidden_dim,
            output_dim=num_vels,
            mesh_size=mesh_size,
            layers=["CLinear"],
            padding=padding
        )

        self.up_projection = GMBlock(
            input_dim=num_vels,
            output_dim=hidden_dim,
            mesh_size=mesh_size,
            layers=["CLinear"],
            padding=padding
        )

        self.interpolation = interpolation

        # Neural network that will learn an effective velocity along the trajectory
        # Output 2 channels per hidden dimension for u and v
        self.velocity_net = GMBlock(
            input_dim=hidden_dim,
            output_dim=2 * num_vels,
            hidden_dim=hidden_dim,
            kernel_size=3,
            mesh_size=mesh_size,
            layers=["SepConv"],
            bias_channels=bias_channels,
            activation=False,
            pre_normalize=True,
            padding=padding
        )

    def _transform_to_latlon(
        self,
        lat_prime: torch.Tensor,
        lon_prime: torch.Tensor,
        lat_p: torch.Tensor,
        lon_p: torch.Tensor,
    ) -> tuple:
        """Transform from local rotated coordinates back to standard latlon coordinates."""
        # Pre-compute trigonometric functions
        sin_lat_prime = torch.sin(lat_prime)
        cos_lat_prime = torch.cos(lat_prime)
        sin_lon_prime = torch.sin(lon_prime)
        cos_lon_prime = torch.cos(lon_prime)
        sin_lat_p = torch.sin(lat_p)
        cos_lat_p = torch.cos(lat_p)

        # Compute standard latitude
        sin_lat = sin_lat_prime * cos_lat_p + cos_lat_prime * cos_lon_prime * sin_lat_p
        lat = torch.arcsin(torch.clamp(sin_lat, -1 + 1e-7, 1 - 1e-7))

        # Compute standard longitude
        num = cos_lat_prime * sin_lon_prime
        den = cos_lat_prime * cos_lon_prime * cos_lat_p - sin_lat_prime * sin_lat_p

        lon = lon_p + torch.atan2(num, den)

        # Normalize longitude to [0, 2π]
        lon = torch.remainder(lon + 2 * torch.pi, 2 * torch.pi)

        return lat, lon

    def _interpolata_latlon(self, hidden_features,u, v, lat_grid, lon_grid, dt):
        
        batch_size = hidden_features.shape[0]
        
        # Compute departure points in a local rotated coordinate system in which the origin
        # of latitude and longitude is moved to the arrival point
        lon_prime = -u * dt
        lat_prime = -v * dt

        # Transform from rotated coordinates back to standard coordinates
        # Expand lat/lon grid for broadcasting with per-channel coordinates
        lat_grid = lat_grid.unsqueeze(1).expand(-1, self.num_vels, -1, -1)
        lon_grid = lon_grid.unsqueeze(1).expand(-1, self.num_vels, -1, -1)

        # Get the max and min values for normalization
        min_lat = torch.min(lat_grid)
        max_lat = torch.max(lat_grid)

        min_lon = torch.min(lon_grid)
        max_lon = torch.max(lon_grid)

        lat_dep, lon_dep = self._transform_to_latlon(
            lat_prime, lon_prime, lat_grid, lon_grid
        )

        # Normalize grid to ensure consistency in interpolation
        grid_x = 2 * (lon_dep - min_lon) / (max_lon - min_lon) - 1
        grid_y = 2 * (lat_dep - min_lat) / (max_lat - min_lat) - 1

        # Apply periodicity for outside values along longitude set to [-1, 1]
        grid_x = torch.remainder(grid_x + 1, 2) - 1

        # Apply geocyclic longitude roll for values beyond +/-90 degrees latitude
        geo_mask_left = grid_x <= 0
        geo_mask_right = grid_x > 0
        lat_mask_outer = torch.abs(grid_y) > 1
        grid_x = torch.where(lat_mask_outer & geo_mask_left, grid_x + 1, grid_x)
        grid_x = torch.where(lat_mask_outer & geo_mask_right, grid_x - 1, grid_x)

        # Mirror values outside of the range [-1, 1] in the latitude direction
        grid_y = torch.where(grid_y < -1, -(2 + grid_y), grid_y)
        grid_y = torch.where(grid_y > 1, 2 - grid_y, grid_y)

        # Reshape grid coordinates for interpolation
        # [batch, dynamic_channels, lat, lon] -> [batch*dynamic_channels, lat, lon]
        grid_x = grid_x.view(batch_size * self.num_vels, *grid_x.shape[-2:])
        grid_y = grid_y.view(batch_size * self.num_vels, *grid_y.shape[-2:])

        # Down-project to num_vel channels
        projected_inputs = self.down_projection(hidden_features)

        # Apply padding and reshape hidden features
        dynamic_padded = self.padding_interp(projected_inputs)

        # Make sure interpolation remains in right range after padding
        grid_x = grid_x * hidden_features.size(-1) / dynamic_padded.size(-1)
        grid_y = grid_y * hidden_features.size(-2) / dynamic_padded.size(-2)

        # Create interpolation grid
        grid = torch.stack([grid_x, grid_y], dim=-1)

        # Apply padding and reshape features
        dynamic_padded = dynamic_padded.reshape(
            batch_size * self.num_vels, 1, *dynamic_padded.shape[-2:]
        )

        # Interpolate
        interpolated = torch.nn.functional.grid_sample(
            dynamic_padded,
            grid,
            align_corners=True,
            mode=self.interpolation,
            padding_mode="border",
        )
        
        return interpolated
    
    def _interpolate_cubed_sphere(self, hidden_features, u, v, cubed_sphere, dt):
        
        # Find departure points
        xi_dep = cubed_sphere.Xi - u * dt
        eta_dep = cubed_sphere.Eta - v * dt
        
        # Move departure point to correct face
        panel_dep, xi_dep, eta_dep = cubed_sphere.remap_local(xi_dep, eta_dep)
        
        # Normalize coordinates to [-1, 1] including the padding
        # Our data is cell centered
        # With align_corner=True, the coordinate of cell center i is
        # x = 2i / (N-1) - 1 where N is the number of cell
        # We want lat[0]  at the center of cell i = pad_width     ( x_0 = 2 * pad_width / (N-1) - 1 )
        # and     lat[-1] at the center of cell i = N-1-pad_width ( x_end = 2 * (N-1-pad_width) / (N-1) - 1)
        # This can be done with x = x_0 + (lat-lat[0]) / (lat[-1]-lat[0]) * (x_end - x_0)
        def normalize(x, pad, N, x_min, x_max):
            x_0 = 2 * pad / (N - 1) - 1
            x_diff = 2 * (N-1-2*pad) / (N-1)
            return x_0 + (x - x_min) / (x_max - x_min) * x_diff
        
        pad = self.padding
        N = cubed_sphere.num_elem
        norm_xi = normalize(xi_dep, pad, N + 2*pad, -pi/4, pi/4)
        norm_eta = normalize(eta_dep, pad, N + 2*pad, -pi/4, pi/4)
        norm_panel = normalize(panel_dep, 0, 6, 0, 5)

        # Create sampling grid (x,y) -> (lat, lon)
        sampling_grid = torch.from_numpy(
            np.stack((norm_xi, norm_eta, norm_panel), axis=-1)
        ).float().reshape(1,1,-1,3)
        
        interpolated_data = torch.nn.functional.grid_sample(
            hidden_features,
            sampling_grid, 
            align_corners=True,
            mode=self.interpolation,
            padding_mode="border",
        )
        
        return interpolated_data

    def forward(
        self,
        hidden_features: torch.Tensor,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute advection using rotated coordinate system."""
        
        batch_size = hidden_features.shape[0]

        # Get learned velocities for each channel
        velocities = self.velocity_net(hidden_features)

        # Reshape velocities to separate u,v components per channel
        # [batch, 2*hidden_dim, lat, lon] -> [batch, 2, hidden_dim, lat, lon]
        velocities = velocities.view(batch_size, 2, self.num_vels, *self.mesh_size)

        # Extract learned u,v components
        u = velocities[:, 0]
        v = velocities[:, 1]

        # Interpolate hidden features
        interpolated =  self._interpolata_latlon(hidden_features,u, v, lat_grid, lon_grid, dt)

        # Reshape back to original dimensions
        interpolated = interpolated.view(batch_size, self.num_vels, *self.mesh_size)

        # Project back up to latent space
        interpolated = self.up_projection(interpolated)

        return interpolated
