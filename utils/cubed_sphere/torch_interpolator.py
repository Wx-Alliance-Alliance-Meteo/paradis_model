import torch
import torch.nn.functional as F
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from model.padding import GeoCyclicPadding

class TorchCubedSphereInterpolator:
    """
    A PyTorch-based interpolator for cubed sphere grids using torch.nn.functional.grid_sample.
    Uses GeoCyclicPadding to handle spherical topology.
    Expects input data with dimensions (..., latitude, longitude).
    """
    def __init__(self, lat, lon, cubed_sphere, mode='bilinear', device=None):
        """
        Args:
            lat (np.ndarray): Latitude array of the source grid.
            lon (np.ndarray): Longitude array of the source grid.
            cubed_sphere: An object with 'lat' and 'lon' attributes.
            mode (str): Interpolation mode, 'bilinear' or 'bicubic'.
            device (torch.device, optional): The device to store tensors on.
        """
        self.device = device
        self.mode = mode
        if self.mode == 'bicubic':
            self.pad_width = 2  # Bicubic interpolation requires a larger kernel (4x4)
        elif self.mode == 'bilinear':
            self.pad_width = 1  # Bilinear is sufficient with 1
        elif self.mode == 'nearest':
            self.pad_width = 0  # No padding for nearest
        else:
            raise ValueError(f"Unsupported interpolation mode: {self.mode}")

        self.padding_layer = GeoCyclicPadding(pad_width=self.pad_width)
        self.grid_shape = cubed_sphere.grid_shape

        # Get target grid coordinates
        lon_cs_deg = np.rad2deg(cubed_sphere.lon) % 360
        lat_cs_deg = np.rad2deg(cubed_sphere.lat)

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
        
        pad = self.pad_width
        norm_lon = normalize(lon_cs_deg, pad, lon.size + 2*pad, lon[0], lon[-1])
        norm_lat = normalize(lat_cs_deg, pad, lat.size + 2*pad, lat[0], lat[-1])

        # Create sampling grid (x,y) -> (lat, lon)
        self.sampling_grid = torch.from_numpy(
            np.stack((norm_lon, norm_lat), axis=-1)
        ).float().reshape(1,1,-1,2)
        
        if device:
            self.sampling_grid = self.sampling_grid.to(device)

    def interpolate(self, data: np.ndarray):
        """
        Interpolates data to the cubed sphere grid.

        Args:
            data (np.ndarray): Input data of shape (..., latitude, longitude).

        Returns:
            torch.Tensor: Interpolated data with shape (..., 6, num_elem, num_elem).
        """
        data = torch.from_numpy(data).float()
        if self.device:
            data = data.to(self.device)

        # Reshape for padding layer (N, C, H, W) -> (N, 1, lat, lon)
        H_in, W_in = data.shape[-2:]
        batch_dims = data.shape[:-2]
        data_reshaped_for_pad = data.view(1, -1, H_in, W_in)

        # Apply padding
        data_padded = self.padding_layer(data_reshaped_for_pad)

        # Prepare grid and perform interpolation
        interpolated_data = F.grid_sample(
            data_padded,
            self.sampling_grid,
            mode=self.mode,
            align_corners=True,
            padding_mode='border'
        )

        # Reshape output
        final_shape = batch_dims + self.grid_shape 
        return interpolated_data.view(final_shape)

