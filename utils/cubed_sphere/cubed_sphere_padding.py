import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class CubedSpherePadding(nn.Module):
    def __init__(self, cubed_sphere, pad: int, device: torch.device = None):
        super().__init__()
        self.cs = cubed_sphere
        self.pad = pad
        self.device = device if device is not None else torch.device("cpu")

        x_0, x_end = cubed_sphere.panel_domain
        dx = cubed_sphere.delta_x
        self.num_elem = cubed_sphere.num_elem
        self.num_panel = cubed_sphere.num_panel
        self.num_dir = 4 # West, East, South, North
        
        # Compute coordinates of padding
        # West
        xi = x_0 + dx * (np.arange(-pad, 0) + 0.5)
        eta = x_0 + dx * (np.arange(-pad, self.num_elem+pad) + 0.5)
        Xi_w, Eta_w = np.meshgrid(xi, eta, indexing='ij')

        # East
        xi = x_end + dx * (np.arange(0, pad) + 0.5)
        eta = x_0 + dx * (np.arange(-pad, self.num_elem+pad) + 0.5)
        Xi_e, Eta_e = np.meshgrid(xi, eta, indexing='ij')

        # South 
        xi = x_0 + dx * (np.arange(-pad, self.num_elem+pad) + 0.5)
        eta = x_0 + dx * (np.arange(-pad, 0) + 0.5)
        Xi_s, Eta_s = np.meshgrid(xi, eta, indexing='xy')

        # North 
        xi = x_0 + dx * (np.arange(-pad, self.num_elem+pad) + 0.5)
        eta = x_end + dx * (np.arange(0, pad) + 0.5)
        Xi_n, Eta_n = np.meshgrid(xi, eta, indexing='xy')

        # Xi, Eta have dims: (num_dir * pad, num_elem+2*pad)
        Xi = np.concat((Xi_w, Xi_e, Xi_s, Xi_n))
        Eta = np.concat((Eta_w, Eta_e, Eta_s, Eta_n))

        # Recompute coordinate on the correct panel
        panel_final, xi_final, eta_final = cubed_sphere.remap_local(Xi, Eta)
        
        neighbour_shape = (1, self.num_panel, 1, self.num_dir, pad,self.num_elem+2*pad)
        panel_final = panel_final.reshape(neighbour_shape)
        xi_final = xi_final.reshape(neighbour_shape)
        eta_final = eta_final.reshape(neighbour_shape)

        idx_xi = np.floor((xi_final - x_0)/ dx - 0.5).astype(int)
        idx_eta = np.floor((eta_final - x_0)/ dx - 0.5).astype(int)

        idx_xi = np.clip(idx_xi, 0, self.num_elem-2)
        idx_eta = np.clip(idx_eta, 0, self.num_elem-2)

        # Get the coordinates of the four corners of the cell
        x1 = cubed_sphere.xi[idx_xi]
        x2 = cubed_sphere.xi[idx_xi + 1]
        y1 = cubed_sphere.eta[idx_eta]
        y2 = cubed_sphere.eta[idx_eta + 1]

        # Calculate the normalized distances
        tx = (xi_final - x1) / (x2 - x1)
        ty = (eta_final - y1) / (y2 - y1)
        
        # Convert to torch tensors and register as buffers
        self.register_buffer("panel_final", torch.from_numpy(panel_final).to(device))
        self.register_buffer("idx_xi", torch.from_numpy(idx_xi).to(device))
        self.register_buffer("idx_eta", torch.from_numpy(idx_eta).to(device))
        self.register_buffer("tx", torch.from_numpy(tx).float().to(device))
        self.register_buffer("ty", torch.from_numpy(ty).float().to(device))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Change shape from [batch * panel, features, xi, eta] -> [batch, panel, features, xi, eta]
        s = x.shape
        x = x.view(-1, self.num_panel, *s[1:])
        
        n_batch = x.shape[0]
        n_features = x.shape[2]
        
        # Shape [n_batch, n_panel, n_feactures, n_dir, pad, n_elem+2*pad]
        idx_batch = torch.arange(n_batch, device=x.device).view(n_batch, 1, 1, 1, 1, 1)
        idx_features = torch.arange(n_features, device=x.device).view(1, 1, n_features, 1, 1, 1)

        q11 = x[idx_batch, self.panel_final, idx_features, self.idx_xi, self.idx_eta]        
        q12 = x[idx_batch, self.panel_final, idx_features, self.idx_xi + 1, self.idx_eta]    
        q21 = x[idx_batch, self.panel_final, idx_features, self.idx_xi, self.idx_eta + 1]    
        q22 = x[idx_batch, self.panel_final, idx_features, self.idx_xi + 1, self.idx_eta + 1]

        # Perform linear interpolation along x-direction
        R1 = torch.lerp(q11, q21, self.tx.to(x.dtype))  
        R2 = torch.lerp(q12, q22, self.tx.to(x.dtype))

        # Perform linear interpolation along y-direction
        interpolated_values = torch.lerp(R1, R2, self.ty.to(x.dtype))

        pad = self.pad
        n_with_pad = self.num_elem + 2 * pad
        
        ans = torch.empty((n_batch, self.num_panel, n_features, n_with_pad, n_with_pad), device=x.device)
        ans[:, :, :, pad:-pad, pad:-pad] = x
        ans[:, :, :, : pad, :] = interpolated_values[:, :, :, 0] # West
        ans[:, :, :, -pad:, :] = interpolated_values[:, :, :, 1] # East
        ans[:, :, :, :, : pad] = interpolated_values[:, :, :, 2].transpose(-2, -1) # South
        ans[:, :, :, :, -pad:] = interpolated_values[:, :, :, 3].transpose(-2, -1) # North
        
        return ans.view(n_batch*self.num_panel, n_features, n_with_pad, n_with_pad)
