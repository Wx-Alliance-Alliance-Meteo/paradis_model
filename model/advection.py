import torch
import torch.nn.functional as F
from torch import nn

from model.blocks import GMBlock
from model.padding import GeoCyclicPadding


class NeuralEulerianUpwind(nn.Module):
    """Eulerian advection operator using 5th-order upwind finite differences."""

    def __init__(
        self,
        hidden_dim: int,
        mesh_size: tuple,
        num_vels: int,
        lat_grid: torch.Tensor,
        lon_grid: torch.Tensor,
        project_advection=True,
        epi_tolerance=1e-5,
    ):
        super().__init__()

        self.padding = 3
        self.padding_layer = GeoCyclicPadding(self.padding)
        self.hidden_dim = hidden_dim
        self.num_vels = num_vels
        self.mesh_size = mesh_size
        self.epi_tolerance = epi_tolerance

        if project_advection:
            self.down_projection = GMBlock(
                layers=["CLinear"],
                input_dim=hidden_dim,
                output_dim=num_vels,
                mesh_size=mesh_size,
                kernel_size=1,
            )

            self.up_projection = GMBlock(
                layers=["SepConv"],
                input_dim=num_vels,
                output_dim=hidden_dim,
                mesh_size=mesh_size,
                kernel_size=1,
            )
        else:
            self.num_vels = hidden_dim
            self.down_projection = nn.Identity()
            self.up_projection = nn.Identity()

        H, W = mesh_size

        self.register_buffer(
            "lat_grid", lat_grid.unsqueeze(0).unsqueeze(0).contiguous().clone()
        )
        self.register_buffer(
            "lon_grid", lon_grid.unsqueeze(0).unsqueeze(0).contiguous().clone()
        )

        dlat = torch.diff(lat_grid[:, 0])[0]
        dlon = torch.diff(lon_grid[0, :])[0]
        
        self.register_buffer("dlat", dlat)
        self.register_buffer("dlon", dlon)

        # Positive velocity: stencil [i-3, i-2, i-1, i, i+1, i+2]
        self.register_buffer(
            "coeff_pos",
            torch.tensor([-2.0, 15.0, -60.0, 20.0, 30.0, -3.0], dtype=lat_grid.dtype) / 60.0
        )
        # Negative velocity: stencil [i-2, i-1, i, i+1, i+2, i+3]
        self.register_buffer(
            "coeff_neg",
            torch.tensor([3.0, -30.0, -20.0, 60.0, -15.0, 2.0], dtype=lat_grid.dtype) / 60.0
        )

    def _compute_derivative_vectorized(self, q_padded, velocity, dx, dim):
        """Compute 5th-order upwind derivative."""
        B, C, H, W = velocity.shape
        q_reshaped = q_padded.reshape(B * C, 1, q_padded.shape[2], q_padded.shape[3])
        
        if dim == 3:  # Longitude direction
            k_pos = self.coeff_pos.view(1, 1, 1, 6)
            k_neg = self.coeff_neg.view(1, 1, 1, 6)
            
            # Apply convolution - output will have shape (B*C, 1, H_pad, W_pad-5)
            d_pos_all = F.conv2d(q_reshaped, k_pos)
            d_neg_all = F.conv2d(q_reshaped, k_neg)
            
            # Extract the correct columns
            # For positive: stencil starts at padded_i-3, so output column k corresponds to original column k
            # For negative: stencil starts at padded_i-2, so output column k corresponds to original column k-1
            d_pos = d_pos_all[:, :, self.padding:-self.padding, :][:, :, :, :W]
            d_neg = d_neg_all[:, :, self.padding:-self.padding, :][:, :, :, 1:W+1]
            
            d_pos = d_pos.reshape(B, C, H, W)
            d_neg = d_neg.reshape(B, C, H, W)
            
            mask_pos = (velocity > 0).float()
            dq_dx = (mask_pos * d_pos + (1 - mask_pos) * d_neg) / dx

        else:  # Latitude direction
            k_pos = self.coeff_pos.view(1, 1, 6, 1)
            k_neg = self.coeff_neg.view(1, 1, 6, 1)
            
            d_pos_all = F.conv2d(q_reshaped, k_pos)
            d_neg_all = F.conv2d(q_reshaped, k_neg)
            
            # Extract the correct rows
            d_pos = d_pos_all[:, :, :, self.padding:-self.padding][:, :, :H, :]
            d_neg = d_neg_all[:, :, :, self.padding:-self.padding][:, :, 1:H+1, :]
            
            d_pos = d_pos.reshape(B, C, H, W)
            d_neg = d_neg.reshape(B, C, H, W)
            
            mask_pos = (velocity > 0).float()
            dq_dx = (mask_pos * d_pos + (1 - mask_pos) * d_neg) / dx
            
        return dq_dx

    def _compute_rhs(self, q, u, v):
        """Compute RHS: -u * dq/dlambda - v * dq/dtheta"""
        q_padded = self.padding_layer(q)
        
        dq_dlon = self._compute_derivative_vectorized(q_padded, u, self.dlon, dim=3)
        dq_dlat = self._compute_derivative_vectorized(q_padded, v, self.dlat, dim=2)
        
        rhs = -u * dq_dlon - v * dq_dlat
        
        return rhs

    def _phi1_times_vector(self, dt, jacobian_fn, vector, q):
        """
        Compute phi_1(dt * J) * vector using Taylor series.
        phi_1(z) = (exp(z) - 1) / z = 1 + z/2! + z^2/3! + z^3/4! + ...
        """
        result = vector.clone()
        term = vector.clone()
        
        for k in range(1, 20): # TODO : mettre dans la config
            J_term = jacobian_fn(term, q)
            term = J_term * (dt / (k + 1))
            
            term_norm = torch.max(torch.abs(term))
            if term_norm < self.epi_tolerance:
                break
                
            result = result + term
        
        return result

    def forward(
        self,
        hidden_features: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
        dt: float,
    ) -> torch.Tensor:
        """Compute advection using EPI2 time integration."""
        
        projected_inputs = self.down_projection(hidden_features)
        
        F_n = self._compute_rhs(projected_inputs, u, v)
        
        def jacobian_action(vec, state):
            return self._compute_rhs(vec, u, v)
        
        phi1_F = self._phi1_times_vector(dt, jacobian_action, F_n, projected_inputs)
        
        updated = projected_inputs + phi1_F * dt
        
        result = self.up_projection(updated)
        
        return result
