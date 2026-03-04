"""Paradis neural architecture — BDF2 semi-Lagrangian scheme."""

import math

import torch
import torch.distributed
from torch import nn

from model.advection import NeuralSemiLagrangian
from model.blocks import GMBlock


def get_scaled_timestep(original_timestep_seconds: float) -> float:
    return original_timestep_seconds * 7.29212e-5


class Paradis(nn.Module):
    """Paradis model — BDF2 neural semi-Lagrangian architecture.

    The latent budget is split evenly between the two history states so that
    the concatenated pair [hidden_n | hidden_nm1] always has exactly
    ``latent_size`` channels. Each individual state lives in a ``latent_size // 2``
    dimensional space.  All internal modules are sized accordingly:

    * encoder output:           latent_size // 2
    * velocity_nets input:      latent_size          (concat of both states)
    * velocity_nets output:     2 * num_vels         (chunked 4-ways → num_vels // 2 each)
    * NeuralSemiLagrangian:     latent_size // 2,  num_vels // 2
    * diffusion / reaction in:  latent_size          (concat of both advected states)
    * diffusion / reaction out: latent_size // 2
    * output_proj input:        latent_size // 2

    Requires ``dataset.n_time_inputs >= 2``.

    ERA5Dataset channel layout (after permute to [B, C, H, W]):
        x[:, :D]              — dynamic features at t_{n-1}
        x[:, D:2D]            — dynamic features at t_n
        x[:, 2D:2D+F*2]       — forcings, interleaved per variable:
                                 [var1_t_{n-1}, var1_t_n, var2_t_{n-1}, var2_t_n, ...]
        x[:, 2D+F*2:]         — static / constant features

    where D = num_dyn_inputs_single, F = number of forcing variables.

    Because forcings are interleaved per-variable (not grouped by time step),
    the full forcing block is passed unchanged to both encoders.  Only the
    dynamic block differs between t_n and t_{n-1}.
    """

    def __init__(self, datamodule, cfg, lat_grid, lon_grid):
        super().__init__()

        self.nlat = lat_grid.shape[0]
        self.nlon = lat_grid.shape[1]

        self.grid = "equiangular"

        if self.grid != "equiangular":
            raise ValueError(
                f"Paradis model only supports 'equiangular' grid, got '{self.grid}'. "
                "Please set data.grid='equiangular' in your config."
            )

        if cfg.dataset.n_time_inputs < 2:
            raise ValueError("Paradis requires dataset.n_time_inputs >= 2.")

        mesh_size = (self.nlat, self.nlon)

        # latent_size is the total budget shared between the two history
        # states.  Each individual state uses half of it.
        latent_size = cfg.model.get("latent_size")
        if latent_size % 2 != 0:
            raise ValueError(
                f"model.latent_size must be even for the BDF2 budget split, got {latent_size}."
            )
        half_dim = latent_size // 2

        num_vels = cfg.model.get("velocity_vectors")
        if num_vels % 2 != 0:
            raise ValueError(
                f"model.velocity_vectors must be even for the BDF2 budget split, got {num_vels}."
            )
        half_vels = num_vels // 2

        diffusion_size = cfg.model.get("diffusion_size")
        reaction_size = cfg.model.get("reaction_size")
        adv_interpolation = cfg.model.get("adv_interpolation")
        bias_channels = cfg.model.get("bias_channels", 4)
        num_encoder_layers = cfg.model.get("num_encoder_layers", 1)

        self.velocity_num_layers = cfg.model.get("velocity_num_layers", 1)
        self.diffusion_num_layers = cfg.model.get("diffusion_num_layers", 1)
        self.reaction_num_layers = cfg.model.get("reaction_num_layers", 1)
        self.register_buffer(
            "dt", torch.tensor(get_scaled_timestep(cfg.model.get("base_dt")))
        )

        # Store split dimensions for use in forward().
        self.half_dim = half_dim
        self.half_vels = half_vels

        # Forcings are interleaved per-variable (2 channels each), NOT
        # grouped as [all_forc_t_{n-1} | all_forc_t_n].  The only clean
        # split boundary is between the dynamic block and the forcing block.
        # We pass the full forcing block and static block identically to both
        # encoders; only the dynamic features differ between t_n and t_{n-1}.
        # TODO : this is not optimal
        self.num_dyn_single = datamodule.dataset.num_dyn_inputs_single
        # Total forcing channels = num_forcing_vars * n_time_inputs
        num_forcing_total = len(cfg.features.input.forcings) * cfg.dataset.n_time_inputs
        self.num_forcing_total = num_forcing_total
        num_static = datamodule.dataset.num_in_static_features

        # Each encoder call receives: dyn (one step) + all forcings + static
        encoder_input_dim = self.num_dyn_single + num_forcing_total + num_static

        self.activation_function = nn.SiLU

        current_dim = encoder_input_dim
        encoder_layers = []
        for _ in range(num_encoder_layers - 1):
            fc = nn.Conv2d(current_dim, half_dim, 1, bias=True)  # output is half_dim
            scale = math.sqrt(2.0 / current_dim)
            nn.init.normal_(fc.weight, mean=0.0, std=scale)
            nn.init.constant_(fc.bias, 0.0)
            encoder_layers.append(fc)
            encoder_layers.append(self.activation_function())
            current_dim = half_dim

        fc = nn.Conv2d(current_dim, half_dim, 1, bias=False)
        scale = math.sqrt(1.0 / current_dim)
        nn.init.normal_(fc.weight, mean=0.0, std=scale)
        encoder_layers.append(fc)

        self.input_proj = nn.Sequential(*encoder_layers)

        self.velocity_nets = GMBlock(
            layers=["SepConv" for _ in range(self.velocity_num_layers)],
            input_dim=latent_size,
            output_dim=4 * half_vels,
            hidden_dim=latent_size,
            kernel_size=3,
            mesh_size=mesh_size,
            bias_channels=bias_channels,
            pre_normalize=True,
        )

        self.advection = NeuralSemiLagrangian(
            half_dim,
            mesh_size,
            num_vels=half_vels,
            lat_grid=lat_grid,
            lon_grid=lon_grid,
            interpolation=adv_interpolation,
            project_advection=cfg.model.get("projected_advection", True),
        )

        self.diffusion = GMBlock(
            layers=["SepConv" for _ in range(self.diffusion_num_layers)],
            input_dim=latent_size,
            output_dim=half_dim,
            hidden_dim=diffusion_size,
            mesh_size=mesh_size,
            pre_normalize=True,
            bias_channels=bias_channels,
        )

        self.reaction = GMBlock(
            layers=["CLinear" for _ in range(self.reaction_num_layers)],
            input_dim=latent_size,
            output_dim=half_dim,
            hidden_dim=reaction_size,
            kernel_size=1,
            mesh_size=mesh_size,
            pre_normalize=True,
            bias_channels=bias_channels,
        )

        self.output_proj = GMBlock(
            layers=["SepConv", "CLinear"],
            input_dim=half_dim,
            output_dim=datamodule.num_out_features,
            hidden_dim=half_dim,
            mesh_size=mesh_size,
            kernel_size=3,
            activation=False,
            bias_channels=bias_channels,
        )

    def _scale_final_layer(self, scaling_factor: float) -> None:
        """
        Multiply all learnable parameters of the output_proj's final
        Conv2d in-place by ``scaling_factor``.
        """
        last_conv = None
        for m in self.output_proj.modules():
            if isinstance(m, nn.Conv2d):
                last_conv = m
        if last_conv is None:
            return
        with torch.no_grad():
            last_conv.weight.mul_(scaling_factor)
            if last_conv.bias is not None:
                last_conv.bias.mul_(scaling_factor)

    @torch.no_grad()
    def calibrate_output_scaling(
        self,
        sample_input: torch.Tensor,
        sample_target: torch.Tensor,
        process_group=None,
    ) -> float:
        """
        Estimate and apply a scaling factor to the output projection so
        that the predicted increment magnitude matches the target increment
        magnitude at initialisation.
        """
        import logging as _logging

        was_training = self.training
        self.eval()

        pred = self(sample_input)

        # Recover x_dyn_n using the same slice as forward().
        D = self.num_dyn_single
        x_dyn_n = sample_input[:, D : 2 * D]

        # Both sides are increments: output_proj predicts (target - x_dyn_n).
        pred_increment = pred - x_dyn_n
        target_increment = sample_target - x_dyn_n

        # Global sum-of-squares — all-reduce before sqrt so the norm is
        # computed over the full distributed batch, not per-rank.
        pred_ss = (pred_increment**2).sum()
        target_ss = (target_increment**2).sum()

        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                pred_ss, op=torch.distributed.ReduceOp.SUM, group=process_group
            )
            torch.distributed.all_reduce(
                target_ss, op=torch.distributed.ReduceOp.SUM, group=process_group
            )

        eps = 1e-8
        norm_pred = pred_ss.sqrt()
        norm_target = target_ss.sqrt()

        # s * norm(pred_increment) = norm(target_increment)
        scaling_factor = float(norm_target / norm_pred.clamp_min(eps))

        self._scale_final_layer(scaling_factor)

        # Trace log on rank 0 only.
        is_rank_zero = (
            not torch.distributed.is_available()
            or not torch.distributed.is_initialized()
            or torch.distributed.get_rank() == 0
        )
        if is_rank_zero:
            norm_after = float(norm_pred) * scaling_factor
            _logging.info(
                f"Output scaling calibration: "
                f"norm(pred_increment)={float(norm_pred):.4e}, "
                f"norm(target_increment)={float(norm_target):.4e}, "
                f"scaling_factor={scaling_factor:.4f}, "
                f"norm(pred_increment) after scaling={norm_after:.4e}"
            )

        if was_training:
            self.train()

        return scaling_factor

    def forward(self, fields: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        fields : torch.Tensor
            Input fields of shape (batch, in_channels, nlat, nlon).

        Returns
        -------
        torch.Tensor
            Output fields of shape (batch, out_channels, nlat, nlon).
        """
        x = fields
        D = self.num_dyn_single
        F = self.num_forcing_total

        # Split the dynamic block into t_{n-1} and t_n.
        x_dyn_nm1 = x[:, :D]
        x_dyn_n = x[:, D : 2 * D]
        # TODO : For the time being, forcings (interleaved per-variable) and static are passed to both
        # encoders unchanged for simplicity
        x_forcing_static = x[:, 2 * D :]

        x_n = torch.cat([x_dyn_n, x_forcing_static], dim=1)
        x_nm1 = torch.cat([x_dyn_nm1, x_forcing_static], dim=1)

        # Encode both time steps with the shared encoder → half_dim each.
        hidden = self.input_proj(x_n)
        hidden_prev = self.input_proj(x_nm1)

        # PathNet sees the full latent_size budget (both states concatenated).
        vel_out = self.velocity_nets(torch.cat([hidden, hidden_prev], dim=1))

        # Chunk into four half_vels components: u_vel, v_vel, u_acc, v_acc.
        u_vel, v_vel, u_acc, v_acc = vel_out.chunk(4, dim=1)

        # Advection operates on half_dim states with half_vels velocity channels.
        z_tilde_n, z_tilde_nm1 = self.advection(
            hidden, hidden_prev, u_vel, v_vel, u_acc, v_acc
        )

        # Physics networks see the full latent_size budget (both advected states).
        combined = torch.cat([z_tilde_n, z_tilde_nm1], dim=1)
        diff_term = self.diffusion(combined)
        reac_term = self.reaction(combined)

        # BDF2 combination:
        # z^{n+1} = (4/3)*z_tilde_n - (1/3)*z_tilde_nm1
        #         + (2/3)*dt*(Net_diff + Net_reac)
        hidden = (
            (4.0 / 3.0) * z_tilde_n
            - (1.0 / 3.0) * z_tilde_nm1
            + (2.0 / 3.0) * self.dt * (diff_term + reac_term)
        )

        return x_dyn_n + self.output_proj(hidden)
