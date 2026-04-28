"""Model training implementation."""

import datetime
import logging
import re
import time
from collections import defaultdict, OrderedDict

import lightning as L
from lightning.pytorch.utilities import rank_zero_only
import omegaconf.dictconfig
import torch
import torch.nn as nn

from data.datamodule import Era5DataModule
from model.paradis import Paradis
from utils.loss import ParadisLoss
from utils.normalization import denormalize_humidity, denormalize_precipitation
from utils.file_output import ZarrForecastWriter

# Configure torch.compile to handle dynamic shapes in Dion optimizers
torch._dynamo.config.cache_size_limit = 64
torch._dynamo.config.force_parameter_static_shapes = False


def build_param_groups(model, lr, weight_decay, optimizer_name):
    muon_params = []
    adamw_params = []
    seen = set()

    for module_name, module in model.named_modules():
        # Muon: standard hidden-layer weights
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            if (
                getattr(module, "weight", None) is not None
                and module.weight.requires_grad
            ):
                muon_params.append(module.weight)
                seen.add(id(module.weight))

            if getattr(module, "bias", None) is not None and module.bias.requires_grad:
                adamw_params.append(module.bias)
                seen.add(id(module.bias))

    # Everything else defaults to AdamW
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        adamw_params.append(p)
        seen.add(id(p))

    return [
        dict(
            params=muon_params,
            algorithm=optimizer_name,
            lr=lr,
            weight_decay=weight_decay,
            flatten=True,
        ),
        dict(
            params=adamw_params,
            algorithm="adamw",
            lr=lr,
            weight_decay=weight_decay,
        ),
    ]


def _strip_orig_mod_prefix(state_dict: dict[str, torch.Tensor]) -> OrderedDict:
    # Ensure compatibility between checkpoints that were or were not compiled
    fixed = OrderedDict()
    for k, v in state_dict.items():
        new_k = k.replace("._orig_mod.", ".")

        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod.") :]
        fixed[new_k] = v
    return fixed


class LitParadis(L.LightningModule):
    """Lightning module for Paradis model training."""

    model: torch.nn.Module

    def __init__(
        self, datamodule: Era5DataModule, cfg: omegaconf.dictconfig.DictConfig
    ) -> None:
        """Initialize the training module.

        Args:
            datamodule: Lightning datamodule containing dataset information
            cfg: Model configuration dictionary
        """
        super().__init__()

        # Instantiate the model
        self.min_dt = 1e10
        self.datamodule = datamodule
        lat_grid = datamodule.dataset.lat_rad_grid
        lon_grid = datamodule.dataset.lon_rad_grid
        self.model = Paradis(datamodule, cfg, lat_grid, lon_grid)
        self.cfg = cfg
        self.n_inputs = cfg.dataset.n_time_inputs

        # Log metrics
        num_parameters = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        if self.global_rank == 0:
            logging.info("Number of trainable parameters: {:,}".format(num_parameters))

        # Access output_name_order from configuration
        self.output_name_order = datamodule.output_name_order

        num_levels = len(cfg.features.pressure_levels)

        # Construct variable loss weight tensor from YAML configuration
        atmospheric_weights = torch.tensor(
            [
                cfg.training.variable_loss_weights.atmospheric[var]
                for var in cfg.features.output.atmospheric
            ],
            dtype=torch.float32,
        )

        surface_weights = torch.tensor(
            [
                cfg.training.variable_loss_weights.surface[var]
                for var in cfg.features.output.surface
            ],
            dtype=torch.float32,
        )

        # Create a mapping of variable names to their weights
        atmospheric_vars = cfg.features.output.atmospheric
        surface_vars = cfg.features.output.surface
        var_name_to_weight = {
            **{var: atmospheric_weights[i] for i, var in enumerate(atmospheric_vars)},
            **{var: surface_weights[i] for i, var in enumerate(surface_vars)},
        }

        # Initialize reordered weights tensor
        num_features = len(atmospheric_weights) * num_levels + len(surface_weights)
        var_loss_weights_reordered = torch.zeros(num_features, dtype=torch.float32)

        # Reorder based on self.output_name_order
        for i, var in enumerate(self.output_name_order):
            # Get the variable name without the level
            var_name = re.sub(r"_h\d+$", "", var)
            if var_name in var_name_to_weight:
                var_loss_weights_reordered[i] = var_name_to_weight[var_name]

        # Initialize loss function with delta schedule parameters
        if not cfg.forecast.enable:
            self.loss_fn = ParadisLoss(
                loss_function=cfg.training.loss_function.type,
                lat_grid=datamodule.lat,
                pressure_levels=torch.tensor(
                    cfg.features.pressure_levels, dtype=torch.float32
                ),
                num_features=datamodule.num_out_features,
                num_surface_vars=len(cfg.features.output.surface),
                var_loss_weights=var_loss_weights_reordered,
                output_name_order=datamodule.output_name_order,
                delta_loss=cfg.training.loss_function.delta_loss,
                apply_latitude_weights=cfg.training.loss_function.lat_weights,
            )

            # Possibly use a different loss for validation
            validation_loss_type = cfg.training.loss_function.get(
                "validation_loss", None
            )
            if validation_loss_type is not None:

                self.val_loss_fn = ParadisLoss(
                    loss_function=validation_loss_type,
                    lat_grid=datamodule.lat,
                    pressure_levels=torch.tensor(
                        cfg.features.pressure_levels, dtype=torch.float32
                    ),
                    num_features=datamodule.num_out_features,
                    num_surface_vars=len(cfg.features.output.surface),
                    var_loss_weights=var_loss_weights_reordered,
                    output_name_order=datamodule.output_name_order,
                    delta_loss=cfg.training.loss_function.delta_loss,
                    apply_latitude_weights=cfg.training.loss_function.lat_weights,
                )

            else:
                self.val_loss_fn = self.loss_fn

        self.num_common_features = datamodule.num_common_features
        self.print_losses = cfg.training.print_losses

        # Load weights only but reset lightning configuration
        if (cfg.init.checkpoint_path and not cfg.init.restart) or cfg.forecast.enable:
            # Load into CPU, then Lightning will transfer to GPU
            checkpoint = torch.load(
                cfg.init.checkpoint_path, weights_only=False, map_location="cpu"
            )

            sd = checkpoint["state_dict"]

            # Make sure model can read checkpoint whether it has been previously compiled or not
            sd = _strip_orig_mod_prefix(sd)

            # Interpolate GlobalBias parameters for resolution change
            # We look for any keys ending in .U or .V belonging to a GlobalBias module
            for k in list(sd.keys()):
                if k.endswith(".U") or k.endswith(".V"):
                    old_param = sd[k]  # Shape: [rank, old_size]

                    # Determine target size from the current model's parameters
                    # self.state_dict() reflects the NEW 0.25 degree initialization
                    target_size = self.state_dict()[k].shape[-1]

                    if old_param.shape[-1] != target_size:
                        # unsqueeze to [1, rank, old_size] for F.interpolate
                        # then interpolate to [1, rank, target_size]
                        new_param = torch.nn.functional.interpolate(
                            old_param.unsqueeze(0),
                            size=target_size,
                            mode="linear",
                            align_corners=True,
                        ).squeeze(0)

                        sd[k] = new_param
                        print(
                            f"Interpolated {k}: {old_param.shape[-1]} -> {target_size}"
                        )

            a, b = self.load_state_dict(sd, strict=True)
        # Compile model in place
        if cfg.compute.compile:
            self.model.compile(
                mode="default",
                fullgraph=True,
                dynamic=False,
                backend="inductor",
            )
        else:
            # If full model cannot be compiled, at least compile heavy functions
            self.model._layer_step = torch.compile(self.model._layer_step)
            self.model.static_encoder = torch.compile(self.model.static_encoder)
            self.model.input_proj = torch.compile(self.model.input_proj)
            self.model.output_proj = torch.compile(self.model.output_proj)

        self.epoch_start_time = None

        # Store the index and stats of the report quantities
        if not cfg.forecast.enable and cfg.training.reports.enable:
            self.report_features = cfg.training.reports.features
            self.report_ind = [
                datamodule.dataset.dyn_input_features.index(feature)
                for feature in cfg.training.reports.features
            ]
            self.report_ind = torch.tensor(self.report_ind, dtype=torch.long)
            self.report_mean = torch.from_numpy(datamodule.dataset.report_stats["mean"])
            self.report_std = torch.from_numpy(datamodule.dataset.report_stats["std"])

        self.custom_norms = not cfg.normalization.standard
        self.log_statistics = cfg.training.get("log_additional_stats", False)

        if cfg.forecast.enable:
            self.forecast_writer = ZarrForecastWriter(cfg, datamodule.dataset)

    def _get_report_rmse(self, output_data, pred_data):

        lat_weights = self.loss_fn.lat_weights.view(1, 1, -1, 1).to(output_data.device)

        # Compute the batch error
        errors = torch.empty(
            len(self.report_ind), dtype=output_data.dtype, device=output_data.device
        )
        for i, ind in enumerate(self.report_ind):
            if self.custom_norms and "specific_humidity" in self.report_features[i]:
                q_min = self.datamodule.dataset.q_min
                q_max = self.datamodule.dataset.q_max
                o_data = denormalize_humidity(output_data[:, ind], q_min, q_max)
                p_data = denormalize_humidity(pred_data[:, ind], q_min, q_max)
                errors[i] = torch.mean((o_data - p_data) ** 2 * lat_weights)
            elif self.custom_norms and "precipitation" in self.report_features[i]:
                o_data = denormalize_precipitation(output_data[:, ind])
                p_data = denormalize_precipitation(pred_data[:, ind])
                errors[i] = torch.mean((o_data - p_data) ** 2 * lat_weights)
            else:
                errors[i] = torch.mean(
                    ((output_data[:, ind] - pred_data[:, ind]) * self.report_std[i])
                    ** 2
                    * lat_weights
                )

        return torch.sqrt(errors).detach()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the model."""
        return self.model(x)

    def configure_optimizers(self):  # type: ignore
        """Configure optimizer and learning rate scheduler."""
        cfg = self.cfg.training

        if cfg.optimizer.name == "adamw":

            if cfg.optimizer.get("custom_adamw", True):
                param_groups = build_adamw_param_groups(
                    self.model,
                    lr=cfg.optimizer.lr,
                    weight_decay=cfg.optimizer.weight_decay,
                )
            else:
                param_groups = self.model.parameters()

            optimizer = torch.optim.AdamW(
                param_groups,
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            )

        elif "muon" in cfg.optimizer.name:
            from dion import Dion2, Muon, NorMuon, Dion

            param_groups = build_param_groups(
                self.model,
                lr=cfg.optimizer.lr,
                weight_decay=cfg.optimizer.weight_decay,
                optimizer_name=cfg.optimizer.name,
            )

            if cfg.optimizer.name == "muon":
                optimizer = Muon(
                    param_groups,
                    lr=cfg.optimizer.lr,
                    weight_decay=cfg.optimizer.weight_decay,
                    betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
                    use_triton=True,
                )

            elif cfg.optimizer.name == "normuon":
                optimizer = NorMuon(
                    param_groups,
                    lr=cfg.optimizer.lr,
                    weight_decay=cfg.optimizer.weight_decay,
                    betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
                    use_triton=True,
                )

        enabled_schedulers = sum(
            [
                cfg.scheduler.one_cycle.enabled,
                cfg.scheduler.reduce_lr.enabled,
                cfg.scheduler.wsd.enabled,
            ]
        )

        # Ensure only one is enabled
        if enabled_schedulers != 1:
            raise ValueError(
                f"Invalid config: Exactly one scheduler must "
                + f"be enabled, but found {enabled_schedulers} enabled."
            )

        if cfg.scheduler.one_cycle.enabled:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                total_steps=int(self.trainer.estimated_stepping_batches),
                max_lr=cfg.optimizer.lr,
                pct_start=cfg.scheduler.one_cycle.warmup_pct_start,
                div_factor=cfg.scheduler.one_cycle.lr_div_factor,
                final_div_factor=cfg.scheduler.one_cycle.lr_final_div,
                anneal_strategy="cos",
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        elif cfg.scheduler.reduce_lr.enabled:

            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=cfg.scheduler.reduce_lr.factor,
                patience=cfg.scheduler.reduce_lr.patience,
                threshold=cfg.scheduler.reduce_lr.threshold,
                threshold_mode=cfg.scheduler.reduce_lr.threshold_mode,
                min_lr=cfg.scheduler.reduce_lr.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss_epoch",  # Monitor epoch-level validation loss
                    "interval": "epoch",  # When the scheduler should make decisions
                    "frequency": 1,
                },
            }
        elif cfg.scheduler.wsd.enabled:
            total_steps = self.trainer.estimated_stepping_batches

            # Set warmup and decay periods
            if cfg.scheduler.wsd.warmup >= 1:
                # Value >= 1, so it's a number of steps
                warmup_steps = cfg.scheduler.wsd.warmup
            else:
                warmup_steps = cfg.scheduler.wsd.warmup * total_steps

            if cfg.scheduler.wsd.decay >= 1:
                # Value >= 1, so it's a number of steps
                decay_steps = cfg.scheduler.wsd.decay
            else:
                decay_steps = cfg.scheduler.wsd.decay * total_steps

            # Sanity checks
            assert warmup_steps >= 0
            assert decay_steps >= 0
            assert warmup_steps + decay_steps <= total_steps

            steady_steps = total_steps - (warmup_steps + decay_steps)

            def lr_lambda(step):
                if step < warmup_steps:
                    # Increasing learning rate phase
                    return (step + 1) / warmup_steps
                elif step <= warmup_steps + steady_steps:
                    # Constant learning rate
                    return 1.0
                else:
                    # Decay learning rate
                    decay_ratio = (total_steps - step) / decay_steps
                    return decay_ratio  # Linear decay

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }
        else:
            # No known scheduler was active
            active_schedulers = [
                k for (k, v) in cfg.scheduler.items() if "enabled" in v and v["enabled"]
            ]
            if len(active_schedulers) == 0:
                # Should not happen if enabled_schedulers check above is still present
                raise ValueError(f"No scheduler activated")
            else:
                raise ValueError(
                    f'Unknown schedule activated: {", ".join(active_schedulers)}'
                )

    @rank_zero_only
    def on_fit_start(self):
        total = sum(p.numel() for p in self.parameters())

        if self.logger and hasattr(self.logger, "experiment"):
            tb = self.logger.experiment
            tb.add_scalar("model/num_parameters", total, global_step=0)

    def on_predict_start(self):
        if not self.cfg.forecast.enable:
            return

        if self.trainer.is_global_zero and not self.forecast_writer.store_initialized:
            self.forecast_writer.init_store(self.datamodule.dataset)

        self.trainer.strategy.barrier()

    def on_train_epoch_start(self):
        """Record the start time of the epoch."""
        if self.print_losses:
            self.epoch_start_time = time.time()

    def _autoregression_next_input(
        self,
        input_data: torch.Tensor,
        output_data: torch.Tensor,
        step: int,
        num_steps: int,
    ) -> torch.Tensor:
        """Update autoregressive channel stack."""
        common = output_data[:, : self.num_common_features]

        if self.n_inputs == 1:
            return common

        lag_channels = self.num_common_features * self.n_inputs

        return torch.cat(
            [
                input_data[:, self.num_common_features : lag_channels],
                common,
            ],
            dim=1,
        )

    def training_step(self, batch, batch_idx):
        input_data, true_data, forcings, constant_data = batch

        constants = constant_data[:, :1].permute(0, 1, 4, 2, 3)
        forcings = forcings.permute(0, 1, 4, 2, 3)

        train_loss = 0.0
        train_loss_reference = 0.0
        num_steps = true_data.size(1)

        for step in range(num_steps):
            forcings_step = forcings[:, step].unsqueeze(1)

            input_data = torch.cat(
                [input_data, forcings_step, constants], dim=2
            ).squeeze(1)

            output_data = self(input_data)

            loss = self.loss_fn(output_data, true_data[:, step])
            train_loss += loss

            input_data = self._autoregression_next_input(
                input_data, output_data, step, num_steps
            ).unsqueeze(1)

        train_loss = train_loss / num_steps

        self.log(
            "train_loss",
            train_loss,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True)

        self.log(
            "forecast_steps",
            num_steps,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

        return train_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""

        input_data, true_data, forcings, constant_data = batch
        constants = constant_data[:, :1].permute(0, 1, 4, 2, 3)
        forcings = forcings.permute(0, 1, 4, 2, 3)

        val_loss = 0.0
        report_loss = 0.0
        num_steps = true_data.size(1)

        for step in range(num_steps):
            forcings_step = forcings[:, step].unsqueeze(1)

            input_data = torch.cat(
                [input_data, forcings_step, constants], dim=2
            ).squeeze(1)

            # Forward pass
            output_data = self(input_data)

            loss = self.val_loss_fn(output_data, true_data[:, step])
            # Log requested scaled RMSE losses for validation
            report_loss += self._get_report_rmse(output_data, true_data[:, step])

            # Compute loss (data is already transformed by dataset)
            val_loss += loss

            input_data = self._autoregression_next_input(
                input_data, output_data, step, num_steps
            ).unsqueeze(1)

        batch_loss = val_loss / num_steps

        self.log(
            "val_loss",
            batch_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        # Log requested reports
        for i, name in enumerate(self.cfg.training.reports.features):
            self.log(
                name,
                report_loss[i] / num_steps,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        return batch_loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        sample_indices, input_data, forcings, constant_data = batch

        dataset = self.datamodule.dataset
        atmospheric_vars = self.cfg.features.output.atmospheric
        surface_vars = self.cfg.features.output.surface
        pressure_levels = self.cfg.features.pressure_levels

        num_levels = len(pressure_levels)
        num_atm_features = len(atmospheric_vars) * num_levels
        num_sur_features = len(surface_vars)
        num_features = num_atm_features + num_sur_features

        num_forecast_steps = self.cfg.model.forecast_steps
        output_frequency = self.cfg.forecast.output_frequency
        write_every_n = self.cfg.forecast.get("write_every_n", num_forecast_steps)

        from utils.postprocessing import (
            denormalize_datasets,
            convert_cartesian_to_spherical_winds,
        )

        output_features = list(dataset.dyn_output_features)
        constants = constant_data[:, :1].permute(0, 1, 4, 2, 3)

        chunk_buffer = []
        chunk_start_idx = None
        stored_step_idx = 0

        for step in range(num_forecast_steps):
            forcings_step = forcings[:, step].unsqueeze(1).permute(0, 1, 4, 2, 3)

            model_input = torch.cat(
                [input_data, forcings_step, constants], dim=2
            ).squeeze(1)
            output_data = self(model_input)

            input_data = self._autoregression_next_input(
                model_input, output_data, step, num_forecast_steps
            ).unsqueeze(1)

            if step % output_frequency == 0:
                if chunk_start_idx is None:
                    chunk_start_idx = stored_step_idx

                chunk_buffer.append(output_data.detach())
                stored_step_idx += 1

                if len(chunk_buffer) == write_every_n:
                    chunk_tensor = torch.stack(chunk_buffer, dim=1).cpu()
                    denormalize_datasets(None, chunk_tensor, dataset)

                    chunk_np = chunk_tensor.numpy()
                    convert_cartesian_to_spherical_winds(
                        dataset.lat, dataset.lon, self.cfg, chunk_np, output_features
                    )

                    self.forecast_writer.write_forecast_chunk(
                        forecast=chunk_np,
                        sample_indices=(
                            None
                            if sample_indices is None
                            else sample_indices.cpu().numpy()
                        ),
                        start_idx=chunk_start_idx,
                        dataset=dataset,
                    )

                    del chunk_tensor, chunk_np
                    chunk_buffer.clear()
                    chunk_start_idx = None

            del output_data

        if chunk_buffer:
            chunk_tensor = torch.stack(chunk_buffer, dim=1).cpu()
            denormalize_datasets(None, chunk_tensor, dataset)

            chunk_np = chunk_tensor.numpy()
            convert_cartesian_to_spherical_winds(
                dataset.lat, dataset.lon, self.cfg, chunk_np, output_features
            )

            self.forecast_writer.write_forecast_chunk(
                forecast=chunk_np,
                sample_indices=(
                    None if sample_indices is None else sample_indices.cpu().numpy()
                ),
                start_idx=chunk_start_idx,
                dataset=dataset,
            )

        return None

    def on_train_epoch_end(self):
        """Log epoch time and metrics if printing losses."""
        if self.print_losses and self.epoch_start_time is not None:
            elapsed_time = time.time() - self.epoch_start_time
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]

            # Get the losses using the logged metrics
            train_loss = self.trainer.callback_metrics.get("train_loss")
            val_loss = self.trainer.callback_metrics.get("val_loss")

            if (
                self.trainer.is_global_zero
                and train_loss is not None
                and val_loss is not None
            ):
                print(
                    f"Epoch {self.current_epoch:4d} | "
                    f"Train Loss: {train_loss.item():.6f} | "
                    f"Val Loss: {val_loss.item():.6f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Elapsed time: {elapsed_time:.4f}s"
                )

    def on_train_end(self):
        """Called when training ends."""
        logging.info(f"Training completed after {self.current_epoch + 1} epochs")

    def on_before_optimizer_step(self, optimizer):

        if self.log_statistics:
            grad_sq = defaultdict(lambda: torch.zeros((), device=self.device))
            param_sq = defaultdict(lambda: torch.zeros((), device=self.device))
            momentum_sq = defaultdict(lambda: torch.zeros((), device=self.device))
            dot_product_total = defaultdict(lambda: torch.zeros((), device=self.device))

            for name, p in self.named_parameters():
                if p is None or p.data is None:
                    continue
                key = name.split(".")[1]

                # param norm (for grad/param ratio)
                param_sq[key] = param_sq[key] + (p.detach().float() ** 2).sum()

                # grad norm
                if p.grad is not None:
                    g = p.grad.detach()
                    if g.dtype != torch.float32:
                        g = g.float()
                    grad_sq[key] = grad_sq[key] + (g**2).sum()

                    # Compute grad-momentum alignment (cosine similarity)
                    if p in optimizer.state and "exp_avg" in optimizer.state[p]:
                        m = optimizer.state[p]["exp_avg"].detach()
                        if m.dtype != torch.float32:
                            m = m.float()

                        # Accumulate for cosine similarity computation
                        dot_product_total[key] = dot_product_total[key] + (g * m).sum()
                        momentum_sq[key] = momentum_sq[key] + (m**2).sum()

            total_grad = (
                torch.stack(
                    list(grad_sq.values()) or [torch.zeros((), device=self.device)]
                )
                .sum()
                .sqrt()
            )

            metrics = {"grad/total": total_grad}
            eps = 1e-12
            total_dot = torch.zeros((), device=self.device)
            total_grad_sq = torch.zeros((), device=self.device)
            total_momentum_sq = torch.zeros((), device=self.device)

            for k in sorted(grad_sq.keys()):
                gnorm = grad_sq[k].sqrt()
                pnorm = param_sq[k].sqrt().clamp_min(eps)
                metrics[f"grad/{k}"] = gnorm
                metrics[f"gradratio/{k}"] = gnorm / pnorm
                metrics[f"pnorm/{k}"] = pnorm

                # Add grad-momentum alignment metrics (per-layer cosine similarity)
                if momentum_sq[k] > 0:
                    g_norm = grad_sq[k].sqrt()
                    m_norm = momentum_sq[k].sqrt()
                    per_layer_alignment = dot_product_total[k] / (g_norm * m_norm + eps)
                    metrics[f"grad_alignment/{k}"] = per_layer_alignment

                # Accumulate for total cosine similarity
                total_dot = total_dot + dot_product_total[k]
                total_grad_sq = total_grad_sq + grad_sq[k]
                total_momentum_sq = total_momentum_sq + momentum_sq[k]

            # Compute overall grad-momentum alignment (true cosine similarity)
            if total_momentum_sq > 0:
                total_grad_norm = total_grad_sq.sqrt()
                total_momentum_norm = total_momentum_sq.sqrt()
                total_alignment = total_dot / (
                    total_grad_norm * total_momentum_norm + eps
                )
                metrics["grad_alignment/total"] = total_alignment

            self.log_dict(
                metrics, on_step=True, logger=True, prog_bar=False, sync_dist=True
            )

        return super().on_before_optimizer_step(optimizer)

    def on_train_batch_start(self, batch, batch_idx):
        # Record current time for time-per-step calculation
        self.tic = datetime.datetime.now()

        return super().on_train_batch_start(batch, batch_idx)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        return super().on_train_batch_end(outputs, batch, batch_idx)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        # After the optimizer step, compute and log how long the step took
        toc = datetime.datetime.now()
        dt = (toc - self.tic).total_seconds()
        self.log(
            "dt",
            dt,
            on_step=True,
        )

        # Keep track of the minimum time
        self.min_dt = min(dt, self.min_dt)
        self.log(
            "min_dt",
            self.min_dt,
            on_step=True,
        )
