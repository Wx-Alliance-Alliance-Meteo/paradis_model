"""Model training implementation."""

import datetime
import time
import re
import logging

import torch
import lightning as L

from model.paradis import Paradis
from utils.loss import ParadisLoss
from utils.normalization import denormalize_humidity, denormalize_precipitation
from data.datamodule import Era5DataModule

import omegaconf.dictconfig


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

        if self.global_rank == 0:
            logging.info(
                "Number of trainable parameters: {:,}".format(
                    sum(p.numel() for p in self.model.parameters() if p.requires_grad)
                )
            )

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

        self.forecast_steps = cfg.model.forecast_steps

        self.num_common_features = datamodule.num_common_features
        self.print_losses = cfg.training.print_losses

        # Compile model in place
        if cfg.compute.compile:
            self.model.compile(
                mode="default",
                fullgraph=True,
                dynamic=False,
                backend="inductor",
            )

        # Load weights only but reset lightning configuration
        if (cfg.init.checkpoint_path and not cfg.init.restart) or cfg.forecast.enable:
            # Load into CPU, then Lightning will transfer to GPU
            checkpoint = torch.load(
                cfg.init.checkpoint_path, weights_only=True, map_location="cpu"
            )
            self.load_state_dict(checkpoint["state_dict"])

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

    def _autoregression_input_from_output(
        self,
        input_data: torch.Tensor,
        output_data: torch.Tensor,
        step: int,
        num_steps: int,
    ) -> torch.Tensor:
        """Process the next input in autoregression."""
        # Add features needed from the output.
        # Common features have been previously sorted to ensure they are first
        # and hence simplify adding them

        # Update future inputs with the current output
        steps_left = num_steps - step - 1
        for i in range(min(steps_left, self.n_inputs)):
            beg_i = self.num_common_features * (self.n_inputs - i - 1)
            end_i = self.num_common_features * (self.n_inputs - i)
            input_data[:, step + i + 1, beg_i:end_i] = output_data[
                :, : self.num_common_features
            ]

        return input_data

    def _get_persistence_loss(
        self, input_data: torch.Tensor, pred_data: torch.Tensor
    ) -> torch.Tensor:
        import torch

        p_loss = torch.tensor(0.0)
        num_steps = input_data.size(1)
        for step in range(num_steps):
            loss = self.loss_fn(
                input_data[:, step, : self.num_common_features], pred_data[:, step]
            )
            p_loss += loss
        return p_loss.detach()

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

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            betas=(cfg.optimizer.beta1, cfg.optimizer.beta2),
            lr=cfg.optimizer.lr,
            weight_decay=cfg.optimizer.weight_decay,
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

    def on_train_epoch_start(self):
        """Record the start time of the epoch."""
        if self.print_losses:
            self.epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):
        input_data, true_data = batch

        train_loss = 0.0

        num_steps = input_data.size(1)
        for step in range(num_steps):
            # Forward pass
            output_data = self(input_data[:, step])

            loss = self.loss_fn(output_data, true_data[:, step])

            # Compute loss (data is already transformed by dataset)
            train_loss += loss

            input_data = self._autoregression_input_from_output(
                input_data, output_data, step, num_steps
            )

        # Log metrics
        self.log(
            "train_loss",
            train_loss / num_steps,
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

        return train_loss / num_steps

    def validation_step(self, batch, batch_idx):
        """Validation step."""

        input_data, true_data = batch

        val_loss = 0.0
        report_loss = 0.0
        num_steps = input_data.size(1)

        for step in range(num_steps):

            # Forward pass
            output_data = self(input_data[:, step])

            loss = self.loss_fn(output_data, true_data[:, step])

            # Log requested scaled RMSE losses for validation
            report_loss += self._get_report_rmse(output_data, true_data[:, step])

            # Compute loss (data is already transformed by dataset)
            val_loss += loss

            input_data = self._autoregression_input_from_output(
                input_data, output_data, step, num_steps
            )

        self.log(
            "val_loss",
            val_loss / num_steps,
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

        return val_loss / num_steps

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
        import torch

        gradmag = (
            sum(
                torch.sum(v.grad**2) if v.grad is not None else 0
                for (k, v) in self.named_parameters()
                if torch.is_tensor(v)
            )
            ** 0.5
        )

        self.log(
            "gradmag",
            gradmag,
            on_step=True,
        )

        return super().on_before_optimizer_step(optimizer)

    def on_train_batch_start(self, batch, batch_idx):
        # Record current time for time-per-step calculation
        self.tic = datetime.datetime.now()

        return super().on_train_batch_start(batch, batch_idx)

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure=None):
        super().optimizer_step(epoch, batch_idx, optimizer, optimizer_closure)

        # After the optimzier step, comptue and log how long the step took
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
