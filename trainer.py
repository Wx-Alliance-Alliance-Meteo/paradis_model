"""Model training implementation."""

import time
import re
import logging

import torch
import lightning as L

from model.paradis_multilayer import Paradis
from utils.loss import ParadisLoss


class LitParadis(L.LightningModule):
    """Lightning module for Paradis model training."""

    def __init__(self, datamodule: L.LightningDataModule, cfg: dict) -> None:
        """Initialize the training module.

        Args:
            datamodule: Lightning datamodule containing dataset information
            cfg: Model configuration dictionary
        """
        super().__init__()

        # Instantiate the model
        self.model = Paradis(datamodule, cfg)
        self.cfg = cfg
        self.variational = cfg.ensemble.enable
        self.beta = cfg.ensemble.get("beta", None)

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
        )

        self.forecast_steps = cfg.model.forecast_steps
        self.num_common_features = datamodule.num_common_features
        self.print_losses = cfg.training.print_losses

        if cfg.compute.compile:
            self.model = torch.compile(
                self.model,
                mode="default",
                fullgraph=True,
                dynamic=False,
                backend="inductor",
            )

        self.epoch_start_time = None

        # Store the index of the GZ100 quantity to
        self.gz500_ind = datamodule.dataset.dyn_input_features.index(
            "geopotential_h500"
        )
        self.gz500_mean = torch.from_numpy(datamodule.dataset.gz500_mean)
        self.gz500_std = torch.from_numpy(datamodule.dataset.gz500_std)

    def _autoregression_input_from_output(self, input_data, output_data):
        """Process the next input in autoregression."""
        # Add features needed from the output.
        # Common features have been previously sorted to ensure they are first
        # and hence simplify adding them
        input_data = input_data.clone()
        input_data[:, : self.num_common_features, ...] = output_data[
            :, : self.num_common_features, ...
        ]
        return input_data

    def _get_persistence_loss(self, input_data, pred_data):
        p_loss = 0.0
        for step in range(self.forecast_steps):
            loss = self.loss_fn(
                input_data[:, step, : self.num_common_features], pred_data[:, step]
            )
            p_loss += loss
        return p_loss.detach()

    def _get_gz500_rmse(self, output_data, pred_data):

        lat_weights = self.loss_fn.lat_weights.view(1, 1, -1, 1).to(output_data.device)

        # Compute the batch error
        error = torch.mean(
            (
                (output_data[:, self.gz500_ind] - pred_data[:, self.gz500_ind])
                / 10
                * self.gz500_std
            )
            ** 2
            * lat_weights
        )

        return torch.sqrt(error).detach()

    def forward(self, x):
        """Forward pass through the model."""
        return self.model(x)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        cfg = self.cfg.training

        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            betas=[cfg.optimizer.beta1, cfg.optimizer.beta2],
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
                total_steps=self.trainer.estimated_stepping_batches,
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

            assert (cfg.scheduler.wsd.warmup_pct + cfg.scheduler.wsd.decay_pct) <= 1.0

            warmup_steps = int(cfg.scheduler.wsd.warmup_pct * total_steps)
            decay_steps = int(cfg.scheduler.wsd.decay_pct * total_steps)
            steady_steps = total_steps - (warmup_steps + decay_steps)

            def lr_lambda(step):
                if step < warmup_steps:
                    # Increasing learning rate phase
                    return (step + 1) / warmup_steps
                elif step < warmup_steps + steady_steps:
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
            raise ValueError(f"Unknown scheduler type: {cfg.scheduler.type}")

    def on_train_epoch_start(self):
        """Record the start time of the epoch."""
        if self.print_losses:
            self.epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):

        input_data, true_data = batch

        train_loss = 0.0
        input_data_step = input_data[:, 0]  # Start with first timestep

        for step in range(self.forecast_steps):
            # Forward pass
            if self.variational:
                output_data, kl_loss = self(input_data_step)
            else:
                output_data = self(input_data_step)

            loss = self.loss_fn(output_data, true_data[:, step])

            if self.variational:
                loss += self.beta * kl_loss

            # Compute loss (data is already transformed by dataset)
            train_loss += loss

            # Prepare next step input
            if step + 1 < self.forecast_steps:
                input_data_step = self._autoregression_input_from_output(
                    input_data[:, step + 1], output_data
                )

        # Log metrics
        self.log(
            "train_loss",
            train_loss / self.forecast_steps,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        if self.variational:
            self.log(
                "kl_loss",
                kl_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )

        # Clip gradients manually if gradient_clip_val is set
        if self.cfg.training.gradient_clip_val > 0:
            self.clip_gradients(
                self.optimizers(),
                gradient_clip_val=self.cfg.training.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""

        input_data, true_data = batch

        val_loss = 0.0
        gz500_loss = 0.0
        input_data_step = input_data[:, 0]  # Start with first timestep

        for step in range(self.forecast_steps):
            # Forward pass
            if self.variational:
                output_data, kl_loss = self(input_data_step)
            else:
                output_data = self(input_data_step)

            loss = self.loss_fn(output_data, true_data[:, step])

            # Log additional GZ500 loss for validation
            gz500_loss += self._get_gz500_rmse(output_data, true_data[:, step])

            if self.variational:
                loss += self.beta * kl_loss

            # Compute loss (data is already transformed by dataset)
            val_loss += loss

            # Prepare next step input
            if step + 1 < self.forecast_steps:
                input_data_step = self._autoregression_input_from_output(
                    input_data[:, step + 1], output_data
                )

        self.log(
            "val_loss",
            val_loss / self.forecast_steps,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

        # Log GZ500 RMSE
        self.log(
            "GZ500-RMSE",
            gz500_loss / self.forecast_steps,
            on_step=True,
            on_epoch=False,
            prog_bar=True,
            sync_dist=True,
        )

        if self.variational:
            self.log(
                "kl_loss",
                kl_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
            )
        return loss

    def on_train_epoch_end(self):
        """Log epoch time and metrics if printing losses."""
        if self.print_losses and self.epoch_start_time is not None:
            elapsed_time = time.time() - self.epoch_start_time
            current_lr = self.optimizers().param_groups[0]["lr"]

            # Get the losses using the logged metrics
            train_loss = self.trainer.callback_metrics.get("train_loss")
            val_loss = self.trainer.callback_metrics.get("val_loss")

            if train_loss is not None and val_loss is not None:
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
