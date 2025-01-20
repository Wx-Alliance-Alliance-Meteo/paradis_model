"""Model training implementation."""

import time
import re

import logging
import torch
import lightning as L

from model.paradis import Paradis
from utils.loss import WeightedHybridLoss


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

        print(f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")

        # Access output_name_order from configuration
        self.output_name_order = datamodule.output_name_order

        num_levels = len(cfg.features.pressure_levels)

        # Construct variable_loss_weights tensor from YAML configuration
        atmospheric_weights = torch.tensor(
            [
                cfg.variable_loss_weights.atmospheric[var]
                for var in cfg.features.output.atmospheric
            ],
            dtype=torch.float32,
        )

        surface_weights = torch.tensor(
            [
                cfg.variable_loss_weights.surface[var]
                for var in cfg.features.output.surface
            ],
            dtype=torch.float32,
        )
        # Concatenate atmospheric and surface weights
        var_loss_weights = torch.cat([atmospheric_weights, surface_weights])

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

        self.loss_fn = WeightedHybridLoss(
            grid_lat=torch.from_numpy(datamodule.lat),
            pressure_levels=torch.tensor(
                cfg.features.pressure_levels, dtype=torch.float32
            ),
            num_features=datamodule.num_out_features,
            num_surface_vars=len(cfg.features.output.surface),
            var_loss_weights=var_loss_weights_reordered,
            output_name_order=datamodule.output_name_order,
            alpha=cfg.model.get("loss_alpha"),
        )

        self.forecast_steps = cfg.model.forecast_steps
        self.num_common_features = datamodule.num_common_features
        self.print_losses = cfg.trainer.print_losses

        if cfg.model.compile:
            self.model = torch.compile(
                self.model,
                mode="default",
                fullgraph=True,
                dynamic=False,
                backend="inductor",
            )

        self.epoch_start_time = None

    def forward(self, x, t=None):
        """Forward pass through the model."""
        return self.model(x, t)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            betas=[0.9, 0.999],
            lr=self.cfg.trainer.lr,
            weight_decay=self.cfg.trainer.weight_decay,
        )

        scheduler_cfg = self.cfg.trainer.scheduler
        if scheduler_cfg.type == "one_cycle":
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                total_steps=self.trainer.estimated_stepping_batches,
                max_lr=self.cfg.trainer.lr,
                pct_start=scheduler_cfg.warmup_pct_start,
                div_factor=scheduler_cfg.lr_div_factor,
                final_div_factor=scheduler_cfg.lr_final_div,
                anneal_strategy="cos",
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        elif scheduler_cfg.type == "reduce_lr":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_cfg.factor,
                patience=scheduler_cfg.patience,
                min_lr=scheduler_cfg.min_lr,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_cfg.type}")

    def on_train_epoch_start(self):
        """Record the start time of the epoch."""
        if self.print_losses:
            self.epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):
        """Training step."""
        loss = self._common_step(batch, batch_idx)

        # Log metrics
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        self.log("lr", self.optimizers().param_groups[0]["lr"], prog_bar=True)

        # Clip gradients manually if gradient_clip_val is set
        if self.cfg.trainer.gradient_clip_val > 0:
            self.clip_gradients(
                self.optimizers(),
                gradient_clip_val=self.cfg.trainer.gradient_clip_val,
                gradient_clip_algorithm="norm",
            )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss = self._common_step(batch, batch_idx)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
        )
        return loss

    def _common_step(self, batch, batch_idx):
        """Common step for both training and validation."""
        input_data, true_data = batch

        batch_loss = 0.0
        input_data_step = input_data[:, 0]  # Start with first timestep

        for step in range(self.forecast_steps):
            # Forward pass
            output_data = self(input_data_step, torch.tensor(step, device=self.device))

            # Compute loss (data is already transformed by dataset)
            loss = self.loss_fn(output_data, true_data[:, step])
            batch_loss += loss

            # Prepare next step input
            if step + 1 < self.forecast_steps:
                input_data_step = self._autoregression_input_from_output(
                    input_data[:, step + 1], output_data
                )

        return batch_loss / self.forecast_steps

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
                    f"Epoch {self.current_epoch:4d} | ",
                    f"Train Loss: {train_loss.item():.6f} | ",
                    f"Val Loss: {val_loss.item():.6f} | ",
                    f"LR: {current_lr:.2e} | ",
                    f"Elapsed time: {elapsed_time:.4f}s"
                )

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

    def on_train_end(self):
        """Called when training ends."""
        logging.info(f"Training completed after {self.current_epoch + 1} epochs")
