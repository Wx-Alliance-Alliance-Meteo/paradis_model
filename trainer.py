"""Model training implementation."""

import time

import lightning as L
import torch

from model.paradis import Paradis
from utils.loss import WeightedMSELoss


class LitParadis(L.LightningModule):
    def __init__(self, datamodule: L.LightningDataModule, cfg: dict) -> None:
        super().__init__()

        # Instantiate the model
        self.model = Paradis(datamodule, cfg)
        self.lr = cfg.model.lr

        print(
            f"Number of trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}"
        )

        self.warmup_steps = cfg.model.get("warmup_steps", 1000)
        self.gradient_clip_val = cfg.model.get("gradient_clip_val", 1.0)

        # Custom loss function
        num_surface_vars = len(cfg.features.output.surface)

        # Access output_name_order from configuration
        self.output_name_order = datamodule.output_name_order

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
        var_loss_weights_reordered = torch.zeros_like(var_loss_weights)

        # Reorder based on self.output_name_order
        for i, var in enumerate(self.output_name_order):
            if var in var_name_to_weight:
                var_loss_weights_reordered[i] = var_name_to_weight[var]

        self.loss_fn = WeightedMSELoss(
            grid_lat=torch.from_numpy(datamodule.lat),
            pressure_levels=torch.tensor(
                cfg.features.pressure_levels, dtype=torch.float32
            ),
            num_features=datamodule.num_out_features,
            num_surface_vars=num_surface_vars,
            var_loss_weights=var_loss_weights_reordered,
        )

        self.forecast_steps = cfg.model.forecast_steps
        self.num_common_features = datamodule.num_common_features
        self.print_losses = cfg.trainer.print_losses

        self.epoch_start_time = None

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        no_decay = [
            "bias",
            "LayerNorm.weight",
            "BatchNorm.weight",
            "InstanceNorm.weight",
        ]
        grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.1,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer = torch.optim.AdamW(
            grouped_parameters, betas=[0.9, 0.999], lr=self.lr, eps=1e-8
        )

        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.lr,
            pct_start=0.3,
            div_factor=25,
            final_div_factor=1e3,
            anneal_strategy="cos",
            three_phase=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }

    def forward(self, x, t=None):
        """Forward pass through the model."""
        return self.model(x, t)

    def on_train_epoch_start(self):
        """Record the start time of the epoch."""
        if self.print_losses:
            self.epoch_start_time = time.time()

    # pylint: disable=W0613
    def _common_step(self, batch, batch_idx):
        """Common step for both training and validation."""
        input_data, true_data = batch

        batch_loss = 0.0

        input_data_step = input_data[:, 0]
        for step in range(self.forecast_steps):
            # Call the model
            output_data = self(input_data_step, torch.tensor(step, device=self.device))
            loss = self.loss_fn(output_data, true_data[:, step])
            batch_loss += loss

            # Use output dataset as input for next forecasting step
            if step + 1 < self.forecast_steps:
                input_data_step = self._autoregression_input_from_output(
                    input_data[:, step + 1], output_data
                )

        return batch_loss / self.forecast_steps

    def training_step(self, batch, batch_idx):
        """Training step using automatic mixed precision."""
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

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        loss = self._common_step(batch, batch_idx)
        self.log(
            "val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True
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
                    f"Elapsed time: {elapsed_time:.2f}s"
                )

    def _autoregression_input_from_output(self, input_data, output_data):
        """Process the next input in autoregression"""

        # Add features needed from the output.
        # Common features have been previously sorted to ensure they are first
        # and hence simplify adding them
        input_data = input_data.clone()
        input_data[:, : self.num_common_features, ...] = output_data[
            :, : self.num_common_features, ...
        ]
        return input_data
