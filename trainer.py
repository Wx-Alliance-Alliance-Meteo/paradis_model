"""Model training implementation."""

import time
import torch
import lightning as L
from model.Paradis import Paradis
from data.era5_dataset import input_from_output
from utils.loss import WeightedMSELoss


class LitParadis(L.LightningModule):
    def __init__(self, datamodule: L.LightningDataModule, cfg: dict) -> None:
        super().__init__()

        # Initialize PARADIS model
        self.model = Paradis(datamodule, cfg)
        self.lr = cfg.model.lr

        # Custom loss function
        num_surface_vars = len(cfg.features.output.surface)
        self.loss_fn = WeightedMSELoss(
            grid_lat=torch.from_numpy(datamodule.lat),
            pressure_levels=torch.tensor(
                cfg.features.pressure_levels, dtype=torch.float32
            ),
            num_features=datamodule.num_out_features,
            num_surface_vars=num_surface_vars,
        )

        self.forecast_steps = cfg.model.forecast_steps
        self.autoreg_maps = datamodule.autoreg_maps
        self.print_losses = cfg.trainer.print_losses

        # To compute elapsed time for each epoch
        self.epoch_start_time = None
        self.train_loss = None

    def setup(self, stage=None):
        # Keep track of loss within epoch
        self.train_loss_sum = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.val_loss_sum = torch.zeros(1, dtype=torch.float32, device=self.device)
        self.train_step_count = torch.zeros(1, dtype=torch.int, device=self.device)
        self.val_step_count = torch.zeros(1, dtype=torch.int, device=self.device)

    def forward(self, x, t=None):
        return self.model(x, t)

    def configure_optimizers(self):
        """Configure optimizer and scheduler."""
        optimizer = torch.optim.AdamW(
            self.model.parameters(), betas=[0.9, 0.95], lr=self.lr, weight_decay=0.1
        )

        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            total_steps=self.trainer.estimated_stepping_batches,
            max_lr=self.lr,
            pct_start=0.05,
            div_factor=25,
            final_div_factor=1e4,
            anneal_strategy="cos",
        )
        return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def on_train_epoch_start(self):
        # Record the start time of the epoch
        if self.print_losses:
            self.epoch_start_time = time.time()

    def training_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.train_loss_sum += loss.detach()
        self.train_step_count += 1
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._common_step(batch, batch_idx)
        self.val_loss_sum += loss.detach()
        self.val_step_count += 1
        return loss

    # pylint: disable=W0613
    def _common_step(self, batch, batch_idx):
        # Extract the input and true dataset for this batch
        input_data, true_data, static_data = batch

        # Perform a number of autoregressive steps
        batch_loss = 0.0
        current_input = input_data

        for step in range(self.forecast_steps):
            # Call the model
            output_data = self(current_input, torch.tensor(step, device=self.device))

            # Compute the loss with respect to the expected output
            loss = self.loss_fn(output_data, true_data[step])
            batch_loss += loss

            # Use output dataset as input for next forecasting step
            if step + 1 < self.forecast_steps:
                # Reshape for input_from_output function
                output_flat = output_data.permute(0, 2, 3, 1).reshape(
                    output_data.shape[0], -1, output_data.shape[1]
                )
                static_flat = (
                    static_data[step]
                    .permute(0, 2, 3, 1)
                    .reshape(static_data.shape[1], -1, static_data.shape[2])
                )

                next_input = input_from_output(
                    current_input.shape,
                    output_flat,
                    static_flat,
                    self.autoreg_maps,
                    self.device,
                )
                current_input = next_input

        return batch_loss / self.forecast_steps

    def on_train_epoch_end(self):
        # Log mean error after all train epoch calls
        self.train_loss = self.train_loss_sum / self.train_step_count
        self.log(
            "train_loss", self.train_loss, on_epoch=True, prog_bar=True, sync_dist=True
        )

    def on_validation_epoch_end(self):
        # Log mean error after all validation epoch calls
        val_loss = self.val_loss_sum / self.val_step_count
        self.log("val_loss", val_loss, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.print_losses and self.epoch_start_time and self.train_loss is not None:
            current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
            elapsed_time = time.time() - self.epoch_start_time
            print(
                f"Epoch {self.current_epoch:4d} | "
                f"Train Loss: {self.train_loss.item():.6f} | "
                f"Val Loss: {val_loss.item():.6f} | "
                f"LR: {current_lr:.2e} | "
                f"Time: {elapsed_time:.2f}s"
            )
