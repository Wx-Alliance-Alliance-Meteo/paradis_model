"""Lightning data module for ERA5 dataset."""

import logging

import lightning as L
from omegaconf import DictConfig
from multiprocessing import Manager
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

from data.era5_dataset import ERA5Dataset


def sync_forecast_steps(shared_config, device):
    # Sync the number of forecast steps to each process' dataloader memory pool
    if not dist.is_available() or not dist.is_initialized():
        return
    tensor = torch.tensor(
        [shared_config.forecast_steps], dtype=torch.int32, device=device
    )
    dist.broadcast(tensor, src=0)
    shared_config.forecast_steps = int(tensor.cpu().item())


def truncate_collate_fn(batch):
    # Truncate all samples to the minimum number of forecast steps in the batch
    xs, ys = zip(*batch)

    # Find minimum forecast_steps (assumes xs are [T, C, H, W])
    min_len = min(x.shape[0] for x in xs)

    # Truncate inputs and targets to min_len
    xs = [x[:min_len] for x in xs]
    ys = [
        y[:min_len] if y.shape[0] >= min_len else y for y in ys
    ]  # optional: truncate target if time-dependent

    return torch.stack(xs), torch.stack(ys)


class Era5DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.manager = Manager()

        # This configuration is shared in the pool of workers associated with a gpu
        # Hence, an update to its attributes will be visible to all
        self.shared_config = self.manager.Namespace()

        # Extract configuration parameters for data
        self.cfg = cfg
        self.root_dir = cfg.dataset.root_dir
        self.batch_size = cfg.compute.batch_size
        self.max_forecast_steps = cfg.model.forecast_steps
        self.prefetch_factor = 2

        if cfg.forecast.enable or cfg.training.autoregression.init_steps < 0:
            self.forecast_steps = cfg.model.forecast_steps
        else:
            self.forecast_steps = cfg.training.autoregression.init_steps

        # Store number of forecast steps in shared configuration namespace
        self.shared_config.forecast_steps = self.forecast_steps

        self.num_workers = cfg.compute.num_workers

        # Drop last batch when using compiled model
        self.drop_last = cfg.compute.compile

        self.has_setup_been_called = {"fit": False, "predict": False}

    def setup(self, stage=None):

        if not self.has_setup_been_called[stage]:
            logging.info(f"Loading dataset from {self.root_dir}")

            if stage == "fit":
                # Generate training dataset
                train_start_date = self.cfg.training.dataset.start_date
                train_end_date = self.cfg.training.dataset.end_date
                logging.info(
                    f"Training date range: {train_start_date} to {train_end_date}"
                )

                train_era5_dataset = ERA5Dataset(
                    root_dir=self.root_dir,
                    start_date=train_start_date,
                    end_date=train_end_date,
                    max_forecast_steps=self.max_forecast_steps,
                    preload=self.cfg.training.dataset.preload,
                    cfg=self.cfg,
                    shared_config=self.shared_config,
                )

                # Generate validation dataset
                val_start_date = self.cfg.training.validation_dataset.start_date
                val_end_date = self.cfg.training.validation_dataset.end_date

                logging.info(
                    f"Validation date range: {val_start_date} to {val_end_date}"
                )

                self.val_dataset = ERA5Dataset(
                    root_dir=self.root_dir,
                    start_date=val_start_date,
                    end_date=val_end_date,
                    max_forecast_steps=self.max_forecast_steps,
                    preload=self.cfg.training.validation_dataset.preload,
                    cfg=self.cfg,
                    shared_config=self.shared_config,
                )

                # Make certain attributes available at the datamodule level
                self.dataset = train_era5_dataset
                self.num_common_features = train_era5_dataset.num_common_features
                self.num_in_features = train_era5_dataset.num_in_features
                self.num_out_features = train_era5_dataset.num_out_features
                self.output_name_order = train_era5_dataset.dyn_output_features
                self.lat = train_era5_dataset.lat
                self.lon = train_era5_dataset.lon
                self.lat_size = train_era5_dataset.lat_size
                self.lon_size = train_era5_dataset.lon_size

            if stage == "predict":
                pred_start_date = self.cfg.forecast.start_date
                pred_end_date = self.cfg.forecast.get("end_date", None)

                if pred_end_date is None:
                    logging.info(f"Forecast from {pred_start_date}")
                else:
                    logging.info(f"Forecast from {pred_start_date} to {pred_end_date}")
                self.dataset = ERA5Dataset(
                    root_dir=self.root_dir,
                    start_date=pred_start_date,
                    end_date=pred_end_date,
                    max_forecast_steps=self.max_forecast_steps,
                    cfg=self.cfg,
                    shared_config=self.shared_config,
                )

                self.num_common_features = self.dataset.num_common_features
                self.num_in_features = self.dataset.num_in_features
                self.num_out_features = self.dataset.num_out_features
                self.output_name_order = self.dataset.dyn_output_features
                self.lat = self.dataset.lat
                self.lon = self.dataset.lon
                self.lat_size = self.dataset.lat_size
                self.lon_size = self.dataset.lon_size

            logging.info(
                "Dataset contains: %d input features, %d output features.",
                self.num_in_features,
                self.num_out_features,
            )

            self.has_setup_been_called[stage] = True

            logging.info(f"Dataset setup completed successfully for stage {stage}")

    def train_dataloader(self):
        """Return the training dataloader."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=self.drop_last,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=truncate_collate_fn,
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            # Shuffle if we're using less than all validation data
            shuffle=self.cfg.training.validation_dataset.validation_batches is not None,
            pin_memory=True,
            drop_last=self.drop_last,
            persistent_workers=True,
            prefetch_factor=self.prefetch_factor,
            collate_fn=truncate_collate_fn,
        )

    def predict_dataloader(self):
        """Return the forecasting dataloader (includes all data)."""
        logging.info("Batch size set to 1 automatically for inference mode.")
        return DataLoader(
            self.dataset,
            batch_size=1,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            prefetch_factor=self.prefetch_factor,
        )
