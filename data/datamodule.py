"""Lightning data module for ERA5 dataset."""

import logging
import lightning as L
from torch.utils.data import DataLoader

from data.era5_dataset import ERA5Dataset


class Era5DataModule(L.LightningDataModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        # Extract configuration parameters for data
        self.cfg = cfg
        self.root_dir = cfg.dataset.root_dir
        self.batch_size = cfg.compute.batch_size
        self.forecast_steps = cfg.model.forecast_steps
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
                    forecast_steps=self.forecast_steps,
                    cfg=self.cfg,
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
                    forecast_steps=self.forecast_steps,
                    cfg=self.cfg,
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
                    forecast_steps=self.forecast_steps,
                    cfg=self.cfg,
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
        )

    def val_dataloader(self):
        """Return the validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def predict_dataloader(self):
        """Return the forecasting dataloader (includes all data)."""
        return DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
