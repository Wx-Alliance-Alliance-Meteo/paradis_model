"""Lightning data module for ERA5 dataset."""

import logging
import lightning as L
from torch.utils.data import DataLoader

from data.era5_dataset import ERA5Dataset


class Era5DataModule(L.LightningDataModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        # Extract configuration parameters for data
        self.root_dir = cfg.dataset.dataset_dir
        self.batch_size = cfg.dataset.batch_size
        self.training = cfg.dataset.training
        self.validation = cfg.dataset.validation
        self.testing = cfg.dataset.testing
        self.num_workers = cfg.dataset.num_workers
        self.features_cfg = cfg.features

        self.forecast_steps = cfg.model.forecast_steps
        self.drop_last = cfg.model.compile  # Drop last batch when using compiled model

        self.has_setup_been_called = {"fit": False, "test": False}

    def setup(self, stage=None):

        if not self.has_setup_been_called[stage]:
            logging.info(f"Loading dataset from {self.root_dir}")
            logging.info(
                f"Training date range: {self.training.start_date} to {self.training.end_date}"
            )

            # Generate dataset
            train_era5_dataset = ERA5Dataset(
                root_dir=self.root_dir,
                start_date=self.training.start_date,
                end_date=self.training.end_date,
                forecast_steps=self.forecast_steps,
                features_cfg=self.features_cfg,
            )

            # Make the autoregression maps available at a higher level
            self.dataset = train_era5_dataset
            self.num_common_features = train_era5_dataset.num_common_features
            self.num_in_features = train_era5_dataset.num_in_features
            self.num_out_features = train_era5_dataset.num_out_features
            self.output_name_order = train_era5_dataset.dyn_output_features
            self.lat = train_era5_dataset.lat
            self.lon = train_era5_dataset.lon
            self.lat_size = train_era5_dataset.lat_size
            self.lon_size = train_era5_dataset.lon_size

            if self.validation:
                logging.info(
                    f"Validation date range: {self.validation.start_date} to {self.validation.end_date}"
                )
                self.val_dataset = ERA5Dataset(
                    root_dir=self.root_dir,
                    start_date=self.validation.start_date,
                    end_date=self.validation.end_date,
                    forecast_steps=self.forecast_steps,
                    features_cfg=self.features_cfg,
                )

            if self.testing:
                logging.info(
                    f"Testing date range: {self.testing.start_date} to {self.testing.end_date}"
                )
                self.test_dataset = ERA5Dataset(
                    root_dir=self.root_dir,
                    start_date=self.testing.start_date,
                    end_date=self.testing.end_date,
                    forecast_steps=self.forecast_steps,
                    features_cfg=self.features_cfg,
                )


            logging.info(
                "Dataset contains: %d input features, %d output features.",
                train_era5_dataset.num_in_features,
                train_era5_dataset.num_out_features,
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

    def test_dataloader(self):
        """Return the test dataloader (includes all data)."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
        )
