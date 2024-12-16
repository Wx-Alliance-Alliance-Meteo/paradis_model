"""Lightning data module for ERA5 dataset."""

import logging
import lightning as L
from torch.utils.data import DataLoader

from data.era5_dataset import ERA5Dataset, split_dataset


class Era5DataModule(L.LightningDataModule):
    def __init__(self, cfg: dict) -> None:
        super().__init__()

        # Extract configuration parameters for data
        self.root_dir = cfg.dataset.dataset_dir
        self.batch_size = cfg.dataset.batch_size
        self.start_date = cfg.dataset.start_date
        self.end_date = cfg.dataset.end_date
        self.num_workers = cfg.dataset.num_workers
        self.train_ratio = cfg.dataset.train_ratio
        self.features_cfg = cfg.features

        self.forecast_steps = cfg.model.forecast_steps
        self.drop_last = cfg.model.compile  # Drop last batch when using compiled model

        self.has_setup_been_called = {"fit": False, "test": False}

    def setup(self, stage=None):
        if stage == "fit" and not self.has_setup_been_called["fit"]:
            # Generate dataset
            era5_dataset = ERA5Dataset(
                root_dir=self.root_dir,
                start_date=self.start_date,
                end_date=self.end_date,
                forecast_steps=self.forecast_steps,
                features_cfg=self.features_cfg,
            )

            # Make the autoregression maps available at a higher level
            self.num_common_features = era5_dataset.num_common_features
            self.num_in_features = era5_dataset.num_in_features
            self.num_out_features = era5_dataset.num_out_features
            self.output_name_order = era5_dataset.output_name_order
            self.lat = era5_dataset.lat
            self.lon = era5_dataset.lon
            self.lat_size = era5_dataset.lat_size
            self.lon_size = era5_dataset.lon_size

            # Split into training and validation sets
            self.train_dataset, self.val_dataset = split_dataset(
                era5_dataset, train_ratio=self.train_ratio
            )

            self.has_setup_been_called["fit"] = True

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=True,
            drop_last=self.drop_last,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=True,
            drop_last=self.drop_last,
        )
