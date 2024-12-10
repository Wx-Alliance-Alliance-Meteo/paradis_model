"""Training script for the model."""

import logging

import hydra
import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from omegaconf import DictConfig

from trainer import LitParadis
from data.datamodule import Era5DataModule


# pylint: disable=E1120
@hydra.main(version_base=None, config_path="config/", config_name="train")
def main(cfg: DictConfig):
    """Train the model on ERA5 dataset."""

    # Instantiate data module
    datamodule = Era5DataModule(cfg)

    # Early setup call for datamodule attribute access
    datamodule.setup(stage="fit")

    # Initialize model
    litmodel = LitParadis(datamodule, cfg)

    if cfg.model.compile:
        # Will throw error if graph breaks present
        litmodel = torch.compile(litmodel, fullgraph=True)

    # Define callbacks
    callbacks = []
    if cfg.trainer.early_stopping:
        # Stop epochs when validation loss is not decreasing during three epochs
        callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=3))

    # Choose double (32-true) or mixed (16-mixed) precision via AMP
    if cfg.trainer.use_amp:
        precision = "16-mixed"
        torch.set_float32_matmul_precision("medium")
    else:
        precision = "32-true"
        torch.set_float32_matmul_precision("high")

    # Instantiate lightning trainer with options
    trainer = L.Trainer(
        default_root_dir="logs/",
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.num_devices,
        strategy="ddp",
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=1.0,
        gradient_clip_algorithm="norm",
        log_every_n_steps=1,
        callbacks=callbacks,
        precision=precision,
        enable_progress_bar=not cfg.trainer.print_losses,
        enable_model_summary=not cfg.trainer.print_losses,
        logger=not cfg.trainer.print_losses,
    )

    # Train model
    trainer.fit(litmodel, datamodule=datamodule)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
