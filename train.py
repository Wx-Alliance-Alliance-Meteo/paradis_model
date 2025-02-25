"""Training script for the model."""

import random
import logging
import hydra
import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from omegaconf import DictConfig

from trainer import LitParadis
from data.datamodule import Era5DataModule


# pylint: disable=E1120
@hydra.main(version_base=None, config_path="config/", config_name="paradis_settings")
def main(cfg: DictConfig):
    """Train the model on ERA5 dataset."""

    # Set random seeds for reproducibility
    seed = 42  # This model will answer the ultimate question about life, the universe, and everything
    L.seed_everything(seed, workers=True)

    # Instantiate data module
    datamodule = Era5DataModule(cfg)

    # Early setup call for datamodule attribute access
    datamodule.setup(stage="fit")

    # Initialize model
    litmodel = LitParadis(datamodule, cfg)

    # Load the model weights if a checkpoint path is provided
    if cfg.model.checkpoint_path:
        # Load into CPU, then Lightning will transfer to GPU
        checkpoint = torch.load(
            cfg.model.checkpoint_path, weights_only=True, map_location="cpu"
        )
        litmodel.load_state_dict(checkpoint["state_dict"])

    # Define callbacks
    callbacks = []
    if cfg.training.parameters.early_stopping.enabled:
        # Stop epochs when validation loss is not decreasing during a coupe of epochs
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=cfg.training.parameters.early_stopping.patience,
                check_finite=True,  # Make sure validation has not gone to nan
            )
        )

    # Keep the last 10 checkpoints and the top "best" checkpoint
    callbacks.append(
        ModelCheckpoint(
            filename="{epoch}",  # Filename format for the checkpoints
            monitor="train_loss",
            save_top_k=10,  # Keep the last 10 checkpoints
            save_last=True,  # Always save the most recent checkpoint
            every_n_epochs=1,  # Save at the end of every epoch
        )
    )
    callbacks.append(
        ModelCheckpoint(
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,  # Keep only the best checkpoint
        )
    )

    # Choose double (32-true) or mixed (16-mixed) precision via AMP
    if cfg.compute.use_amp:
        precision = "16-mixed"
        torch.set_float32_matmul_precision("medium")
    else:
        precision = "32-true"
        torch.set_float32_matmul_precision("high")

    # Instantiate lightning trainer with options
    train_params = cfg.training.parameters

    trainer = L.Trainer(
        default_root_dir="logs/",
        accelerator=cfg.compute.accelerator,
        devices=cfg.compute.num_devices,
        strategy="ddp",
        max_epochs=train_params.max_epochs,
        gradient_clip_val=train_params.gradient_clip_val,
        gradient_clip_algorithm="norm",
        log_every_n_steps=cfg.training.parameters.log_every_n_steps,
        callbacks=callbacks,
        precision=precision,
        enable_progress_bar=not train_params.print_losses,
        enable_model_summary=True,
        logger=True,
    )

    # Train model
    trainer.fit(litmodel, datamodule=datamodule)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
