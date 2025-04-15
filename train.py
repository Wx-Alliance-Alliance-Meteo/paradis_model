"""Training script for the model."""

import logging
import hydra
import lightning as L
from omegaconf import DictConfig

from trainer import LitParadis
from data.datamodule import Era5DataModule
from utils.callbacks import enable_callbacks
from utils.system import setup_system, save_train_config


# pylint: disable=E1120
@hydra.main(version_base=None, config_path="config/", config_name="paradis_settings")
def main(cfg: DictConfig):
    """Train the model on ERA5 dataset."""

    # Initiate seed for reproducibility and set torch precision
    setup_system(cfg)

    # Instantiate data module
    datamodule = Era5DataModule(cfg)

    # Early setup call for datamodule attribute access
    datamodule.setup(stage="fit")

    # Initialize model
    litmodel = LitParadis(datamodule, cfg)

    # Prepare callbacks
    callbacks = enable_callbacks(cfg)

    # Instantiate lightning trainer with options
    trainer = L.Trainer(
        default_root_dir="logs/",
        accelerator=cfg.compute.accelerator,
        devices=cfg.compute.num_devices,
        strategy="auto" if cfg.compute.num_devices == 1 else "fsdp",
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.max_steps,
        gradient_clip_val=cfg.training.gradient_clip_val,
        gradient_clip_algorithm="norm",
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        precision="16-mixed" if cfg.compute.use_amp else "32-true",
        enable_progress_bar=cfg.training.progress_bar and not cfg.training.print_losses,
        enable_model_summary=True,
        logger=True,
        val_check_interval=cfg.training.validation_dataset.validation_every_n_steps,
        limit_val_batches=cfg.training.validation_dataset.validation_batches,
        enable_checkpointing=cfg.training.checkpointing.enabled,
    )

    # Keep track of configuration parameters in logging directory
    save_train_config(trainer.logger.log_dir, cfg)  # type: ignore

    # Train model
    trainer.fit(litmodel, datamodule=datamodule, ckpt_path=cfg.model.checkpoint_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
