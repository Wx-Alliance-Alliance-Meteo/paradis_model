"""Training script for the model."""
import os # Intel GPUs
# the following works but forces serial GPU computation
os.environ["TORCH_ATTENTION_MODE"] = "sdpa" # Intel GPUs
os.environ["TORCH_COMPILE_DISABLE"] = "1" # Intel GPUs

import logging

import hydra
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import DictConfig

from data.datamodule import Era5DataModule
from trainer import LitParadis
from utils.callbacks import enable_callbacks
from utils.system import save_train_config, setup_system
from utils.Intel_GPU_utils import SingleXPUStrategy # Intel GPUs

#import torch

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

    # Configure logger with optional experiment name
    logger = TensorBoardLogger(
        save_dir=cfg.training.log_dir,
        name="lightning_logs",
        version=cfg.training.get("experiment_name", None),
    )

    # Keyword arguments common to all hardware
    trainer_kwargs = {
        "default_root_dir": cfg.training.log_dir,
        "devices": cfg.compute.num_devices,
        "num_nodes": cfg.compute.num_nodes,
        "max_epochs": cfg.training.max_epochs,
        "max_steps": cfg.training.max_steps,
        "gradient_clip_val": cfg.training.gradient_clip_val,
        "gradient_clip_algorithm": "norm",
        "log_every_n_steps": cfg.training.log_every_n_steps,
        "callbacks": callbacks,
        "precision": "bf16-mixed" if cfg.compute.use_amp else "32-true",
        "enable_progress_bar": cfg.training.progress_bar and not cfg.training.print_losses,
        "enable_model_summary": True,
        "logger": logger,
        "val_check_interval": cfg.training.validation_dataset.validation_every_n_steps,
        "limit_val_batches": cfg.training.validation_dataset.validation_batches,
        "enable_checkpointing": cfg.training.checkpointing.enabled,
        "num_sanity_val_steps": 0,
        "accumulate_grad_batches": cfg.training.get("accumulate_grad_batches", 1)
    }
            

    # Keyword arguments that depend on GPU hardware choice
    if cfg.compute.Intel_GPU:
        xpu_strategy = SingleXPUStrategy(device_index=0)
        trainer_kwargs.update({
            "strategy": xpu_strategy
        })
    else:
        trainer_kwargs.update({
            "accelerator": cfg.compute.accelerator,
            "strategy": auto if cfg.compute.num_devices == 1 else "ddp"
        })
        
    # Instantiate lightning trainer with options
    trainer = L.Trainer(**trainer_kwargs)

    # Keep track of configuration parameters in logging directory
    save_train_config(trainer.logger.log_dir, cfg)  # type: ignore

    # Train model
    checkpoint_path = cfg.init.checkpoint_path if cfg.init.restart else None
    trainer.fit(litmodel, datamodule=datamodule, ckpt_path=checkpoint_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
