"""Training script for the model."""

import os # Intel GPUs
os.environ["TORCH_ATTENTION_MODE"] = "sdpa" # Intel GPUs
os.environ["TORCH_COMPILE_DISABLE"] = "1" # Intel GPUs

import logging

import hydra
import lightning as L
import torch # Intel GPUs
#from torch.utils.data import DataLoader, TensorDataset # Intel GPUs -- may not be needed
from lightning.pytorch.accelerators import Accelerator # Intel GPUs
from lightning.pytorch.strategies import SingleDeviceStrategy # Intel GPUs
from lightning.pytorch.loggers import TensorBoardLogger
from typing import Any, List, Optional # Intel GPUs
from omegaconf import DictConfig

from data.datamodule import Era5DataModule
from trainer import LitParadis
from utils.callbacks import enable_callbacks
from utils.system import save_train_config, setup_system

# For compatibility with Intel ARC GPUs
class IntelXPUAccelerator(Accelerator):
    
    # REQUIRED: Lightning uses this name property internally
    @property
    def name(self) -> str:
        return "intel_xpu"

    # FIXED: Properly parses the integer count into a clean list of device indices
    @staticmethod
    def parse_devices(devices: Any) -> Optional[List[int]]:
        if isinstance(devices, int):
            return list(range(devices))  # Turns 1 into [0]
        if isinstance(devices, list):
            return devices
        return None

    # FIXED: Expects a structured iterable list from parse_devices instead of an int
    @staticmethod
    def get_parallel_devices(devices: List[int]) -> List[torch.device]:
        return [torch.device("xpu", idx) for idx in devices]

    @staticmethod
    def auto_device_count() -> int:
        return torch.xpu.device_count()

    @staticmethod
    def is_available() -> bool:
        return torch.xpu.is_available()

    # REQUIRED: Tells Lightning how to register the device to the runtime environment
    def setup_device(self, device: torch.device) -> None:
        if device.type != "xpu":
            raise ValueError(f"Invalid device passed to XPU Accelerator: {device}")
        torch.xpu.set_device(device)

    # REQUIRED: Tells Lightning how to clean up memory assets after execution
    def teardown(self) -> None:
        # Clear out the active XPU cache layer on exit
        if torch.xpu.is_available():
            torch.xpu.empty_cache()

    def get_device_stats(self, device: torch.device) -> dict:
        return {}

# For compatibility with Intel ARC GPUs
class SingleXPUStrategy(SingleDeviceStrategy):
    def __init__(self, device_index: int = 0):
        super().__init__(
            accelerator=IntelXPUAccelerator(),
            device=torch.device("xpu", device_index)
        )

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

    # Instantiate lightning trainer with options
    trainer = L.Trainer(
        default_root_dir=cfg.training.log_dir,
        #accelerator=cfg.compute.accelerator,
        devices=cfg.compute.num_devices,
        num_nodes=cfg.compute.num_nodes,
        strategy=xpu_strategy if cfg.compute.num_devices == 1 else "ddp",# OG strategy="auto"
        max_epochs=cfg.training.max_epochs,
        max_steps=cfg.training.max_steps,
        gradient_clip_val=cfg.training.gradient_clip_val,
        gradient_clip_algorithm="norm",
        log_every_n_steps=cfg.training.log_every_n_steps,
        callbacks=callbacks,
        precision="bf16-mixed" if cfg.compute.use_amp else "32-true",
        enable_progress_bar=cfg.training.progress_bar and not cfg.training.print_losses,
        enable_model_summary=True,
        logger=logger,
        val_check_interval=cfg.training.validation_dataset.validation_every_n_steps,
        limit_val_batches=cfg.training.validation_dataset.validation_batches,
        enable_checkpointing=cfg.training.checkpointing.enabled,
        num_sanity_val_steps=0,
        accumulate_grad_batches=cfg.training.get("accumulate_grad_batches", 1),
    )

    # Keep track of configuration parameters in logging directory
    save_train_config(trainer.logger.log_dir, cfg)  # type: ignore

    # Train model
    checkpoint_path = cfg.init.checkpoint_path if cfg.init.restart else None
    trainer.fit(litmodel, datamodule=datamodule, ckpt_path=checkpoint_path)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    # For Intel ARC GPU compatibility
    xpu_strategy = SingleXPUStrategy(device_index=0)
    main()
