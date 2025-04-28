import os

import torch
import lightning as L
from omegaconf import OmegaConf


def setup_system(cfg):
    # Set random seeds for reproducibility
    if "init" in cfg and "seed" in cfg["init"]:
        seed = cfg["init"]["seed"]
        L.seed_everything(seed, workers=True)

    # Choose double (32-true) or mixed (16-mixed) precision via AMP
    if cfg.compute.use_amp:
        torch.set_float32_matmul_precision("medium")
    else:
        torch.set_float32_matmul_precision("high")


def save_train_config(log_dir: str, cfg):

    config_save_path = os.path.join(log_dir, "config.yaml")
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)

    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))
