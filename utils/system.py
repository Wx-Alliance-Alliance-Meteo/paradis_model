import glob
import os
import shutil
import subprocess

import lightning as L
import torch
from lightning.pytorch.utilities import rank_zero_only
from omegaconf import DictConfig, OmegaConf


def setup_system(cfg: DictConfig):
    # Set random seeds for reproducibility
    if "init" in cfg and "seed" in cfg["init"]:
        seed = cfg["init"]["seed"]
        L.seed_everything(seed, workers=True)

    # Choose double (32-true) or mixed (16-mixed) precision via AMP
    if cfg.compute.use_amp:
        torch.set_float32_matmul_precision("medium")
    else:
        torch.set_float32_matmul_precision("high")

    # Make sure number of steps or epochs is defined
    if not cfg.forecast.enable:
        assert not (
            cfg.training.max_steps < 0 and cfg.training.max_epochs < 0
        ), "Please indicate max_epochs or max_steps"

    # Check whether experiment name has been used (if provided)
    _check_log_dir(cfg)


@rank_zero_only
def _check_log_dir(cfg: DictConfig):
    experiment_name = cfg.training.get("experiment_name", None)
    if experiment_name:
        exp_dir = os.path.join(cfg.training.log_dir, "lightning_logs", experiment_name)
        if os.path.exists(exp_dir):
            raise RuntimeError(
                f"Experiment directory already exists: {exp_dir}\n"
                f"Please choose a different experiment_name or delete the existing directory."
            )


@rank_zero_only
def save_train_config(log_dir: str, cfg: DictConfig):

    config_save_path = os.path.join(log_dir, "config.yaml")
    os.makedirs(os.path.dirname(config_save_path), exist_ok=True)

    with open(config_save_path, "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # Save source code snapshot
    code_snapshot_dir = os.path.join(log_dir, "code_snapshot")
    _save_code_snapshot(code_snapshot_dir)


@rank_zero_only
def _save_code_snapshot(code_snapshot_dir: str):
    """
    Save tracked files from git to code_snapshot directory.
    If git is not available, falls back to copying .py and .yaml files.
    """
    os.makedirs(code_snapshot_dir, exist_ok=True)

    try:
        # 1. Try Git approach
        project_root = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()

        result = subprocess.run(
            ["git", "ls-files"],
            cwd=project_root,
            capture_output=True,
            text=True,
            check=True,
        )
        tracked_files = (
            result.stdout.strip().split("\n") if result.stdout.strip() else []
        )

        for tracked_file in tracked_files:
            if tracked_file:
                src = os.path.join(project_root, tracked_file)
                dst = os.path.join(code_snapshot_dir, tracked_file)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(src, dst)

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # 2. Fallback: Git failed (not a repo or git not installed)
        print(f"Warning: Git snapshot failed ({e}). Falling back to manual copy.")

        # Get current working directory as root
        root_dir = os.getcwd()

        # Walk through directory and copy relevant files
        for dirpath, dirnames, filenames in os.walk(root_dir):
            # Ignore hidden directories, logs, and cache
            dirnames[:] = [
                d
                for d in dirnames
                if not d.startswith(".")
                and d not in ["__pycache__", "lightning_logs", "wandb", "outputs"]
            ]

            for filename in filenames:
                if filename.endswith((".py", ".yaml", ".yml", ".sh", ".md")):
                    src_file = os.path.join(dirpath, filename)

                    # Calculate relative path to maintain structure
                    rel_path = os.path.relpath(src_file, root_dir)
                    dst_file = os.path.join(code_snapshot_dir, rel_path)

                    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                    shutil.copy2(src_file, dst_file)

