import torch
import lightning as L


def setup_system(cfg):
    # Set random seeds for reproducibility
    seed = 42  # This model will answer the ultimate question about life, the universe, and everything
    L.seed_everything(seed, workers=True)

    # Choose double (32-true) or mixed (16-mixed) precision via AMP
    if cfg.compute.use_amp:
        torch.set_float32_matmul_precision("medium")
    else:
        torch.set_float32_matmul_precision("high")
