import os # Intel GPUs

# PyTorch settings for specific GPU vendors
torch_wheel = os.environ.get("TORCH_WHEEL","cuda") # get environment variable for PyTorch wheel (NVIDIA: "cuda", Intel: "xpu")
if torch_wheel == "xpu":
    os.environ["TORCH_ATTENTION_MODE"] = "sdpa" # Intel GPUs
    os.environ["TORCH_COMPILE_DISABLE"] = "1"   # Intel GPUs

import argparse
import logging
from omegaconf import OmegaConf
import lightning as L
import torch

from trainer import LitParadis
from data.datamodule import Era5DataModule
from utils.Intel_GPU_utils import SingleXPUStrategy # Intel GPUs

torch.set_float32_matmul_precision("high")

def parse_args():
    parser = argparse.ArgumentParser(description="Run forecasts with a trained model.")

    parser.add_argument("--config", help="Path to config YAML", required=True)
    parser.add_argument(
        "--checkpoint-path", help="Path to model checkpoint", required=True
    )
    parser.add_argument("--output-file", help="Output Zarr path", required=True)

    parser.add_argument("--root-dir", default=None, help="Override root dir")
    parser.add_argument(
        "--forecast-steps", type=int, default=40, help="Autoregressive forecast steps"
    )
    parser.add_argument(
        "--sampling-interval",
        type=str,
        default="36h",
        help='Dataset sampling interval, e.g. "36h"',
    )

    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Forecast start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="Forecast end date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Prediction batch size"
    )
    parser.add_argument(
        "--num-devices", type=int, default=1, help="Number of devices", required=True
    )
    parser.add_argument(
        "--flush-every-n-steps", type=int, default=0, help="Write a forecast every n steps to reduce CPU memory usage", required=True,
    )

    return parser.parse_args()


def main():

    args = parse_args()

    cfg = OmegaConf.load(args.config)
    cfg.forecast.enable = True

    cfg.init.checkpoint_path = args.checkpoint_path
    cfg.forecast.output_file = args.output_file

    if args.root_dir is not None:
        cfg.dataset.root_dir = args.root_dir

    cfg.model.forecast_steps = args.forecast_steps
    cfg.dataset.sampling_interval = args.sampling_interval

    cfg.forecast.start_date = args.start_date
    cfg.forecast.end_date = args.end_date
    cfg.forecast.write_every_n = args.flush_every_n_steps

    cfg.compute.batch_size = args.batch_size
    cfg.compute.num_devices = args.num_devices

    cfg.compute.use_amp = False
    cfg.compute.num_workers = 4
    cfg.compute.compile = True

    # Only supporting single node for now
    cfg.compute.num_nodes = 1

    # Restart true must be set so that the checkpoint isn't loaded twice
    cfg.init.restart = True

    # Enable forecasting mode
    cfg.forecast.enable = True

    datamodule = Era5DataModule(cfg)
    datamodule.setup(stage="predict")

    model = LitParadis(datamodule, cfg)

    # Keyword arguments common to all hardware
    trainer_kwargs = {
        "devices": cfg.compute.num_devices,
        "num_nodes": cfg.compute.num_nodes,
        "precision": "16-mixed" if cfg.compute.use_amp else "32-true",
        "logger": False,
        "enable_checkpointing": False
    }

    # Keyword arguments that depend on GPU hardware choice
    if cfg.compute.Intel_GPU:
        xpu_strategy = SingleXPUStrategy(device_index=0)
        trainer_kwargs.update({
            "strategy": xpu_strategy
        })
    else:
        trainer_kwargs.update({
            "accelerator": cfg.compute.accelerator
        })

    # Instantiate lightning trainer with options
    trainer = L.Trainer(**trainer_kwargs)

    trainer.predict(
        model,
        datamodule=datamodule,
        return_predictions=False,
        ckpt_path=cfg.init.checkpoint_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
