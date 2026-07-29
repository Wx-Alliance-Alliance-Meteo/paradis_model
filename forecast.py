import argparse
import logging
from omegaconf import OmegaConf
import lightning as L
import torch

from trainer import LitParadis
from data.datamodule import Era5DataModule

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
        "--num-devices", type=int, help="Number of devices", required=True
    )
    parser.add_argument(
        "--flush-every-n-steps", type=int, help="Write a forecast every n steps to reduce CPU memory usage", required=True,
    )

    parser.add_argument(
        "--num-workers", type=int, default=1, help="Number of dataloader workers"
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
    cfg.compute.num_workers = args.num_workers

    # Only supporting single node for now
    cfg.compute.num_nodes = 1

    # Restart continues the training from the current step
    cfg.init.restart = True

    datamodule = Era5DataModule(cfg)
    datamodule.setup(stage="predict")

    model = LitParadis(datamodule, cfg)

    trainer = L.Trainer(
        accelerator=cfg.compute.accelerator,
        devices=cfg.compute.num_devices,
        num_nodes=cfg.compute.num_nodes,
        precision="16-mixed" if cfg.compute.use_amp else "32-true",
        logger=False,
        enable_checkpointing=False,
    )

    trainer.predict(
        model,
        datamodule=datamodule,
        return_predictions=False,
        ckpt_path=cfg.init.checkpoint_path,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
