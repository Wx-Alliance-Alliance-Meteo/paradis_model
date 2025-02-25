#!/bin/bash

set -e

rm -fr logs

root_dir=/path/to/dataset/

echo "Starting Phase 1: Training on 10 years"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.max_epochs=500 \
    training.print_losses=True \
    training.dataset.start_date=2010-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.optimizer.lr=1e-3 \


PREV_CHECKPOINT="logs/lightning_logs/version_0/checkpoints/best.ckpt"

echo "Starting Phase 2: Final finetuning with more forecast steps"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.print_losses=True \
    training.max_epochs=500 \
    training.dataset.start_date=2010-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.optimizer.lr=1e-4 \
    model.forecast_steps=2 \
    model.checkpoint_path=$PREV_CHECKPOINT

echo "Training completed successfully!"
