#!/bin/bash

set -e

if [ -z "$1" ]; then
    echo "Error: Dataset directory path is required"
    echo "Usage: $0 /path/to/dataset/"
    exit 1
fi

root_dir=$1

if [ ! -d "$root_dir" ]; then
    echo "Error: Directory $root_dir does not exist"
    exit 1
fi

rm -fr logs

echo "Starting Phase 1: Training on 10 years"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.max_epochs=500 \
    training.print_losses=True \
    training.dataset.start_date=2010-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.optimizer.lr=1e-3 \
    training.scheduler.wsd.enabled=true


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
    model.checkpoint_path=$PREV_CHECKPOINT \
    training.scheduler.wsd.enabled=true

echo "Training completed successfully!"
