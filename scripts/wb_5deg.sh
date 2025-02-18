#!/bin/bash

set -e

rm -fr logs

root_dir=/home/stef/code/ERA5/ 

echo "Starting Phase 1: Initial warmup on 3 year of data"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.parameters.print_losses=True \
    training.dataset.start_date=2016-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.parameters.scheduler.lr_final_div=1 \
    training.parameters.max_epochs=25 \

PREV_CHECKPOINT="logs/lightning_logs/version_0/checkpoints/best.ckpt"

echo "Starting Phase 2: Training on 10 years"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.parameters.print_losses=True \
    training.dataset.start_date=2010-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.parameters.lr=1e-3 \
    training.parameters.max_epochs=500 \
    model.checkpoint_path=$PREV_CHECKPOINT

PREV_CHECKPOINT="logs/lightning_logs/version_3/checkpoints/best.ckpt"

echo "Starting Phase 3: Final finetuning with more forecast steps"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.parameters.print_losses=True \
    training.dataset.start_date=2010-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.parameters.lr=1e-4 \
    training.parameters.max_epochs=500 \
    model.forecast_steps=2 \
    model.checkpoint_path=$PREV_CHECKPOINT

echo "Training completed successfully!"
