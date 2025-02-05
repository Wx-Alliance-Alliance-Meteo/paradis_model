#!/bin/bash

set -ex

rm -fr logs


root_dir=/home/stef/code/ERA5/ 

echo "Starting Phase 1: Initial training on 1 year of data"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.parameters.print_losses=True \
    training.dataset.start_date=2019-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.parameters.max_epochs=25 \
    training.parameters.loss_function.reversed=True \
    training.parameters.early_stopping.patience=5

PREV_CHECKPOINT="logs/lightning_logs/version_0/checkpoints/best.ckpt"

echo "Starting Phase 2: Training on 2 years of data"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.parameters.print_losses=True \
    training.dataset.start_date=2018-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.parameters.max_epochs=25 \
    training.parameters.loss_function.reversed=True \
    training.parameters.early_stopping.patience=5 \
    model.checkpoint_path=$PREV_CHECKPOINT

PREV_CHECKPOINT="logs/lightning_logs/version_1/checkpoints/best.ckpt"

echo "Starting Phase 3: Training on 5 years of data"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.parameters.print_losses=True \
    training.dataset.start_date=2015-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.parameters.max_epochs=25 \
    training.parameters.loss_function.reversed=True \
    training.parameters.early_stopping.patience=5 \
    model.checkpoint_path=$PREV_CHECKPOINT

PREV_CHECKPOINT="logs/lightning_logs/version_2/checkpoints/best.ckpt"

echo "Starting Phase 4: Training on 10 years"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.parameters.print_losses=True \
    training.dataset.start_date=2010-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.parameters.max_epochs=500 \
    training.parameters.loss_function.reversed=True \
    training.parameters.early_stopping.patience=8 \
    model.checkpoint_path=$PREV_CHECKPOINT

PREV_CHECKPOINT="logs/lightning_logs/version_3/checkpoints/best.ckpt"

echo "Starting Phase 5: Finetuning with lower learning rate"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.parameters.print_losses=True \
    training.dataset.start_date=2010-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.parameters.lr=1e-4 \
    training.parameters.max_epochs=500 \
    training.parameters.loss_function.reversed=False \
    training.parameters.early_stopping.patience=8 \
    model.checkpoint_path=$PREV_CHECKPOINT

PREV_CHECKPOINT="logs/lightning_logs/version_4/checkpoints/best.ckpt"

echo "Starting Phase 6: Final finetuning with more forecast steps"
python3.12 train.py \
    dataset.root_dir="${root_dir}" \
    training.parameters.print_losses=True \
    training.dataset.start_date=2010-01-01 \
    training.dataset.end_date=2019-12-31 \
    training.parameters.lr=1e-5 \
    training.parameters.max_epochs=500 \
    training.parameters.loss_function.reversed=False \
    training.parameters.early_stopping.patience=8 \
    model.forecast_steps=2 \
    model.checkpoint_path=$PREV_CHECKPOINT

echo "Training completed successfully!"
