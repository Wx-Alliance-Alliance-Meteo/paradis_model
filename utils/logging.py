"""Logging utilities for the model.

This module provides comprehensive logging functionality including:
- File and console logging
- Training metrics tracking
- TensorBoard support
- CSV logging for metrics
- Pretty progress bar
"""

import csv
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union, TextIO

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class MetricsLogger:
    """Handles logging of training/validation metrics."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
    ) -> None:
        """Initialize metrics logger.
        
        Args:
            log_dir: Directory to store logs
            experiment_name: Name of current experiment
            use_tensorboard: Whether to use TensorBoard logging
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Create log directory
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup CSV logging
        self.csv_path = self.log_dir / f"{experiment_name}_metrics.csv"
        self.csv_file = open(self.csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['epoch', 'step', 'metric', 'value'])
        
        # Setup TensorBoard
        self.use_tensorboard = use_tensorboard
        if use_tensorboard:
            self.tb_writer = SummaryWriter(
                log_dir=str(self.log_dir / 'tensorboard' / experiment_name)
            )

    def log_metric(
        self,
        metric_name: str,
        value: float,
        epoch: int,
        step: Optional[int] = None,
    ) -> None:
        """Log a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            epoch: Current epoch number
            step: Optional step within epoch
        """
        # Log to CSV
        self.csv_writer.writerow([epoch, step or 0, metric_name, value])
        self.csv_file.flush()
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_scalar(
                f"metrics/{metric_name}",
                value,
                epoch if step is None else epoch * step
            )

    def log_model_graph(
        self,
        model: torch.nn.Module,
        input_size: tuple
    ) -> None:
        """Log model architecture to TensorBoard.
        
        Args:
            model: Model to log
            input_size: Input tensor size for model visualization
        """
        if self.use_tensorboard:
            self.tb_writer.add_graph(
                model,
                torch.zeros(input_size, dtype=torch.float32)
            )

    def log_hyperparameters(self, hparams: Dict[str, Any]) -> None:
        """Log hyperparameters.
        
        Args:
            hparams: Dictionary of hyperparameters
        """
        # Save to JSON
        with open(self.log_dir / f"{self.experiment_name}_hparams.json", 'w') as f:
            json.dump(hparams, f, indent=2)
        
        # Log to TensorBoard
        if self.use_tensorboard:
            self.tb_writer.add_hparams(hparams, {})

    def close(self) -> None:
        """Close all open file handles."""
        self.csv_file.close()
        if self.use_tensorboard:
            self.tb_writer.close()


class TrainingLogger:
    """Handles logging during model training."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ) -> None:
        """Initialize training logger.
        
        Args:
            log_dir: Directory to store logs
            experiment_name: Name of current experiment
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        
        # Create logger
        self.logger = logging.getLogger(experiment_name)
        self.logger.setLevel(logging.DEBUG)
        
        # Create formatters
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler
        self.log_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(
            self.log_dir / f"{experiment_name}_{datetime.now():%Y%m%d_%H%M%S}.log"
        )
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # Progress bar
        self.pbar = None

    def info(self, msg: str) -> None:
        """Log info message.
        
        Args:
            msg: Message to log
        """
        self.logger.info(msg)

    def debug(self, msg: str) -> None:
        """Log debug message.
        
        Args:
            msg: Message to log
        """
        self.logger.debug(msg)

    def warning(self, msg: str) -> None:
        """Log warning message.
        
        Args:
            msg: Message to log
        """
        self.logger.warning(msg)

    def error(self, msg: str) -> None:
        """Log error message.
        
        Args:
            msg: Message to log
        """
        self.logger.error(msg)

    def start_epoch(self, epoch: int, total_epochs: int) -> None:
        """Start logging a new epoch.
        
        Args:
            epoch: Current epoch number
            total_epochs: Total number of epochs
        """
        if self.pbar is not None:
            self.pbar.close()
        
        self.pbar = tqdm(
            total=total_epochs,
            initial=epoch,
            desc=f"Epoch {epoch}/{total_epochs}",
            unit="epoch"
        )

    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update progress bar with current metrics.
        
        Args:
            metrics: Dictionary of metric names and values
        """
        if self.pbar is not None:
            self.pbar.set_postfix(metrics)

    def end_epoch(self) -> None:
        """End current epoch logging."""
        if self.pbar is not None:
            self.pbar.update(1)

    def close(self) -> None:
        """Close logger and progress bar."""
        if self.pbar is not None:
            self.pbar.close()


def create_experiment_logger(
    experiment_name: str,
    base_dir: str = "logs",
    use_tensorboard: bool = True,
) -> tuple[TrainingLogger, MetricsLogger]:
    """Create loggers for a new experiment.
    
    Args:
        experiment_name: Name of the experiment
        base_dir: Base directory for logs
        use_tensorboard: Whether to use TensorBoard logging
        
    Returns:
        Tuple of (TrainingLogger, MetricsLogger)
    """
    # Create timestamped experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(base_dir) / f"{experiment_name}_{timestamp}"
    
    # Create loggers
    training_logger = TrainingLogger(log_dir, experiment_name)
    metrics_logger = MetricsLogger(log_dir, experiment_name, use_tensorboard)
    
    return training_logger, metrics_logger


def write_stats(
    filename: Union[str, TextIO],
    epoch: int,
    loss: float,
    metrics: Optional[Dict[str, float]] = None,
    mode: str = 'a'
) -> None:
    """Write training statistics to file.
    
    Args:
        filename: Output file path or file object
        epoch: Current epoch number
        loss: Loss value to record
        metrics: Optional additional metrics to record
        mode: File opening mode
    """
    metrics = metrics or {}
    metrics['loss'] = loss
    
    if isinstance(filename, str):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, mode) as f:
            _write_stats_line(f, epoch, metrics)
    else:
        _write_stats_line(filename, epoch, metrics)


def _write_stats_line(f: TextIO, epoch: int, metrics: Dict[str, float]) -> None:
    """Write a single line of stats to file.
    
    Args:
        f: File object to write to
        epoch: Current epoch number
        metrics: Dictionary of metrics to record
    """
    metric_str = ','.join(f"{k}={v:.6f}" for k, v in metrics.items())
    f.write(f"Epoch {epoch}: {metric_str}\n")
