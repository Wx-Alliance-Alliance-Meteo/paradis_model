#!/usr/bin/env python
"""
Sensitivity Analysis for Paradis Weather Model

This script performs Jacobian-based sensitivity analysis to understand
how input features influence output predictions in the Paradis model.
It computes gradients of model outputs with respect to inputs and
visualizes the results as a heatmap.
"""

import argparse
import os
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from data.datamodule import Era5DataModule
from trainer import LitParadis
from utils.system import setup_system

# Constants
LOG_EPSILON = 1e-10  # Small epsilon to avoid log(0)
TOP_FEATURES_COUNT = 15  # Number of top/least influential features to show
TOP_PAIRS_COUNT = 20  # Number of top input-output pairs to show
TOP_OUTPUTS_COUNT = 10  # Number of most/least influenced outputs to show
MAX_VAR_TYPES = 15  # Maximum number of variable types to show in analysis


def load_model_and_config(
    config: dict, force_cpu: bool = False
) -> Tuple[LitParadis, DictConfig, Era5DataModule, torch.device]:
    """Load the model, configuration, and data module."""
    print("Loading model configuration...")
    cfg = OmegaConf.load(config["config_path"])

    # Override configuration values
    cfg.forecast.start_date = config["start_date"]
    cfg.forecast.end_date = config["end_date"]
    cfg.compute.num_workers = 1
    cfg.compute.num_nodes = 1
    cfg.compute.batch_size = config["batch_size"]
    cfg.compute.compile = config["compile_model"]

    setup_system(cfg)

    # Setup data module
    print("Setting up data module...")
    datamodule = Era5DataModule(cfg)
    datamodule.setup(stage="predict")

    # Load model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model on device: {device}")

    litmodel = LitParadis.load_from_checkpoint(
        config["model_path"], cfg=cfg, datamodule=datamodule
    ).to(device)

    return litmodel, cfg, datamodule, device


def create_dataloader(datamodule: Era5DataModule, cfg: DictConfig) -> DataLoader:
    """Create and return a DataLoader for the dataset."""
    return DataLoader(
        datamodule.dataset,
        batch_size=cfg.compute.batch_size,
        shuffle=True,
        pin_memory=torch.cuda.is_available(),
        num_workers=cfg.compute.num_workers,
    )


def compute_jacobian_matrix(
    litmodel: LitParadis, dataloader: DataLoader, max_batches: int, device: torch.device, 
    n_in: int, n_out: int, abs_input: bool = True, abs_output: bool = True, normalize: bool = True
) -> torch.Tensor:
    """Compute the Jacobian matrix showing input-output sensitivities.

    Args:
        litmodel: The trained Paradis model
        dataloader: DataLoader for input data
        max_batches: Maximum number of batches to process
        device: Torch device for computation
        abs_input: If True, take absolute value of gradients (input sensitivity magnitudes).
                  If False, preserve gradient signs for directional sensitivity.
        abs_output: If True, take absolute value of outputs before spatial averaging.
                   If False, preserve output signs (may cause spatial cancellation).
        normalize: If True, normalize by input standard deviations computed from the data.

    Returns:
        Jacobian matrix of shape (n_out, n_in), normalized if normalize=True
    """
    
    J = torch.zeros((n_out, n_in), device=device)
    w = litmodel.loss_fn.lat_weights.view(1, 1, -1, 1).to(device)
    
    # For computing input standard deviations over space and time
    input_sum = torch.zeros(n_in, device=device)
    input_sum_sq = torch.zeros(n_in, device=device)
    total_elements = 0

    def model_avg(tensor: torch.Tensor) -> torch.Tensor:
        """Compute weighted spatial average of model output."""
        return (tensor * w).mean((0, 2, 3))

    processed_batches = 0

    for idx_batch, batch in enumerate(dataloader):
        if processed_batches >= max_batches:
            break

        print(f"Processing batch {idx_batch + 1}/{max_batches}")
        
        inputs, _ = batch
        batch_size = inputs.shape[0]

        # Prepare input with gradient tracking
        litmodel.zero_grad()
        model_input = inputs[:, 0].detach().clone().to(device).requires_grad_()
    
        # Compute running statistics for std over space and time
        with torch.no_grad():
            # Apply latitude weights first, then flatten
            # model_input shape: (batch_size, n_in, lat, lon)
            weighted_input = model_input * w  # Apply weights to spatial dimensions
            flattened = weighted_input.view(batch_size, n_in, -1)  # (batch_size, n_in, lat*lon)
            
            # Sum over batch and spatial dimensions
            batch_sum = flattened.sum(dim=(0, 2))  # Shape: (n_in,)
            batch_sum_sq = (flattened ** 2).sum(dim=(0, 2))  # Shape: (n_in,)
            
            input_sum += batch_sum
            input_sum_sq += batch_sum_sq
            total_elements += batch_size * flattened.shape[2]  # batch_size * (lat * lon)
        
        # Compute model output once per batch
        model_output = litmodel(model_input)
        if abs_output:
            avg_out = model_avg(model_output.abs())
        else:
            avg_out = model_avg(model_output)

        # Compute gradients for each output (back to sequential for stability)
        for i in range(n_out):
            if model_input.grad is not None:
                model_input.grad.zero_()

            avg_out[i].backward(retain_graph=(i != n_out - 1))

            if model_input.grad is not None:
                if abs_input:
                    grad_contribution = model_avg(model_input.grad.abs())
                else:
                    grad_contribution = model_avg(model_input.grad)
                
                J[i, :] += grad_contribution
            
        processed_batches += 1

    print(f"Processed {processed_batches} batches successfully")
    
    # Compute input standard deviations efficiently using running statistics over space and time
    if normalize:
        print("Computing input standard deviations from running statistics over space and time...")
        input_mean = input_sum / total_elements
        input_variance = (input_sum_sq / total_elements) - (input_mean ** 2)
        input_std = torch.sqrt(input_variance + 1e-8)  # Add small epsilon for numerical stability
        
        print(f"Computed standard deviations: min={input_std.min():.6f}, max={input_std.max():.6f}")
        
        # Apply normalization
        J = J / input_std.unsqueeze(0)
    
    return J


def generate_feature_names(
    datamodule: Era5DataModule, cfg: DictConfig
) -> Tuple[List[str], List[str]]:
    """Generate input and output feature names for labeling.

    Args:
        datamodule: The data module containing feature information
        cfg: Configuration object

    Returns:
        Tuple of (input_names, output_names)
    """
    # Input feature names with time indices
    input_names = []
    input_names += [f"{f}_t_n-2" for f in datamodule.dataset.ds_input.features.values]
    input_names += [f"{f}_t_n-1" for f in datamodule.dataset.ds_input.features.values]
    input_names += [
        f"{f}_t_n-{2-i}" for f in datamodule.dataset.forcing_inputs for i in range(2)
    ]
    input_names += list(cfg.features.input.constants)

    # Output feature names
    output_names = [str(f) for f in datamodule.dataset.ds_input.features.values]

    return input_names, output_names


def load_sensitivity_from_csv(
    csv_path: str,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Load sensitivity data from a CSV file.

    Args:
        csv_path: Path to the CSV file containing sensitivity data

    Returns:
        Tuple of (dataframe, input_names, output_names)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading sensitivity data from {csv_path}")
    df = pd.read_csv(csv_path, index_col=0)

    input_names = df.index.tolist()
    output_names = df.columns.tolist()

    print(
        f"Loaded sensitivity matrix: {len(input_names)} inputs × {len(output_names)} outputs"
    )
    return df, input_names, output_names


def generate_sensitivity_summary(
    sensitivity_df: pd.DataFrame, summary_filename: str
) -> None:
    """Generate and save a text summary of sensitivity analysis results.

    Args:
        sensitivity_df: DataFrame containing sensitivity data
        summary_filename: Output filename for the summary
    """
    print("Generating sensitivity summary...")

    # Calculate overall sensitivity scores for each input
    input_importance = sensitivity_df.mean(axis=1).sort_values(ascending=False)

    # Calculate overall influence scores for each output
    output_influence = sensitivity_df.mean(axis=0).sort_values(ascending=False)

    # Find most influential input-output pairs
    sensitivity_flat = sensitivity_df.stack().sort_values(ascending=False)

    with open(summary_filename, "w") as f:
        f.write("SENSITIVITY ANALYSIS SUMMARY\n")
        f.write("=" * 50 + "\n\n")

        # Overall statistics
        f.write(f"Analysis Overview:\n")
        f.write(f"- Total input features: {len(sensitivity_df.index)}\n")
        f.write(f"- Total output features: {len(sensitivity_df.columns)}\n")
        f.write(f"- Mean sensitivity: {sensitivity_df.mean().mean():.6f}\n")
        f.write(f"- Max sensitivity: {sensitivity_df.max().max():.6f}\n")
        f.write(f"- Min sensitivity: {sensitivity_df.min().min():.6f}\n\n")

        # Most influential inputs (overall)
        f.write("TOP 15 MOST INFLUENTIAL INPUT FEATURES:\n")
        f.write("-" * 40 + "\n")
        for i, (input_name, score) in enumerate(
            input_importance.head(TOP_FEATURES_COUNT).items(), 1
        ):
            f.write(f"{i:2d}. {input_name:<40} {score:8.6f}\n")

        f.write("\nTOP 15 LEAST INFLUENTIAL INPUT FEATURES:\n")
        f.write("-" * 40 + "\n")
        least_influential = input_importance.sort_values(ascending=True).head(
            TOP_FEATURES_COUNT
        )
        for i, (input_name, score) in enumerate(least_influential.items(), 1):
            f.write(f"{i:2d}. {input_name:<40} {score:8.6f}\n")

        # Most influenced outputs
        f.write("\nMOST INFLUENCED OUTPUT FEATURES:\n")
        f.write("-" * 40 + "\n")
        for i, (output_name, score) in enumerate(
            output_influence.head(TOP_OUTPUTS_COUNT).items(), 1
        ):
            f.write(f"{i:2d}. {output_name:<30} {score:8.6f}\n")

        f.write("\nLEAST INFLUENCED OUTPUT FEATURES:\n")
        f.write("-" * 40 + "\n")
        least_influenced_outputs = output_influence.sort_values(ascending=True).head(
            TOP_OUTPUTS_COUNT
        )
        for i, (output_name, score) in enumerate(least_influenced_outputs.items(), 1):
            f.write(f"{i:2d}. {output_name:<30} {score:8.6f}\n")

        # Top input-output pairs
        f.write("\nTOP 20 MOST SENSITIVE INPUT-OUTPUT PAIRS:\n")
        f.write("-" * 60 + "\n")
        f.write(
            "Rank  Input Feature                    Output Feature           Sensitivity\n"
        )
        f.write("-" * 75 + "\n")
        for i, (index, score) in enumerate(
            sensitivity_flat.head(TOP_PAIRS_COUNT).items(), 1
        ):
            input_name, output_name = index
            f.write(f"{i:2d}.   {input_name:<30} -> {output_name:<20} {score:8.6f}\n")

    print(f"Sensitivity summary saved to {summary_filename}")


def create_heatmap_plot(sensitivity_df: pd.DataFrame, plot_filename: str) -> None:
    """Create and save the sensitivity heatmap plot.

    Args:
        sensitivity_df: DataFrame containing sensitivity data in original scale
        plot_filename: Output filename for the plot
    """
    print("Creating sensitivity heatmap...")

    # Create the plot with log transformation for visualization only
    plt.figure(figsize=(30, 28))

    # Apply log10 transformation only for the plot
    df_log = np.log10(sensitivity_df + LOG_EPSILON)  # Add small epsilon to avoid log(0)
    
    # Create heatmap with log-transformed data
    sns.heatmap(
        df_log,
        cmap="viridis",
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Log10(Normalized Sensitivity)"},
    )

    # Customize plot
    plt.xlabel("Output Features", fontsize=14)
    plt.ylabel("Input Features", fontsize=14)
    title = "Jacobian Matrix Heatmap: Input-Output Feature Sensitivity"
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    # Save with high DPI
    plt.tight_layout()
    plt.savefig(plot_filename, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Visualization saved to {plot_filename}")


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Perform sensitivity analysis on Paradis weather model"
    )

    parser.add_argument("--run-path", type=str, help="Path to model run directory")

    parser.add_argument(
        "--max-batches",
        type=int,
        default=10,
        help="Maximum number of batches to process",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="sensitivity",
        help="Output filename for visualization",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for data loading (default: 1)",
    )

    parser.add_argument(
        "--load-csv",
        type=str,
        help="Load sensitivity data from existing CSV file instead of recomputing",
    )

    parser.add_argument(
        "--normalize-by-std",
        action="store_true",
        default=True,
        help="Normalize sensitivity by input standard deviations (default: True)",
    )

    parser.add_argument(
        "--no-abs-output",
        dest="abs_output",
        action="store_false",
        default=True,
        help="Preserve output signs in spatial averaging (may cause cancellation)",
    )
    
    parser.add_argument(
        "--no-abs-input", 
        dest="abs_input",
        action="store_false",
        default=True,
        help="Preserve gradient signs for directional sensitivity",
    )

    return parser.parse_args()
        

def main() -> None:
    """Main function to run sensitivity analysis."""
    args = parse_arguments()

    # Initialize configuration
    config = {
        "input_data_path": "/home/cap003/hall6/weatherbench_paradis/era5_1deg_13level/",
        "run_path": "/home/saz001/ss5/paradis_studies/week_2025_06_16/logs/lightning_logs/run_7_66hour_new_control_run_lr2e-3_ls966_ds_966_vv322_300Ksteps",
        "start_date": "2022-01-01",
        "end_date": "2022-12-31",
        "compile_model": False,
    }

    if args.run_path:
        config["run_path"] = args.run_path

    config["model_path"] = os.path.join(config["run_path"], "checkpoints", "best.ckpt")
    config["config_path"] = os.path.join(config["run_path"], "config.yaml")

    config["max_batches"] = args.max_batches
    config["batch_size"] = args.batch_size
    config["output_filename"] = args.output

    if args.load_csv:
        print("Loading sensitivity data from CSV file")
        sensitivity_df, input_names, output_names = load_sensitivity_from_csv(args.load_csv)

        create_heatmap_plot(sensitivity_df, config["output_filename"])

        summary_filename = f'{config["output_filename"]}.txt'
        generate_sensitivity_summary(sensitivity_df, summary_filename)

    else:
        # Load model and data
        litmodel, cfg, datamodule, device = load_model_and_config(config)
        dataloader = create_dataloader(datamodule, cfg)

        print(f"Dataset contains {len(dataloader)} batches")

        # Generate feature names
        input_names, output_names = generate_feature_names(datamodule, cfg)
        n_in = len(input_names)
        n_out = len(output_names)

        print(f"Input channels: {n_in}, Output channels: {n_out}")
        
        # Compute sensitivity analysis with retry capability
        jacobian_matrix = compute_jacobian_matrix(
            litmodel, dataloader, config["max_batches"], device, 
            n_in, n_out, args.abs_input, args.abs_output, args.normalize_by_std)

        # Convert to numpy and normalize to [0,1] for comparison
        J_numpy = jacobian_matrix.abs().T.cpu().numpy()
        J_normalized = J_numpy / J_numpy.max()

        # Create DataFrame with normalized data
        df = pd.DataFrame(J_normalized, index=input_names, columns=output_names)

        # Create the heatmap plot using shared function
        create_heatmap_plot(df, f'{config["output_filename"]}.png')

        # Also save the raw data
        csv_filename = f'{config["output_filename"]}.csv'
        df.to_csv(csv_filename)
        print(f"Raw sensitivity data saved to {csv_filename}")

        # Generate and save summary
        summary_filename = f'{config["output_filename"]}.txt'
        generate_sensitivity_summary(df, summary_filename)

    print("Sensitivity analysis completed successfully")


if __name__ == "__main__":
    main()
