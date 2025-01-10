import os
import wandb
import torch
from typing import Optional
from pathlib import Path
import yaml

def _convert_config_to_dict(obj: object) -> dict:
    """
    Recursively convert a config object and its nested attributes to a dictionary.
    
    Args:
        obj: Config object or nested config object
        
    Returns:
        Dictionary representation of the config
    """
    if not hasattr(obj, '__dict__'):
        return obj
        
    result = {}
    for key, value in obj.__dict__.items():
        if not key.startswith('_'):  # Skip private attributes
            if hasattr(value, '__dict__'):
                result[key] = _convert_config_to_dict(value)
            else:
                result[key] = value
    return result

def init_wandb(
    project_name: str,
    run_name: str,
    config_class: object,
    save_path: str
) -> None:
    """
    Initialize a new WandB run.
    
    Args:
        project_name: Name of the WandB project
        run_name: Custom name for this run
        config_class: Configuration class instance
        save_path: Path to save configuration to locally
    """
    # Convert config class to dictionary
    config_dict = _convert_config_to_dict(config_class)
    
    wandb.init(
        project=project_name,
        name=run_name,
        config=config_dict
    )    
    
    # Update wandb config
    wandb.config.update(config_dict)
    
    # Save locally as YAML
    with open(save_path+'config_mod.yaml', 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    # Upload to wandb
    wandb.save(save_path+'config_mod.yaml')
    
    # Also log config as wandb summary
    wandb.summary.update({"model_config": config_dict})

def log_losses(
    train_loss: float,
    valid_loss: float,
    step: Optional[int] = None,
    epoch: Optional[int] = None,
) -> None:
    """
    Log training and validation losses to WandB.
    
    Args:
        train_loss: Training loss value
        valid_loss: Validation loss value
        step: Current training step (optional)
        epoch: Current epoch number (optional)
    """
    log_dict = {
        'train_loss': train_loss,
        'valid_loss': valid_loss
    }
    
    # Add step/epoch info if provided
    if step is not None:
        log_dict['step'] = step
    if epoch is not None:
        log_dict['epoch'] = epoch
    
    wandb.log(log_dict)
    
def save_model_architecture(model: torch.nn.Module, save_path: str) -> None:
    """
    Save model architecture to a text file and log it to WandB.
    
    Args:
        model: PyTorch model
        save_path: Path where to save the architecture file
    """
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Get model architecture as string
    model_str = str(model)
    
    # Save to file
    with open(save_path+'arch.txt', 'w') as f:
        f.write(model_str)
        
    wandb.save(save_path+'arch.txt')
    
    # Also log as a wandb summary
    wandb.summary['model_architecture'] = model_str

def finish_run():
    """
    Properly finish the WandB run.
    """
    wandb.finish()