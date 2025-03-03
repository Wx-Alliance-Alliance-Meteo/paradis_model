"""Normalization and denormalization functions."""

import torch


def normalize_standard(data, mean, std):
    """Normalize data using Z-score normalization."""
    return (data - mean) / std


def denormalize_standard(norm_data, mean, std):
    """Denormalize data that was normalized using Z-score normalization."""
    return norm_data * std + mean


def normalize_humidity(data, q_min, q_max, eps=1e-12):
    """Normalize specific humidity using physically-motivated logarithmic transform.

    This normalization accounts for the exponential variation of specific humidity
    with altitude, mapping values from ~10^-5 (upper atmosphere) to ~10^-2 (surface)
    onto a normalized range while preserving relative variations at all scales.

    Args:
        data: Specific humidity data in kg/kg
        q_min: Minimum specific humidity value
        q_max: Maximum specific humidity value
        eps: Small positive constant to avoid log(0)

    Returns:
        Normalized specific humidity data
    """
    q_norm = (torch.log(torch.clip(data, 0, q_max) + eps) - torch.log(q_min)) / (
        torch.log(q_max) - torch.log(q_min)
    )

    return q_norm


def denormalize_humidity(data, q_min, q_max, eps=1e-12):
    """Denormalize specific humidity data from normalized space back to kg/kg.

    Args:
        data: Normalized specific humidity data
        q_min: Minimum specific humidity value used in normalization
        q_max: Maximum specific humidity value used in normalization
        eps: Small positive constant used in normalization

    Returns:
        Specific humidity data in kg/kg
    """
    q = torch.exp(data * (torch.log(q_max) - torch.log(q_min)) + torch.log(q_min)) - eps
    return torch.clip(q, min=0, max=q_max)


def normalize_precipitation(data, shift=10, eps=1e-6):
    """Normalize precipitation using logarithmic transform.

    Args:
        data: Precipitation data
        shift: Constant shift applied to log-transformed values
        eps: Small positive constant to avoid log(0)

    Returns:
        Normalized precipitation data
    """
    return torch.log(data + eps) + shift


def denormalize_precipitation(data, shift=10, eps=1e-6):
    """Denormalize precipitation data.

    Args:
        data: Normalized precipitation data
        shift: Constant shift applied during normalization
        eps: Small positive constant used in normalization

    Returns:
        Precipitation data in original scale
    """
    return torch.clip(torch.exp(data - shift) - eps, min=0)
