"""Automatic hardware configuration for optimal training parameters."""

from __future__ import annotations

import torch
from typing import Optional, Tuple


def get_optimal_batch_size(
    device: str = "cuda",
    model_size: str = "medium",
    available_memory_gb: Optional[float] = None,
) -> Tuple[int, int, int]:
    """
    Automatically determine optimal batch size and worker count.
    
    Args:
        device: "cuda" or "cpu"
        model_size: "small", "medium", or "large"
        available_memory_gb: Override detected memory
    
    Returns:
        (batch_size, eval_batch_size, num_workers)
    """
    if device == "cuda" and torch.cuda.is_available():
        if available_memory_gb is None:
            # Get GPU memory in GB
            available_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        
        # Base configurations for different model sizes and GPU memory
        # Format: (train_batch, eval_batch, workers)
        if model_size == "small":
            if available_memory_gb >= 40:  # A100 or similar
                return 256, 256, 8
            elif available_memory_gb >= 24:  # RTX 3090/4090, A5000
                return 192, 256, 8
            elif available_memory_gb >= 16:  # RTX 4080, A4000
                return 128, 192, 6
            elif available_memory_gb >= 12:  # RTX 3080, 4070 Ti
                return 96, 128, 4
            elif available_memory_gb >= 8:  # RTX 3070, 4060 Ti
                return 64, 96, 4
            else:  # Lower memory GPUs
                return 32, 64, 2
        
        elif model_size == "medium":
            if available_memory_gb >= 40:
                return 192, 256, 8
            elif available_memory_gb >= 24:
                return 128, 192, 8
            elif available_memory_gb >= 16:
                return 96, 128, 6
            elif available_memory_gb >= 12:
                return 64, 96, 4
            elif available_memory_gb >= 8:
                return 48, 64, 4
            else:
                return 24, 32, 2
        
        else:  # large
            if available_memory_gb >= 40:
                return 128, 192, 8
            elif available_memory_gb >= 24:
                return 64, 96, 8
            elif available_memory_gb >= 16:
                return 48, 64, 6
            elif available_memory_gb >= 12:
                return 32, 48, 4
            elif available_memory_gb >= 8:
                return 24, 32, 4
            else:
                return 16, 24, 2
    
    else:  # CPU
        import os
        cpu_count = os.cpu_count() or 4
        
        if model_size == "small":
            return 32, 64, min(cpu_count, 8)
        elif model_size == "medium":
            return 16, 32, min(cpu_count, 6)
        else:  # large
            return 8, 16, min(cpu_count, 4)


def get_mixed_precision_support(device: str = "cuda") -> bool:
    """Check if mixed precision training is supported."""
    if device == "cuda" and torch.cuda.is_available():
        # Check for compute capability >= 7.0 (Volta or newer)
        capability = torch.cuda.get_device_properties(0).major
        return capability >= 7
    return False


def print_hardware_info():
    """Print detected hardware information."""
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}")
        print(f"Memory: {props.total_memory / (1024**3):.1f} GB")
        print(f"Compute Capability: {props.major}.{props.minor}")
        print(f"Mixed Precision Supported: {get_mixed_precision_support('cuda')}")
    else:
        import os
        print("Device: CPU")
        print(f"CPU Cores: {os.cpu_count()}")


def auto_configure(device: Optional[str] = None, model_size: str = "medium") -> dict:
    """
    Automatically configure all training parameters.
    
    Args:
        device: Override device selection
        model_size: "small", "medium", or "large"
    
    Returns:
        Dictionary of recommended parameters
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    batch_size, eval_batch_size, num_workers = get_optimal_batch_size(device, model_size)
    mixed_precision = get_mixed_precision_support(device)
    
    return {
        "device": device,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "num_workers": num_workers,
        "mixed_precision": mixed_precision,
    }

