"""
Device abstraction module.

Transparently selects the best available compute backend:
  1. CUDA  (NVIDIA GPU)
  2. MPS   (Apple Silicon / Metal Performance Shaders)
  3. CPU   (fallback)

Usage:
    from utils.device import get_device, device_info
    device = get_device()
"""

from __future__ import annotations

import platform
import torch


def get_device() -> torch.device:
    """Return the best available torch device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_info() -> dict:
    """Return a dict describing the resolved device and environment."""
    device = get_device()
    info: dict = {
        "device": str(device),
        "platform": platform.system(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_count"] = torch.cuda.device_count()
        info["cuda_version"] = torch.version.cuda or "N/A"
    elif device.type == "mps":
        info["apple_silicon"] = True
    return info


def to_device(obj, device: torch.device | None = None):
    """Move a tensor / module to the resolved device."""
    if device is None:
        device = get_device()
    return obj.to(device)
