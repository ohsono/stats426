"""
Learning rate schedulers for curriculum training.

Provides factory functions for creating schedulers that align with the
curriculum stages defined in Phase 3.
"""

from __future__ import annotations

from typing import Optional

import torch.optim as optim


def build_scheduler(
    optimizer: optim.Optimizer,
    name: str = "cosine",
    total_epochs: int = 40,
    warmup_epochs: int = 5,
    eta_min: float = 1e-6,
) -> Optional[optim.lr_scheduler._LRScheduler]:
    """
    Build a learning rate scheduler by name.

    Supported schedulers:
        • ``cosine``   — CosineAnnealingLR (per Phase 3 spec).
        • ``step``     — StepLR with step_size = total_epochs // 3.
        • ``plateau``  — ReduceLROnPlateau (monitors val_loss).
        • ``none``     — No scheduler.
    """
    name = name.lower()

    if name == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_epochs, eta_min=eta_min
        )
    elif name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, total_epochs // 3), gamma=0.1
        )
    elif name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5
        )
    elif name == "none":
        return None
    else:
        raise ValueError(f"Unknown scheduler: {name}")


def build_warmup_cosine_scheduler(
    optimizer: optim.Optimizer,
    warmup_epochs: int = 5,
    total_epochs: int = 40,
    eta_min: float = 1e-6,
) -> optim.lr_scheduler.SequentialLR:
    """
    Warmup + Cosine Annealing in sequence:
        • Linear warmup for ``warmup_epochs``.
        • Cosine decay for the remainder.
    """
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, total_iters=warmup_epochs
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=total_epochs - warmup_epochs, eta_min=eta_min
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
    )
