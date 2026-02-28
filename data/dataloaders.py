"""
DataLoader factories implementing the 70-10-10-10 split strategy and
weighted sampling for class-imbalanced datasets.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from data.datasets import TrafficSignDataset, UnifiedTrafficSignDataset
from utils.config import DataConfig


# ---------------------------------------------------------------------------
# Split helper
# ---------------------------------------------------------------------------

def stratified_split(
    dataset: TrafficSignDataset,
    ratios: Tuple[float, ...] = (0.70, 0.10, 0.10, 0.10),
    seed: int = 42,
) -> List[Subset]:
    """
    Split a dataset into N subsets with the given ratios using a
    deterministic random generator.

    Returns a list of ``torch.utils.data.Subset`` in the same order as
    the ratios tuple (train, val, test, ood).
    """
    assert abs(sum(ratios) - 1.0) < 1e-6, "Ratios must sum to 1.0"

    n = len(dataset)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=generator).tolist()

    subsets: List[Subset] = []
    start = 0
    for r in ratios:
        end = start + int(round(n * r))
        end = min(end, n)
        subsets.append(Subset(dataset, indices[start:end]))
        start = end

    # Assign remaining to last subset (rounding residual)
    if start < n:
        subsets[-1] = Subset(dataset, indices[start - int(round(n * ratios[-1])):n])

    return subsets


# ---------------------------------------------------------------------------
# Weighted sampler
# ---------------------------------------------------------------------------

def build_weighted_sampler(
    labels: List[int],
) -> WeightedRandomSampler:
    """
    Build a ``WeightedRandomSampler`` that oversamples minority classes
    and undersamples majority classes (per Phase 1 spec for BDD100K).
    """
    counts = Counter(labels)
    total = sum(counts.values())
    weights = [total / counts[l] for l in labels]
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(labels),
        replacement=True,
    )


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def create_dataloaders(
    dataset: TrafficSignDataset | UnifiedTrafficSignDataset,
    config: Optional[DataConfig] = None,
    use_weighted_sampler: bool = False,
    seed: int = 42,
) -> Dict[str, DataLoader]:
    """
    Create train / val / test / ood DataLoaders from a single dataset
    using the 70-10-10-10 split.

    Parameters
    ----------
    dataset
        The dataset to split.
    config
        DataConfig with batch_size, num_workers, etc.
    use_weighted_sampler
        If True, apply ``WeightedRandomSampler`` on the **training** split.
    seed
        Random seed for reproducible splits.

    Returns
    -------
    dict
        Keys: ``"train"``, ``"val"``, ``"test"``, ``"ood"``.
    """
    if config is None:
        config = DataConfig()

    ratios = (config.train_ratio, config.val_ratio, config.test_ratio, config.ood_ratio)
    train_sub, val_sub, test_sub, ood_sub = stratified_split(dataset, ratios, seed)

    # Build training sampler
    train_sampler = None
    shuffle_train = True
    if use_weighted_sampler:
        train_labels = [dataset.get_labels()[i] for i in train_sub.indices]
        train_sampler = build_weighted_sampler(train_labels)
        shuffle_train = False  # Sampler and shuffle are mutually exclusive

    loader_kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
    )

    loaders: Dict[str, DataLoader] = {
        "train": DataLoader(
            train_sub,
            shuffle=shuffle_train if train_sampler is None else False,
            sampler=train_sampler,
            **loader_kwargs,
        ),
        "val": DataLoader(val_sub, shuffle=False, **loader_kwargs),
        "test": DataLoader(test_sub, shuffle=False, **loader_kwargs),
        "ood": DataLoader(ood_sub, shuffle=False, **loader_kwargs),
    }
    return loaders
