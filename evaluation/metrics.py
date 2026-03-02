"""
Per-class Precision, Recall, and F1-Score computation (Phase 4.2).

Wraps scikit-learn classification_report with special emphasis on
safety-critical sign classes (stop, yield, donotenter).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data.unify import DOT_CLASSES, global_index_to_name
from utils.device import get_device


def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference on the entire loader and return
    ``(all_labels, all_preds, all_probs)`` as numpy arrays.
    """
    if device is None:
        device = get_device()
    model.eval()
    all_labels, all_preds, all_probs = [], [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    return (
        np.concatenate(all_labels),
        np.concatenate(all_preds),
        np.concatenate(all_probs),
    )


def classification_report_dict(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Dict:
    """
    Generate a classification report as a dict.

    Falls back to a simple implementation if scikit-learn is not available.
    """
    try:
        from sklearn.metrics import classification_report
        present = sorted(set(y_true.tolist()) | set(y_pred.tolist()))
        if class_names is None:
            target_names = [global_index_to_name(i) for i in present]
        else:
            target_names = [class_names[i] if i < len(class_names) else f"class_{i}" for i in present]
        return classification_report(
            y_true, y_pred,
            labels=present,
            target_names=target_names,
            output_dict=True,
            zero_division=0,
        )
    except ImportError:
        # Minimal fallback
        correct = (y_true == y_pred).sum()
        return {"accuracy": float(correct / len(y_true))}


def critical_class_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    critical_indices: Optional[List[int]] = None,
) -> Dict[str, float]:
    """
    Compute recall specifically for safety-critical classes.

    Default critical classes: stop (0), yield (1), donotenter (18).
    """
    if critical_indices is None:
        critical_indices = [0, 1, 18]  # DOT indices

    recalls: Dict[str, float] = {}
    for idx in critical_indices:
        mask = y_true == idx
        if mask.sum() == 0:
            recalls[global_index_to_name(idx)] = float("nan")
            continue
        correct = ((y_pred == idx) & mask).sum()
        recalls[global_index_to_name(idx)] = float(correct / mask.sum())
    return recalls
