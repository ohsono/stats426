"""
Out-of-Distribution (OOD) Degradation Testing (Phase 4.3).

Evaluates the generalization gap between In-Domain test performance and
OOD / Challenge split performance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader

from evaluation.metrics import collect_predictions, classification_report_dict
from utils.device import get_device


@dataclass
class DegradationReport:
    """Summary of in-domain vs. OOD performance."""
    in_domain_accuracy: float
    ood_accuracy: float
    accuracy_gap: float          # absolute drop
    in_domain_f1_macro: float
    ood_f1_macro: float
    f1_gap: float                # absolute drop
    per_class_gaps: Dict[str, float]  # per-class F1 drop


def evaluate_split(
    model: nn.Module,
    loader: DataLoader,
    device=None,
) -> Dict:
    """Run evaluation on a loader and return a classification report dict."""
    if device is None:
        device = get_device()
    y_true, y_pred, _ = collect_predictions(model, loader, device)
    return classification_report_dict(y_true, y_pred)


def ood_degradation_test(
    model: nn.Module,
    in_domain_loader: DataLoader,
    ood_loader: DataLoader,
    device=None,
) -> DegradationReport:
    """
    Compare model performance on in-domain vs OOD splits.

    Returns a ``DegradationReport`` quantifying the generalization gap.
    """
    if device is None:
        device = get_device()

    id_report = evaluate_split(model, in_domain_loader, device)
    ood_report = evaluate_split(model, ood_loader, device)

    id_acc = id_report.get("accuracy", 0.0)
    ood_acc = ood_report.get("accuracy", 0.0)

    # Extract macro-average F1 (scikit-learn key)
    id_f1 = id_report.get("macro avg", {}).get("f1-score", id_acc)
    ood_f1 = ood_report.get("macro avg", {}).get("f1-score", ood_acc)

    # Per-class F1 gaps
    per_class_gaps: Dict[str, float] = {}
    for key in id_report:
        if isinstance(id_report[key], dict) and "f1-score" in id_report[key]:
            id_f1_cls = id_report[key]["f1-score"]
            ood_f1_cls = ood_report.get(key, {}).get("f1-score", 0.0)
            per_class_gaps[key] = id_f1_cls - ood_f1_cls

    return DegradationReport(
        in_domain_accuracy=id_acc,
        ood_accuracy=ood_acc,
        accuracy_gap=id_acc - ood_acc,
        in_domain_f1_macro=id_f1,
        ood_f1_macro=ood_f1,
        f1_gap=id_f1 - ood_f1,
        per_class_gaps=per_class_gaps,
    )
