"""
Classical ML Evaluation — Feature Extraction + ML Model Comparison.

Extracts dense features from a trained CNN (penultimate layer), then trains
and evaluates 4 classical ML models:
  1. XGBoost (gradient-boosted trees)
  2. Logistic Regression (linear baseline)
  3. Random Forest (bagged trees)
  4. SVM (RBF kernel)

Metrics:
  • Accuracy, Precision (macro), Recall (macro), F1 (macro/weighted)
  • AUC-ROC (one-vs-rest, macro)
  • McFadden's pseudo-R² (for logistic regression)
  • Per-class classification report
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.device import get_device


# ---------------------------------------------------------------------------
# Feature extraction from CNN
# ---------------------------------------------------------------------------

def extract_features(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[torch.device] = None,
    layer_name: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features from the penultimate (or specified) layer of a CNN.

    Returns:
        features: (N, D) float32 array
        labels:   (N,)   int array
    """
    if device is None:
        device = get_device()

    model.to(device)
    model.eval()

    features_list: List[np.ndarray] = []
    labels_list: List[np.ndarray] = []
    hook_output: List[torch.Tensor] = []

    # Register forward hook on penultimate layer
    target_layer = _find_penultimate_layer(model, layer_name)

    def hook_fn(module, input, output):
        # Handle both linear and conv layers
        out = output.detach()
        if out.dim() > 2:
            out = out.flatten(1)  # (B, C, H, W) -> (B, C*H*W)
        hook_output.append(out.cpu())

    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        for images, labels in loader:
            hook_output.clear()
            images = images.to(device)
            _ = model(images)
            if hook_output:
                features_list.append(hook_output[0].numpy())
            labels_list.append(labels.numpy())

    handle.remove()

    return np.concatenate(features_list), np.concatenate(labels_list)


def _find_penultimate_layer(model: nn.Module, name: Optional[str] = None) -> nn.Module:
    """Find the penultimate layer (last layer before the final classifier)."""
    if name:
        for n, m in model.named_modules():
            if n == name:
                return m
        raise ValueError(f"Layer '{name}' not found in model")

    # Auto-detect: find the layer just before the last Linear
    modules = list(model.named_modules())
    last_linear_idx = None
    for i, (n, m) in enumerate(modules):
        if isinstance(m, nn.Linear):
            last_linear_idx = i

    if last_linear_idx and last_linear_idx > 0:
        # Walk backwards to find the previous meaningful layer
        for i in range(last_linear_idx - 1, -1, -1):
            n, m = modules[i]
            if isinstance(m, (nn.Linear, nn.Conv2d, nn.BatchNorm1d,
                              nn.BatchNorm2d, nn.AdaptiveAvgPool2d,
                              nn.ReLU, nn.Dropout)):
                if isinstance(m, (nn.Linear, nn.AdaptiveAvgPool2d)):
                    return m
                if isinstance(m, (nn.ReLU, nn.Dropout, nn.BatchNorm1d)):
                    continue
                return m

    # Fallback: use AdaptiveAvgPool if present
    for n, m in modules:
        if isinstance(m, nn.AdaptiveAvgPool2d):
            return m

    # Last resort: second-to-last child
    children = list(model.children())
    if len(children) >= 2:
        return children[-2]

    raise RuntimeError("Cannot determine penultimate layer — pass layer_name explicitly")


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

@dataclass
class ModelResult:
    """Results for a single classical ML model."""
    name: str
    accuracy: float = 0.0
    precision_macro: float = 0.0
    recall_macro: float = 0.0
    f1_macro: float = 0.0
    f1_weighted: float = 0.0
    auc_roc_macro: float = 0.0
    pseudo_r2: float = 0.0
    train_time: float = 0.0
    y_pred: Optional[np.ndarray] = field(default=None, repr=False)
    y_proba: Optional[np.ndarray] = field(default=None, repr=False)

    def summary_dict(self) -> Dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision_macro": self.precision_macro,
            "recall_macro": self.recall_macro,
            "f1_macro": self.f1_macro,
            "f1_weighted": self.f1_weighted,
            "auc_roc_macro": self.auc_roc_macro,
            "pseudo_r2": self.pseudo_r2,
            "train_time_s": self.train_time,
        }


def _compute_metrics(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    train_time: float,
    n_classes: int = 0,
    null_log_likelihood: Optional[float] = None,
    model_log_likelihood: Optional[float] = None,
) -> ModelResult:
    """Compute all classification metrics for a model's predictions."""
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
    )

    result = ModelResult(name=name, train_time=train_time)
    result.y_pred = y_pred
    result.y_proba = y_proba

    result.accuracy = accuracy_score(y_true, y_pred)
    result.precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    result.recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    result.f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    result.f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    # AUC-ROC (one-vs-rest) — use all class labels to handle train/test mismatch
    if y_proba is not None:
        try:
            all_labels = list(range(n_classes)) if n_classes > 0 else sorted(set(y_true))
            if len(all_labels) > 2 and y_proba.shape[1] > 2:
                result.auc_roc_macro = roc_auc_score(
                    y_true, y_proba, multi_class="ovr",
                    average="macro", labels=all_labels,
                )
            elif len(all_labels) == 2:
                result.auc_roc_macro = roc_auc_score(y_true, y_proba[:, 1])
        except (ValueError, IndexError):
            result.auc_roc_macro = 0.0

    # McFadden's pseudo-R² (only meaningful for logistic regression)
    if null_log_likelihood is not None and model_log_likelihood is not None:
        if null_log_likelihood != 0:
            result.pseudo_r2 = 1.0 - (model_log_likelihood / null_log_likelihood)

    return result


# ---------------------------------------------------------------------------
# Classical ML models
# ---------------------------------------------------------------------------

def _train_xgboost(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_classes: int,
) -> ModelResult:
    """Train and evaluate XGBoost classifier."""
    from xgboost import XGBClassifier

    t0 = time.time()
    model = XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective="multi:softprob",
        num_class=n_classes,
        eval_metric="mlogloss",
        use_label_encoder=False,
        tree_method="hist",
        verbosity=0,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return _compute_metrics("XGBoost", y_test, y_pred, y_proba, train_time, n_classes=n_classes)


def _train_logistic_regression(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_classes: int,
) -> ModelResult:
    """Train and evaluate Logistic Regression."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    t0 = time.time()
    model = LogisticRegression(
        max_iter=2000,
        solver="lbfgs",
        C=1.0,
        n_jobs=-1,
    )
    model.fit(X_train_s, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)

    # Compute pseudo-R² via log-likelihoods
    from sklearn.metrics import log_loss
    all_labels = list(range(n_classes))
    null_ll = -log_loss(y_test, np.full_like(y_proba, 1.0 / n_classes),
                        normalize=False, labels=all_labels)
    model_ll = -log_loss(y_test, y_proba, normalize=False, labels=all_labels)

    return _compute_metrics(
        "Logistic Regression", y_test, y_pred, y_proba, train_time,
        n_classes=n_classes,
        null_log_likelihood=null_ll,
        model_log_likelihood=model_ll,
    )


def _train_random_forest(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_classes: int,
) -> ModelResult:
    """Train and evaluate Random Forest."""
    from sklearn.ensemble import RandomForestClassifier

    t0 = time.time()
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=5,
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    return _compute_metrics("Random Forest", y_test, y_pred, y_proba, train_time, n_classes=n_classes)


def _train_svm(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    n_classes: int,
) -> ModelResult:
    """Train and evaluate SVM with RBF kernel."""
    from sklearn.svm import SVC
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    t0 = time.time()
    model = SVC(
        kernel="rbf",
        C=10.0,
        gamma="scale",
        probability=True,
        decision_function_shape="ovr",
        random_state=42,
    )
    model.fit(X_train_s, y_train)
    train_time = time.time() - t0

    y_pred = model.predict(X_test_s)
    y_proba = model.predict_proba(X_test_s)

    return _compute_metrics("SVM (RBF)", y_test, y_pred, y_proba, train_time, n_classes=n_classes)


# ---------------------------------------------------------------------------
# Main comparison pipeline
# ---------------------------------------------------------------------------

@dataclass
class ComparisonReport:
    """Full comparison report across all models."""
    cnn_source: str
    feature_dim: int
    n_train: int
    n_test: int
    n_classes: int
    results: List[ModelResult]

    def summary_table(self) -> str:
        """Return a formatted summary table."""
        header = (
            f"{'Model':<25} {'Accuracy':>8} {'F1-Macro':>8} {'F1-Wt':>8} "
            f"{'Prec':>8} {'Recall':>8} {'AUC-ROC':>8} {'R²':>8} {'Time(s)':>8}"
        )
        sep = "─" * len(header)
        lines = [sep, header, sep]

        for r in self.results:
            lines.append(
                f"{r.name:<25} {r.accuracy:>8.4f} {r.f1_macro:>8.4f} "
                f"{r.f1_weighted:>8.4f} {r.precision_macro:>8.4f} "
                f"{r.recall_macro:>8.4f} {r.auc_roc_macro:>8.4f} "
                f"{r.pseudo_r2:>8.4f} {r.train_time:>8.1f}"
            )

        lines.append(sep)

        # Best model
        best = max(self.results, key=lambda r: r.f1_macro)
        lines.append(f"\n🏆 Best model: {best.name} (F1-macro: {best.f1_macro:.4f})")

        return "\n".join(lines)


def run_comparison(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    model_name: str = "CNN",
    device: Optional[torch.device] = None,
    layer_name: Optional[str] = None,
    verbose: bool = False,
) -> ComparisonReport:
    """
    Full classical ML comparison pipeline.

    1. Extract features from the CNN's penultimate layer
    2. Train 4 classical ML models on the extracted features
    3. Evaluate and compare all models

    Args:
        model: Trained CNN model
        train_loader: Training DataLoader
        test_loader: Test/val DataLoader
        model_name: Name of the CNN source model
        device: Torch device
        layer_name: Optional specific layer name for feature extraction
        verbose: Print progress

    Returns:
        ComparisonReport with all results
    """
    if device is None:
        device = get_device()

    # 1. Extract features
    if verbose:
        print("📊 Extracting features from CNN...")
    X_train, y_train = extract_features(model, train_loader, device, layer_name)
    X_test, y_test = extract_features(model, test_loader, device, layer_name)

    if verbose:
        print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

    # Remap labels to contiguous range for sklearn
    all_labels = np.concatenate([y_train, y_test])
    unique_labels = sorted(set(all_labels.tolist()))
    label_map = {old: new for new, old in enumerate(unique_labels)}
    y_train_mapped = np.array([label_map[y] for y in y_train])
    y_test_mapped = np.array([label_map[y] for y in y_test])
    n_classes = len(unique_labels)

    if verbose:
        print(f"   {n_classes} classes, Feature dim: {X_train.shape[1]}")

    # 2. Train models
    results: List[ModelResult] = []

    models_to_train = [
        ("XGBoost", _train_xgboost),
        ("Logistic Regression", _train_logistic_regression),
        ("Random Forest", _train_random_forest),
        ("SVM (RBF)", _train_svm),
    ]

    for name, train_fn in models_to_train:
        if verbose:
            print(f"   Training {name}...")
        try:
            result = train_fn(X_train, y_train_mapped, X_test, y_test_mapped, n_classes)
            results.append(result)
            if verbose:
                print(f"   ✅ {name}: F1={result.f1_macro:.4f}, Acc={result.accuracy:.4f}")
        except Exception as e:
            if verbose:
                print(f"   ❌ {name} failed: {e}")
            results.append(ModelResult(name=name))

    return ComparisonReport(
        cnn_source=model_name,
        feature_dim=X_train.shape[1],
        n_train=len(X_train),
        n_test=len(X_test),
        n_classes=n_classes,
        results=results,
    )
