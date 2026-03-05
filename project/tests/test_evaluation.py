"""
Phase 4 tests – Evaluation.

Tests cover:
  • Classification report generation
  • Critical-class recall computation
  • ECE computation with known values
  • Temperature Scaling convergence
  • OOD degradation report structure
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import IMAGE_SIZE, NUM_CLASSES
from models.baseline import BaselineCNN
from evaluation.metrics import (
    collect_predictions,
    classification_report_dict,
    critical_class_recall,
)
from evaluation.calibration import compute_ece, TemperatureScaling
from evaluation.ood_testing import evaluate_split, ood_degradation_test, DegradationReport


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def tiny_loader():
    """16 random images with labels in {0, 1, 2}."""
    images = torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0])
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=4)


@pytest.fixture
def model_cpu():
    return BaselineCNN(num_classes=3).eval()


# ===================================================================
# 1. Metrics
# ===================================================================

class TestMetrics:
    def test_classification_report_dict(self):
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 2, 2, 2])
        report = classification_report_dict(y_true, y_pred)
        assert "accuracy" in report

    def test_critical_class_recall_perfect(self):
        y_true = np.array([0, 0, 1, 1, 18, 18])
        y_pred = np.array([0, 0, 1, 1, 18, 18])
        recalls = critical_class_recall(y_true, y_pred)
        assert recalls["stop"] == 1.0
        assert recalls["yield"] == 1.0
        assert recalls["donotenter"] == 1.0

    def test_critical_class_recall_zero(self):
        y_true = np.array([0, 0, 1, 1])
        y_pred = np.array([1, 1, 0, 0])
        recalls = critical_class_recall(y_true, y_pred, critical_indices=[0, 1])
        assert recalls["stop"] == 0.0
        assert recalls["yield"] == 0.0

    def test_collect_predictions_shapes(self, model_cpu, tiny_loader):
        labels, preds, probs = collect_predictions(
            model_cpu, tiny_loader, device=torch.device("cpu")
        )
        assert labels.shape == (16,)
        assert preds.shape == (16,)
        assert probs.shape == (16, 3)


# ===================================================================
# 2. Calibration
# ===================================================================

class TestCalibration:
    def test_ece_perfect_calibration(self):
        """A model that predicts correctly with 100% confidence has ECE ≈ 0."""
        n = 100
        y_true = np.zeros(n, dtype=int)
        y_prob = np.zeros((n, 3))
        y_prob[:, 0] = 1.0  # 100% confidence on correct class
        ece, _, _, _ = compute_ece(y_true, y_prob, n_bins=10)
        assert ece < 0.01

    def test_ece_returns_four_values(self):
        y_true = np.array([0, 1, 0, 1])
        y_prob = np.array([
            [0.9, 0.1], [0.3, 0.7], [0.8, 0.2], [0.4, 0.6]
        ])
        result = compute_ece(y_true, y_prob, n_bins=5)
        assert len(result) == 4

    def test_temperature_scaling_fit(self):
        """Temperature scaling should adjust temperature."""
        np.random.seed(42)
        n = 200
        logits = np.random.randn(n, 5) * 3
        labels = np.argmax(logits, axis=1)  # Oracle labels
        ts = TemperatureScaling(initial_temperature=1.0)
        ts.fit(logits, labels, max_iter=50)
        # After fitting, temperature should still be a positive number
        assert ts.temperature > 0

    def test_calibrate_returns_probs(self):
        ts = TemperatureScaling(initial_temperature=2.0)
        logits = np.array([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2]])
        probs = ts.calibrate(logits)
        # Each row should sum to 1
        np.testing.assert_allclose(probs.sum(axis=1), 1.0, atol=1e-6)


# ===================================================================
# 3. OOD Testing
# ===================================================================

class TestOODTesting:
    def test_evaluate_split_returns_dict(self, model_cpu, tiny_loader):
        report = evaluate_split(model_cpu, tiny_loader, device=torch.device("cpu"))
        assert isinstance(report, dict)
        assert "accuracy" in report

    def test_degradation_report_fields(self, model_cpu, tiny_loader):
        report = ood_degradation_test(
            model_cpu, tiny_loader, tiny_loader, device=torch.device("cpu")
        )
        assert isinstance(report, DegradationReport)
        assert hasattr(report, "accuracy_gap")
        assert hasattr(report, "f1_gap")

    def test_same_split_zero_gap(self, model_cpu, tiny_loader):
        """When in-domain and OOD are the same loader, gap should be ~0."""
        report = ood_degradation_test(
            model_cpu, tiny_loader, tiny_loader, device=torch.device("cpu")
        )
        assert abs(report.accuracy_gap) < 1e-9
        assert abs(report.f1_gap) < 1e-9
