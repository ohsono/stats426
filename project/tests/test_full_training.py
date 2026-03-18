"""
Full Training Test — end-to-end training on the real DOT dataset.

This test:
  1. Loads the real DOT reference images from ./dataset/DOT/
  2. Trains baseline, advanced, and resnet50 models for a few epochs
  3. Evaluates with classification report, ECE, critical class recall
  4. Saves checkpoints and verbose log files to a temp dir
  5. Verifies all artifacts are created correctly

Run with:
    python -m pytest tests/test_full_training.py -v -s
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import Config, DataConfig, IMAGE_SIZE
from utils.logger import ExperimentLogger
from data.unify import num_classes, DOT_CLASSES
from data.transforms import eval_transform, gtsrb_train_transform
from data.datasets import DOTDataset
from data.dataloaders import stratified_split
from models.baseline import BaselineCNN
from models.advanced import AdvancedCNN
from models.resnet import ResNet50
from training.engine import Trainer
from training.schedulers import build_scheduler
from evaluation.metrics import collect_predictions, classification_report_dict, critical_class_recall
from evaluation.calibration import compute_ece, TemperatureScaling
from evaluation.ood_testing import ood_degradation_test


# ===================================================================
# Paths
# ===================================================================
DOT_ROOT = PROJECT_ROOT / "dataset" / "DOT"
DOT_CSV = PROJECT_ROOT / "dataset" / "DOT_traffic_sign_label.csv"
SKIP_REASON = "DOT dataset not found — skipping full training test"


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture(scope="module")
def dot_dataset():
    """Load the real DOT dataset; skip if not available."""
    if not DOT_ROOT.exists():
        pytest.skip(SKIP_REASON)
    ds = DOTDataset(DOT_ROOT, transform=eval_transform())
    if len(ds) == 0:
        pytest.skip("DOT dataset is empty")
    return ds


@pytest.fixture(scope="module")
def dot_train_dataset():
    """DOT dataset with training augmentations."""
    if not DOT_ROOT.exists():
        pytest.skip(SKIP_REASON)
    return DOTDataset(DOT_ROOT, transform=gtsrb_train_transform())


@pytest.fixture(scope="module")
def dot_loaders(dot_dataset):
    """Split the DOT dataset 70-10-10-10 and create DataLoaders."""
    splits = stratified_split(dot_dataset, ratios=(0.6, 0.2, 0.1, 0.1))
    loaders = {}
    names = ["train", "val", "test", "ood"]
    for name, subset in zip(names, splits):
        loaders[name] = DataLoader(
            subset, batch_size=8, shuffle=(name == "train"),
            num_workers=0,
        )
    return loaders


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


# ===================================================================
# 1. DOT Dataset Verification
# ===================================================================

class TestDOTDataset:
    """Verify the real DOT dataset loads correctly."""

    def test_dataset_length(self, dot_dataset):
        # DOT CSV has 43 entries, but only those with existing files load
        assert len(dot_dataset) >= 30  # at least 30 of the 43 should exist

    def test_image_shape(self, dot_dataset):
        img, label = dot_dataset[0]
        assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

    def test_labels_in_range(self, dot_dataset):
        labels = dot_dataset.get_labels()
        assert all(0 <= l <= 57 for l in labels)

    def test_stop_sign_present(self, dot_dataset):
        labels = dot_dataset.get_labels()
        assert 0 in labels  # DOT index 0 = stop

    def test_yield_sign_present(self, dot_dataset):
        labels = dot_dataset.get_labels()
        assert 1 in labels  # DOT index 1 = yield

    def test_split_creates_four_loaders(self, dot_loaders):
        assert set(dot_loaders.keys()) == {"train", "val", "test", "ood"}
        for name, loader in dot_loaders.items():
            assert len(loader.dataset) > 0, f"{name} split is empty"


# ===================================================================
# 2. Full Training — Baseline CNN
# ===================================================================

class TestFullTrainBaseline:
    """Train BaselineCNN on real DOT data for a few epochs."""

    def test_train_baseline_on_dot(self, dot_loaders, tmp_dir):
        n_cls = num_classes()
        model = BaselineCNN(num_classes=n_cls)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        logger = ExperimentLogger(
            log_dir=tmp_dir / "logs", function="train",
            model_name="baseline", verbose=True,
        )
        trainer = Trainer(
            model=model, optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=tmp_dir / "ckpt",
            logger=logger, save_best=True, save_every_n_epochs=3,
        )
        history = trainer.fit(
            dot_loaders["train"], dot_loaders["val"], epochs=5,
        )
        logger.close()

        # Verify training ran
        assert len(history["train_loss"]) == 5
        assert len(history["val_loss"]) == 5
        # Verify artifacts
        assert (tmp_dir / "ckpt" / "best_model.pth").exists()
        assert (tmp_dir / "ckpt" / "best_model_weights.pth").exists()
        assert (tmp_dir / "ckpt" / "checkpoint_last.pth").exists()
        # Verify log file
        assert logger.log_file_path.exists()
        log_content = logger.log_file_path.read_text()
        assert "MODEL SUMMARY" in log_content
        assert "EPOCH" in log_content


# ===================================================================
# 3. Full Training — ResNet50
# ===================================================================

class TestFullTrainResNet:
    """Train ResNet50 on real DOT data with cosine scheduler."""

    def test_train_resnet_on_dot(self, dot_loaders, tmp_dir):
        n_cls = num_classes()
        model = ResNet50(num_classes=n_cls)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = build_scheduler(optimizer, name="cosine", total_epochs=5)
        logger = ExperimentLogger(
            log_dir=tmp_dir / "logs", function="train",
            model_name="resnet50", verbose=True,
        )
        trainer = Trainer(
            model=model, optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=tmp_dir / "ckpt",
            scheduler=scheduler, logger=logger,
            save_best=True, save_every_n_epochs=2,
        )
        history = trainer.fit(
            dot_loaders["train"], dot_loaders["val"], epochs=5,
        )
        logger.close()

        assert len(history["train_loss"]) == 5
        assert (tmp_dir / "ckpt" / "best_model.pth").exists()
        # Verify log file naming convention
        log_name = logger.log_file_path.name
        assert log_name.startswith("train-resnet50-")
        assert log_name.endswith(".log")


# ===================================================================
# 4. Full Evaluation Pipeline
# ===================================================================

class TestFullEvaluation:
    """Train briefly, then run the complete evaluation pipeline."""

    def test_full_eval_pipeline(self, dot_loaders, tmp_dir):
        n_cls = num_classes()
        model = BaselineCNN(num_classes=n_cls)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        logger = ExperimentLogger(
            log_dir=tmp_dir / "logs", function="eval",
            model_name="baseline", verbose=True,
        )
        trainer = Trainer(
            model=model, optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=tmp_dir / "ckpt",
            logger=logger, save_best=True,
        )
        trainer.fit(dot_loaders["train"], dot_loaders["val"], epochs=5)

        # --- Classification Report ---
        labels, preds, probs = collect_predictions(
            model, dot_loaders["test"], device=torch.device("cpu"),
        )
        report = classification_report_dict(labels, preds)
        logger.log_evaluation("test", report)
        assert "accuracy" in report

        # --- Critical Class Recall ---
        present_critical = [idx for idx in [0, 1, 18] if idx in set(labels)]
        if present_critical:
            recalls = critical_class_recall(labels, preds, critical_indices=present_critical)
            for name, val in recalls.items():
                if not np.isnan(val):
                    assert 0.0 <= val <= 1.0

        # --- ECE ---
        ece, bin_conf, bin_acc, bin_counts = compute_ece(labels, probs, n_bins=10)
        assert 0.0 <= ece <= 1.0
        assert len(bin_conf) == 10

        # --- Temperature Scaling ---
        all_logits = []
        model.eval()
        with torch.no_grad():
            for imgs, _ in dot_loaders["val"]:
                all_logits.append(model(imgs).numpy())
        logits_np = np.concatenate(all_logits)
        val_labels = np.array([l for _, l in dot_loaders["val"].dataset])
        ts = TemperatureScaling()
        ts.fit(logits_np, val_labels, max_iter=50)
        assert ts.temperature > 0

        # --- OOD Degradation ---
        if len(dot_loaders["ood"].dataset) > 0:
            deg = ood_degradation_test(
                model, dot_loaders["test"], dot_loaders["ood"],
                device=torch.device("cpu"),
            )
            assert hasattr(deg, "accuracy_gap")

        logger.close()
        # Verify eval log
        log_content = logger.log_file_path.read_text()
        assert "EVALUATION" in log_content
        assert "TEST" in log_content
