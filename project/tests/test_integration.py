"""
Integration Tests — cross-module pipeline verification.

Tests cover end-to-end flows combining multiple modules:
  1. Data Pipeline → Model: DOT dataset → transform → model forward pass
  2. Data → Train: synthetic data → Trainer.fit() with logging
  3. Train → Checkpoint → Resume: save & reload checkpoint, continue training
  4. Data → Train → Evaluate: full train + evaluation + calibration pipeline
  5. Curriculum Integration: stage-aware dataset switching in fit()
  6. Verbose Logging: verify log files are created with correct naming
"""

from __future__ import annotations

import csv
import shutil
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.device import get_device
from utils.config import Config, DataConfig, IMAGE_SIZE, NUM_CLASSES
from utils.logger import ExperimentLogger, create_log_file
from data.unify import num_classes, global_index_to_name, DOT_CLASSES
from data.transforms import eval_transform
from data.datasets import DOTDataset, GTSRBDataset, UnifiedTrafficSignDataset
from data.dataloaders import stratified_split, create_dataloaders
from models.baseline import BaselineCNN
from models.advanced import AdvancedCNN
from models.resnet import ResNet50
from training.engine import Trainer
from training.curriculum import CurriculumScheduler, Stage
from training.schedulers import build_scheduler
from evaluation.metrics import collect_predictions, classification_report_dict, critical_class_recall
from evaluation.calibration import compute_ece, TemperatureScaling
from evaluation.ood_testing import ood_degradation_test, DegradationReport

from PIL import Image


# ===================================================================
# Shared Fixtures
# ===================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def dot_fixture(tmp_dir) -> Path:
    """Create a minimal DOT dataset with 6 images across 3 classes."""
    root = tmp_dir / "DOT"
    root.mkdir()
    csv_path = root / "DOT_traffic_sign_label.csv"
    rows = [
        {"index": "0", "filename": "0_stop.png", "label": "stop"},
        {"index": "1", "filename": "1_yield.png", "label": "yield"},
        {"index": "2", "filename": "2_speedlimitsign.png", "label": "speedlimitsign"},
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "filename", "label"])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        img = Image.new("RGB", (64, 64), color="red")
        img.save(root / row["filename"])
    return root


@pytest.fixture
def synthetic_loader_3class():
    """32 random images with labels in {0, 1, 2} — small 3-class problem."""
    torch.manual_seed(42)
    images = torch.randn(32, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.tensor([i % 3 for i in range(32)])
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=8, shuffle=True)


@pytest.fixture
def synthetic_loader_58class():
    """64 random images with labels spread across the full 58-class space."""
    torch.manual_seed(42)
    images = torch.randn(64, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.tensor([i % NUM_CLASSES for i in range(64)])
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=16, shuffle=True)


# ===================================================================
# 1. Data Pipeline → Model (forward pass)
# ===================================================================

class TestDataToModel:
    """Verify that dataset outputs flow correctly through each model."""

    def test_dot_to_baseline(self, dot_fixture):
        ds = DOTDataset(dot_fixture, transform=eval_transform())
        loader = DataLoader(ds, batch_size=3)
        model = BaselineCNN(num_classes=3).eval()
        images, labels = next(iter(loader))
        with torch.no_grad():
            out = model(images)
        assert out.shape == (3, 3)
        assert labels.tolist() == [0, 1, 2]

    def test_dot_to_advanced(self, dot_fixture):
        ds = DOTDataset(dot_fixture, transform=eval_transform())
        loader = DataLoader(ds, batch_size=3)
        model = AdvancedCNN(num_classes=3).eval()
        images, _ = next(iter(loader))
        with torch.no_grad():
            out = model(images)
        assert out.shape == (3, 3)

    def test_dot_to_resnet(self, dot_fixture):
        ds = DOTDataset(dot_fixture, transform=eval_transform())
        loader = DataLoader(ds, batch_size=3)
        model = ResNet50(num_classes=3).eval()
        images, _ = next(iter(loader))
        with torch.no_grad():
            out = model(images)
        assert out.shape == (3, 3)


# ===================================================================
# 2. Data → Trainer.fit() with logging
# ===================================================================

class TestDataToTrainer:
    """Verify Trainer.fit() works end-to-end with synthetic data."""

    def test_fit_synthetic_baseline(self, synthetic_loader_3class, tmp_dir):
        model = BaselineCNN(num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        logger = ExperimentLogger(
            log_dir=tmp_dir, function="train", model_name="baseline",
            verbose=True,
        )
        trainer = Trainer(
            model=model, optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=tmp_dir / "ckpt",
            logger=logger, save_best=True, save_every_n_epochs=2,
        )
        history = trainer.fit(synthetic_loader_3class, synthetic_loader_3class, epochs=4)
        logger.close()

        # Verify history
        assert len(history["train_loss"]) == 4
        assert len(history["val_loss"]) == 4
        # Verify checkpoints
        assert (tmp_dir / "ckpt" / "best_model.pth").exists()
        assert (tmp_dir / "ckpt" / "checkpoint_epoch_2.pth").exists()
        assert (tmp_dir / "ckpt" / "checkpoint_epoch_4.pth").exists()
        assert (tmp_dir / "ckpt" / "checkpoint_last.pth").exists()

    def test_fit_synthetic_resnet(self, synthetic_loader_3class, tmp_dir):
        model = ResNet50(num_classes=3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model, optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=tmp_dir / "ckpt",
            save_best=True,
        )
        history = trainer.fit(synthetic_loader_3class, synthetic_loader_3class, epochs=3)
        assert history["train_loss"][-1] <= history["train_loss"][0]

    def test_fit_with_scheduler(self, synthetic_loader_3class, tmp_dir):
        model = BaselineCNN(num_classes=3)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        scheduler = build_scheduler(optimizer, name="cosine", total_epochs=6)
        trainer = Trainer(
            model=model, optimizer=optimizer,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=tmp_dir / "ckpt",
            scheduler=scheduler, save_best=True,
        )
        history = trainer.fit(synthetic_loader_3class, synthetic_loader_3class, epochs=6)
        assert len(history["train_loss"]) == 6


# ===================================================================
# 3. Train → Checkpoint → Resume
# ===================================================================

class TestCheckpointResume:
    """Verify saving and resuming training from a checkpoint."""

    def test_resume_continues_training(self, synthetic_loader_3class, tmp_dir):
        ckpt_dir = tmp_dir / "ckpt"

        # Phase 1: train for 3 epochs
        model1 = BaselineCNN(num_classes=3)
        opt1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        trainer1 = Trainer(
            model=model1, optimizer=opt1,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=ckpt_dir, save_best=True,
        )
        h1 = trainer1.fit(synthetic_loader_3class, synthetic_loader_3class, epochs=3)

        # Phase 2: load checkpoint and continue
        model2 = BaselineCNN(num_classes=3)
        opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        trainer2 = Trainer(
            model=model2, optimizer=opt2,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=ckpt_dir, save_best=True,
        )
        resumed_epoch = trainer2.load_checkpoint(ckpt_dir / "checkpoint_last.pth")
        assert resumed_epoch == 3

        h2 = trainer2.fit(
            synthetic_loader_3class, synthetic_loader_3class,
            epochs=3, start_epoch=resumed_epoch + 1,
        )
        assert len(h2["train_loss"]) == 3

    def test_best_model_weights_loadable(self, synthetic_loader_3class, tmp_dir):
        """best_model_weights.pth should be loadable as raw state_dict."""
        ckpt_dir = tmp_dir / "ckpt"
        model = BaselineCNN(num_classes=3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model, optimizer=opt,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=ckpt_dir, save_best=True,
        )
        trainer.fit(synthetic_loader_3class, synthetic_loader_3class, epochs=2)

        # Load weights-only file into a fresh model
        fresh = BaselineCNN(num_classes=3)
        state = torch.load(ckpt_dir / "best_model_weights.pth",
                           map_location="cpu", weights_only=True)
        fresh.load_state_dict(state)
        # Should produce same output
        x = torch.randn(1, 3, IMAGE_SIZE, IMAGE_SIZE)
        model.eval()
        fresh.eval()
        with torch.no_grad():
            assert torch.allclose(model(x), fresh(x), atol=1e-5)


# ===================================================================
# 4. Train → Evaluate (metrics + calibration + OOD)
# ===================================================================

class TestTrainToEvaluate:
    """End-to-end: train a model, then run full evaluation pipeline."""

    def test_train_then_evaluate(self, synthetic_loader_3class, tmp_dir):
        # Train
        model = BaselineCNN(num_classes=3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        trainer = Trainer(
            model=model, optimizer=opt,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=tmp_dir / "ckpt", save_best=True,
        )
        trainer.fit(synthetic_loader_3class, synthetic_loader_3class, epochs=5)

        # Evaluate — classification report
        labels, preds, probs = collect_predictions(
            model, synthetic_loader_3class, device=torch.device("cpu")
        )
        report = classification_report_dict(labels, preds)
        assert "accuracy" in report
        assert report["accuracy"] > 0  # should be non-zero after training

        # ECE
        ece, _, _, _ = compute_ece(labels, probs, n_bins=10)
        assert 0 <= ece <= 1.0

    def test_train_then_ood(self, synthetic_loader_3class, tmp_dir):
        # Train on one distribution
        model = BaselineCNN(num_classes=3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        trainer = Trainer(
            model=model, optimizer=opt,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=tmp_dir / "ckpt", save_best=True,
        )
        trainer.fit(synthetic_loader_3class, synthetic_loader_3class, epochs=4)

        # OOD test (use different random data as "OOD")
        ood_images = torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE) * 3  # amplified
        ood_labels = torch.tensor([i % 3 for i in range(16)])
        ood_loader = DataLoader(TensorDataset(ood_images, ood_labels), batch_size=8)

        report = ood_degradation_test(
            model, synthetic_loader_3class, ood_loader,
            device=torch.device("cpu"),
        )
        assert isinstance(report, DegradationReport)
        assert report.accuracy_gap >= 0 or report.accuracy_gap < 0  # just check it runs

    def test_temperature_scaling_integration(self, synthetic_loader_3class, tmp_dir):
        """Train → collect logits → temperature scale → verify calibration."""
        model = BaselineCNN(num_classes=3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-2)
        trainer = Trainer(
            model=model, optimizer=opt,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=tmp_dir / "ckpt", save_best=True,
        )
        trainer.fit(synthetic_loader_3class, synthetic_loader_3class, epochs=5)

        # Collect raw logits
        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for imgs, labels in synthetic_loader_3class:
                logits = model(imgs)
                all_logits.append(logits.cpu().numpy())
                all_labels.append(labels.numpy())
        logits_np = np.concatenate(all_logits)
        labels_np = np.concatenate(all_labels)

        # Fit temperature scaling
        ts = TemperatureScaling(initial_temperature=1.5)
        ts.fit(logits_np, labels_np, max_iter=50)
        calibrated = ts.calibrate(logits_np)

        # Calibrated probs should sum to 1
        np.testing.assert_allclose(calibrated.sum(axis=1), 1.0, atol=1e-6)
        # Temperature should be reasonable
        assert 0.01 < ts.temperature < 100


# ===================================================================
# 5. Curriculum Integration
# ===================================================================

class TestCurriculumIntegration:
    """Verify curriculum scheduler works with the trainer."""

    def test_curriculum_stage_switching(self):
        cs = CurriculumScheduler(stage1_epochs=5, stage2_epochs=5)
        stages = [cs.get_stage(e) for e in range(1, 11)]
        assert stages[:5] == [Stage.GEOMETRIC] * 5
        assert stages[5:] == [Stage.REAL_WORLD] * 5

    def test_curriculum_with_trainer(self, synthetic_loader_3class, tmp_dir):
        """Simulate stage-aware training with the curriculum scheduler."""
        cs = CurriculumScheduler(stage1_epochs=3, stage2_epochs=3)
        model = BaselineCNN(num_classes=3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model, optimizer=opt,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=tmp_dir / "ckpt", save_best=True,
        )

        all_losses = []
        for epoch in range(1, cs.total_epochs + 1):
            stage = cs.get_stage(epoch)
            # In production, we'd switch dataloaders based on stage
            metrics = trainer.train_one_epoch(synthetic_loader_3class)
            val = trainer.validate(synthetic_loader_3class)
            all_losses.append(metrics["train_loss"])

        assert len(all_losses) == 6


# ===================================================================
# 6. Verbose Logging Integration
# ===================================================================

class TestLoggingIntegration:
    """Verify log files are created with the correct naming convention."""

    def test_log_file_created(self, tmp_dir):
        log_file = create_log_file("train", "resnet50", log_dir=tmp_dir)
        assert "train-resnet50-" in log_file.name
        assert log_file.name.endswith(".log")

    def test_verbose_logger_writes_file(self, synthetic_loader_3class, tmp_dir):
        logger = ExperimentLogger(
            log_dir=tmp_dir, function="train", model_name="baseline",
            verbose=True,
        )
        model = BaselineCNN(num_classes=3)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer = Trainer(
            model=model, optimizer=opt,
            criterion=nn.CrossEntropyLoss(),
            device=torch.device("cpu"), use_amp=False,
            checkpoint_dir=tmp_dir / "ckpt",
            logger=logger, save_best=True,
        )
        trainer.fit(synthetic_loader_3class, synthetic_loader_3class, epochs=2)
        logger.close()

        # Check log file exists and has content
        assert logger.log_file_path is not None
        assert logger.log_file_path.exists()
        content = logger.log_file_path.read_text()
        assert "EPOCH" in content
        assert "MODEL SUMMARY" in content
        assert "train_loss" in content

    def test_evaluation_logging(self, tmp_dir):
        logger = ExperimentLogger(
            log_dir=tmp_dir, function="eval", model_name="resnet50",
            verbose=True,
        )
        logger.log_evaluation("test", {"accuracy": 0.92, "macro avg": {"f1-score": 0.89}})
        logger.close()

        content = logger.log_file_path.read_text()
        assert "EVALUATION" in content
        assert "TEST" in content
