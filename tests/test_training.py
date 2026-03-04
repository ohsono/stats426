"""
Phase 3 tests – Training Logic.

Tests cover:
  • Trainer forward/backward on a tiny synthetic DataLoader
  • Checkpoint save / load round-trip
  • CurriculumScheduler stage resolution
  • Gradient Reversal Layer reverses gradients
  • Domain Classifier + DANN loss
  • LR scheduler factory
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline import BaselineCNN
from models.resnet import ResNet10
from training.engine import Trainer, DANNTrainer
from training.curriculum import CurriculumScheduler, Stage
from training.domain_adv import (
    GradientReversalLayer,
    DomainClassifier,
    dann_loss,
    gradient_reversal,
)
from training.schedulers import build_scheduler, build_warmup_cosine_scheduler
from utils.config import IMAGE_SIZE, NUM_CLASSES


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def tiny_loader():
    """16 random 3×64×64 images with random labels."""
    images = torch.randn(16, 3, IMAGE_SIZE, IMAGE_SIZE)
    labels = torch.randint(0, NUM_CLASSES, (16,))
    ds = TensorDataset(images, labels)
    return DataLoader(ds, batch_size=4, shuffle=False)


@pytest.fixture
def trainer_on_cpu(tiny_loader):
    """A Trainer with a BaselineCNN on CPU for fast tests."""
    model = BaselineCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    return Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
        use_amp=False,
    )


# ===================================================================
# 1. Trainer
# ===================================================================

class TestTrainer:
    def test_train_one_epoch_returns_metrics(self, trainer_on_cpu, tiny_loader):
        metrics = trainer_on_cpu.train_one_epoch(tiny_loader)
        assert "train_loss" in metrics
        assert "train_acc" in metrics
        assert metrics["train_loss"] >= 0

    def test_validate_returns_metrics(self, trainer_on_cpu, tiny_loader):
        metrics = trainer_on_cpu.validate(tiny_loader)
        assert "val_loss" in metrics
        assert "val_acc" in metrics

    def test_checkpoint_round_trip(self, trainer_on_cpu, tiny_loader):
        trainer_on_cpu.train_one_epoch(tiny_loader)
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            trainer_on_cpu.checkpoint_dir = ckpt_dir
            trainer_on_cpu.save_checkpoint(epoch=1, filename="test.pth")

            # Create fresh trainer and load
            model2 = BaselineCNN()
            opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            trainer2 = Trainer(
                model=model2,
                optimizer=opt2,
                criterion=nn.CrossEntropyLoss(),
                device=torch.device("cpu"),
                use_amp=False,
            )
            epoch = trainer2.load_checkpoint(ckpt_dir / "test.pth")
            assert epoch == 1

    def test_loss_decreases_over_epochs(self, tiny_loader):
        """Smoke test: loss should generally decrease over a few epochs."""
        model = BaselineCNN()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()
        trainer = Trainer(
            model=model, optimizer=optimizer, criterion=criterion,
            device=torch.device("cpu"), use_amp=False,
        )
        first = trainer.train_one_epoch(tiny_loader)
        for _ in range(5):
            last = trainer.train_one_epoch(tiny_loader)
        # Loss should be lower after 6 epochs on 16 samples
        assert last["train_loss"] <= first["train_loss"]

    def test_fit_returns_history(self, tiny_loader):
        """fit() should return a history dict with per-epoch metric lists."""
        with tempfile.TemporaryDirectory() as tmp:
            model = BaselineCNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = Trainer(
                model=model, optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
                device=torch.device("cpu"), use_amp=False,
                checkpoint_dir=Path(tmp),
                save_best=True,
                save_every_n_epochs=2,
            )
            history = trainer.fit(tiny_loader, tiny_loader, epochs=4)
            assert len(history["train_loss"]) == 4
            assert len(history["val_loss"]) == 4

    def test_fit_saves_best_model(self, tiny_loader):
        """fit() should create best_model.pth and best_model_weights.pth."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            model = BaselineCNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = Trainer(
                model=model, optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
                device=torch.device("cpu"), use_amp=False,
                checkpoint_dir=ckpt_dir,
                save_best=True,
            )
            trainer.fit(tiny_loader, tiny_loader, epochs=3)
            assert (ckpt_dir / "best_model.pth").exists()
            assert (ckpt_dir / "best_model_weights.pth").exists()
            assert trainer.best_epoch >= 1

    def test_fit_periodic_checkpoints(self, tiny_loader):
        """fit() should save checkpoint_epoch_N.pth every N epochs."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            model = BaselineCNN()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = Trainer(
                model=model, optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
                device=torch.device("cpu"), use_amp=False,
                checkpoint_dir=ckpt_dir,
                save_every_n_epochs=2,
            )
            trainer.fit(tiny_loader, tiny_loader, epochs=6)
            assert (ckpt_dir / "checkpoint_epoch_2.pth").exists()
            assert (ckpt_dir / "checkpoint_epoch_4.pth").exists()
            assert (ckpt_dir / "checkpoint_epoch_6.pth").exists()
            assert (ckpt_dir / "checkpoint_last.pth").exists()

    def test_fit_early_stopping(self, tiny_loader):
        """fit() should stop early when patience is exceeded."""
        with tempfile.TemporaryDirectory() as tmp:
            model = BaselineCNN()
            # lr=0 means weights never change -> val_loss stays constant
            optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
            trainer = Trainer(
                model=model, optimizer=optimizer,
                criterion=nn.CrossEntropyLoss(),
                device=torch.device("cpu"), use_amp=False,
                checkpoint_dir=Path(tmp),
                save_best=True,
                early_stopping_patience=3,
            )
            history = trainer.fit(tiny_loader, tiny_loader, epochs=20)
            # Epoch 1 sets best, then 3 more with no improvement -> stop at 4
            assert len(history["train_loss"]) <= 5
            assert trainer.stopped_early

    def test_load_checkpoint_restores_best_tracking(self, tiny_loader):
        """Checkpoint round-trip should preserve best_value and best_epoch."""
        with tempfile.TemporaryDirectory() as tmp:
            ckpt_dir = Path(tmp)
            model = BaselineCNN()
            opt = torch.optim.Adam(model.parameters(), lr=1e-3)
            trainer = Trainer(
                model=model, optimizer=opt,
                criterion=nn.CrossEntropyLoss(),
                device=torch.device("cpu"), use_amp=False,
                checkpoint_dir=ckpt_dir, save_best=True,
            )
            trainer.fit(tiny_loader, tiny_loader, epochs=3)
            saved_best = trainer.best_value

            # Load into fresh trainer
            model2 = BaselineCNN()
            opt2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
            trainer2 = Trainer(
                model=model2, optimizer=opt2,
                criterion=nn.CrossEntropyLoss(),
                device=torch.device("cpu"), use_amp=False,
            )
            trainer2.load_checkpoint(ckpt_dir / "checkpoint_last.pth")
            assert trainer2.best_value == saved_best


# ===================================================================
# 2. Curriculum Scheduler
# ===================================================================

class TestCurriculum:
    def test_stage1_geometric(self):
        cs = CurriculumScheduler(stage1_epochs=20, stage2_epochs=20)
        assert cs.get_stage(1) == Stage.GEOMETRIC
        assert cs.get_stage(20) == Stage.GEOMETRIC

    def test_stage2_real_world(self):
        cs = CurriculumScheduler(stage1_epochs=20, stage2_epochs=20)
        assert cs.get_stage(21) == Stage.REAL_WORLD
        assert cs.get_stage(40) == Stage.REAL_WORLD

    def test_stage3_domain_adv(self):
        cs = CurriculumScheduler(enable_domain_adv=True)
        assert cs.get_stage(100) == Stage.DOMAIN_ADV

    def test_active_datasets_stage1(self):
        cs = CurriculumScheduler()
        datasets = cs.get_active_datasets(Stage.GEOMETRIC)
        assert datasets == ["gtsrb", "lisa"]

    def test_active_datasets_stage2(self):
        cs = CurriculumScheduler()
        datasets = cs.get_active_datasets(Stage.REAL_WORLD)
        assert "bdd100k" in datasets

    def test_freeze_unfreeze(self):
        model = BaselineCNN()
        CurriculumScheduler.freeze_backbone(model, freeze=True)
        # conv1 should be frozen, fc2 should still be trainable
        assert not model.conv1.weight.requires_grad
        assert model.fc2.weight.requires_grad

        CurriculumScheduler.freeze_backbone(model, freeze=False)
        assert model.conv1.weight.requires_grad


# ===================================================================
# 3. Gradient Reversal & Domain Adversarial
# ===================================================================

class TestGRL:
    def test_grl_forward_identity(self):
        x = torch.randn(4, 16, requires_grad=True)
        out = gradient_reversal(x, alpha=1.0)
        assert torch.allclose(out, x)

    def test_grl_backward_negates(self):
        x = torch.randn(4, 16, requires_grad=True)
        out = gradient_reversal(x, alpha=1.0)
        out.sum().backward()
        # Gradient should be negated: all -1
        assert torch.allclose(x.grad, -torch.ones_like(x.grad))


class TestDomainClassifier:
    def test_output_shape(self):
        dc = DomainClassifier(feature_dim=128, num_domains=3)
        features = torch.randn(8, 128)
        out = dc(features)
        assert out.shape == (8, 3)


class TestDANNLoss:
    def test_dann_loss_scalar(self):
        class_logits = torch.randn(4, NUM_CLASSES)
        class_labels = torch.randint(0, NUM_CLASSES, (4,))
        domain_logits = torch.randn(4, 3)
        domain_labels = torch.randint(0, 3, (4,))
        loss = dann_loss(class_logits, class_labels, domain_logits, domain_labels)
        assert loss.dim() == 0  # scalar


# ===================================================================
# 4. DANN Trainer
# ===================================================================

class TestDANNTrainer:

    # --- helpers ---

    @staticmethod
    def _make_loader(n: int = 16, batch_size: int = 4):
        images = torch.randn(n, 3, IMAGE_SIZE, IMAGE_SIZE)
        labels = torch.randint(0, NUM_CLASSES, (n,))
        return DataLoader(TensorDataset(images, labels), batch_size=batch_size)

    @staticmethod
    def _make_dann_trainer(model, feature_dim, tmpdir, **kwargs):
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.CrossEntropyLoss()
        domain_cls = DomainClassifier(feature_dim=feature_dim, num_domains=2)
        optimizer.add_param_group({"params": domain_cls.parameters()})
        return DANNTrainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            device=torch.device("cpu"),
            use_amp=False,
            checkpoint_dir=Path(tmpdir),
            domain_classifier=domain_cls,
            lambda_domain=0.1,
            grl_alpha_max=1.0,
            **kwargs,
        )

    # --- tests ---

    def test_extract_features_resnet10(self):
        model = ResNet10()
        x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        feats = model.extract_features(x)
        assert feats.shape == (4, 512)

    def test_extract_features_baseline(self):
        model = BaselineCNN()
        x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        feats = model.extract_features(x)
        assert feats.shape == (4, 256)

    def test_classify_features_consistent_with_forward(self):
        """extract_features + classify_features should match forward() output."""
        model = BaselineCNN()
        model.eval()
        x = torch.randn(4, 3, IMAGE_SIZE, IMAGE_SIZE)
        with torch.no_grad():
            expected = model(x)
            feats = model.extract_features(x)
            got = model.classify_features(feats)
        assert torch.allclose(expected, got, atol=1e-5)

    def test_dann_train_one_epoch_returns_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = BaselineCNN()
            loader = self._make_loader()
            domain_loader = self._make_loader(n=8)
            trainer = self._make_dann_trainer(model, feature_dim=256, tmpdir=tmp)
            metrics = trainer.train_one_epoch_dann(loader, domain_loader)
            for key in ("train_loss", "train_acc", "dann_ce_loss", "dann_dom_loss", "grl_alpha"):
                assert key in metrics, f"Missing key: {key}"
            assert metrics["train_loss"] >= 0
            assert 0.0 <= metrics["grl_alpha"] <= 1.0

    def test_dann_fit_returns_history(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = BaselineCNN()
            loader = self._make_loader()
            domain_loader = self._make_loader(n=8)
            trainer = self._make_dann_trainer(model, feature_dim=256, tmpdir=tmp)
            history = trainer.fit(loader, loader, epochs=2, domain_loader=domain_loader)
            assert len(history["train_loss"]) == 2
            assert len(history["val_loss"]) == 2

    def test_grl_alpha_schedule(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = BaselineCNN()
            trainer = self._make_dann_trainer(model, feature_dim=256, tmpdir=tmp)
            # At step 0, alpha should be ~0
            trainer._dann_step = 0
            trainer._total_dann_steps = 100
            alpha_start = trainer._grl_alpha()
            # At step 100 (end), alpha should be near grl_alpha_max
            trainer._dann_step = 100
            alpha_end = trainer._grl_alpha()
            assert alpha_start < 0.5
            assert alpha_end > 0.9 * trainer.grl_alpha_max

    def test_dann_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = BaselineCNN()
            loader = self._make_loader()
            domain_loader = self._make_loader(n=8)
            trainer = self._make_dann_trainer(model, feature_dim=256, tmpdir=tmp)
            trainer.fit(loader, loader, epochs=2, domain_loader=domain_loader)
            trainer.save_checkpoint(epoch=2, filename="dann_ckpt.pth")

            # Load into a fresh DANNTrainer
            model2 = BaselineCNN()
            trainer2 = self._make_dann_trainer(model2, feature_dim=256, tmpdir=tmp)
            trainer2.load_checkpoint(Path(tmp) / "dann_ckpt.pth")

            # domain_classifier weights should match
            for (n1, p1), (n2, p2) in zip(
                trainer.domain_classifier.named_parameters(),
                trainer2.domain_classifier.named_parameters(),
            ):
                assert torch.allclose(p1, p2), f"Mismatch in {n1}"

    def test_dann_fit_without_domain_loader_falls_back(self):
        """fit() without domain_loader should fall back to supervised Trainer.fit()."""
        with tempfile.TemporaryDirectory() as tmp:
            model = BaselineCNN()
            loader = self._make_loader()
            trainer = self._make_dann_trainer(model, feature_dim=256, tmpdir=tmp)
            history = trainer.fit(loader, loader, epochs=2, domain_loader=None)
            assert len(history["train_loss"]) == 2


# ===================================================================
# 5. LR Schedulers
# ===================================================================

class TestSchedulers:
    def test_build_cosine(self):
        opt = torch.optim.Adam([torch.randn(2, requires_grad=True)], lr=1e-3)
        sched = build_scheduler(opt, name="cosine", total_epochs=40)
        assert sched is not None

    def test_build_step(self):
        opt = torch.optim.Adam([torch.randn(2, requires_grad=True)], lr=1e-3)
        sched = build_scheduler(opt, name="step", total_epochs=30)
        assert sched is not None

    def test_build_none(self):
        opt = torch.optim.Adam([torch.randn(2, requires_grad=True)], lr=1e-3)
        assert build_scheduler(opt, name="none") is None

    def test_unknown_raises(self):
        opt = torch.optim.Adam([torch.randn(2, requires_grad=True)], lr=1e-3)
        with pytest.raises(ValueError):
            build_scheduler(opt, name="foobar")

    def test_warmup_cosine(self):
        opt = torch.optim.Adam([torch.randn(2, requires_grad=True)], lr=1e-3)
        sched = build_warmup_cosine_scheduler(opt, warmup_epochs=3, total_epochs=20)
        assert sched is not None
        # Step through a few epochs without error
        for _ in range(5):
            sched.step()
