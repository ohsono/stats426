"""
Training engine — core forward / backward / optimization loops.

Supports:
    • Single-device training (CPU, MPS, CUDA)
    • Automatic Mixed Precision (AMP) on CUDA
    • Checkpoint saving / loading (cross-device via map_location)
    • Auto-save best model (by val_loss or val_acc)
    • Periodic checkpoint saving every N epochs
    • Early stopping with configurable patience
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.device import get_device
from utils.logger import ExperimentLogger

_log = logging.getLogger("tsr.engine")


class Trainer:
    """Core training and validation engine with auto-save and early stopping."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: Optional[torch.device] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        logger: Optional[ExperimentLogger] = None,
        use_amp: bool = True,
        checkpoint_dir: Optional[Path] = None,
        # Auto-save settings
        save_best: bool = True,
        save_every_n_epochs: int = 5,
        best_metric: str = "val_loss",       # "val_loss" or "val_acc"
        early_stopping_patience: int = 0,    # 0 = disabled
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.logger = logger
        self.checkpoint_dir = checkpoint_dir

        # AMP (only effective on CUDA)
        self.use_amp = use_amp and self.device.type == "cuda"
        self.scaler = torch.amp.GradScaler("cuda") if self.use_amp else None

        self._global_step = 0

        # Auto-save state
        self.save_best = save_best
        self.save_every_n_epochs = save_every_n_epochs
        self.best_metric = best_metric
        self._best_value: Optional[float] = None
        self._best_epoch: int = 0

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self._epochs_no_improve = 0
        self.stopped_early = False

    # ------------------------------------------------------------------
    # Single epoch
    # ------------------------------------------------------------------
    def train_one_epoch(self, loader: DataLoader) -> Dict[str, float]:
        """Run one epoch of training.  Returns a metrics dict."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        t0 = time.time()

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast("cuda"):
                    outputs = self.model(images)
                    loss = self.criterion(outputs, labels)
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)
            self._global_step += 1

        epoch_loss = running_loss / max(total, 1)
        epoch_acc = correct / max(total, 1)
        elapsed = time.time() - t0

        metrics = {
            "train_loss": epoch_loss,
            "train_acc": epoch_acc,
            "train_time": elapsed,
        }
        if self.logger:
            self.logger.log_metrics(metrics, step=self._global_step)
        return metrics

    @torch.no_grad()
    def validate(self, loader: DataLoader) -> Dict[str, float]:
        """Run validation.  Returns a metrics dict."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum().item()
            total += images.size(0)

        val_loss = running_loss / max(total, 1)
        val_acc = correct / max(total, 1)

        metrics = {"val_loss": val_loss, "val_acc": val_acc}
        if self.logger:
            self.logger.log_metrics(metrics, step=self._global_step)
        return metrics

    # ------------------------------------------------------------------
    # Full training loop with auto-save
    # ------------------------------------------------------------------
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 1,
    ) -> Dict[str, list]:
        """
        Full training loop with automatic best-model saving, periodic
        checkpointing, and optional early stopping.

        Returns a history dict with per-epoch metrics lists.
        """
        total_epochs = start_epoch + epochs - 1
        history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

        # Log model summary at the start of training
        if self.logger:
            self.logger.log_model_summary(self.model)

        for epoch in range(start_epoch, start_epoch + epochs):
            train_metrics = self.train_one_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            # Step the LR scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            # Record history
            for k in history:
                if k in train_metrics:
                    history[k].append(train_metrics[k])
                elif k in val_metrics:
                    history[k].append(val_metrics[k])

            _log.info(
                "Epoch %d/%d — train_loss=%.4f train_acc=%.4f "
                "val_loss=%.4f val_acc=%.4f",
                epoch, total_epochs,
                train_metrics["train_loss"], train_metrics["train_acc"],
                val_metrics["val_loss"], val_metrics["val_acc"],
            )

            # Verbose per-epoch logging
            if self.logger:
                self.logger.log_epoch(epoch, total_epochs, train_metrics, val_metrics)

            # --- Auto-save best model ---
            if self.save_best:
                self._check_and_save_best(epoch, val_metrics)

            # --- Periodic checkpoint ---
            if self.save_every_n_epochs > 0 and epoch % self.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, filename=f"checkpoint_epoch_{epoch}.pth")

            # --- Early stopping (checked immediately after best-model update) ---
            if (self.early_stopping_patience > 0
                    and self._epochs_no_improve >= self.early_stopping_patience):
                _log.info(
                    "Early stopping triggered at epoch %d "
                    "(no improvement for %d epochs)",
                    epoch, self.early_stopping_patience,
                )
                if self.logger:
                    self.logger.log_early_stop(epoch, self.early_stopping_patience)
                self.stopped_early = True
                break

        # Always save final checkpoint
        self.save_checkpoint(epoch, filename="checkpoint_last.pth")
        return history

    # ------------------------------------------------------------------
    # Best-model tracking
    # ------------------------------------------------------------------
    def _check_and_save_best(self, epoch: int, val_metrics: Dict[str, float]):
        """Compare current metrics to the best seen and save if improved."""
        current = val_metrics.get(self.best_metric, val_metrics.get("val_loss", 0))

        # For loss: lower is better. For accuracy: higher is better.
        is_loss = "loss" in self.best_metric

        improved = False
        if self._best_value is None:
            improved = True
        elif is_loss and current < self._best_value:
            improved = True
        elif not is_loss and current > self._best_value:
            improved = True

        if improved:
            self._best_value = current
            self._best_epoch = epoch
            self._epochs_no_improve = 0
            self.save_checkpoint(epoch, filename="best_model.pth")
            self._save_model_only(filename="best_model_weights.pth")
            _log.info(
                "✓ New best %s=%.6f at epoch %d — saved best_model.pth",
                self.best_metric, current, epoch,
            )
            if self.logger and self.checkpoint_dir:
                self.logger.log_checkpoint(
                    self.checkpoint_dir / "best_model.pth", epoch, is_best=True
                )
        else:
            self._epochs_no_improve += 1

    def _save_model_only(self, filename: str = "model.pth"):
        """Save just the model weights (no optimizer state) for deployment."""
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename
        torch.save(self.model.state_dict(), path)

    @property
    def best_value(self) -> Optional[float]:
        return self._best_value

    @property
    def best_epoch(self) -> int:
        return self._best_epoch

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------
    def save_checkpoint(self, epoch: int, filename: str = "checkpoint.pth"):
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self._global_step,
            "best_value": self._best_value,
            "best_epoch": self._best_epoch,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: Path):
        """Load a checkpoint with device-agnostic map_location.

        If the saved optimizer has a different number of param groups than the
        current optimizer (e.g. transitioning between supervised and DANN), the
        optimizer state is skipped and the optimizer starts fresh.  Model weights
        are always restored.
        """
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        try:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        except ValueError as exc:
            if "different number of parameter groups" in str(exc):
                _log.warning(
                    "Checkpoint optimizer param groups (%d) != current (%d); "
                    "skipping optimizer state — optimizer will start fresh.",
                    len(state["optimizer_state_dict"]["param_groups"]),
                    len(self.optimizer.param_groups),
                )
            else:
                raise
        self._global_step = state.get("global_step", 0)
        self._best_value = state.get("best_value")
        self._best_epoch = state.get("best_epoch", 0)
        if self.scheduler and "scheduler_state_dict" in state:
            saved_sched = state["scheduler_state_dict"]
            saved_base_lrs = saved_sched.get("base_lrs", [])
            if len(saved_base_lrs) == len(self.optimizer.param_groups):
                try:
                    self.scheduler.load_state_dict(saved_sched)
                except Exception:
                    pass
            else:
                _log.warning(
                    "Skipping scheduler state restore: saved base_lrs length %d "
                    "!= current optimizer param groups %d; scheduler will start fresh.",
                    len(saved_base_lrs), len(self.optimizer.param_groups),
                )
        return state.get("epoch", 0)


# ---------------------------------------------------------------------------
# Domain-Adversarial Neural Network Trainer (Phase 3 – Stage 3)
# ---------------------------------------------------------------------------

class DANNTrainer(Trainer):
    """
    Extends Trainer with Domain-Adversarial Neural Network (DANN) training.

    Uses a Gradient Reversal Layer to simultaneously minimise classification
    loss and maximise domain confusion, producing domain-invariant features.
    """

    def __init__(
        self,
        *args,
        domain_classifier: nn.Module,
        lambda_domain: float = 0.1,
        grl_alpha_max: float = 1.0,
        total_dann_steps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.domain_classifier = domain_classifier.to(self.device)
        self.lambda_domain = lambda_domain
        self.grl_alpha_max = grl_alpha_max
        self._dann_step = 0
        self._total_dann_steps = total_dann_steps or 1

    def _grl_alpha(self) -> float:
        """Smooth ramp from 0 → grl_alpha_max using the standard DANN schedule."""
        p = self._dann_step / self._total_dann_steps
        return self.grl_alpha_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

    def train_one_epoch_dann(
        self,
        train_loader: DataLoader,
        domain_loader: DataLoader,
    ) -> Dict[str, float]:
        """One DANN training epoch.  Returns metrics dict."""
        self.model.train()
        self.domain_classifier.train()
        running_loss = running_ce = running_dom = 0.0
        correct = total = 0

        domain_iter = iter(domain_loader)
        for images_s, labels_s in train_loader:
            # Cycle domain loader if shorter than supervised loader
            try:
                images_d, _ = next(domain_iter)
            except StopIteration:
                domain_iter = iter(domain_loader)
                images_d, _ = next(domain_iter)

            images_s = images_s.to(self.device)
            labels_s = labels_s.to(self.device)
            images_d = images_d.to(self.device)
            alpha = self._grl_alpha()

            self.optimizer.zero_grad()

            # Feature extraction (shared backbone)
            feat_s = self.model.extract_features(images_s)   # (B_s, D)
            feat_d = self.model.extract_features(images_d)   # (B_d, D)

            # Classification loss — supervised source batches only
            class_logits = self.model.classify_features(feat_s)
            ce_loss = self.criterion(class_logits, labels_s)

            # Domain adversarial loss — binary: source=0, target=1
            dom_logits_s = self.domain_classifier(feat_s, alpha)
            dom_logits_d = self.domain_classifier(feat_d, alpha)
            dom_labels_s = torch.zeros(feat_s.size(0), dtype=torch.long, device=self.device)
            dom_labels_d = torch.ones(feat_d.size(0), dtype=torch.long, device=self.device)
            dom_logits = torch.cat([dom_logits_s, dom_logits_d])
            dom_labels = torch.cat([dom_labels_s, dom_labels_d])
            dom_loss = nn.CrossEntropyLoss()(dom_logits, dom_labels)

            loss = ce_loss + self.lambda_domain * dom_loss
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images_s.size(0)
            running_ce   += ce_loss.item() * images_s.size(0)
            running_dom  += dom_loss.item() * images_s.size(0)
            _, preds = class_logits.max(1)
            correct += preds.eq(labels_s).sum().item()
            total += images_s.size(0)
            self._dann_step += 1
            self._global_step += 1

        n = max(total, 1)
        metrics = {
            "train_loss": running_loss / n,
            "train_acc": correct / n,
            "dann_ce_loss": running_ce / n,
            "dann_dom_loss": running_dom / n,
            "grl_alpha": alpha,
        }
        if self.logger:
            self.logger.log_metrics(metrics, step=self._global_step)
        return metrics

    def fit(  # type: ignore[override]
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        start_epoch: int = 1,
        domain_loader: Optional[DataLoader] = None,
    ) -> Dict[str, list]:
        """
        DANN-aware fit loop.  Falls back to supervised Trainer.fit() when
        no domain_loader is provided.
        """
        if domain_loader is None:
            return super().fit(train_loader, val_loader, epochs, start_epoch)

        # Pre-compute total DANN steps for the alpha schedule
        steps_per_epoch = max(len(train_loader), 1)
        self._total_dann_steps = max(epochs * steps_per_epoch, 1)

        history: Dict[str, list] = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }
        if self.logger:
            self.logger.log_model_summary(self.model)

        for epoch in range(start_epoch, start_epoch + epochs):
            train_metrics = self.train_one_epoch_dann(train_loader, domain_loader)
            val_metrics = self.validate(val_loader)

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics["val_loss"])
                else:
                    self.scheduler.step()

            for k in history:
                if k in train_metrics:
                    history[k].append(train_metrics[k])
                elif k in val_metrics:
                    history[k].append(val_metrics[k])

            _log.info(
                "DANN Epoch %d/%d — ce=%.4f dom=%.4f alpha=%.3f val_acc=%.4f",
                epoch, start_epoch + epochs - 1,
                train_metrics["dann_ce_loss"], train_metrics["dann_dom_loss"],
                train_metrics["grl_alpha"], val_metrics["val_acc"],
            )
            if self.logger:
                self.logger.log_epoch(epoch, start_epoch + epochs - 1,
                                      train_metrics, val_metrics)
            if self.save_best:
                self._check_and_save_best(epoch, val_metrics)
            if self.save_every_n_epochs > 0 and epoch % self.save_every_n_epochs == 0:
                self.save_checkpoint(epoch, f"checkpoint_epoch_{epoch}.pth")
            if (self.early_stopping_patience > 0
                    and self._epochs_no_improve >= self.early_stopping_patience):
                self.stopped_early = True
                break

        self.save_checkpoint(epoch, "checkpoint_last.pth")
        return history

    # ------------------------------------------------------------------
    # Override checkpoint to also persist domain_classifier
    # ------------------------------------------------------------------

    def save_checkpoint(self, epoch: int, filename: str = "checkpoint.pth"):
        if self.checkpoint_dir is None:
            return
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        path = self.checkpoint_dir / filename
        state = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "global_step": self._global_step,
            "best_value": self._best_value,
            "best_epoch": self._best_epoch,
            "domain_classifier_state_dict": self.domain_classifier.state_dict(),
            "dann_step": self._dann_step,
        }
        if self.scheduler is not None:
            state["scheduler_state_dict"] = self.scheduler.state_dict()
        torch.save(state, path)

    def load_checkpoint(self, path: Path):
        """Load checkpoint, restoring domain_classifier state if present."""
        epoch = super().load_checkpoint(path)
        state = torch.load(path, map_location=self.device, weights_only=False)
        if "domain_classifier_state_dict" in state:
            self.domain_classifier.load_state_dict(state["domain_classifier_state_dict"])
        self._dann_step = state.get("dann_step", 0)
        return epoch
