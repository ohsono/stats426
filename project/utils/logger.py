"""
Experiment logger module.

Provides structured logging to local files with the naming convention:
    ``{function}-{model}-{datetime}.log``

When verbose mode is enabled, logs per-epoch training metrics, evaluation
results, model shape info, and checkpoint events.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch.nn as nn

from utils.config import LOG_DIR


# ---------------------------------------------------------------------------
# File-based verbose logger factory
# ---------------------------------------------------------------------------

def create_log_file(
    function: str,
    model_name: str,
    log_dir: Path = LOG_DIR,
) -> Path:
    """
    Create a log file path with the convention:
        ``{function}-{model}-{YYYYMMDD_HHMMSS}.log``
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{function}-{model_name}-{ts}.log"
    return log_dir / filename


def setup_logging(
    name: str = "tsr",
    level: int = logging.INFO,
    log_file: Optional[Path] = None,
    verbose: bool = False,
) -> logging.Logger:
    """
    Configure and return a logger with console + optional file handler.

    Parameters
    ----------
    name : Logger name.
    level : Logging level.
    log_file : If provided, attach a FileHandler writing to this path.
    verbose : If True, set level to DEBUG for more detailed output.
    """
    logger = logging.getLogger(name)

    # Clear existing handlers to avoid duplication on re-init
    logger.handlers.clear()

    if verbose:
        level = logging.DEBUG
    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(level)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (when log_file is provided)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# Model shape logger
# ---------------------------------------------------------------------------

def log_model_summary(
    logger: logging.Logger,
    model: nn.Module,
    model_name: str = "model",
):
    """Log model architecture summary: layer names, shapes, and param counts."""
    total_params = 0
    trainable = 0

    logger.info("=" * 60)
    logger.info("MODEL SUMMARY: %s", model_name)
    logger.info("=" * 60)
    logger.info("%-40s %-20s %s", "Layer", "Shape", "Params")
    logger.info("-" * 60)

    for name, param in model.named_parameters():
        p_count = param.numel()
        total_params += p_count
        if param.requires_grad:
            trainable += p_count
        logger.info("%-40s %-20s %s", name, str(list(param.shape)), f"{p_count:,}")

    logger.info("-" * 60)
    logger.info("Total parameters:     %s", f"{total_params:,}")
    logger.info("Trainable parameters: %s", f"{trainable:,}")
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# Experiment Logger (enhanced)
# ---------------------------------------------------------------------------

class ExperimentLogger:
    """
    Experiment tracker with JSON-lines metrics, verbose file logging,
    and optional TensorBoard integration.
    """

    def __init__(
        self,
        log_dir: Path = LOG_DIR,
        experiment_name: str = "default",
        function: str = "train",
        model_name: str = "model",
        verbose: bool = False,
    ):
        self.log_dir = Path(log_dir)
        self.experiment_name = experiment_name
        self.function = function
        self.model_name = model_name
        self.verbose = verbose

        # Metrics JSONL dir
        self._metrics_dir = self.log_dir / experiment_name
        self._metrics_dir.mkdir(parents=True, exist_ok=True)
        self._metrics_file = self._metrics_dir / "metrics.jsonl"

        # Verbose log file
        self._log_file: Optional[Path] = None
        if verbose:
            self._log_file = create_log_file(function, model_name, self.log_dir)

        self._logger = setup_logging(
            f"tsr.{experiment_name}",
            log_file=self._log_file,
            verbose=verbose,
        )
        self._step = 0
        self._tb_writer: Optional[Any] = None

        if verbose and self._log_file:
            self._logger.info(
                "Verbose logging enabled → %s", self._log_file
            )

    @property
    def log_file_path(self) -> Optional[Path]:
        """Return the path to the verbose log file (None if not verbose)."""
        return self._log_file

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Append a metrics dict as a JSON-lines entry and log verbosely."""
        if step is not None:
            self._step = step
        record = {"step": self._step, **metrics}

        with open(self._metrics_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Pretty-print metrics
        parts = [f"{k}={v:.6f}" if isinstance(v, float) else f"{k}={v}"
                 for k, v in metrics.items()]
        self._logger.info("step=%d  %s", self._step, "  ".join(parts))
        self._step += 1

        # Mirror to TensorBoard if active
        if self._tb_writer is not None:
            for k, v in metrics.items():
                self._tb_writer.add_scalar(k, v, self._step)

    def log_epoch(
        self,
        epoch: int,
        total_epochs: int,
        train_metrics: Dict[str, float],
        val_metrics: Dict[str, float],
    ):
        """Log a full epoch summary (designed for verbose per-epoch output)."""
        self._logger.info("─" * 60)
        self._logger.info("EPOCH %d / %d", epoch, total_epochs)
        self._logger.info("─" * 60)
        for k, v in train_metrics.items():
            self._logger.info("  [train] %-20s %.6f", k, v)
        for k, v in val_metrics.items():
            self._logger.info("  [val]   %-20s %.6f", k, v)

    def log_checkpoint(self, path: Path, epoch: int, is_best: bool = False):
        """Log a checkpoint save event."""
        tag = "BEST " if is_best else ""
        self._logger.info(
            "💾 %sCheckpoint saved → %s  (epoch %d)", tag, path, epoch
        )

    def log_early_stop(self, epoch: int, patience: int):
        """Log an early-stopping event."""
        self._logger.info(
            "⏹ Early stopping at epoch %d (patience=%d exhausted)", epoch, patience
        )

    def log_evaluation(self, split: str, report: Dict):
        """Log evaluation metrics for a split (test / ood)."""
        self._logger.info("=" * 60)
        self._logger.info("EVALUATION — %s split", split.upper())
        self._logger.info("=" * 60)
        for key, val in report.items():
            if isinstance(val, dict):
                parts = "  ".join(f"{k}={v:.4f}" for k, v in val.items())
                self._logger.info("  %-30s %s", key, parts)
            else:
                self._logger.info("  %-30s %.4f", key, val)

    def log_model_summary(self, model: nn.Module):
        """Convenience: log model architecture summary."""
        log_model_summary(self._logger, model, self.model_name)

    def log_text(self, tag: str, text: str):
        self._logger.info("[%s] %s", tag, text)

    # ------------------------------------------------------------------
    # TensorBoard helpers
    # ------------------------------------------------------------------
    def enable_tensorboard(self):
        """Lazy-import TensorBoard to avoid hard dependency."""
        try:
            from torch.utils.tensorboard import SummaryWriter
            self._tb_writer = SummaryWriter(log_dir=str(self._metrics_dir / "tb"))
            self._logger.info("TensorBoard enabled at %s", self._metrics_dir / "tb")
        except ImportError:
            self._logger.warning(
                "tensorboard not installed — skipping TensorBoard logging."
            )

    def close(self):
        if self._tb_writer is not None:
            self._tb_writer.close()
        # Flush file handlers
        for h in self._logger.handlers:
            h.flush()
