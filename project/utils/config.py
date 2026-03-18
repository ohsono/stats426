"""
Centralized configuration module.

All hyperparameters, paths, and project-wide constants live here so that
every other module imports from a single source of truth.

Paths are loaded from a ``.env`` file (via python-dotenv) so they can be
changed without modifying code.  Falls back to sensible defaults.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

# Load .env file if python-dotenv is available
try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    load_dotenv(_env_path)
except ImportError:
    pass  # dotenv not installed — use env vars or defaults


# ---------------------------------------------------------------------------
# Paths (from .env → environment → defaults)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "dataset"))
if not DATA_DIR.is_absolute():
    DATA_DIR = (PROJECT_ROOT / DATA_DIR).resolve()

CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", PROJECT_ROOT / "checkpoints"))
if not CHECKPOINT_DIR.is_absolute():
    CHECKPOINT_DIR = (PROJECT_ROOT / CHECKPOINT_DIR).resolve()

LOG_DIR = Path(os.environ.get("LOG_DIR", PROJECT_ROOT / "logs"))
if not LOG_DIR.is_absolute():
    LOG_DIR = (PROJECT_ROOT / LOG_DIR).resolve()

# Ensure critical dirs exist
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Label / Class Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 58  # Unified label space (0-57)
IMAGE_SIZE = 64   # Default resize target


# ---------------------------------------------------------------------------
# Dataclass-based config for easy CLI / YAML override
# ---------------------------------------------------------------------------
@dataclass
class DataConfig:
    """Settings for the data pipeline."""
    image_size: int = IMAGE_SIZE
    num_classes: int = NUM_CLASSES
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    ood_ratio: float = 0.10
    batch_size: int = 64
    num_workers: int = 4
    pin_memory: bool = True
    data_dir: Path = DATA_DIR


@dataclass
class TrainConfig:
    """Settings for training."""
    epochs_stage1: int = 20      # GTSRB + LISA only
    epochs_stage2: int = 20      # + BDD100K
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    use_amp: bool = True         # Automatic Mixed Precision
    checkpoint_dir: Path = CHECKPOINT_DIR
    log_dir: Path = LOG_DIR
    curriculum_stages: List[str] = field(
        default_factory=lambda: ["geometric", "real_world", "domain_adv"]
    )
    # DANN (Stage 3)
    epochs_stage3: int = 20
    enable_domain_adv: bool = False
    lambda_domain: float = 0.1    # weight for domain loss
    grl_alpha_max: float = 1.0    # final GRL alpha (linearly ramped)
    num_domains: int = 2          # binary: source=0, target=1


@dataclass
class EvalConfig:
    """Settings for evaluation."""
    ece_n_bins: int = 15
    temperature: float = 1.0     # Post-hoc Temperature Scaling initial value
    critical_classes: List[str] = field(
        default_factory=lambda: ["stop", "yield", "do_not_enter"]
    )


@dataclass
class Config:
    """Top-level config aggregating all sub-configs."""
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    model_name: str = "resnet50"  # baseline | advanced | resnet50 | orion
    seed: int = 42
