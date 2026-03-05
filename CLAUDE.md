# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Traffic Sign Recognition system for UCLA MASDS STAT426. A monolithic PyTorch application that trains and evaluates CNN models on unified traffic sign datasets (DOT, GTSRB, LISA, BDD100K). Designed for device-agnostic execution across CUDA, Apple Silicon (MPS), and CPU.

## Repository Layout

```
stats426/
├── project/       # All source code (entry point, models, training, evaluation, tests)
├── docs/
│   ├── course/    # HW/, Lecture/, Quiz/ — course materials
│   └── proposal/  # Project proposal documents
├── .env           # Local config (DATA_DIR, CHECKPOINT_DIR, LOG_DIR)
└── .env.example
```

## Key Commands

All commands run from `project/`:

```bash
cd project

# Environment setup (conda or pyenv auto-detected)
source setup_env.sh

# Install dependencies
pip install -r requirements.txt

# Check device info
python main.py info

# Train models (baseline, advanced, resnet10, orion)
python main.py train --model resnet10 --stage full --verbose
python main.py train --model baseline --stage geometric --epochs 20
python main.py train --model advanced --epochs 30 --patience 10
python main.py train --model resnet10 --continue --epochs 10  # resume from checkpoint

# Evaluate
python main.py evaluate --model resnet10 --split test --checkpoint checkpoints/resnet10/best_model.pth

# Run all tests
python -m pytest tests/ -v

# Run a single test file or test class
python -m pytest tests/test_models.py -v
python -m pytest tests/test_models.py::TestResNet10 -v
python -m pytest tests/test_models.py::TestResNet10::test_output_shape -v
```

## Architecture

**Entry point**: `project/main.py` — CLI with subcommands `info`, `train`, `evaluate`. Models are lazy-imported via `build_model()`.

**Four-package structure** (mirrors the four project phases):

- `project/data/` — Dataset classes (`DOTDataset`, `GTSRBDataset`, `LISADataset`, `BDD100KDataset`), all inheriting from `TrafficSignDataset`. `unify.py` defines the canonical DOT label space (0-57, 58 classes) and cross-dataset label maps (`GTSRB_TO_DOT`, `LISA_LABEL_MAP`, `BDD100K_LABEL_MAP`). Transforms in `transforms.py` are domain-specific (heavier augmentation for BDD100K dashcam, lighter for clean GTSRB).

- `project/models/` — Progressive scaling: `BaselineCNN` (3 conv layers) → `AdvancedCNN` (CNN + `SpatialTransformerNetwork`) → `ResNet10` (modified stem: 3x3 stride-1 to preserve spatial info at 64x64) → `OrionVLMStub` (VQA with `LoRALinear`). All default to `num_classes=58`, `image_size=64`.

- `project/training/` — `Trainer` in `engine.py` handles the train/val loop with AMP (CUDA only), checkpoint save/load, early stopping, and best-model tracking. `CurriculumScheduler` manages staged dataset introduction (geometric → real-world → domain adversarial). `domain_adv.py` implements gradient reversal for domain-invariant features.

- `project/evaluation/` — `metrics.py` (precision/recall/F1, critical class recall for safety signs), `calibration.py` (ECE, reliability diagrams), `ood_testing.py` (OOD degradation analysis).

**Cross-cutting modules** in `project/utils/`: `device.py` (auto-selects cuda > mps > cpu), `config.py` (dataclass-based `Config`/`DataConfig`/`TrainConfig`/`EvalConfig`, paths from `.env`), `logger.py` (TensorBoard integration).

## Critical Conventions

- **Label space**: All datasets map to DOT unified indices (0-57). The label indices have gaps (e.g., 16, 17 exist but 22, 23 don't). `num_classes()` returns 58 (max index + 1). Never assume contiguous labels.
- **Image size**: Standard input is 64x64 RGB. All transforms resize to this. Models assume `(B, 3, 64, 64)` input tensors.
- **ImageNet normalization**: All transforms normalize with `mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`.
- **Loss function**: `CrossEntropyLoss` (models output raw logits, not softmax).
- **MPS fallback**: `main.py` sets `PYTORCH_ENABLE_MPS_FALLBACK=1` for ops not yet on Metal (e.g., `grid_sampler_2d_backward` used by STN).
- **Checkpoint format**: Dicts with keys `model_state_dict`, `optimizer_state_dict`, `epoch`, `global_step`, `best_value`, `best_epoch`. Loaded with `map_location=device` for cross-device compatibility.
- **Data split**: 70-10-10-10 (train/val/test/OOD challenge). The OOD split is for measuring real-world robustness degradation.

## Configuration

Paths are set via `.env` at the repo root (see `.env.example`): `DATA_DIR`, `CHECKPOINT_DIR`, `LOG_DIR`. Defaults resolve to `./dataset`, `./checkpoints`, `./logs` relative to `project/`. Hyperparameters live in `project/utils/config.py` dataclasses.

## Git Workflow

- Default branch: `main`
- Data directories (`dataset/`, `checkpoints/`, `logs/`) are gitignored

## Course Materials

Course-related files are in `docs/`:
- `docs/course/HW/` — Homework assignments (HW1–HW4)
- `docs/course/Lecture/` — Lecture PDFs and summaries
- `docs/course/Quiz/` — Quiz PDFs and answer keys
- `docs/proposal/` — Project proposal documents

## Legacy HW Code

`docs/course/HW/HW1/` contains standalone MNIST binary classification scripts (separate from the main pipeline). These use relative CSV paths — run from that directory. Uses `BCEWithLogitsLoss` (binary), unlike the main pipeline's `CrossEntropyLoss` (multi-class).
