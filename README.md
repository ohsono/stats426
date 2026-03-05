# Traffic Sign Recognition — UCLA MASDS STAT426

A PyTorch-based traffic sign classification system that trains and evaluates CNN models on a unified multi-source dataset (DOT, GTSRB, LISA, BDD100K). Designed for device-agnostic execution on CUDA, Apple Silicon (MPS), and CPU.

---

## Overview

| Item | Detail |
|------|--------|
| **Task** | 58-class traffic sign classification |
| **Label space** | DOT canonical indices 0–57 (non-contiguous) |
| **Input** | 64×64 RGB images, ImageNet normalization |
| **Datasets** | DOT, GTSRB, LISA (domain), BDD100K (domain) |
| **Models** | BaselineCNN → AdvancedCNN (STN) → ResNet10 → OrionVLM |
| **Training** | Curriculum staged training + optional DANN domain adaptation |

---

## Repository Structure

```
stats426/
├── project/                       # Main source code
│   ├── main.py                    # CLI entry point (info / train / evaluate)
│   ├── requirements.txt
│   ├── setup_env.sh               # Conda/pyenv environment bootstrap
│   ├── .gitignore
│   │
│   ├── data/
│   │   ├── datasets.py            # DOTDataset, GTSRBDataset, LISADataset, BDD100KDataset
│   │   ├── dataloaders.py         # stratified_split (70/10/10/10) + DataLoader factory
│   │   ├── transforms.py          # Per-dataset augmentation pipelines
│   │   ├── unify.py               # DOT label space + cross-dataset label maps
│   │   └── preprocess_bdd100k.py  # Offline BDD100K crop extractor
│   │
│   ├── models/
│   │   ├── baseline.py            # BaselineCNN (3 conv layers)
│   │   ├── advanced.py            # AdvancedCNN (CNN + SpatialTransformerNetwork)
│   │   ├── resnet.py              # ResNet10 (modified 3×3 stride-1 stem)
│   │   └── orion_vlm.py           # OrionVLMStub (LoRA VQA stub)
│   │
│   ├── training/
│   │   ├── engine.py              # Trainer: train/val loop, AMP, checkpointing, early stopping
│   │   ├── curriculum.py          # CurriculumScheduler: staged dataset introduction
│   │   └── domain_adv.py          # Gradient Reversal Layer + DomainClassifier (DANN)
│   │
│   ├── evaluation/
│   │   ├── metrics.py             # Precision/Recall/F1, critical class recall (safety signs)
│   │   ├── calibration.py         # ECE, reliability diagrams
│   │   └── ood_testing.py         # OOD degradation analysis
│   │
│   ├── utils/
│   │   ├── device.py              # Auto-selects cuda > mps > cpu
│   │   ├── config.py              # Dataclass config (Config, DataConfig, TrainConfig, EvalConfig)
│   │   └── logger.py              # TensorBoard integration
│   │
│   └── tests/                     # pytest suite
│
├── docs/                          # Course materials and documentation
│   ├── course/
│   │   ├── HW/                    # Homework assignments (HW1–HW4)
│   │   ├── Lecture/               # Lecture notes and summaries
│   │   └── Quiz/                  # Quiz materials and answer keys
│   └── proposal/                  # Project proposal documents
│
├── .env                           # Local environment config (not committed)
└── .env.example                   # Config template
```

---

## Datasets

### DOT (primary labeled source)
- 58-class canonical label space (indices 0–57, non-contiguous)
- Small annotated set (~43 images), augmented heavily during training

### GTSRB — German Traffic Sign Recognition Benchmark
- 43-class German signs mapped to DOT indices via `GTSRB_TO_DOT`
- Expected format: Kaggle CSV (`Train.csv`, `Test.csv`) with ROI columns
- Path: `$DATA_DIR/gtsrb/`

### LISA — Laboratory for Intelligent and Safe Automobiles
- Traffic light dataset; **no fine-grained sign classes**
- Used as domain-adaptation source only (`label=-1`)
- Path: `$DATA_DIR/lisa/` (auto-extracted from `lisa-traffic-light-dataset.zip`)

### BDD100K
- Dashcam dataset with `"traffic sign"` bounding boxes (generic — no fine-grained type)
- Used as domain-adaptation source only (`label=-1`)
- Two load modes:
  - **Live scan** (default): reads `100k/train/*.json`, crops on-the-fly
  - **Pre-extracted**: run `preprocess_bdd100k.py` once, then loads from `annotations.json`
- Path: `$DATA_DIR/BDD_100K/`

To pre-extract BDD100K crops:
```bash
cd project
python data/preprocess_bdd100k.py \
    --bdd100k-dir /path/to/BDD_100K \
    --output-dir  /path/to/BDD_100K/preextracted \
    --split train --min-size 16
```

---

## Models

| Model | Params (approx.) | Notes |
|-------|-----------------|-------|
| `baseline` | ~200K | 3 conv + 2 FC layers |
| `advanced` | ~400K | Adds Spatial Transformer Network (STN) |
| `resnet10` | ~2M | Modified ResNet with 3×3 stride-1 stem for 64×64 inputs |
| `orion` | ~50M+ | VLM stub with LoRA (experimental) |

All models output raw logits for 58 classes (`CrossEntropyLoss`).

---

## Setup

### 1. Environment

```bash
cd project
source setup_env.sh      # auto-detects conda or pyenv
pip install -r requirements.txt
```

### 2. Configuration

Copy `.env.example` to `.env` (at the repo root) and set dataset paths:

```bash
cp .env.example .env
# Edit .env:
DATA_DIR=/path/to/your/datasets
CHECKPOINT_DIR=./project/checkpoints
LOG_DIR=./project/logs
```

### 3. Dataset layout expected

```
$DATA_DIR/
├── dot/                         # DOT traffic signs
├── gtsrb/
│   ├── Train.csv
│   ├── Test.csv
│   └── Train/00000/*.png ...
├── lisa/
│   └── lisa-traffic-light-dataset.zip   # auto-extracted
└── BDD_100K/
    └── 100k/
        ├── train/*.jpg  *.json           # 70K pairs
        └── val/*.jpg    *.json
```

---

## Quick Start

```bash
cd project

# Check detected device and dataset info
python main.py info

# Train ResNet10 (recommended) for 30 epochs
python main.py train --model resnet10 --epochs 30 --batch-size 64 --patience 10 --verbose

# Resume training from last checkpoint
python main.py train --model resnet10 --continue --epochs 20

# Evaluate on test split
python main.py evaluate --model resnet10 --split test \
    --checkpoint checkpoints/resnet10/best_model.pth

# Run all tests
python -m pytest tests/ -v
```

---

## CLI Reference

### `python main.py info`
Prints device info, dataset paths, and detected hardware.

### `python main.py train`

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `resnet10` | `baseline`, `advanced`, `resnet10`, `orion` |
| `--stage` | `full` | `geometric`, `real_world`, `domain_adv`, `full` |
| `--epochs` | `30` | Number of training epochs |
| `--batch-size` | `32` | Batch size |
| `--lr` | `1e-3` | Learning rate |
| `--patience` | `10` | Early stopping patience (0 = disabled) |
| `--continue` | — | Resume from latest checkpoint |
| `--verbose` | — | Print per-batch loss |

**Training stages** (`--stage`):
- `geometric` — GTSRB + DOT supervised (clean, augmented)
- `real_world` — adds BDD100K domain loader (label=-1, unsupervised)
- `domain_adv` — enables DANN gradient reversal
- `full` — all of the above in sequence

### `python main.py evaluate`

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | `resnet10` | Model architecture |
| `--split` | `test` | `val`, `test`, `ood` |
| `--checkpoint` | auto-detect | Path to `.pth` checkpoint |

---

## Configuration

All hyperparameters live in `project/utils/config.py` dataclasses:

```python
@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 10
    # ...

@dataclass
class DataConfig:
    train_ratio: float = 0.70
    val_ratio: float = 0.10
    test_ratio: float = 0.10
    ood_ratio: float = 0.10
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
```

---

## Evaluation Metrics

- **Top-1 accuracy** — primary metric
- **Per-class F1** — macro and weighted
- **Critical class recall** — recall on safety-critical signs (stop, yield, speed limit)
- **ECE** (Expected Calibration Error) — confidence calibration
- **OOD degradation** — accuracy drop on the held-out 10% OOD split vs. test split

---

## Training Results

| Model | Val Acc | Epochs | Notes |
|-------|---------|--------|-------|
| ResNet10 | 98.3% | 10 | DOT + GTSRB supervised; early stop |
| BaselineCNN | ~90% | 30 | DOT + GTSRB |
| AdvancedCNN (STN) | ~93% | 30 | DOT + GTSRB |

---

## GPU Notes

- **CUDA 13.x required** for Blackwell (SM 12.1) GPUs
- Install: `pip install torch==2.10.0+cu130 torchvision==0.25.0+cu130 --index-url https://download.pytorch.org/whl/cu130`
- AMP (Automatic Mixed Precision) is enabled automatically on CUDA; disabled on MPS/CPU
- MPS fallback: `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in `main.py` for ops missing on Metal (e.g., `grid_sampler_2d_backward` in STN)
- If GPU OOM occurs, reduce `--batch-size` or kill competing processes before running training

---

## Testing

```bash
cd project

# Full suite
python -m pytest tests/ -v

# Single test file
python -m pytest tests/test_models.py -v

# Single test class or method
python -m pytest tests/test_models.py::TestResNet10 -v
python -m pytest tests/test_models.py::TestResNet10::test_output_shape -v
```

---

## License

UCLA MASDS STAT426 course project. For academic use only.
