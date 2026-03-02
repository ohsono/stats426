# Software Architecture Design

## 1. Overview
This document defines the software architecture for the unified traffic sign recognition system. Based on the plans established in Phases 1-4, the system is designed as a **Monolithic Application** capable of handling data harmonization, progressive model scaling, cross-domain curriculum learning, and trustworthy evaluation.

The architecture is carefully designed to be **environment-independent** at the logic level, but **environment-aware** at the execution level, seamlessly supporting two primary target systems:
1.  **macOS Apple Silicon**: Utilizing the `mps` (Metal Performance Shaders) backend in PyTorch or Apple's `mlx` framework.
2.  **CUDA-Compatible Systems**: Utilizing NVIDIA GPUs (e.g., PyTorch 2.9.1+ with CUDA).

---

## 2. Environment & Hardware Abstraction
A core tenet of this monolithic design is writing device-agnostic code. A central `utils/device.py` module will handle the hardware detection and abstraction.

### 2.1 Supported Backends
*   **CUDA (NVIDIA):** Used for heavy lifting, particularly when training the foundational layers or running Domain Adversarial networks on large batches. Supports AMP (Automatic Mixed Precision) for accelerated training.
*   **MPS/MLX (Apple Silicon):** Used for localized development, debugging, and edge-like inference simulation on macOS. 

### 2.2 Device Abstraction Logic
All model instantiation and tensor creation will reference a globally resolved device standard:
```python
import torch

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")
```

---

## 3. Monolithic Directory Structure
The repository will be structured as a single comprehensive application.

```text
stats426/
│
├── data/                   # Phase 1: Data Harmonization
│   ├── datasets.py         # Custom PyTorch Datasets (GTSRB, LISA, BDD100K)
│   ├── dataloaders.py      # 70-10-10-10 Split Logic & WeightedRandomSampler
│   ├── transforms.py       # Domain-specific Augmentations (ColorJitter, Blur)
│   └── unify.py            # Label mapping to 0-57 unified index
│
├── models/                 # Phase 2: Architecture Scaling
│   ├── baseline.py         # Simple CNN
│   ├── advanced.py         # CNN + Spatial Transformer Network (STN)
│   ├── resnet.py           # Modified ResNet10 (3x3 stride-1 stem)
│   └── orion_vlm.py        # VQA / LoRA implementation for VLM
│
├── training/               # Phase 3: Cross-Domain Training
│   ├── engine.py           # Main training loops (forward, backward, optimization)
│   ├── curriculum.py       # Epoch-based logic (GTSRB+LISA -> BDD100K)
│   ├── domain_adv.py       # Gradient Reversal Layer (GRL) & Domain Classifier
│   └── schedulers.py       # CosineAnnealingLR and other custom schedulers
│
├── evaluation/             # Phase 4: Trustworthy Evaluation
│   ├── metrics.py          # Precision, Recall, F1-Score per class
│   ├── calibration.py      # ECE (Expected Calibration Error) & Reliability Diagrams
│   └── ood_testing.py      # Degradation testing against Challenge Split
│
├── utils/                  # Shared Utility Modules
│   ├── device.py           # CUDA vs MPS/MLX environment selection
│   ├── logger.py           # Weights & Biases (WandB) or TensorBoard integration
│   └── config.py           # Centralized hyperparameters and path variables
│
└── main.py                 # Monolithic Entry Point
```

---

## 4. Module Specifications

### 4.1 Data Pipeline (`data/`)
*   **Ingestion:** Data can be pulled from AWS S3 or a local cache.
*   **Harmonization:** Implement a `UnifiedTrafficSignDataset` that accepts an enum or string indicating the source (GTSRB, LISA, BDD100K).
*   **Transforms:** Applied dynamically. GTSRB gets standard resizing to `64x64`. LISA gets color jitter. BDD100K requires bounding box extraction followed by Gaussian Blur and heavy contrast shifts.
*   **Samplers:** A custom highly-tuned PyTorch `DataLoader` will utilize `WeightedRandomSampler` for BDD100K inside the training loop to address class imbalances.

### 4.2 Models (`models/`)
*   Models inherit from standard `torch.nn.Module`.
*   All intermediate tensor sizes must be documented. The modified ResNet10 will feature a custom stem `nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)` to preserve spatial information for `64x64` inputs.
*   The STN (Spatial Transformer Network) will be modularized so it can be optionally prepended to any backbone.

### 4.3 Training Logic (`training/`)
*   The monolithic `main.py` will accept arguments defining the training stage (e.g., `--stage geometric` or `--stage real-world`).
*   **Curriculum Learning:** Managed by a specialized trainer class that automatically adjusts datasets (mixing in BDD100K) and unfreezes deeper layers based on the current epoch.
*   **Checkpoints:** Model weights will be saved frequently. Due to the environment abstraction, a model trained on a CUDA machine must be cleanly loadable on an Apple Silicon machine via `map_location=device`.

### 4.4 Evaluation & Trust (`evaluation/`)
*   Evaluation is separated from the core training loop to allow for extensive post-processing without memory overhead.
*   A dedicated testing script will run the in-domain test split followed by the OOD (Out-of-Distribution) test split, generating a comparison report (gap analysis).
*   Calibration functions will analyze the output logits/softmax distributions to map confidence vs. accuracy.

---

## 5. Execution Flow
A typical execution command via the CLI will orchestrate the modules:

```bash
# Train on Apple Silicon (automatically detects MPS)
python main.py --mode train --stage full_curriculum

# Evaluate OOD on CUDA Server
python main.py --mode evaluate --split ood --checkpoint weights/best_resnet10.pth
```
