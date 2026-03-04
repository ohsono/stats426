"""
Phase 1 tests – Utils & Data Pipeline (including DOT dataset).

Tests cover:
  • Device abstraction
  • Config defaults
  • Label unification from DOT CSV canonical source
  • Transforms produce correct tensor shapes
  • DOT, GTSRB, LISA, BDD100K dataset classes on synthetic fixtures
  • DataLoader 70-10-10-10 split ratios
  • WeightedRandomSampler construction
"""

from __future__ import annotations

import csv
import json
import shutil
import tempfile
from pathlib import Path

import pytest
import torch
from PIL import Image

import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.device import get_device, device_info, to_device
from utils.config import Config, DataConfig, IMAGE_SIZE, NUM_CLASSES
from data.unify import (
    global_index_to_name,
    name_to_global_index,
    num_classes,
    get_all_class_names,
    DOT_CLASSES,
    GTSRB_CLASSES,
    LISA_LABEL_MAP,
    BDD100K_LABEL_MAP,
)
from data.transforms import (
    gtsrb_train_transform,
    lisa_train_transform,
    bdd100k_train_transform,
    eval_transform,
)
from data.datasets import (
    TrafficSignDataset,
    DOTDataset,
    GTSRBDataset,
    LISADataset,
    BDD100KDataset,
    UnifiedTrafficSignDataset,
)
from data.dataloaders import stratified_split, build_weighted_sampler, create_dataloaders


# ===================================================================
# Fixtures
# ===================================================================

@pytest.fixture
def tmp_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d)


@pytest.fixture
def dot_root(tmp_dir) -> Path:
    """Create a minimal fake DOT directory with CSV."""
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
def gtsrb_root(tmp_dir) -> Path:
    root = tmp_dir / "gtsrb"
    for cls_id in range(3):
        cls_dir = root / "Train" / f"{cls_id:05d}"
        cls_dir.mkdir(parents=True)
        for i in range(5):
            img = Image.new("RGB", (30, 30), color=(cls_id * 80, 100, 50))
            img.save(cls_dir / f"img_{i:04d}.ppm")
    return root


@pytest.fixture
def lisa_root(tmp_dir) -> Path:
    """Create a minimal fake LISA with annotation CSVs and frame images."""
    root = tmp_dir / "lisa"
    # Create annotation CSV
    ann_dir = root / "Annotations" / "Annotations" / "testSeq"
    ann_dir.mkdir(parents=True)
    frames_dir = root / "testSeq" / "testSeq" / "frames"
    frames_dir.mkdir(parents=True)
    # Create frame images and annotation entries
    rows = []
    for i in range(4):
        fname = f"testSeq--{i:05d}.jpg"
        img = Image.new("RGB", (200, 200), color="blue")
        img.save(frames_dir / fname)
        rows.append({
            "Filename": f"testSeq/{fname}",
            "Annotation tag": "stop",
            "Upper left corner X": "10",
            "Upper left corner Y": "10",
            "Lower right corner X": "50",
            "Lower right corner Y": "50",
            "Origin file": "test.mp4",
            "Origin frame number": str(i),
            "Origin track": "test.mp4",
            "Origin track frame number": str(i),
        })
    csv_path = ann_dir / "frameAnnotationsBOX.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "Filename", "Annotation tag",
            "Upper left corner X", "Upper left corner Y",
            "Lower right corner X", "Lower right corner Y",
            "Origin file", "Origin frame number",
            "Origin track", "Origin track frame number",
        ], delimiter=";")
        writer.writeheader()
        writer.writerows(rows)
    return root


@pytest.fixture
def bdd100k_root(tmp_dir) -> Path:
    root = tmp_dir / "bdd100k"
    crops_dir = root / "crops"
    crops_dir.mkdir(parents=True)
    annotations = []
    for i, cat in enumerate(["stop", "yield", "construction"]):
        fname = f"crop_{i}.jpg"
        img = Image.new("RGB", (64, 64), color="green")
        img.save(crops_dir / fname)
        annotations.append({"image": f"crops/{fname}", "category": cat})
    with open(root / "annotations.json", "w") as f:
        json.dump(annotations, f)
    return root


# ===================================================================
# 1. Device Abstraction
# ===================================================================

class TestDevice:
    def test_get_device_returns_torch_device(self):
        dev = get_device()
        assert isinstance(dev, torch.device)
        assert dev.type in ("cuda", "mps", "cpu")

    def test_device_info_keys(self):
        info = device_info()
        assert "device" in info
        assert "platform" in info
        assert "torch" in info

    def test_to_device_moves_tensor(self):
        t = torch.zeros(2, 3)
        t2 = to_device(t)
        assert t2.device.type == get_device().type


# ===================================================================
# 2. Config
# ===================================================================

class TestConfig:
    def test_default_config(self):
        cfg = Config()
        assert cfg.data.image_size == IMAGE_SIZE
        assert cfg.data.num_classes == NUM_CLASSES
        assert cfg.data.train_ratio == 0.70

    def test_split_ratios_sum_to_one(self):
        dc = DataConfig()
        total = dc.train_ratio + dc.val_ratio + dc.test_ratio + dc.ood_ratio
        assert abs(total - 1.0) < 1e-9

    def test_train_config_defaults(self):
        cfg = Config()
        assert cfg.train.epochs_stage1 == 20
        assert cfg.train.lr == 1e-3
        assert cfg.train.optimizer == "adamw"


# ===================================================================
# 3. Label Unification (DOT-based)
# ===================================================================

class TestUnify:
    def test_dot_classes_loaded(self):
        assert len(DOT_CLASSES) > 0
        assert 0 in DOT_CLASSES  # stop
        assert DOT_CLASSES[0] == "stop"

    def test_dot_canonical_labels(self):
        assert DOT_CLASSES[1] == "yield"
        assert DOT_CLASSES[18] == "donotenter"

    def test_shared_labels_across_datasets(self):
        assert LISA_LABEL_MAP["stop"] == 0
        assert BDD100K_LABEL_MAP["stop"] == 0
        assert LISA_LABEL_MAP["yield"] == 1

    def test_global_index_to_name(self):
        assert global_index_to_name(0) == "stop"
        assert global_index_to_name(1) == "yield"

    def test_name_to_global_index_dot(self):
        assert name_to_global_index("stop", "dot") == 0
        assert name_to_global_index("yield", "dot") == 1

    def test_num_classes(self):
        assert num_classes() == 58

    def test_get_all_class_names_nonempty(self):
        names = get_all_class_names()
        assert len(names) > 0
        assert "stop" in names


# ===================================================================
# 4. Transforms
# ===================================================================

class TestTransforms:
    def _apply(self, transform, size=64):
        img = Image.new("RGB", (100, 100), color="red")
        out = transform(img)
        assert out.shape == (3, size, size)
        return out

    def test_gtsrb_transform_shape(self):
        self._apply(gtsrb_train_transform())

    def test_lisa_transform_shape(self):
        self._apply(lisa_train_transform())

    def test_bdd100k_transform_shape(self):
        self._apply(bdd100k_train_transform())

    def test_eval_transform_shape(self):
        self._apply(eval_transform())

    def test_eval_transform_deterministic(self):
        img = Image.new("RGB", (80, 80), color="blue")
        t = eval_transform()
        a = t(img)
        b = t(img)
        assert torch.allclose(a, b)


# ===================================================================
# 5. Datasets
# ===================================================================

class TestDatasets:
    def test_dot_scan(self, dot_root):
        ds = DOTDataset(dot_root, transform=eval_transform())
        assert len(ds) == 3  # 3 images in CSV

    def test_dot_getitem(self, dot_root):
        ds = DOTDataset(dot_root, transform=eval_transform())
        img, label = ds[0]
        assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert label == 0  # stop

    def test_dot_labels(self, dot_root):
        ds = DOTDataset(dot_root, transform=eval_transform())
        labels = ds.get_labels()
        assert labels == [0, 1, 2]

    def test_gtsrb_scan(self, gtsrb_root):
        ds = GTSRBDataset(gtsrb_root, transform=eval_transform(),
                          split="train", download=False)
        assert len(ds) == 15  # 3 classes × 5 images

    def test_gtsrb_getitem(self, gtsrb_root):
        ds = GTSRBDataset(gtsrb_root, transform=eval_transform(),
                          split="train", download=False)
        img, label = ds[0]
        assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
        assert isinstance(label, int)

    def test_lisa_scan(self, lisa_root):
        ds = LISADataset(lisa_root, transform=eval_transform())
        assert len(ds) == 4  # 4 stop annotations

    def test_bdd100k_scan(self, bdd100k_root):
        ds = BDD100KDataset(bdd100k_root, transform=eval_transform())
        assert len(ds) == 3

    def test_unified_dataset(self, gtsrb_root, lisa_root):
        g = GTSRBDataset(gtsrb_root, transform=eval_transform(),
                         split="train", download=False)
        l = LISADataset(lisa_root, transform=eval_transform())
        unified = UnifiedTrafficSignDataset([g, l])
        assert len(unified) == 15 + 4

    def test_unified_with_dot(self, dot_root, gtsrb_root):
        d = DOTDataset(dot_root, transform=eval_transform())
        g = GTSRBDataset(gtsrb_root, transform=eval_transform(),
                         split="train", download=False)
        unified = UnifiedTrafficSignDataset([d, g])
        assert len(unified) == 3 + 15


# ===================================================================
# 6. DataLoaders & Splitting
# ===================================================================

class TestDataLoaders:
    def test_stratified_split_sums(self, gtsrb_root):
        ds = GTSRBDataset(gtsrb_root, transform=eval_transform(),
                          split="train", download=False)
        subsets = stratified_split(ds, ratios=(0.7, 0.1, 0.1, 0.1))
        total = sum(len(s) for s in subsets)
        assert abs(total - len(ds)) <= 1

    def test_create_dataloaders_keys(self, gtsrb_root):
        ds = GTSRBDataset(gtsrb_root, transform=eval_transform(),
                          split="train", download=False)
        cfg = DataConfig(batch_size=4, num_workers=0)
        loaders = create_dataloaders(ds, config=cfg)
        assert set(loaders.keys()) == {"train", "val", "test", "ood"}

    def test_weighted_sampler(self):
        labels = [0] * 50 + [1] * 10
        sampler = build_weighted_sampler(labels)
        assert len(sampler) == 60

    def test_create_dataloaders_with_sampler(self, gtsrb_root):
        ds = GTSRBDataset(gtsrb_root, transform=eval_transform(),
                          split="train", download=False)
        cfg = DataConfig(batch_size=4, num_workers=0)
        loaders = create_dataloaders(ds, config=cfg, use_weighted_sampler=True)
        assert loaders["train"] is not None
