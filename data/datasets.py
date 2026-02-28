"""
Custom PyTorch Datasets for DOT, GTSRB, LISA, and BDD100K.

Each dataset class loads images from local storage, applies the appropriate
transforms, and returns ``(image, label)`` tuples where ``label`` is in
the unified DOT global index (0–57).

Data sources:
  • DOT    — Canonical reference images (43 signs)
  • GTSRB  — German Traffic Sign Recognition Benchmark (via torchvision)
  • LISA   — LISA Traffic Sign Detection (bbox crops from video frames)
  • BDD100K — Full scene images (needs crop extraction)
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

from data.unify import (
    BDD100K_LABEL_MAP,
    DOT_CLASSES,
    LISA_LABEL_MAP,
    GTSRB_LABEL_MAP,
)

# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class TrafficSignDataset(Dataset):
    """Abstract base for all traffic sign datasets."""

    source: str = "unknown"

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        samples: Optional[List[Tuple[str, int]]] = None,
    ):
        self.root = Path(root)
        self.transform = transform
        # samples = [(relative_image_path, global_label_index), ...]
        self.samples: List[Tuple[str, int]] = samples if samples is not None else []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rel_path, label = self.samples[idx]
        img_path = self.root / rel_path
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def get_labels(self) -> List[int]:
        """Return list of all labels (useful for WeightedRandomSampler)."""
        return [s[1] for s in self.samples]

# ---------------------------------------------------------------------------
# DOT (Canonical reference images)
# ---------------------------------------------------------------------------

class DOTDataset(TrafficSignDataset):
    """
    DOT canonical traffic sign reference images.

    Expected layout::
        root/
            0_stop.png
            1_yield.png
            ...
            DOT_traffic_sign_label.csv

    Each image filename is ``{index}_{label}.png``.  Labels are read from
    the companion CSV (``DOT_traffic_sign_label.csv``) located either in
    ``root`` or one directory above ``root``.
    """

    source = "dot"

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        csv_filename: str = "DOT_traffic_sign_label.csv",
    ):
        super().__init__(root, transform)
        self.csv_path = self.root / csv_filename
        if not self.csv_path.exists():
            # Also check parent directory
            self.csv_path = self.root.parent / csv_filename
        self._scan()

    def _scan(self):
        if not self.csv_path.exists():
            return
        with open(self.csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["index"])
                fname = row["filename"].strip()
                img_path = self.root / fname
                if img_path.exists():
                    rel = fname
                    self.samples.append((rel, idx))


# ---------------------------------------------------------------------------
# GTSRB (via torchvision with auto-download)
# ---------------------------------------------------------------------------

class GTSRBDataset(TrafficSignDataset):
    """
    German Traffic Sign Recognition Benchmark.

    Uses torchvision's built-in GTSRB loader which handles download and
    label mapping automatically.  GTSRB classes (0–42) are mapped into
    the DOT unified index via GTSRB_LABEL_MAP.

    Falls back to local directory scan if torchvision loading fails.
    """

    source = "gtsrb"

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        split: str = "train",
        download: bool = True,
    ):
        super().__init__(root, transform)
        self.split = split
        self._tv_dataset = None
        self._load(download)

    def _load(self, download: bool):
        """Load GTSRB via torchvision."""
        try:
            import torchvision
            self._tv_dataset = torchvision.datasets.GTSRB(
                root=str(self.root),
                split=self.split,
                download=download,
                transform=None,  # We apply our own transform in __getitem__
            )
            # Build samples list with DOT-mapped labels
            for i in range(len(self._tv_dataset)):
                # torchvision GTSRB stores (path, label) in _samples
                if hasattr(self._tv_dataset, '_samples'):
                    gtsrb_label = self._tv_dataset._samples[i][1]
                else:
                    # Fallback: access via __getitem__ (slower)
                    _, gtsrb_label = self._tv_dataset[i]
                dot_label = GTSRB_LABEL_MAP.get(gtsrb_label, -1)
                if dot_label >= 0:
                    self.samples.append((str(i), dot_label))
                else:
                    # Keep GTSRB class ID if no DOT mapping exists
                    self.samples.append((str(i), gtsrb_label))
        except Exception:
            # Fallback: scan directory
            self._scan_directory()

    def _scan_directory(self):
        """Fallback: scan Train/ directory with class subdirectories."""
        base = self.root / "Train" if self.split == "train" else self.root / "Test"
        if not base.exists():
            return
        for class_dir in sorted(base.iterdir()):
            if not class_dir.is_dir():
                continue
            try:
                gtsrb_label = int(class_dir.name)
            except ValueError:
                continue
            dot_label = GTSRB_LABEL_MAP.get(gtsrb_label, gtsrb_label)
            for ext in ("*.ppm", "*.png", "*.jpg"):
                for img_file in class_dir.glob(ext):
                    rel = img_file.relative_to(self.root)
                    self.samples.append((str(rel), dot_label))

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        if self._tv_dataset is not None:
            # Use torchvision dataset directly
            real_idx = int(self.samples[idx][0])
            label = self.samples[idx][1]
            image, _ = self._tv_dataset[real_idx]
            if self.transform is not None:
                image = self.transform(image)
            return image, label
        return super().__getitem__(idx)


# ---------------------------------------------------------------------------
# LISA (bbox crop from video frame annotations)
# ---------------------------------------------------------------------------

class LISADataset(TrafficSignDataset):
    """
    LISA Traffic Sign Detection dataset.

    Expected layout::
        root/
            Annotations/Annotations/<sequence>/frameAnnotationsBOX.csv
            dayTrain/dayTrain/<clip>/frames/<clip>--<frame>.jpg
            nightTrain/nightTrain/<clip>/frames/<clip>--<frame>.jpg
            daySequence1/daySequence1/frames/<sequence>--<frame>.jpg
            ...

    Annotation CSV columns (semicolon-delimited):
        Filename; Annotation tag; Upper left corner X; Upper left corner Y;
        Lower right corner X; Lower right corner Y; Origin file; ...

    The loader reads all BOX CSVs, resolves image paths, and crops sign
    regions using the bounding box coordinates.
    """

    source = "lisa"

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        label_map: Optional[Dict[str, int]] = None,
        min_crop_size: int = 8,
    ):
        super().__init__(root, transform)
        self.label_map = label_map or LISA_LABEL_MAP
        self.min_crop_size = min_crop_size
        # Store bbox info for cropping at load time
        self._bbox_info: List[Tuple[str, int, int, int, int]] = []
        self._scan()

    def _scan(self):
        if not self.root.exists():
            return

        # Find all frameAnnotationsBOX.csv files
        ann_root = self.root / "Annotations" / "Annotations"
        if not ann_root.exists():
            ann_root = self.root / "Annotations"

        csv_files = list(Path(ann_root).rglob("frameAnnotationsBOX.csv")) if ann_root.exists() else []

        # Also check sample dirs
        for sample_dir in self.root.glob("sample-*"):
            csv_files.extend(sample_dir.rglob("frameAnnotationsBOX.csv"))

        for csv_path in csv_files:
            self._parse_annotation_csv(csv_path)

    def _parse_annotation_csv(self, csv_path: Path):
        """Parse a LISA BOX annotation CSV and register samples."""
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f, delimiter=";")
            for row in reader:
                tag = row.get("Annotation tag", "").strip().lower()
                dot_label = self.label_map.get(tag)
                if dot_label is None:
                    continue

                # Parse bbox
                try:
                    x1 = int(row["Upper left corner X"])
                    y1 = int(row["Upper left corner Y"])
                    x2 = int(row["Lower right corner X"])
                    y2 = int(row["Lower right corner Y"])
                except (ValueError, KeyError):
                    continue

                if (x2 - x1) < self.min_crop_size or (y2 - y1) < self.min_crop_size:
                    continue

                # Resolve image path
                filename = row.get("Filename", "").strip()
                if not filename:
                    continue

                # LISA filenames look like: dayTest/daySequence1--00000.jpg
                # The actual file is at: root/daySequence1/daySequence1/frames/daySequence1--00000.jpg
                # Try multiple path resolutions
                img_path = self._resolve_image_path(filename)
                if img_path and img_path.exists():
                    rel = str(img_path.relative_to(self.root))
                    self.samples.append((rel, dot_label))
                    self._bbox_info.append((rel, x1, y1, x2, y2))

    def _resolve_image_path(self, filename: str) -> Optional[Path]:
        """Resolve a LISA annotation filename to an actual file path."""
        # Direct path
        p = self.root / filename
        if p.exists():
            return p

        # Strip leading directory and search
        basename = Path(filename).name
        # Extract sequence name from basename (e.g., "daySequence1--00000.jpg" -> "daySequence1")
        parts = basename.split("--")
        if len(parts) >= 2:
            seq_name = parts[0]
            # Try: root/<seq_name>/<seq_name>/frames/<basename>
            p = self.root / seq_name / seq_name / "frames" / basename
            if p.exists():
                return p
            # Try: root/dayTrain/dayTrain/<clip>/frames/<basename> (for clips like dayClip1)
            for train_dir in ("dayTrain", "nightTrain"):
                train_root = self.root / train_dir / train_dir
                if train_root.exists():
                    for clip_dir in train_root.iterdir():
                        p = clip_dir / "frames" / basename
                        if p.exists():
                            return p

        return None

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rel_path, label = self.samples[idx]
        img_path = self.root / rel_path
        image = Image.open(img_path).convert("RGB")

        # Crop to bounding box
        if idx < len(self._bbox_info):
            _, x1, y1, x2, y2 = self._bbox_info[idx]
            w, h = image.size
            # Clamp to image bounds
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            image = image.crop((x1, y1, x2, y2))

        if self.transform is not None:
            image = self.transform(image)
        return image, label


# ---------------------------------------------------------------------------
# BDD100K (placeholder — needs crop extraction pipeline)
# ---------------------------------------------------------------------------

class BDD100KDataset(TrafficSignDataset):
    """
    BDD100K traffic sign crops.

    This dataset requires a preprocessing step to extract traffic sign
    crops from the full 1280×720 scene images using bounding box annotations.

    Expected layout after preprocessing::
        root/
            crops/
                <image_id>_<box_id>.jpg
            annotations.json

    If no preprocessed crops exist, this loader returns an empty dataset.
    """

    source = "bdd100k"

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        annotation_file: str = "annotations.json",
        label_map: Optional[Dict[str, int]] = None,
    ):
        super().__init__(root, transform)
        self.annotation_file = self.root / annotation_file
        self.label_map = label_map or BDD100K_LABEL_MAP
        self._scan()

    def _scan(self):
        if not self.annotation_file.exists():
            return
        with open(self.annotation_file) as f:
            annotations = json.load(f)
        for entry in annotations:
            img_rel = entry.get("image", "")
            cat = entry.get("category", "").lower()
            label = self.label_map.get(cat)
            if label is None:
                continue
            if (self.root / img_rel).exists():
                self.samples.append((img_rel, label))


# ---------------------------------------------------------------------------
# Unified wrapper
# ---------------------------------------------------------------------------

class UnifiedTrafficSignDataset(Dataset):
    """
    Concatenates one or more ``TrafficSignDataset`` instances into a single
    Dataset, keeping track of origin metadata.
    """

    def __init__(self, datasets: List[TrafficSignDataset]):
        self.datasets = datasets
        self._flat: List[Tuple[TrafficSignDataset, int]] = []
        for ds in datasets:
            for i in range(len(ds)):
                self._flat.append((ds, i))

    def __len__(self) -> int:
        return len(self._flat)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        ds, local_idx = self._flat[idx]
        return ds[local_idx]

    def get_labels(self) -> List[int]:
        labels = []
        for ds in self.datasets:
            labels.extend(ds.get_labels())
        return labels
