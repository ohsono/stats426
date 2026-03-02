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

# Sentinel label: marks samples used for domain adaptation only.
# Samples with this label are NOT included in the supervised CE loss.
DOMAIN_LABEL: int = -1

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
    German Traffic Sign Recognition Benchmark (Kaggle CSV format).

    Primary path: reads Train.csv / Test.csv with columns
        Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path
    Crops each image to the ROI bounding box for a tighter sign crop.

    Only GTSRB classes that have a mapping in GTSRB_LABEL_MAP are
    included when ``dot_only=True`` (the default).

    Fallback: scans Train/ or Test/ subdirectories (no ROI applied).
    Used when no CSV exists — keeps all classes, preserving test fixture
    compatibility.
    """

    source = "gtsrb"

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        split: str = "train",
        download: bool = True,       # retained for API compatibility; ignored
        dot_only: bool = True,       # filter to only DOT-mapped classes (CSV mode)
        apply_roi_crop: bool = True, # crop to ROI bounding box (CSV mode)
    ):
        super().__init__(root, transform)
        self.split = split
        self.dot_only = dot_only
        self.apply_roi_crop = apply_roi_crop
        # ROI info parallel to self.samples; None means no crop to apply
        self._roi_info: List[Optional[Tuple[int, int, int, int]]] = []
        self._load()

    def _load(self):
        """Try CSV-based loading first; fall back to directory scan."""
        csv_name = "Train.csv" if self.split == "train" else "Test.csv"
        csv_path = self.root / csv_name
        if csv_path.exists():
            self._load_from_csv(csv_path)
        else:
            self._scan_directory()

    def _load_from_csv(self, csv_path: Path):
        """
        Load samples from Kaggle-format GTSRB CSV.

        CSV columns: Width, Height, Roi.X1, Roi.Y1, Roi.X2, Roi.Y2, ClassId, Path
        Path is relative to self.root (e.g. ``Train/20/00020_00000_00000.png``).
        """
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                gtsrb_label = int(row["ClassId"])
                dot_label = GTSRB_LABEL_MAP.get(gtsrb_label)
                if self.dot_only and dot_label is None:
                    continue  # skip classes with no DOT mapping
                label = dot_label if dot_label is not None else gtsrb_label

                rel_path = row["Path"].strip()
                img_path = self.root / rel_path
                if not img_path.exists():
                    continue

                self.samples.append((rel_path, label))

                if self.apply_roi_crop:
                    roi: Optional[Tuple[int, int, int, int]] = (
                        int(row["Roi.X1"]),
                        int(row["Roi.Y1"]),
                        int(row["Roi.X2"]),
                        int(row["Roi.Y2"]),
                    )
                else:
                    roi = None
                self._roi_info.append(roi)

    def _scan_directory(self):
        """
        Fallback: scan Train/ or Test/ subdirectory with class subdirs.

        No ROI information is available here — no additional crop applied.
        Used primarily by synthetic test fixtures (no CSV present).
        Keeps all GTSRB class IDs regardless of ``dot_only`` setting.
        """
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
                    self._roi_info.append(None)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rel_path, label = self.samples[idx]
        img_path = self.root / rel_path
        image = Image.open(img_path).convert("RGB")

        roi = self._roi_info[idx] if idx < len(self._roi_info) else None
        if roi is not None:
            x1, y1, x2, y2 = roi
            w, h = image.size
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(x1 + 1, min(x2, w))
            y2 = max(y1 + 1, min(y2, h))
            image = image.crop((x1, y1, x2, y2))

        if self.transform is not None:
            image = self.transform(image)
        return image, label


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
        domain_only: bool = False,
    ):
        super().__init__(root, transform)
        self.label_map = label_map or LISA_LABEL_MAP
        self.min_crop_size = min_crop_size
        self.domain_only = domain_only
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
                    effective_label = DOMAIN_LABEL if self.domain_only else dot_label
                    self.samples.append((rel, effective_label))
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
    BDD100K traffic sign crops for domain adaptation.

    Supports two loading modes selected automatically:

    **Mode A — on-the-fly (primary)**:
        Uses per-image JSON annotations co-located next to images in
        ``root/100k/{split}/``.  Scans all JSON files at init and stores
        bounding-box coordinates; crops are extracted in ``__getitem__``.
        All crops receive ``DOMAIN_LABEL = -1`` (domain adaptation only).

        Layout::
            root/100k/train/{name}.jpg
            root/100k/train/{name}.json   ← {"frames": [{"objects": [...]}]}

    **Mode B — pre-extracted (fallback)**:
        Reads from ``root/annotations.json`` index and ``root/crops/`` dir.
        Supports fine-grained labels when ``category`` is in
        ``BDD100K_LABEL_MAP`` and ``domain_only=False``.

        Layout::
            root/annotations.json   ← [{"image": "crops/x.jpg", "category": "stop"}]
            root/crops/x.jpg

    **Note**: BDD100K bounding-box annotations use the generic ``"traffic sign"``
    category (no fine-grained type).  Fine-grained labels require an external
    annotation file; run ``data/preprocess_bdd100k.py`` for details.
    """

    source = "bdd100k"

    def __init__(
        self,
        root: Path,
        transform: Optional[Callable] = None,
        split: str = "train",
        annotation_file: str = "annotations.json",
        label_map: Optional[Dict[str, int]] = None,
        min_crop_size: int = 16,
        domain_only: bool = True,
        max_samples: Optional[int] = None,
    ):
        super().__init__(root, transform)
        self.split = split
        self.annotation_file = self.root / annotation_file
        self.label_map = label_map or BDD100K_LABEL_MAP
        self.min_crop_size = min_crop_size
        self.domain_only = domain_only
        self.max_samples = max_samples
        # Parallel to self.samples: None = already-cropped file; 5-tuple = live crop
        self._bbox_info: List[Optional[Tuple[str, int, int, int, int]]] = []
        self._scan()

    def _scan(self):
        """Select Mode A (live) or Mode B (pre-extracted)."""
        if self.annotation_file.exists():
            self._scan_preextracted()
        else:
            live_dir = self.root / "100k" / self.split
            if live_dir.exists():
                self._scan_live(live_dir)
            else:
                # No data available — empty dataset
                pass

    def _scan_preextracted(self):
        """Mode B: load from annotations.json and pre-cropped images."""
        with open(self.annotation_file) as f:
            annotations = json.load(f)
        for entry in annotations:
            img_rel = entry.get("image", "")
            cat = entry.get("category", "").lower()
            if self.domain_only:
                label = DOMAIN_LABEL
            else:
                label = self.label_map.get(cat)
                if label is None:
                    continue
            img_abs = self.root / img_rel
            if img_abs.exists():
                self.samples.append((img_rel, label))
                self._bbox_info.append(None)
            if self.max_samples and len(self.samples) >= self.max_samples:
                break

    def _scan_live(self, live_dir: Path):
        """
        Mode A: scan per-image JSON files in ``100k/{split}/``.

        For each JSON, extracts all ``"traffic sign"`` objects with
        bounding-box dimensions ≥ ``min_crop_size``.  macOS resource-fork
        files (``._*.json``) are skipped.
        """
        for json_file in live_dir.iterdir():
            if json_file.suffix != ".json":
                continue
            if json_file.stem.startswith("._"):
                continue

            jpg_path = json_file.with_suffix(".jpg")
            if not jpg_path.exists():
                continue

            try:
                with open(json_file, encoding="utf-8", errors="ignore") as f:
                    ann = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            for frame in ann.get("frames", []):
                for obj in frame.get("objects", []):
                    if obj.get("category") != "traffic sign":
                        continue
                    box = obj.get("box2d")
                    if box is None:
                        continue
                    x1, y1 = box["x1"], box["y1"]
                    x2, y2 = box["x2"], box["y2"]
                    if (x2 - x1) < self.min_crop_size or (y2 - y1) < self.min_crop_size:
                        continue

                    rel = str(jpg_path.relative_to(self.root))
                    self.samples.append((rel, DOMAIN_LABEL))
                    self._bbox_info.append((
                        str(jpg_path),
                        int(x1), int(y1), int(x2), int(y2),
                    ))
                    if self.max_samples and len(self.samples) >= self.max_samples:
                        return

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        rel_path, label = self.samples[idx]
        bbox = self._bbox_info[idx] if idx < len(self._bbox_info) else None

        if bbox is not None:
            # Mode A: load full scene and crop to sign region
            jpg_path, x1, y1, x2, y2 = bbox
            image = Image.open(jpg_path).convert("RGB")
            iw, ih = image.size
            x1 = max(0, min(x1, iw - 1))
            y1 = max(0, min(y1, ih - 1))
            x2 = max(x1 + 1, min(x2, iw))
            y2 = max(y1 + 1, min(y2, ih))
            image = image.crop((x1, y1, x2, y2))
        else:
            # Mode B: pre-extracted crop, load directly
            image = Image.open(self.root / rel_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)
        return image, label


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
