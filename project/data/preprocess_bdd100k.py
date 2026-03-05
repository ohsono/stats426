#!/usr/bin/env python3
"""
BDD100K Traffic Sign Crop Extractor.

Reads per-image JSON annotations from the extracted BDD100K ``100k/`` directory,
crops ``"traffic sign"`` bounding boxes from full 1280×720 images, saves them to
a ``crops/`` directory, and writes an ``annotations.json`` index.

Usage::

    python data/preprocess_bdd100k.py \\
        --bdd100k-dir /home/ohsono/dataset/stats426/BDD_100K \\
        --output-dir  /home/ohsono/dataset/stats426/BDD_100K/preextracted \\
        --split train \\
        --min-size 16

Output structure::

    output_dir/
        crops/
            train_{name}_{obj_id}.jpg
        annotations.json   ← [{"image": "crops/...", "category": "traffic_sign"}]

NOTE on fine-grained labels
---------------------------
BDD100K per-image JSON uses ``"traffic sign"`` as the category (not stop/yield/etc.).
To obtain fine-grained traffic sign types you need the BDD100K Det-20 label files
from https://bdd-data.berkeley.edu/ which annotate 20 specific sign categories.
Without those, all crops are tagged ``category: "traffic_sign"`` and receive
``label = -1`` (DOMAIN_LABEL) — suitable for domain adversarial training but
*not* for supervised 58-class classification.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

from PIL import Image

DOMAIN_LABEL: int = -1


def extract_crops(
    bdd100k_dir: Path,
    output_dir: Path,
    split: str = "train",
    min_size: int = 16,
    max_crops: Optional[int] = None,
) -> List[Dict]:
    """
    Extract traffic sign crops from BDD100K to ``output_dir/crops/``.

    Returns a list of annotation dicts suitable for ``annotations.json``.
    """
    live_dir = bdd100k_dir / "100k" / split
    if not live_dir.exists():
        raise FileNotFoundError(f"BDD100K {split} dir not found: {live_dir}")

    crops_dir = output_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    annotations: List[Dict] = []
    n_skipped = 0

    for json_file in sorted(live_dir.iterdir()):
        if json_file.suffix != ".json" or json_file.stem.startswith("._"):
            continue
        jpg_path = json_file.with_suffix(".jpg")
        if not jpg_path.exists():
            continue

        try:
            with open(json_file, encoding="utf-8", errors="ignore") as f:
                ann = json.load(f)
        except (json.JSONDecodeError, OSError):
            n_skipped += 1
            continue

        for frame in ann.get("frames", []):
            for obj in frame.get("objects", []):
                if obj.get("category") != "traffic sign":
                    continue
                box = obj.get("box2d")
                if box is None:
                    continue
                x1, y1 = int(box["x1"]), int(box["y1"])
                x2, y2 = int(box["x2"]), int(box["y2"])
                if (x2 - x1) < min_size or (y2 - y1) < min_size:
                    continue

                try:
                    img = Image.open(jpg_path).convert("RGB")
                    iw, ih = img.size
                    x1c = max(0, min(x1, iw - 1))
                    y1c = max(0, min(y1, ih - 1))
                    x2c = max(x1c + 1, min(x2, iw))
                    y2c = max(y1c + 1, min(y2, ih))
                    crop = img.crop((x1c, y1c, x2c, y2c))

                    obj_id = obj.get("id", len(annotations))
                    crop_name = f"{split}_{json_file.stem}_{obj_id}.jpg"
                    crop.save(crops_dir / crop_name, quality=90)

                    annotations.append({
                        "image": f"crops/{crop_name}",
                        "category": "traffic_sign",
                        "label": DOMAIN_LABEL,
                        "source_image": str(jpg_path),
                        "box2d": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    })
                except (OSError, ValueError):
                    n_skipped += 1
                    continue

                if max_crops and len(annotations) >= max_crops:
                    return annotations

        if len(annotations) % 1000 == 0 and len(annotations) > 0:
            print(f"  {len(annotations)} crops extracted ...", end="\r")

    if n_skipped:
        print(f"\n  Skipped {n_skipped} files (corrupted/missing)")
    return annotations


def main():
    parser = argparse.ArgumentParser(description="Extract BDD100K traffic sign crops")
    parser.add_argument(
        "--bdd100k-dir", type=Path,
        default=Path("/home/ohsono/dataset/stats426/BDD_100K"),
        help="Root of BDD100K dataset (must contain 100k/{split}/ subdirs)",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("/home/ohsono/dataset/stats426/BDD_100K/preextracted"),
        help="Directory to save crops/ and annotations.json",
    )
    parser.add_argument(
        "--split", default="train", choices=["train", "val", "test"],
        help="BDD100K split to process",
    )
    parser.add_argument(
        "--min-size", type=int, default=16,
        help="Minimum bounding-box dimension in pixels (smaller crops discarded)",
    )
    parser.add_argument(
        "--max-crops", type=int, default=None,
        help="Cap total number of crops (useful for quick testing)",
    )
    args = parser.parse_args()

    print(f"Extracting BDD100K '{args.split}' crops")
    print(f"  Source : {args.bdd100k_dir / '100k' / args.split}")
    print(f"  Output : {args.output_dir}")

    try:
        annotations = extract_crops(
            args.bdd100k_dir, args.output_dir,
            args.split, args.min_size, args.max_crops,
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)

    ann_path = args.output_dir / "annotations.json"
    with open(ann_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"\nDone: {len(annotations)} crops saved to {args.output_dir / 'crops'}")
    print(f"Index written to {ann_path}")
    print()
    print("NOTE: All crops are labeled DOMAIN_LABEL=-1 (domain adaptation only).")
    print("  BDD100K bounding-box annotations use the generic 'traffic sign' category.")
    print("  To assign fine-grained labels (stop/yield/etc.) download the BDD100K")
    print("  Det-20 annotation files from https://bdd-data.berkeley.edu/ and update")
    print("  the 'category' and 'label' fields in annotations.json manually.")


if __name__ == "__main__":
    main()
