"""
Unified label mapping module.

The canonical label space (0–57) is defined by the DOT reference dataset
located at ``dataset/DOT/DOT_traffic_sign_label.csv``.  Each external
dataset (GTSRB, LISA, BDD100K) is mapped into this DOT index.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# DOT canonical labels (source of truth)
# ---------------------------------------------------------------------------
_DOT_CSV = Path(__file__).resolve().parent.parent / "dataset" / "DOT_traffic_sign_label.csv"

DOT_CLASSES: Dict[int, str] = {}  # index -> label string


def _load_dot_classes():
    """Parse the DOT CSV once and populate DOT_CLASSES."""
    if DOT_CLASSES:
        return
    if _DOT_CSV.exists():
        with open(_DOT_CSV, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                idx = int(row["index"])
                label = row["label"].strip()
                DOT_CLASSES[idx] = label
    else:
        # Fallback hardcoded table (when CSV is not available, e.g. CI)
        _FALLBACK = {
            0: "stop", 1: "yield", 2: "speedlimitsign",
            3: "yieldtopedestrians", 4: "speedlimitchange",
            5: "norightturn", 6: "noleftturn", 7: "noUturn",
            8: "leftstraightoptionallane", 9: "laneusecontrolleft",
            10: "advancedintersectionlanecontrolleft",
            11: "intersectionlanecontrolleftcenterright",
            12: "leftturnonly", 13: "straightthroughonly",
            14: "noleftorUturn", 15: "nostraightthrough",
            18: "donotenter", 19: "notrucks",
            20: "oneway-1", 21: "oneway-2",
            24: "norightonred", 25: "noleftonred",
            26: "railroadcrossing-1", 28: "lightrailonlylane",
            30: "laneusecontrol2left1right", 31: "leftonlyrightonly",
            32: "advancedintersectionlanecontrol2left2right",
            34: "leftturnnoUturn", 35: "leftandstraightnoUturn",
            36: "HOVlanedescription", 37: "handicapparkingonly",
            38: "crosswalk", 39: "railroadcrossing-2",
            40: "pedestriancrossing", 48: "deadend",
            50: "constructiondetour-1", 51: "constructiondetour-2",
            52: "laneclosed", 53: "freewayexit",
            54: "freewayentrance-1", 55: "freewayentrance-2",
            56: "speedlimitschoolzone", 57: "rightlaneend",
        }
        DOT_CLASSES.update(_FALLBACK)


_load_dot_classes()

# Total number of classes in unified space
NUM_DOT_CLASSES = max(DOT_CLASSES.keys()) + 1 if DOT_CLASSES else 58

# ---------------------------------------------------------------------------
# Reverse map: label_name -> DOT index
# ---------------------------------------------------------------------------
_NAME_TO_INDEX: Dict[str, int] = {v: k for k, v in DOT_CLASSES.items()}


# ---------------------------------------------------------------------------
# GTSRB mapping: GTSRB class ID -> DOT index
# Shared concepts mapped to their DOT equivalents; others unchanged.
# ---------------------------------------------------------------------------
GTSRB_TO_DOT: Dict[int, int] = {
    14: 0,   # GTSRB "stop" -> DOT 0 "stop"
    13: 1,   # GTSRB "yield" -> DOT 1 "yield"
    17: 18,  # GTSRB "no_entry" -> DOT 18 "donotenter"
    38: 16,  # GTSRB "keep_right" -> DOT 16 "keepright" (via 58-class DOT index 16)
    39: 17,  # GTSRB "keep_left" -> DOT 17 "keepleft"
}

# Alias for datasets.py import
GTSRB_LABEL_MAP = GTSRB_TO_DOT

# Original GTSRB class names (for reference / documentation)
GTSRB_CLASSES: Dict[int, str] = {
    0: "speed_limit_20", 1: "speed_limit_30", 2: "speed_limit_50",
    3: "speed_limit_60", 4: "speed_limit_70", 5: "speed_limit_80",
    6: "end_speed_limit_80", 7: "speed_limit_100", 8: "speed_limit_120",
    9: "no_passing", 10: "no_passing_heavy", 11: "right_of_way",
    12: "priority_road", 13: "yield", 14: "stop",
    15: "no_vehicles", 16: "no_heavy_vehicles", 17: "no_entry",
    18: "general_caution", 19: "dangerous_curve_left",
    20: "dangerous_curve_right", 21: "double_curve",
    22: "bumpy_road", 23: "slippery_road", 24: "road_narrows_right",
    25: "road_work", 26: "traffic_signals", 27: "pedestrians",
    28: "children_crossing", 29: "bicycles_crossing",
    30: "beware_ice_snow", 31: "wild_animals",
    32: "end_all_restrictions", 33: "turn_right_ahead",
    34: "turn_left_ahead", 35: "ahead_only", 36: "go_straight_or_right",
    37: "go_straight_or_left", 38: "keep_right", 39: "keep_left",
    40: "roundabout", 41: "end_no_passing",
    42: "end_no_passing_heavy",
}

# ---------------------------------------------------------------------------
# LISA mapping: LISA string label -> DOT index
# ---------------------------------------------------------------------------
LISA_LABEL_MAP: Dict[str, int] = {
    "stop":                0,
    "stopleft":            0,   # stop sign seen from the left — same class
    "yield":               1,
    "speedlimitsign":      2,
    "norightturn":         5,
    "noleftturn":          6,
    "noUturn":             7,
    "donotenter":         18,
    "pedestriancrossing": 40,
    "keepright":          16,
    "keepleft":           17,
    # "go" / "goLeft" / "goForward" / "warning" / "warningLeft" are traffic
    # lights or generic warning shapes with no DOT equivalent — they are
    # excluded here and collected as DOMAIN_LABEL=-1 in preprocess_lisa.py.
}

# ---------------------------------------------------------------------------
# BDD100K mapping: BDD100K category string -> DOT index
# ---------------------------------------------------------------------------
BDD100K_LABEL_MAP: Dict[str, int] = {
    "stop":               0,
    "yield":              1,
    "donotenter":        18,
    "norightturn":        5,
    "noleftturn":         6,
    "pedestriancrossing": 40,
    "construction":       50,  # constructiondetour-1
    "speedlimitsign":     2,
}

# ---------------------------------------------------------------------------
# Reverse map: global DOT index -> human-readable label (from DOT CSV)
# ---------------------------------------------------------------------------
_INDEX_TO_NAME: Dict[int, str] = dict(DOT_CLASSES)


def global_index_to_name(index: int) -> str:
    """Return the human-readable DOT label for a global index."""
    return _INDEX_TO_NAME.get(index, f"unknown_{index}")


def name_to_global_index(name: str, source: str = "dot") -> Optional[int]:
    """Map a label name to its global DOT index, optionally by source."""
    source = source.lower()
    if source == "dot":
        return _NAME_TO_INDEX.get(name)
    elif source == "gtsrb":
        # Find GTSRB class id by name, then map through GTSRB_TO_DOT
        for gid, gname in GTSRB_CLASSES.items():
            if gname == name:
                return GTSRB_TO_DOT.get(gid, gid)
        return None
    elif source == "lisa":
        return LISA_LABEL_MAP.get(name)
    elif source == "bdd100k":
        return BDD100K_LABEL_MAP.get(name)
    return None


def get_all_class_names() -> List[str]:
    """Return a sorted list of all DOT class names."""
    return [_INDEX_TO_NAME[i] for i in sorted(_INDEX_TO_NAME.keys())]


def num_classes() -> int:
    """Return the total number of classes in the unified DOT label space."""
    return NUM_DOT_CLASSES
