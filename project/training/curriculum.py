"""
Curriculum Learning scheduler (Phase 3).

Manages the staged introduction of datasets and controls layer freezing /
unfreezing across training stages.

Stages:
    1. Geometric (Epochs 1–20):  GTSRB + LISA only, high LR.
    2. Real-World (Epochs 21–40): + BDD100K, cosine LR decay, all layers unfrozen.
    3. Domain Adversarial (optional): Attach GRL if generalization gap is large.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import List, Optional

import torch.nn as nn


class Stage(Enum):
    GEOMETRIC = auto()
    REAL_WORLD = auto()
    DOMAIN_ADV = auto()


class CurriculumScheduler:
    """
    Drives curriculum learning by resolving which stage is active
    based on the current epoch.
    """

    def __init__(
        self,
        stage1_epochs: int = 20,
        stage2_epochs: int = 20,
        enable_domain_adv: bool = False,
    ):
        self.stage1_epochs = stage1_epochs
        self.stage2_epochs = stage2_epochs
        self.enable_domain_adv = enable_domain_adv
        self.total_epochs = stage1_epochs + stage2_epochs

    def get_stage(self, epoch: int) -> Stage:
        """Return the curriculum stage for the given epoch (1-indexed)."""
        if epoch <= self.stage1_epochs:
            return Stage.GEOMETRIC
        elif epoch <= self.total_epochs:
            return Stage.REAL_WORLD
        else:
            return Stage.DOMAIN_ADV if self.enable_domain_adv else Stage.REAL_WORLD

    def get_active_datasets(self, stage: Stage) -> List[str]:
        """Return dataset names active in this stage."""
        if stage == Stage.GEOMETRIC:
            return ["gtsrb", "lisa"]
        return ["gtsrb", "lisa", "bdd100k"]

    # ------------------------------------------------------------------
    # Layer freeze / unfreeze utilities
    # ------------------------------------------------------------------
    @staticmethod
    def freeze_backbone(model: nn.Module, freeze: bool = True):
        """Freeze or unfreeze all parameters except the final classifier."""
        for name, param in model.named_parameters():
            if "fc" not in name and "classifier" not in name and "head" not in name:
                param.requires_grad = not freeze

    def apply_stage(self, model: nn.Module, epoch: int):
        """
        Apply curriculum stage side-effects to the model:
          Stage 1: Backbone frozen (if pre-trained).
          Stage 2: All layers unfrozen.
        """
        stage = self.get_stage(epoch)
        if stage == Stage.GEOMETRIC:
            # In Stage 1 we typically train from scratch so nothing to freeze,
            # but if using a pre-trained backbone, freeze it.
            pass
        elif stage == Stage.REAL_WORLD:
            self.freeze_backbone(model, freeze=False)
        return stage
