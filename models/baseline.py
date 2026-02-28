"""
Baseline CNN for traffic sign classification.

Architecture (Phase 2.1):
    2-3 Conv layers → MaxPool → FC head.
    Validates the data pipeline and establishes a performance floor.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import IMAGE_SIZE, NUM_CLASSES


class BaselineCNN(nn.Module):
    """Simple 3-conv-layer CNN baseline."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        image_size: int = IMAGE_SIZE,
        in_channels: int = 3,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.25)

        # Compute flattened dim:  image_size / 2^3  (3 pooling layers)
        flat_size = image_size // 8
        self.fc1 = nn.Linear(128 * flat_size * flat_size, 256)
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
