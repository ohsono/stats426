"""
Advanced CNN with Spatial Transformer Network (STN).

Architecture (Phase 2.2):
    STN module → BatchNorm → Conv blocks → FC head.
    The STN learns affine transformations to auto-center / zoom on the sign.
    BatchNorm stabilizes training across mixed color distributions.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.config import IMAGE_SIZE, NUM_CLASSES


class SpatialTransformerNetwork(nn.Module):
    """
    Learns a 2-D affine transformation to spatially align the input.
    Outputs a transformed image of the same spatial size.
    """

    def __init__(self, in_channels: int = 3, image_size: int = IMAGE_SIZE):
        super().__init__()
        self.image_size = image_size

        # Localisation network
        self.localization = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.MaxPool2d(2),
            nn.ReLU(True),
        )

        loc_out_size = image_size // 4  # After 2 pools of stride 2
        self.fc_loc = nn.Sequential(
            nn.Linear(32 * loc_out_size * loc_out_size, 64),
            nn.ReLU(True),
            nn.Linear(64, 6),  # 2×3 affine matrix
        )

        # Initialize to identity transform
        self.fc_loc[-1].weight.data.zero_()
        self.fc_loc[-1].bias.data.copy_(
            torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xs = self.localization(x)
        xs = xs.view(xs.size(0), -1)
        theta = self.fc_loc(xs).view(-1, 2, 3)

        grid = F.affine_grid(theta, x.size(), align_corners=False)
        x = F.grid_sample(x, grid, align_corners=False)
        return x


class AdvancedCNN(nn.Module):
    """
    CNN with spatial transformer front-end and heavy BatchNorm usage.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        image_size: int = IMAGE_SIZE,
        in_channels: int = 3,
    ):
        super().__init__()
        self.stn = SpatialTransformerNetwork(in_channels, image_size)

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        flat_size = image_size // 8
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * flat_size * flat_size, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stn(x)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
