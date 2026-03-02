"""
Vision Language Model (Orion) stub.

Architecture (Phase 2.4):
    Pre-trained VLM adapted via LoRA for Visual Question Answering (VQA).
    Casts classification as "What traffic sign is this?" prompt.

This is a stub/interface — actual VLM weights and full LoRA logic will be
integrated when the model is selected for training.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from utils.config import NUM_CLASSES


class LoRALinear(nn.Module):
    """
    Low-Rank Adaptation (LoRA) wrapper around a nn.Linear layer.

    Adds trainable low-rank decomposition A·B to the frozen original weight.
    """

    def __init__(self, original: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Freeze original
        self.original.weight.requires_grad_(False)
        if self.original.bias is not None:
            self.original.bias.requires_grad_(False)

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.original(x)
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return base_out + lora_out * self.scaling


class OrionVLMStub(nn.Module):
    """
    Placeholder VLM classifier.

    Uses a simple image encoder (could be swapped for a real VLM backbone)
    with a LoRA-adapted classification head.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        embed_dim: int = 512,
        lora_rank: int = 8,
    ):
        super().__init__()
        # Minimal image encoder (placeholder for real VLM vision tower)
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, embed_dim),
            nn.ReLU(inplace=True),
        )

        # Classification head with LoRA
        base_linear = nn.Linear(embed_dim, num_classes)
        self.head = LoRALinear(base_linear, rank=lora_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.encoder(x)
        return self.head(features)
