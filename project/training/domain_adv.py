"""
Domain Adversarial Training (Phase 3 – Stage 3).

Implements:
    • Gradient Reversal Layer (GRL)
    • Domain Classifier head
    • Combined loss computation for DANN-style training
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.autograd import Function


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradientReversal(Function):
    """Reverses gradient direction during backward pass."""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def gradient_reversal(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """Apply gradient reversal with scaling factor ``alpha``."""
    return _GradientReversal.apply(x, alpha)


class GradientReversalLayer(nn.Module):
    """Module wrapper around gradient reversal."""

    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return gradient_reversal(x, self.alpha)


# ---------------------------------------------------------------------------
# Domain Classifier
# ---------------------------------------------------------------------------

class DomainClassifier(nn.Module):
    """
    Binary / multi-class domain discriminator.

    Takes features from the shared backbone and predicts which dataset
    the sample originated from (GTSRB=0, LISA=1, BDD100K=2).
    """

    def __init__(self, feature_dim: int, num_domains: int = 3, hidden_dim: int = 256):
        super().__init__()
        self.grl = GradientReversalLayer()
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, features: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        self.grl.alpha = alpha
        reversed_features = self.grl(features)
        return self.classifier(reversed_features)


# ---------------------------------------------------------------------------
# DANN loss combiner
# ---------------------------------------------------------------------------

def dann_loss(
    class_logits: torch.Tensor,
    class_labels: torch.Tensor,
    domain_logits: torch.Tensor,
    domain_labels: torch.Tensor,
    class_criterion: nn.Module = nn.CrossEntropyLoss(),
    domain_criterion: nn.Module = nn.CrossEntropyLoss(),
    lambda_domain: float = 0.1,
) -> torch.Tensor:
    """Combine classification loss and domain-adversarial loss."""
    c_loss = class_criterion(class_logits, class_labels)
    d_loss = domain_criterion(domain_logits, domain_labels)
    return c_loss + lambda_domain * d_loss
