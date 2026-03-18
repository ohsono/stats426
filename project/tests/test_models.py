"""
Phase 2 tests – Model Architectures.

Tests cover:
  • BaselineCNN forward pass & output shape
  • AdvancedCNN + STN forward pass & output shape
  • ResNet50 modified stem, forward pass, & output shape
  • OrionVLM stub + LoRA forward pass & output shape
  • Gradient flow through each model
  • Parameter counts are reasonable
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.config import IMAGE_SIZE, NUM_CLASSES
from models.baseline import BaselineCNN
from models.advanced import AdvancedCNN, SpatialTransformerNetwork
from models.resnet import ResNet50, Bottleneck
from models.orion_vlm import OrionVLMStub, LoRALinear


BATCH = 4
IN_CHANNELS = 3


@pytest.fixture
def dummy_batch():
    """A random (B, 3, 64, 64) input tensor."""
    return torch.randn(BATCH, IN_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


# ===================================================================
# 1. Baseline CNN
# ===================================================================

class TestBaselineCNN:
    def test_output_shape(self, dummy_batch):
        model = BaselineCNN()
        out = model(dummy_batch)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_gradients_flow(self, dummy_batch):
        model = BaselineCNN()
        out = model(dummy_batch)
        loss = out.sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_custom_num_classes(self, dummy_batch):
        model = BaselineCNN(num_classes=10)
        out = model(dummy_batch)
        assert out.shape == (BATCH, 10)


# ===================================================================
# 2. Advanced CNN (with STN)
# ===================================================================

class TestSTN:
    def test_stn_identity_init(self):
        stn = SpatialTransformerNetwork()
        x = torch.randn(2, 3, IMAGE_SIZE, IMAGE_SIZE)
        out = stn(x)
        # Output should have same shape
        assert out.shape == x.shape

    def test_stn_preserves_shape(self):
        stn = SpatialTransformerNetwork()
        x = torch.randn(BATCH, 3, IMAGE_SIZE, IMAGE_SIZE)
        assert stn(x).shape == x.shape


class TestAdvancedCNN:
    def test_output_shape(self, dummy_batch):
        model = AdvancedCNN()
        out = model(dummy_batch)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_gradients_flow(self, dummy_batch):
        model = AdvancedCNN()
        out = model(dummy_batch)
        loss = out.sum()
        loss.backward()
        # Check STN params also get gradients
        for p in model.stn.parameters():
            assert p.grad is not None

    def test_stn_is_submodule(self):
        model = AdvancedCNN()
        assert hasattr(model, "stn")
        assert isinstance(model.stn, SpatialTransformerNetwork)


# ===================================================================
# 3. ResNet50
# ===================================================================

class TestResNet50:
    def test_output_shape(self, dummy_batch):
        model = ResNet50()
        out = model(dummy_batch)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_modified_stem_kernel(self):
        """Verify the stem uses 3×3 conv, NOT 7×7."""
        model = ResNet50()
        stem_conv = model.stem[0]  # First module in stem Sequential
        assert stem_conv.kernel_size == (3, 3)
        assert stem_conv.stride == (1, 1)

    def test_gradients_flow(self, dummy_batch):
        model = ResNet50()
        out = model(dummy_batch)
        loss = out.sum()
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No grad for {name}"

    def test_layer_count(self):
        """ResNet50 = [3,4,6,3] Bottleneck blocks."""
        model = ResNet50()
        assert len(model.layer1) == 3
        assert len(model.layer2) == 4
        assert len(model.layer3) == 6
        assert len(model.layer4) == 3

    def test_parameter_count_reasonable(self):
        model = ResNet50()
        total = sum(p.numel() for p in model.parameters())
        # ResNet50 has ~23M params
        assert total < 30_000_000


# ===================================================================
# 4. Orion VLM Stub
# ===================================================================

class TestLoRA:
    def test_lora_output_shape(self):
        base = torch.nn.Linear(64, 32)
        lora = LoRALinear(base, rank=4)
        x = torch.randn(2, 64)
        out = lora(x)
        assert out.shape == (2, 32)

    def test_lora_freezes_original(self):
        base = torch.nn.Linear(64, 32)
        lora = LoRALinear(base, rank=4)
        assert not lora.original.weight.requires_grad
        assert lora.lora_A.requires_grad
        assert lora.lora_B.requires_grad


class TestOrionVLM:
    def test_output_shape(self, dummy_batch):
        model = OrionVLMStub()
        out = model(dummy_batch)
        assert out.shape == (BATCH, NUM_CLASSES)

    def test_gradients_flow_through_lora(self, dummy_batch):
        model = OrionVLMStub()
        out = model(dummy_batch)
        loss = out.sum()
        loss.backward()
        assert model.head.lora_A.grad is not None
        assert model.head.lora_B.grad is not None
