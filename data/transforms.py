"""
Domain-specific image augmentations.

Returns ``torchvision.transforms.Compose`` pipelines tailored to each
dataset's characteristics as outlined in Phase 1.
"""

from __future__ import annotations

from torchvision import transforms

from utils.config import IMAGE_SIZE


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _base_resize(size: int = IMAGE_SIZE):
    """Resize + convert to tensor + normalize with ImageNet stats."""
    return [
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]


# ---------------------------------------------------------------------------
# Per-dataset transforms
# ---------------------------------------------------------------------------
def gtsrb_train_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    GTSRB: clean, centered crops.
    Light augmentation to introduce slight variability.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def lisa_train_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    LISA: US signs, clean crops.
    RandomAffine + ColorJitter per Phase 1 spec.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def bdd100k_train_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    """
    BDD100K: heavy augmentation to simulate dashcam artifacts.
    GaussianBlur, heavy contrast shifts, artificial noise via ColorJitter.
    """
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.RandomAffine(degrees=20, translate=(0.15, 0.15), scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.3, hue=0.1),
        transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0)),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def eval_transform(size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic transform for validation / testing."""
    return transforms.Compose(_base_resize(size))
