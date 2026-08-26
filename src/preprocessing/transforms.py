"""
VIGIL — Preprocessing Transforms

Image transforms for the MVTec AD 2 Vial anomaly detection pipeline.

Key design decisions:
- Grayscale images are converted to 3-channel (replicate) for ImageNet-pretrained backbones
- No data augmentation: anomaly detection trains on normal distribution only
- ImageNet normalization applied since backbone is pretrained on ImageNet
- Resize to 256x256 by default (configurable)
"""

from __future__ import annotations

from typing import Optional, Tuple

import torchvision.transforms as T


# ImageNet statistics (used because backbone is pretrained on ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_transform(
    resize: Tuple[int, int] = (256, 256),
    center_crop: Optional[int] = None,
    normalize: bool = True,
) -> T.Compose:
    """Build the image transform pipeline.

    This is used for both training (feature extraction) and inference.
    No augmentation is applied — this is standard for anomaly detection
    where we want to model the exact normal distribution.

    Args:
        resize: Target (height, width) for resizing.
        center_crop: If provided, center-crop to this size after resize.
        normalize: Whether to apply ImageNet normalization.

    Returns:
        A torchvision Compose transform.
    """
    transforms = [
        T.Resize(resize, interpolation=T.InterpolationMode.BILINEAR, antialias=True),
    ]

    if center_crop is not None:
        transforms.append(T.CenterCrop(center_crop))

    transforms.append(T.ToTensor())  # Converts to [0, 1] float tensor

    if normalize:
        transforms.append(T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD))

    return T.Compose(transforms)


def get_mask_transform(
    resize: Tuple[int, int] = (256, 256),
    center_crop: Optional[int] = None,
) -> T.Compose:
    """Build the mask transform pipeline.

    Masks are resized using nearest-neighbor interpolation to preserve
    binary values. No normalization is applied.

    Args:
        resize: Target (height, width).
        center_crop: Optional center crop size.

    Returns:
        A torchvision Compose transform for masks.
    """
    transforms = [
        T.Resize(resize, interpolation=T.InterpolationMode.NEAREST),
    ]

    if center_crop is not None:
        transforms.append(T.CenterCrop(center_crop))

    transforms.append(T.ToTensor())  # Converts to [0, 1] float tensor

    return T.Compose(transforms)
