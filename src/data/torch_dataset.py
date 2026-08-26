"""
VIGIL — PyTorch Dataset Adapter

Wraps the MVTecAD2VialDataset's ImageSample objects into a PyTorch Dataset
for use with DataLoader during feature extraction and inference.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .dataset import ImageSample


class VialDataset(Dataset):
    """PyTorch Dataset wrapping a list of ImageSample objects.

    Handles:
    - Grayscale → RGB conversion (3-channel replicate for ImageNet backbone)
    - Applying image and mask transforms
    - Returning label and metadata alongside the image tensor

    Args:
        samples: List of ImageSample objects from MVTecAD2VialDataset.
        transform: Image transform (e.g., from get_transform()).
        mask_transform: Mask transform (e.g., from get_mask_transform()).
    """

    def __init__(
        self,
        samples: List[ImageSample],
        transform: Optional[Callable] = None,
        mask_transform: Optional[Callable] = None,
    ):
        self.samples = samples
        self.transform = transform
        self.mask_transform = mask_transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        """Return a dictionary with image, label, mask, and metadata.

        Returns:
            Dictionary with keys:
                - 'image': Tensor of shape (3, H, W) — float32, normalized
                - 'label': int — 0 for good, 1 for bad
                - 'mask': Tensor of shape (1, H, W) or zero tensor if no mask
                - 'path': str — path to the original image file
                - 'variant': str — domain-shift variant name
                - 'index': str — sample index (e.g., '000')
        """
        sample = self.samples[idx]

        # Load image as RGB (replicate grayscale to 3 channels)
        image = Image.open(sample.path).convert("RGB")

        # Load mask if available
        if sample.mask_path and sample.mask_path.exists():
            mask = Image.open(sample.mask_path).convert("L")
        else:
            mask = None

        # Apply transforms
        if self.transform is not None:
            image = self.transform(image)

        if mask is not None and self.mask_transform is not None:
            mask = self.mask_transform(mask)
            # Binarize: mask values > 0.5 are defect
            mask = (mask > 0.5).float()
        elif mask is not None:
            # Convert to tensor without transform
            mask = torch.from_numpy(np.array(mask)).unsqueeze(0).float() / 255.0
            mask = (mask > 0.5).float()
        else:
            # No mask available — create zero mask
            if self.transform is not None:
                h, w = image.shape[1], image.shape[2]
            else:
                h, w = 256, 256
            mask = torch.zeros(1, h, w, dtype=torch.float32)

        label = 1 if sample.is_anomalous else 0

        return {
            "image": image,
            "label": label,
            "mask": mask,
            "path": str(sample.path),
            "variant": sample.variant,
            "index": sample.index,
        }
