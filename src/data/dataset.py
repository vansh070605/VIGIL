"""
VIGIL — MVTec AD 2 Vial Dataset Loader

Reusable utilities for discovering, loading, validating, and summarizing
the MVTec AD 2 Vial anomaly-detection dataset.

Design decisions:
- pathlib throughout (no hardcoded OS paths)
- Dataset root is configurable via constructor or environment variable
- Lazy loading: images are not loaded into memory until requested
- All label semantics are derived from directory structure, not guesses
"""

from __future__ import annotations

import hashlib
import logging
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Generator, List, Literal, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
DEFAULT_DATA_ROOT_ENV = "VIGIL_DATA_ROOT"

# Domain-shift variant types discovered in the dataset
VARIANT_TYPES = [
    "regular",
    "overexposed",
    "underexposed",
    "shift_1",
    "shift_2",
    "shift_3",
    "shift_4",
]


# ── Data classes ─────────────────────────────────────────────────────────────
@dataclass
class ImageSample:
    """Represents a single image sample with metadata."""

    path: Path
    split: str                           # train, validation, test_public, etc.
    label: Literal["good", "bad"]        # Normal vs anomalous
    variant: str                         # regular, overexposed, shift_1, etc.
    index: str                           # Sample index (e.g., "000")
    mask_path: Optional[Path] = None     # Path to ground-truth mask (if available)

    @property
    def is_anomalous(self) -> bool:
        return self.label == "bad"

    def load_image(self, mode: str = "L") -> np.ndarray:
        """Load the image as a numpy array.

        Args:
            mode: PIL color mode ('L' for grayscale, 'RGB' for color).

        Returns:
            numpy array of shape (H, W) for grayscale or (H, W, 3) for RGB.

        Raises:
            FileNotFoundError: If the image file does not exist.
            IOError: If the image cannot be read.
        """
        if not self.path.exists():
            raise FileNotFoundError(f"Image not found: {self.path}")
        img = Image.open(self.path).convert(mode)
        return np.array(img)

    def load_mask(self) -> Optional[np.ndarray]:
        """Load the ground-truth anomaly mask.

        Returns:
            Binary numpy array (H, W) with values {0, 255}, or None if no mask.
        """
        if self.mask_path is None or not self.mask_path.exists():
            return None
        mask = np.array(Image.open(self.mask_path))
        if mask.ndim == 3:
            mask = mask[:, :, 0]  # Take first channel if multi-channel
        return mask


@dataclass
class DatasetStatistics:
    """Aggregated statistics for a dataset split or the full dataset."""

    total_images: int = 0
    normal_count: int = 0
    anomalous_count: int = 0
    resolutions: Dict[str, int] = field(default_factory=dict)
    color_modes: Dict[str, int] = field(default_factory=dict)
    variants: Dict[str, int] = field(default_factory=dict)
    file_size_min_kb: float = 0.0
    file_size_max_kb: float = 0.0
    file_size_mean_kb: float = 0.0
    has_masks: bool = False
    mask_count: int = 0


# ── Dataset class ────────────────────────────────────────────────────────────
class MVTecAD2VialDataset:
    """Interface for the MVTec AD 2 Vial dataset.

    The dataset is organized as:
        dataset/
        ├── train/good/               (291 normal training images)
        ├── validation/good/          (41 normal validation images)
        ├── test_public/
        │   ├── good/                 (35 normal test images, 7 variants x 5 vials)
        │   ├── bad/                  (105 anomalous test images, 7 variants x 15 defects)
        │   └── ground_truth/bad/     (105 pixel-level anomaly masks)
        ├── test_private/             (276 unlabeled images, regular variant only)
        └── test_private_mixed/       (276 unlabeled images, mixed variants)

    All images are grayscale (mode 'L'), 1400x1900 pixels, PNG format.

    Args:
        root: Path to the dataset root directory.
              Falls back to VIGIL_DATA_ROOT environment variable.
    """

    def __init__(self, root: Optional[Path] = None):
        if root is None:
            env_root = os.environ.get(DEFAULT_DATA_ROOT_ENV)
            if env_root:
                root = Path(env_root)
            else:
                # Default: assume dataset is in <project_root>/dataset
                root = Path(__file__).resolve().parents[2] / "dataset"

        self.root = Path(root).resolve()
        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset root not found: {self.root}\n"
                f"Set the {DEFAULT_DATA_ROOT_ENV} environment variable or pass root= explicitly."
            )

        self._samples: Dict[str, List[ImageSample]] = {}
        self._discover_all()

    def _discover_all(self) -> None:
        """Discover all image samples across all splits."""
        self._samples = {
            "train": self._discover_split("train", "train/good", "good"),
            "validation": self._discover_split("validation", "validation/good", "good"),
            "test_good": self._discover_split("test_public_good", "test_public/good", "good"),
            "test_bad": self._discover_split_with_masks(
                "test_public_bad",
                "test_public/bad",
                "bad",
                "test_public/ground_truth/bad",
            ),
            "test_private": self._discover_split("test_private", "test_private", "good"),
            "test_private_mixed": self._discover_split(
                "test_private_mixed", "test_private_mixed", "good"
            ),
        }
        logger.info(
            "Dataset discovered: %s",
            {k: len(v) for k, v in self._samples.items()},
        )

    def _discover_split(
        self, split_name: str, subdir: str, label: str
    ) -> List[ImageSample]:
        """Discover images in a directory and return ImageSample objects."""
        directory = self.root / subdir
        if not directory.exists():
            logger.warning("Split directory not found: %s", directory)
            return []

        samples = []
        for img_path in sorted(directory.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            parsed = self._parse_filename(img_path.name)
            samples.append(
                ImageSample(
                    path=img_path,
                    split=split_name,
                    label=label,
                    variant=parsed["variant"],
                    index=parsed["index"],
                )
            )
        return samples

    def _discover_split_with_masks(
        self,
        split_name: str,
        img_subdir: str,
        label: str,
        mask_subdir: str,
    ) -> List[ImageSample]:
        """Discover images paired with ground-truth masks."""
        img_dir = self.root / img_subdir
        mask_dir = self.root / mask_subdir

        if not img_dir.exists():
            logger.warning("Split directory not found: %s", img_dir)
            return []

        samples = []
        for img_path in sorted(img_dir.iterdir()):
            if img_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            parsed = self._parse_filename(img_path.name)

            # Construct expected mask filename: NNN_variant_mask.png
            mask_name = img_path.stem + "_mask.png"
            mask_path = mask_dir / mask_name
            if not mask_path.exists():
                logger.warning("Mask not found for %s (expected %s)", img_path.name, mask_path)
                mask_path = None

            samples.append(
                ImageSample(
                    path=img_path,
                    split=split_name,
                    label=label,
                    variant=parsed["variant"],
                    index=parsed["index"],
                    mask_path=mask_path,
                )
            )
        return samples

    @staticmethod
    def _parse_filename(filename: str) -> dict:
        """Parse MVTec AD 2 naming convention: NNN_variant[_mask].png

        Examples:
            '000_regular.png'       -> index='000', variant='regular'
            '003_shift_2.png'       -> index='003', variant='shift_2'
            '005_overexposed.png'   -> index='005', variant='overexposed'
            '042_mixed.png'         -> index='042', variant='mixed'
        """
        stem = Path(filename).stem
        parts = stem.split("_")
        index = parts[0]

        if stem.endswith("_mask"):
            variant_parts = parts[1:-1]
        else:
            variant_parts = parts[1:]

        variant = "_".join(variant_parts) if variant_parts else "unknown"
        return {"index": index, "variant": variant}

    # ── Public API ──────────────────────────────────────────────────────────
    def get_split(self, split: str) -> List[ImageSample]:
        """Get all samples for a given split.

        Args:
            split: One of 'train', 'validation', 'test_good', 'test_bad',
                   'test_private', 'test_private_mixed'.

        Returns:
            List of ImageSample objects.
        """
        if split not in self._samples:
            raise ValueError(
                f"Unknown split '{split}'. Available: {list(self._samples.keys())}"
            )
        return self._samples[split]

    def get_test_public(self) -> List[ImageSample]:
        """Get all test_public samples (good + bad) for evaluation."""
        return self._samples["test_good"] + self._samples["test_bad"]

    def filter_by_variant(
        self, samples: List[ImageSample], variant: str
    ) -> List[ImageSample]:
        """Filter samples to a specific domain-shift variant."""
        return [s for s in samples if s.variant == variant]

    def get_statistics(self, split: Optional[str] = None) -> DatasetStatistics:
        """Compute statistics for a split or the entire dataset.

        Args:
            split: Split name, or None for aggregate statistics.
        """
        if split:
            samples = self.get_split(split)
        else:
            samples = []
            for v in self._samples.values():
                samples.extend(v)

        stats = DatasetStatistics()
        stats.total_images = len(samples)
        sizes = []

        for s in samples:
            if s.label == "good":
                stats.normal_count += 1
            else:
                stats.anomalous_count += 1

            stats.variants[s.variant] = stats.variants.get(s.variant, 0) + 1

            if s.mask_path and s.mask_path.exists():
                stats.has_masks = True
                stats.mask_count += 1

            # Get resolution and size without full load
            try:
                with Image.open(s.path) as img:
                    res = f"{img.width}x{img.height}"
                    stats.resolutions[res] = stats.resolutions.get(res, 0) + 1
                    stats.color_modes[img.mode] = stats.color_modes.get(img.mode, 0) + 1
                sizes.append(s.path.stat().st_size / 1024)
            except Exception as e:
                logger.error("Error reading %s: %s", s.path, e)

        if sizes:
            stats.file_size_min_kb = min(sizes)
            stats.file_size_max_kb = max(sizes)
            stats.file_size_mean_kb = sum(sizes) / len(sizes)

        return stats

    def iterate_samples(
        self,
        split: str,
        load: bool = False,
        mode: str = "L",
    ) -> Generator[Tuple[ImageSample, Optional[np.ndarray], Optional[np.ndarray]], None, None]:
        """Iterate over samples, optionally loading images and masks.

        Args:
            split: Dataset split name.
            load: If True, also yield loaded image and mask arrays.
            mode: PIL color mode for loading.

        Yields:
            (sample, image_array_or_None, mask_array_or_None)
        """
        for sample in self.get_split(split):
            if load:
                img = sample.load_image(mode)
                mask = sample.load_mask()
                yield sample, img, mask
            else:
                yield sample, None, None

    def validate_integrity(self) -> Dict[str, list]:
        """Validate dataset integrity.

        Checks:
        - All image files exist and are readable
        - All masks match their corresponding images
        - Image resolutions are consistent
        - No corrupt/zero-byte files

        Returns:
            Dictionary with keys 'errors' and 'warnings'.
        """
        issues = {"errors": [], "warnings": []}
        expected_resolution = (1400, 1900)  # Width x Height from inspection

        for split_name, samples in self._samples.items():
            for sample in samples:
                # Check existence
                if not sample.path.exists():
                    issues["errors"].append(f"Missing image: {sample.path}")
                    continue

                # Check zero-byte
                if sample.path.stat().st_size == 0:
                    issues["errors"].append(f"Zero-byte file: {sample.path}")
                    continue

                # Check readability and resolution
                try:
                    with Image.open(sample.path) as img:
                        if img.size != expected_resolution:
                            issues["warnings"].append(
                                f"Unexpected resolution {img.size} "
                                f"(expected {expected_resolution}): {sample.path}"
                            )
                except Exception as e:
                    issues["errors"].append(f"Corrupt image {sample.path}: {e}")

                # Check mask correspondence
                if sample.mask_path:
                    if not sample.mask_path.exists():
                        issues["errors"].append(
                            f"Missing mask for {sample.path.name}: {sample.mask_path}"
                        )
                    else:
                        try:
                            with Image.open(sample.mask_path) as mask_img:
                                if mask_img.size != expected_resolution:
                                    issues["warnings"].append(
                                        f"Mask resolution mismatch: {sample.mask_path}"
                                    )
                        except Exception as e:
                            issues["errors"].append(
                                f"Corrupt mask {sample.mask_path}: {e}"
                            )

        n_err = len(issues["errors"])
        n_warn = len(issues["warnings"])
        logger.info("Validation complete: %d errors, %d warnings", n_err, n_warn)
        return issues

    def compute_file_hash(self, filepath: Path, algorithm: str = "md5") -> str:
        """Compute hash of a file for integrity/deduplication checks."""
        h = hashlib.new(algorithm)
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()

    def __repr__(self) -> str:
        parts = []
        for name, samples in self._samples.items():
            parts.append(f"{name}={len(samples)}")
        return f"MVTecAD2VialDataset(root={self.root}, {', '.join(parts)})"
