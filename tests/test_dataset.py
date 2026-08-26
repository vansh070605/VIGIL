"""
Tests for VIGIL dataset loader.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.dataset import MVTecAD2VialDataset, ImageSample


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def dataset():
    """Load the real dataset (skip if not available)."""
    root = Path(__file__).resolve().parents[1] / "dataset"
    if not root.exists():
        pytest.skip("Dataset not found at expected location")
    return MVTecAD2VialDataset(root=root)


# ── Discovery tests ─────────────────────────────────────────────────────────

class TestDatasetDiscovery:

    def test_all_splits_discovered(self, dataset):
        """All expected splits are present."""
        expected_splits = {"train", "validation", "test_good", "test_bad",
                          "test_private", "test_private_mixed"}
        assert set(dataset._samples.keys()) == expected_splits

    def test_train_count(self, dataset):
        """Training set has 291 normal images."""
        assert len(dataset.get_split("train")) == 291

    def test_validation_count(self, dataset):
        """Validation set has 41 normal images."""
        assert len(dataset.get_split("validation")) == 41

    def test_test_good_count(self, dataset):
        """Test public good has 35 images (5 vials x 7 variants)."""
        assert len(dataset.get_split("test_good")) == 35

    def test_test_bad_count(self, dataset):
        """Test public bad has 105 images (15 defects x 7 variants)."""
        assert len(dataset.get_split("test_bad")) == 105

    def test_train_all_good(self, dataset):
        """All training samples are labeled 'good'."""
        for sample in dataset.get_split("train"):
            assert sample.label == "good"
            assert not sample.is_anomalous

    def test_test_bad_all_bad(self, dataset):
        """All test_bad samples are labeled 'bad'."""
        for sample in dataset.get_split("test_bad"):
            assert sample.label == "bad"
            assert sample.is_anomalous


# ── Mask tests ───────────────────────────────────────────────────────────────

class TestMasks:

    def test_all_bad_have_masks(self, dataset):
        """Every anomalous test sample has a corresponding mask."""
        for sample in dataset.get_split("test_bad"):
            assert sample.mask_path is not None, f"No mask for {sample.path.name}"
            assert sample.mask_path.exists(), f"Mask missing: {sample.mask_path}"

    def test_mask_is_binary(self, dataset):
        """Masks contain only values 0 and 255."""
        sample = dataset.get_split("test_bad")[0]
        mask = sample.load_mask()
        unique_vals = set(np.unique(mask))
        assert unique_vals.issubset({0, 255}), f"Unexpected mask values: {unique_vals}"

    def test_mask_has_defect_pixels(self, dataset):
        """Each mask has some non-zero pixels (actual defect)."""
        for sample in dataset.get_split("test_bad")[:5]:
            mask = sample.load_mask()
            assert np.any(mask > 0), f"Empty mask for {sample.path.name}"


# ── Loading tests ────────────────────────────────────────────────────────────

class TestImageLoading:

    def test_load_grayscale(self, dataset):
        """Loading in grayscale produces 2D array."""
        sample = dataset.get_split("train")[0]
        img = sample.load_image("L")
        assert img.ndim == 2
        assert img.shape == (1900, 1400)

    def test_load_rgb(self, dataset):
        """Loading in RGB produces 3D array."""
        sample = dataset.get_split("train")[0]
        img = sample.load_image("RGB")
        assert img.ndim == 3
        assert img.shape == (1900, 1400, 3)

    def test_image_dtype(self, dataset):
        """Loaded images are uint8."""
        sample = dataset.get_split("train")[0]
        img = sample.load_image("L")
        assert img.dtype == np.uint8


# ── Variant tests ────────────────────────────────────────────────────────────

class TestVariants:

    def test_train_all_regular(self, dataset):
        """Training set only has 'regular' variants."""
        variants = {s.variant for s in dataset.get_split("train")}
        assert variants == {"regular"}

    def test_test_bad_all_variants(self, dataset):
        """Test bad has all 7 domain-shift variants."""
        variants = {s.variant for s in dataset.get_split("test_bad")}
        expected = {"regular", "overexposed", "underexposed",
                    "shift_1", "shift_2", "shift_3", "shift_4"}
        assert variants == expected

    def test_filter_by_variant(self, dataset):
        """Variant filtering works correctly."""
        all_bad = dataset.get_split("test_bad")
        regular_only = dataset.filter_by_variant(all_bad, "regular")
        assert all(s.variant == "regular" for s in regular_only)
        assert len(regular_only) == 15


# ── Validation tests ─────────────────────────────────────────────────────────

class TestValidation:

    def test_integrity_no_errors(self, dataset):
        """Dataset passes integrity validation with no errors."""
        issues = dataset.validate_integrity()
        assert len(issues["errors"]) == 0, f"Errors found: {issues['errors'][:5]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
