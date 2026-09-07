"""
test_low_light_enhancement.py
Unit tests for src/low_light_enhancement.py (Module 2).

Run with: pytest tests/test_low_light_enhancement.py -v --cov=src.low_light_enhancement
"""

import numpy as np
import pytest

from src.enhancement.low_light_enhancement import (    apply_clahe,
    apply_gamma_correction,
    is_low_light,
    enhance_frame,
)


@pytest.fixture
def dark_frame():
    """A uniformly dark BGR frame (simulates a poorly lit cabin)."""
    return np.full((100, 100, 3), 20, dtype=np.uint8)


@pytest.fixture
def bright_frame():
    """A uniformly bright BGR frame (simulates good daylight)."""
    return np.full((100, 100, 3), 200, dtype=np.uint8)


@pytest.fixture
def random_frame():
    return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)


class TestApplyClahe:
    def test_clahe_preserves_shape_and_dtype(self, random_frame):
        result = apply_clahe(random_frame)
        assert result.shape == random_frame.shape
        assert result.dtype == random_frame.dtype

    def test_clahe_increases_contrast_on_flat_frame(self):
        flat = np.full((50, 50, 3), 128, dtype=np.uint8)
        flat[10:40, 10:40] = 130
        result = apply_clahe(flat)
        assert result.std() >= flat.std()

    def test_clahe_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            apply_clahe(None)

    def test_clahe_zero_clip_limit_raises(self, random_frame):
        with pytest.raises(ValueError, match="positive"):
            apply_clahe(random_frame, clip_limit=0)

    def test_clahe_negative_clip_limit_raises(self, random_frame):
        with pytest.raises(ValueError, match="positive"):
            apply_clahe(random_frame, clip_limit=-1.0)


class TestApplyGammaCorrection:
    def test_gamma_brightens_dark_frame(self, dark_frame):
        result = apply_gamma_correction(dark_frame, gamma=2.0)
        assert result.mean() > dark_frame.mean()

    def test_gamma_darkens_with_low_gamma(self, bright_frame):
        result = apply_gamma_correction(bright_frame, gamma=0.5)
        assert result.mean() < bright_frame.mean()

    def test_gamma_one_is_near_identity(self, random_frame):
        result = apply_gamma_correction(random_frame, gamma=1.0)
        assert np.allclose(result, random_frame, atol=2)

    def test_gamma_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            apply_gamma_correction(None)

    def test_gamma_zero_raises(self, random_frame):
        with pytest.raises(ValueError, match="positive"):
            apply_gamma_correction(random_frame, gamma=0)

    def test_gamma_negative_raises(self, random_frame):
        with pytest.raises(ValueError, match="positive"):
            apply_gamma_correction(random_frame, gamma=-1.0)


class TestIsLowLight:
    def test_detects_dark_frame(self, dark_frame):
        assert is_low_light(dark_frame) is True

    def test_detects_bright_frame_as_not_low_light(self, bright_frame):
        assert is_low_light(bright_frame) is False

    def test_custom_threshold(self, dark_frame):
        assert is_low_light(dark_frame, brightness_threshold=10.0) is False

    def test_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            is_low_light(None)


class TestEnhanceFrame:
    def test_dark_frame_gets_enhanced(self, dark_frame):
        result = enhance_frame(dark_frame)
        assert result.mean() > dark_frame.mean()

    def test_bright_frame_passes_through_unchanged(self, bright_frame):
        result = enhance_frame(bright_frame)
        assert np.array_equal(result, bright_frame)

    def test_output_shape_and_dtype_preserved(self, dark_frame):
        result = enhance_frame(dark_frame)
        assert result.shape == dark_frame.shape
        assert result.dtype == dark_frame.dtype

    def test_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            enhance_frame(None)
