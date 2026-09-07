"""
test_preprocessing.py
Unit tests for src/preprocessing.py (Module 1).

Run with: pytest tests/test_preprocessing.py -v --cov=src.preprocessing
"""

import numpy as np
import pytest

from src.preprocessing import (
    resize_frame,
    denoise_frame,
    normalize_frame,
    preprocess_pipeline,
    DEFAULT_WIDTH,
    DEFAULT_HEIGHT,
)


@pytest.fixture
def sample_frame():
    """A synthetic 720x1280 BGR frame with random pixel values."""
    return np.random.randint(0, 256, (720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def small_frame():
    """A tiny frame for fast edge-case tests."""
    return np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)


class TestResizeFrame:
    def test_resize_default_dimensions(self, sample_frame):
        result = resize_frame(sample_frame)
        assert result.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)

    def test_resize_custom_dimensions(self, sample_frame):
        result = resize_frame(sample_frame, width=320, height=240)
        assert result.shape == (240, 320, 3)

    def test_resize_preserves_dtype(self, sample_frame):
        result = resize_frame(sample_frame)
        assert result.dtype == sample_frame.dtype

    def test_resize_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            resize_frame(None)

    def test_resize_empty_raises(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            resize_frame(np.array([]))

    def test_resize_invalid_shape_raises(self):
        grayscale = np.zeros((100, 100), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            resize_frame(grayscale)

    def test_resize_zero_width_raises(self, sample_frame):
        with pytest.raises(ValueError, match="positive"):
            resize_frame(sample_frame, width=0, height=100)

    def test_resize_negative_height_raises(self, sample_frame):
        with pytest.raises(ValueError, match="positive"):
            resize_frame(sample_frame, width=100, height=-5)


class TestDenoiseFrame:
    def test_denoise_preserves_shape(self, small_frame):
        result = denoise_frame(small_frame)
        assert result.shape == small_frame.shape

    def test_denoise_preserves_dtype(self, small_frame):
        result = denoise_frame(small_frame)
        assert result.dtype == small_frame.dtype

    def test_denoise_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            denoise_frame(None)

    def test_denoise_zero_strength_raises(self, small_frame):
        with pytest.raises(ValueError, match="positive"):
            denoise_frame(small_frame, strength=0)

    def test_denoise_negative_strength_raises(self, small_frame):
        with pytest.raises(ValueError, match="positive"):
            denoise_frame(small_frame, strength=-3)


class TestNormalizeFrame:
    def test_normalize_output_range(self, sample_frame):
        result = normalize_frame(sample_frame)
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_normalize_output_dtype(self, sample_frame):
        result = normalize_frame(sample_frame)
        assert result.dtype == np.float32

    def test_normalize_known_values(self):
        frame = np.full((5, 5, 3), 255, dtype=np.uint8)
        result = normalize_frame(frame)
        assert np.allclose(result, 1.0)

    def test_normalize_zero_values(self):
        frame = np.zeros((5, 5, 3), dtype=np.uint8)
        result = normalize_frame(frame)
        assert np.allclose(result, 0.0)

    def test_normalize_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            normalize_frame(None)


class TestPreprocessPipeline:
    def test_pipeline_end_to_end_shape(self, sample_frame):
        result = preprocess_pipeline(sample_frame)
        assert result.shape == (DEFAULT_HEIGHT, DEFAULT_WIDTH, 3)

    def test_pipeline_output_normalized(self, sample_frame):
        result = preprocess_pipeline(sample_frame)
        assert result.dtype == np.float32
        assert result.min() >= 0.0
        assert result.max() <= 1.0

    def test_pipeline_custom_size(self, sample_frame):
        result = preprocess_pipeline(sample_frame, width=160, height=120)
        assert result.shape == (120, 160, 3)

    def test_pipeline_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            preprocess_pipeline(None)
