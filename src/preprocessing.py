"""
preprocessing.py
Module 1 of the Context-Aware Driver Risk Assessment pipeline.

Responsible for preparing raw camera frames before they reach the
low-light enhancement and detection stages: resizing to a consistent
working resolution, denoising, and normalizing pixel values.
"""

import cv2
import numpy as np


DEFAULT_WIDTH = 640
DEFAULT_HEIGHT = 480


def resize_frame(frame: np.ndarray, width: int = DEFAULT_WIDTH, height: int = DEFAULT_HEIGHT) -> np.ndarray:
    """
    Resize a frame to a fixed working resolution so downstream
    modules (detector, temporal analyzer) operate on consistent input.

    Args:
        frame: Input BGR frame as a numpy array (H, W, 3).
        width: Target width in pixels.
        height: Target height in pixels.

    Returns:
        Resized frame as a numpy array (height, width, 3).

    Raises:
        ValueError: If frame is None, empty, or not a valid image array.
    """
    _validate_frame(frame)
    if width <= 0 or height <= 0:
        raise ValueError(f"width and height must be positive, got width={width}, height={height}")

    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def denoise_frame(frame: np.ndarray, strength: int = 7) -> np.ndarray:
    """
    Apply denoising to reduce camera sensor noise, especially useful
    for low-quality webcam feeds used in in-cabin monitoring.

    Args:
        frame: Input BGR frame as a numpy array.
        strength: Filter strength (higher = more denoising, more blur).
                  Must be a positive integer.

    Returns:
        Denoised frame as a numpy array, same shape as input.

    Raises:
        ValueError: If frame is None, empty, or strength is not positive.
    """
    _validate_frame(frame)
    if strength <= 0:
        raise ValueError(f"strength must be positive, got {strength}")

    return cv2.fastNlMeansDenoisingColored(
        frame, None, h=strength, hColor=strength,
        templateWindowSize=7, searchWindowSize=21
    )


def normalize_frame(frame: np.ndarray) -> np.ndarray:
    """
    Normalize pixel values to the [0, 1] float range, which most
    detection models (MediaPipe, YOLO) expect at their input layer.

    Args:
        frame: Input BGR frame as a numpy array with uint8 values [0, 255].

    Returns:
        Normalized frame as a float32 numpy array with values in [0, 1].

    Raises:
        ValueError: If frame is None or empty.
    """
    _validate_frame(frame)
    return frame.astype(np.float32) / 255.0


def preprocess_pipeline(frame: np.ndarray, width: int = DEFAULT_WIDTH,
                         height: int = DEFAULT_HEIGHT, denoise_strength: int = 7) -> np.ndarray:
    """
    Run the full preprocessing sequence: resize -> denoise -> normalize.
    This is the single entry point MainPipeline calls for each frame.

    Args:
        frame: Raw input BGR frame from the camera.
        width: Target resize width.
        height: Target resize height.
        denoise_strength: Denoising filter strength.

    Returns:
        Fully preprocessed frame ready for the enhancement stage,
        as a float32 numpy array normalized to [0, 1].
    """
    resized = resize_frame(frame, width, height)
    denoised = denoise_frame(resized, denoise_strength)
    normalized = normalize_frame(denoised)
    return normalized


def _validate_frame(frame: np.ndarray) -> None:
    """Shared validation used by every function in this module."""
    if frame is None:
        raise ValueError("frame cannot be None")
    if not isinstance(frame, np.ndarray):
        raise ValueError(f"frame must be a numpy array, got {type(frame)}")
    if frame.size == 0:
        raise ValueError("frame cannot be empty")
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"frame must have shape (H, W, 3), got {frame.shape}")
