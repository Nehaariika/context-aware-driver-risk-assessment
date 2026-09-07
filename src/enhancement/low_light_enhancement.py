"""
low_light_enhancement.py
Module 2 of the Context-Aware Driver Risk Assessment pipeline.

Improves visibility of the driver's face in poor cabin lighting
(night driving, tunnels, backlit conditions) before detection runs.
Two techniques are provided: CLAHE (adaptive contrast) and Gamma
correction (brightness curve adjustment). They can be used
independently or chained.
"""

import cv2
import numpy as np


DEFAULT_CLIP_LIMIT = 2.0
DEFAULT_TILE_GRID_SIZE = (8, 8)
DEFAULT_GAMMA = 1.5


def apply_clahe(frame: np.ndarray, clip_limit: float = DEFAULT_CLIP_LIMIT,
                 tile_grid_size: tuple = DEFAULT_TILE_GRID_SIZE) -> np.ndarray:
    """
    Apply Contrast Limited Adaptive Histogram Equalization on the
    luminance channel to boost local contrast without blowing out
    already-bright regions (e.g. a dashboard light next to a dark face).

    Args:
        frame: Input BGR frame as a numpy array (uint8, [0,255]).
        clip_limit: Threshold for contrast limiting; higher = stronger effect.
        tile_grid_size: Size of the grid for histogram equalization tiles.

    Returns:
        Contrast-enhanced BGR frame, same shape and dtype as input.

    Raises:
        ValueError: If frame is invalid or clip_limit is not positive.
    """
    _validate_frame(frame)
    if clip_limit <= 0:
        raise ValueError(f"clip_limit must be positive, got {clip_limit}")

    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    l_enhanced = clahe.apply(l_channel)

    enhanced_lab = cv2.merge((l_enhanced, a_channel, b_channel))
    return cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)


def apply_gamma_correction(frame: np.ndarray, gamma: float = DEFAULT_GAMMA) -> np.ndarray:
    """
    Apply gamma correction to brighten (gamma > 1) or darken (gamma < 1)
    a frame. Used to lift shadow detail in dark cabin footage before
    the detector looks for eye/mouth landmarks.

    Args:
        frame: Input BGR frame as a numpy array (uint8, [0,255]).
        gamma: Correction exponent. >1 brightens, <1 darkens, =1 is a no-op.

    Returns:
        Gamma-corrected BGR frame, same shape and dtype as input.

    Raises:
        ValueError: If frame is invalid or gamma is not positive.
    """
    _validate_frame(frame)
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")

    inv_gamma = 1.0 / gamma
    table = np.array([
        ((i / 255.0) ** inv_gamma) * 255 for i in range(256)
    ]).astype(np.uint8)

    return cv2.LUT(frame, table)


def is_low_light(frame: np.ndarray, brightness_threshold: float = 90.0) -> bool:
    """
    Decide whether a frame is dark enough to warrant enhancement,
    so the pipeline can skip this stage in good lighting and save
    processing time.

    Args:
        frame: Input BGR frame as a numpy array.
        brightness_threshold: Mean luminance below which a frame is
                               considered low-light (0-255 scale).

    Returns:
        True if the frame's mean brightness is below the threshold.

    Raises:
        ValueError: If frame is invalid.
    """
    _validate_frame(frame)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray)) < brightness_threshold


def enhance_frame(frame: np.ndarray, clip_limit: float = DEFAULT_CLIP_LIMIT,
                   gamma: float = DEFAULT_GAMMA,
                   brightness_threshold: float = 90.0) -> np.ndarray:
    """
    Pipeline entry point for Module 2. Checks whether the frame is
    low-light and, if so, applies gamma correction followed by CLAHE.
    Bright frames pass through unchanged to avoid over-processing.

    Args:
        frame: Preprocessed BGR frame (uint8, [0,255]) from Module 1.
        clip_limit: CLAHE clip limit.
        gamma: Gamma correction exponent.
        brightness_threshold: Mean luminance threshold to trigger enhancement.

    Returns:
        Enhanced frame if low-light, otherwise the original frame unchanged.
    """
    _validate_frame(frame)
    if not is_low_light(frame, brightness_threshold):
        return frame

    brightened = apply_gamma_correction(frame, gamma)
    contrasted = apply_clahe(brightened, clip_limit)
    return contrasted


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
    if frame.dtype != np.uint8:
        raise ValueError(f"frame must be uint8, got {frame.dtype}")
