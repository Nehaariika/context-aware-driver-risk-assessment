"""
test_detector.py
Unit tests for src/detection/detector.py (Module 3).

Tests target the pure logic functions (EAR/MAR, classification, phone
matching) using synthetic landmark/detection data - no MediaPipe or YOLO
model needs to be loaded to run these.

Run with: pytest tests/test_detector.py -v --cov=src.detection.detector
"""

import numpy as np
import pytest

from src.detection.detector import (
    calculate_ear,
    calculate_mar,
    detect_eye_closure,
    detect_yawn,
    classify_head_pose,
    detect_phone_use,
    EAR_THRESHOLD,
    MAR_THRESHOLD,
)


def make_eye(open_amount: float) -> np.ndarray:
    """Synthetic 6-point eye: open_amount scales vertical gap (0=closed, 1=fully open)."""
    return np.array([
        [0.0, 5.0],
        [3.0, 5.0 - 3.0 * open_amount],
        [7.0, 5.0 - 3.0 * open_amount],
        [10.0, 5.0],
        [7.0, 5.0 + 3.0 * open_amount],
        [3.0, 5.0 + 3.0 * open_amount],
    ])


def make_mouth(open_amount: float) -> np.ndarray:
    """Synthetic 6-point mouth: open_amount scales vertical gap (0=closed, 1=wide open)."""
    return np.array([
        [0.0, 5.0],
        [10.0, 5.0],
        [4.0, 5.0 - 5.0 * open_amount],
        [6.0, 5.0 - 4.0 * open_amount],
        [6.0, 5.0 + 4.0 * open_amount],
        [4.0, 5.0 + 5.0 * open_amount],
    ])


class TestCalculateEar:
    def test_open_eye_has_higher_ear_than_closed(self):
        open_ear = calculate_ear(make_eye(1.0))
        closed_ear = calculate_ear(make_eye(0.05))
        assert open_ear > closed_ear

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            calculate_ear(np.zeros((5, 2)))

    def test_none_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            calculate_ear(None)


class TestCalculateMar:
    def test_open_mouth_has_higher_mar_than_closed(self):
        open_mar = calculate_mar(make_mouth(1.0))
        closed_mar = calculate_mar(make_mouth(0.02))
        assert open_mar > closed_mar

    def test_wrong_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            calculate_mar(np.zeros((4, 2)))


class TestDetectEyeClosure:
    def test_closed_eyes_flagged(self):
        result = detect_eye_closure(make_eye(0.02), make_eye(0.02))
        assert result["is_closed"] is True

    def test_open_eyes_not_flagged(self):
        result = detect_eye_closure(make_eye(1.0), make_eye(1.0))
        assert result["is_closed"] is False

    def test_result_contains_avg_ear(self):
        result = detect_eye_closure(make_eye(1.0), make_eye(1.0))
        assert "avg_ear" in result
        assert isinstance(result["avg_ear"], float)

    def test_custom_threshold(self):
        result = detect_eye_closure(make_eye(0.5), make_eye(0.5), threshold=0.9)
        assert result["is_closed"] is True


class TestDetectYawn:
    def test_wide_open_mouth_flagged_as_yawn(self):
        result = detect_yawn(make_mouth(1.0))
        assert result["is_yawning"] is True

    def test_closed_mouth_not_flagged(self):
        result = detect_yawn(make_mouth(0.02))
        assert result["is_yawning"] is False

    def test_result_contains_mar(self):
        result = detect_yawn(make_mouth(1.0))
        assert "mar" in result


class TestClassifyHeadPose:
    def test_forward_facing_not_distracted(self):
        result = classify_head_pose(yaw_deg=2.0, pitch_deg=1.0)
        assert result["is_distracted"] is False

    def test_large_yaw_flagged_distracted(self):
        result = classify_head_pose(yaw_deg=45.0, pitch_deg=0.0)
        assert result["is_distracted"] is True

    def test_large_pitch_flagged_distracted(self):
        result = classify_head_pose(yaw_deg=0.0, pitch_deg=35.0)
        assert result["is_distracted"] is True

    def test_negative_yaw_also_flagged(self):
        result = classify_head_pose(yaw_deg=-45.0, pitch_deg=0.0)
        assert result["is_distracted"] is True

    def test_custom_thresholds(self):
        result = classify_head_pose(yaw_deg=10.0, pitch_deg=0.0, yaw_threshold=5.0)
        assert result["is_distracted"] is True


class TestDetectPhoneUse:
    def test_phone_detected_above_threshold(self):
        detections = [{"class_id": 67, "confidence": 0.8}]
        result = detect_phone_use(detections)
        assert result["phone_detected"] is True
        assert result["confidence"] == 0.8

    def test_phone_below_confidence_threshold_ignored(self):
        detections = [{"class_id": 67, "confidence": 0.2}]
        result = detect_phone_use(detections)
        assert result["phone_detected"] is False

    def test_no_phone_class_present(self):
        detections = [{"class_id": 0, "confidence": 0.9}]
        result = detect_phone_use(detections)
        assert result["phone_detected"] is False

    def test_empty_detections(self):
        result = detect_phone_use([])
        assert result["phone_detected"] is False
        assert result["confidence"] == 0.0

    def test_multiple_phones_returns_highest_confidence(self):
        detections = [
            {"class_id": 67, "confidence": 0.6},
            {"class_id": 67, "confidence": 0.9},
        ]
        result = detect_phone_use(detections)
        assert result["confidence"] == 0.9

    def test_non_list_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            detect_phone_use("not a list")
