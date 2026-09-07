"""
detector.py
Module 3 of the Context-Aware Driver Risk Assessment pipeline.

Contains the detection logic for four visual cues: eye closure (EAR),
yawning (MAR), head pose (distraction), and phone use (YOLO box overlap).

Design note: the core logic here works on landmark coordinates / bounding
boxes rather than raw frames or loaded models. This keeps it unit-testable
with synthetic data, without requiring MediaPipe/YOLO to be installed or
a webcam to be present. The thin `FrameDetector` wrapper at the bottom is
what wires in the real MediaPipe FaceMesh and YOLO model at runtime.
"""

import numpy as np


# Eye landmark indices from MediaPipe FaceMesh (left eye, 6-point EAR scheme)
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
MOUTH_IDX = [61, 291, 39, 181, 0, 17]  # left, right, top-outer, top-inner, bottom-inner, bottom-outer

EAR_THRESHOLD = 0.21
MAR_THRESHOLD = 0.6
HEAD_YAW_THRESHOLD_DEG = 30.0
HEAD_PITCH_THRESHOLD_DEG = 20.0
PHONE_CLASS_ID = 67  # COCO class id for "cell phone"
PHONE_CONFIDENCE_THRESHOLD = 0.5


def calculate_ear(eye_points: np.ndarray) -> float:
    """
    Compute the Eye Aspect Ratio from 6 (x, y) eye landmark points.
    Low EAR means the eye is closing/closed.

    Args:
        eye_points: (6, 2) array of landmark coordinates
                    [p1_outer, p2_top1, p3_top2, p4_inner, p5_bottom2, p6_bottom1].

    Returns:
        The EAR value as a float. Typically ~0.3 open, ~0.1 closed.

    Raises:
        ValueError: If eye_points doesn't have shape (6, 2).
    """
    _validate_points(eye_points, expected_count=6, name="eye_points")

    vertical_1 = np.linalg.norm(eye_points[1] - eye_points[5])
    vertical_2 = np.linalg.norm(eye_points[2] - eye_points[4])
    horizontal = np.linalg.norm(eye_points[0] - eye_points[3])

    if horizontal == 0:
        raise ValueError("horizontal eye distance cannot be zero")

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def calculate_mar(mouth_points: np.ndarray) -> float:
    """
    Compute the Mouth Aspect Ratio from 6 (x, y) mouth landmark points.
    High MAR means the mouth is open wide (yawn candidate).

    Args:
        mouth_points: (6, 2) array of landmark coordinates
                      [left, right, top_outer, top_inner, bottom_inner, bottom_outer].

    Returns:
        The MAR value as a float.

    Raises:
        ValueError: If mouth_points doesn't have shape (6, 2).
    """
    _validate_points(mouth_points, expected_count=6, name="mouth_points")

    vertical_1 = np.linalg.norm(mouth_points[2] - mouth_points[4])
    vertical_2 = np.linalg.norm(mouth_points[3] - mouth_points[5])
    horizontal = np.linalg.norm(mouth_points[0] - mouth_points[1])

    if horizontal == 0:
        raise ValueError("horizontal mouth distance cannot be zero")

    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def detect_eye_closure(left_eye_points: np.ndarray, right_eye_points: np.ndarray,
                        threshold: float = EAR_THRESHOLD) -> dict:
    """
    Determine whether both eyes are closed based on average EAR.

    Args:
        left_eye_points: (6, 2) left eye landmark coordinates.
        right_eye_points: (6, 2) right eye landmark coordinates.
        threshold: EAR value below which an eye is considered closed.

    Returns:
        dict with keys: is_closed (bool), avg_ear (float).
    """
    left_ear = calculate_ear(left_eye_points)
    right_ear = calculate_ear(right_eye_points)
    avg_ear = (left_ear + right_ear) / 2.0

    return {"is_closed": bool(avg_ear < threshold), "avg_ear": float(avg_ear)}


def detect_yawn(mouth_points: np.ndarray, threshold: float = MAR_THRESHOLD) -> dict:
    """
    Determine whether the driver is yawning based on MAR.

    Args:
        mouth_points: (6, 2) mouth landmark coordinates.
        threshold: MAR value above which the mouth is considered a yawn.

    Returns:
        dict with keys: is_yawning (bool), mar (float).
    """
    mar = calculate_mar(mouth_points)
    return {"is_yawning": bool(mar > threshold), "mar": float(mar)}


def classify_head_pose(yaw_deg: float, pitch_deg: float,
                        yaw_threshold: float = HEAD_YAW_THRESHOLD_DEG,
                        pitch_threshold: float = HEAD_PITCH_THRESHOLD_DEG) -> dict:
    """
    Classify whether the head pose angles indicate distraction
    (looking away from the road / head drooping down).

    Args:
        yaw_deg: Head rotation left/right in degrees (0 = facing forward).
        pitch_deg: Head rotation up/down in degrees (0 = level).
        yaw_threshold: Max yaw before flagged as looking away.
        pitch_threshold: Max pitch before flagged as head drooping.

    Returns:
        dict with keys: is_distracted (bool), yaw_deg (float), pitch_deg (float).
    """
    is_distracted = abs(yaw_deg) > yaw_threshold or abs(pitch_deg) > pitch_threshold
    return {"is_distracted": is_distracted, "yaw_deg": yaw_deg, "pitch_deg": pitch_deg}


def detect_phone_use(detections: list, class_id: int = PHONE_CLASS_ID,
                      confidence_threshold: float = PHONE_CONFIDENCE_THRESHOLD) -> dict:
    """
    Determine whether a phone is present in the frame from a list of
    YOLO-style detections.

    Args:
        detections: List of dicts, each with keys "class_id" (int) and
                    "confidence" (float), as produced by a YOLO model's
                    postprocessing step.
        class_id: The class id representing "cell phone" (COCO: 67).
        confidence_threshold: Minimum confidence to count as a detection.

    Returns:
        dict with keys: phone_detected (bool), confidence (float, 0 if none).

    Raises:
        ValueError: If detections is not a list.
    """
    if not isinstance(detections, list):
        raise ValueError(f"detections must be a list, got {type(detections)}")

    phone_matches = [
        d["confidence"] for d in detections
        if d.get("class_id") == class_id and d.get("confidence", 0) >= confidence_threshold
    ]

    if phone_matches:
        return {"phone_detected": True, "confidence": max(phone_matches)}
    return {"phone_detected": False, "confidence": 0.0}


def _validate_points(points: np.ndarray, expected_count: int, name: str) -> None:
    """Shared validation for landmark point arrays."""
    if points is None:
        raise ValueError(f"{name} cannot be None")
    points = np.asarray(points)
    if points.shape != (expected_count, 2):
        raise ValueError(f"{name} must have shape ({expected_count}, 2), got {points.shape}")


class FrameDetector:
    """
    Runtime detector using MediaPipe Tasks FaceLandmarker.

    Compatible with MediaPipe 0.10.35 and Python 3.13.
    """

    def __init__(self, yolo_model=None, model_path=None):
        import os
        import cv2
        import mediapipe as mp

        if model_path is None:
            # detector.py is:
            # src/detection/detector.py
            #
            # Project root is therefore:
            # ../../
            #
            # Model is:
            # models/face_landmarker.task
            model_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "..",
                    "models",
                    "face_landmarker.task",
                )
            )

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Face Landmarker model not found: {model_path}"
            )

        print(f"[INIT] Loading MediaPipe FaceLandmarker: {model_path}")

        base_options = mp.tasks.BaseOptions(
            model_asset_path=model_path
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.IMAGE,
            num_faces=1,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self._face_landmarker = (
            mp.tasks.vision.FaceLandmarker.create_from_options(options)
        )

        self._yolo_model = yolo_model

        print("[INIT] MediaPipe FaceLandmarker: OK")

    def detect(self, frame: np.ndarray) -> dict:
        """
        Run face, eye closure and yawn detection on one BGR frame.
        """

        import cv2
        import mediapipe as mp

        if frame is None:
            raise ValueError("frame cannot be None")

        h, w = frame.shape[:2]

        # OpenCV uses BGR; MediaPipe expects RGB.
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Convert NumPy image to MediaPipe Image.
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )

        # IMAGE mode -> detect()
        results = self._face_landmarker.detect(mp_image)

        output = {
            "face_found": False
        }

        if results.face_landmarks:
            landmarks = results.face_landmarks[0]

            pts = np.array(
                [
                    (lm.x * w, lm.y * h)
                    for lm in landmarks
                ],
                dtype=np.float32,
            )

            output["face_found"] = True

            # Eye closure
            output["eye_closure"] = detect_eye_closure(
                pts[LEFT_EYE_IDX],
                pts[RIGHT_EYE_IDX],
            )

            # Yawning
            output["yawn"] = detect_yawn(
                pts[MOUTH_IDX]
            )

        # YOLO phone detection
        if self._yolo_model is not None:
            yolo_results = self._yolo_model(
                frame,
                verbose=False
            )[0]

            detections = [
                {
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]),
                }
                for box in yolo_results.boxes
            ]

            output["phone_use"] = detect_phone_use(detections)

        return output

    def close(self):
        """Release MediaPipe resources."""
        if hasattr(self, "_face_landmarker"):
            self._face_landmarker.close()
