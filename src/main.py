"""
main.py
Main orchestration pipeline for the Context-Aware Driver Risk Assessment system.

Ties together all 6 modules:
1. Preprocessing (resize, denoise, normalize)
2. Enhancement (low-light adjustment)
3. Detection (eye closure, yawn, head pose, phone use)
4. Temporal Analysis (sliding window metrics)
5. Risk Scoring (D/S/F weighted formula)
6. Warning System (alerts + logging)

Run with: python main.py --source 0  (for webcam)
          python main.py --source video.mp4  (for video file)
"""

import argparse
import cv2
import numpy as np
import time
from typing import Optional
from datetime import datetime

from src.preprocessing import preprocess_pipeline
from src.enhancement.low_light_enhancement import enhance_frame
from src.detection.detector import FrameDetector
from src.temporal_analysis import TemporalAnalyzer, DetectionEvent
from src.risk_scoring import RiskScorer
from src.warning_system import WarningSystem


class MainPipeline:
    """
    Full end-to-end driver monitoring pipeline. Processes frames sequentially
    through all 6 modules and displays results in real-time.
    """

    def __init__(self, yolo_model=None, db_path: str = "driver_monitoring.db"):
        """
        Initialize the pipeline with all 6 components.

        Args:
            yolo_model: Pre-loaded YOLO model for phone detection (optional).
            db_path: Path to SQLite database for logging.
        """
        print("[INIT] Initializing pipeline modules...")

        # Module 3: Detector
        self.detector = FrameDetector(yolo_model=yolo_model)

        # Module 4: Temporal Analyzer
        self.temporal_analyzer = TemporalAnalyzer(window_seconds=30.0)

        # Module 5: Risk Scorer
        self.risk_scorer = RiskScorer(
            w_duration=0.3,
            w_severity=0.5,
            w_frequency=0.2
        )

        # Module 6: Warning System
        self.warning_system = WarningSystem(db_path=db_path)

        # Session tracking
        self.session_id = int(time.time())
        self.frame_count = 0
        self.start_time = time.time()

        print("[INIT] Pipeline ready. Starting capture loop...")

    def process_frame(self, frame: np.ndarray, timestamp: Optional[float] = None) -> dict:
        """
        Run a single frame through the full pipeline.

        Args:
            frame: Raw BGR frame from camera/video.
            timestamp: Optional override for timestamp (defaults to current time).

        Returns:
            dict with all pipeline outputs (preprocessed, enhanced, detections, risk, alerts).
        """
        if timestamp is None:
            timestamp = time.time() - self.start_time

        # MODULE 1: Preprocess
        preprocessed = preprocess_pipeline(frame)
        # Convert back to uint8 for enhancement and detection
        preprocessed_uint8 = (preprocessed * 255).astype(np.uint8)

        # MODULE 2: Enhance (low-light)
        enhanced = enhance_frame(preprocessed_uint8)

        # MODULE 3: Detect (eye closure, yawn, head pose, phone)
        detections = self.detector.detect(enhanced)

        # MODULE 4 & 5: Temporal + Risk Scoring
        risk_score = None
        risk_level = None

        if detections["face_found"]:
            # Log detection events
            if "eye_closure" in detections:
                is_closed = detections["eye_closure"]["is_closed"]
                self.temporal_analyzer.add_event(DetectionEvent(
                    event_type="eye_closure",
                    timestamp=timestamp,
                    is_active=is_closed
                ))
                self.warning_system.log_detection_event(
                    "eye_closure", timestamp,
                    confidence=1.0 - detections["eye_closure"]["avg_ear"],
                    session_id=self.session_id
                )

            if "yawn" in detections:
                is_yawning = detections["yawn"]["is_yawning"]
                self.temporal_analyzer.add_event(DetectionEvent(
                    event_type="yawn",
                    timestamp=timestamp,
                    is_active=is_yawning
                ))
                self.warning_system.log_detection_event(
                    "yawn", timestamp,
                    confidence=detections["yawn"]["mar"],
                    session_id=self.session_id
                )

            # Evaluate temporal metrics
            temporal_metrics = self.temporal_analyzer.evaluate(timestamp)

            # Compute risk score
            risk_score_obj = self.risk_scorer.score(
                duration=temporal_metrics.duration_score,
                severity=temporal_metrics.severity_score,
                frequency=temporal_metrics.frequency_score
            )
            risk_score = risk_score_obj.score
            risk_level = risk_score_obj.level

            # MODULE 6: Alert if needed
            if risk_level.value in ["high", "critical"]:
                alert_event = self.warning_system.trigger_alert(risk_level, timestamp)
                self.warning_system.log_alert_event(
                    risk_level, timestamp,
                    session_id=self.session_id
                )

        if "phone_use" in detections:
            phone_detected = detections["phone_use"]["phone_detected"]
            self.temporal_analyzer.add_event(DetectionEvent(
                event_type="phone_use",
                timestamp=timestamp,
                is_active=phone_detected
            ))
            if phone_detected:
                self.warning_system.log_detection_event(
                    "phone_use", timestamp,
                    confidence=detections["phone_use"]["confidence"],
                    session_id=self.session_id
                )

        self.frame_count += 1

        return {
            "frame": enhanced,
            "timestamp": timestamp,
            "detections": detections,
            "risk_score": risk_score,
            "risk_level": risk_level,
        }

    def run(self, source: str = "0", output_video: Optional[str] = None):
        """
        Main capture loop: read frames, process, display results.

        Args:
            source: "0" for webcam, or path to video file.
            output_video: Optional path to save output video with overlays.
        """
        # Open video source
        if source == "0":
            cap = cv2.VideoCapture(0)
            print("[RUN] Opened webcam (source=0)")
        else:
            cap = cv2.VideoCapture(source)
            print(f"[RUN] Opened video file: {source}")

        if not cap.isOpened():
            print("[ERROR] Failed to open video source")
            return

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

        # Setup video writer if output requested
        out = None
        if output_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))
            print(f"[RUN] Saving output to: {output_video}")

        print(f"[RUN] Video: {frame_width}x{frame_height} @ {fps} fps")
        print("[RUN] Press 'Q' to quit, 'P' to pause")

        paused = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[RUN] End of video stream")
                    break

                if not paused:
                    # Process the frame
                    result = self.process_frame(frame)

                    # Draw overlays
                    display_frame = self._draw_overlay(
                        result["frame"],
                        result["detections"],
                        result["risk_score"],
                        result["risk_level"],
                        result["timestamp"]
                    )

                    # Write to output if requested
                    if out:
                        out.write(display_frame)

                    # Display
                    cv2.imshow("Driver Risk Assessment", display_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    print("[RUN] Quit requested")
                    break
                elif key == ord('p') or key == ord('P'):
                    paused = not paused
                    status = "PAUSED" if paused else "RESUMING"
                    print(f"[RUN] {status}")

        except KeyboardInterrupt:
            print("[RUN] Interrupted by user")
        finally:
            cap.release()
            if out:
                out.release()
            cv2.destroyAllWindows()
            print(f"[RUN] Processed {self.frame_count} frames in {time.time() - self.start_time:.1f}s")

    def _draw_overlay(self, frame, detections, risk_score, risk_level, timestamp):
        """Draw detection results and risk info on the frame."""
        display = frame.copy()
        h, w = display.shape[:2]

        # Risk level color
        if risk_level is None:
            color = (255, 255, 255)  # White (no face)
            level_text = "No Face Detected"
        elif risk_level.value == "low":
            color = (0, 255, 0)  # Green
            level_text = "LOW RISK"
        elif risk_level.value == "moderate":
            color = (0, 165, 255)  # Orange
            level_text = "MODERATE RISK"
        elif risk_level.value == "high":
            color = (0, 0, 255)  # Red
            level_text = "HIGH RISK"
        else:  # critical
            color = (0, 0, 139)  # Dark Red
            level_text = "🔴 CRITICAL 🔴"

        # Draw risk level
        cv2.rectangle(display, (0, 0), (w, 60), color, -1)
        cv2.putText(display, level_text, (10, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    1.5, (255, 255, 255), 2)

        # Draw score
        if risk_score is not None:
            score_text = f"Risk Score: {risk_score:.2f}"
            cv2.putText(display, score_text, (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (200, 200, 200), 2)

        # Draw detection info
        info_lines = [
            f"Frame: {self.frame_count}",
            f"Time: {timestamp:.1f}s",
        ]

        if detections.get("face_found"):
            ear = detections.get("eye_closure", {}).get("avg_ear", 0)
            mar = detections.get("yawn", {}).get("mar", 0)
            info_lines.append(f"EAR: {ear:.2f}")
            info_lines.append(f"MAR: {mar:.2f}")

        if detections.get("phone_use", {}).get("phone_detected"):
            info_lines.append("☎️ PHONE DETECTED")

        for i, line in enumerate(info_lines):
            cv2.putText(display, line, (10, 80 + i*25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (200, 200, 200), 1)

        return display


def main():
    parser = argparse.ArgumentParser(
        description="Driver Risk Assessment Pipeline"
    )
    parser.add_argument("--source", type=str, default="0",
                        help="Video source: '0' for webcam, or path to video file")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional path to save output video")
    parser.add_argument("--db", type=str, default="driver_monitoring.db",
                        help="Path to SQLite database")

    args = parser.parse_args()

    # Initialize pipeline
    pipeline = MainPipeline(db_path=args.db)

    # Run
    pipeline.run(source=args.source, output_video=args.output)

    # Print summary
    print("\n" + "="*60)
    print("SESSION SUMMARY")
    print("="*60)
    summary = pipeline.warning_system.get_session_summary(pipeline.session_id)
    print(f"Total Detections: {len(summary['detections'])}")
    print(f"Total Alerts: {len(summary['alerts'])}")
    if summary['alerts']:
        print(f"Alert Risk Levels: {[a['risk_level'] for a in summary['alerts']]}")
    print("="*60)


if __name__ == "__main__":
    main()
