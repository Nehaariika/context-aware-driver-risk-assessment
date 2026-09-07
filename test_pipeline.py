#!/usr/bin/env python3
"""
Quick verification script to test the pipeline with synthetic data.
No webcam or video file needed.
"""

import numpy as np
from src.preprocessing import preprocess_pipeline
from src.enhancement.low_light_enhancement import enhance_frame, is_low_light
from src.detection.detector import detect_eye_closure, detect_yawn, calculate_ear, calculate_mar
from src.temporal_analysis import TemporalAnalyzer, DetectionEvent
from src.risk_scoring import RiskScorer, classify_risk_level
from src.warning_system import WarningSystem


def test_module_1_preprocessing():
    """Test Module 1: Preprocessing"""
    print("\n[TEST] Module 1: Preprocessing")
    frame = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    result = preprocess_pipeline(frame)
    assert result.shape == (480, 640, 3)
    assert result.dtype == np.float32
    assert 0 <= result.min() and result.max() <= 1
    print("✓ Preprocessing works")


def test_module_2_enhancement():
    """Test Module 2: Low-light Enhancement"""
    print("\n[TEST] Module 2: Enhancement")
    # Dark frame
    dark_frame = np.ones((480, 640, 3), dtype=np.uint8) * 50
    assert is_low_light(dark_frame)
    enhanced = enhance_frame(dark_frame)
    assert enhanced.shape == dark_frame.shape
    print("✓ Low-light enhancement works")


def test_module_3_detection():
    """Test Module 3: Detection"""
    print("\n[TEST] Module 3: Detection")
    # Synthetic eye points (open)
    open_eye = np.array([
        [100, 100],  # outer
        [110, 90],   # top1
        [110, 92],   # top2
        [120, 100],  # inner
        [110, 108],  # bottom2
        [110, 106],  # bottom1
    ], dtype=np.float32)

    ear = calculate_ear(open_eye)
    assert 0 < ear < 1
    print(f"  EAR (open): {ear:.2f}")

    # Synthetic mouth points (closed)
    mouth = np.array([
        [100, 150], [120, 150],  # left, right
        [110, 145], [110, 147],  # top
        [110, 153], [110, 155],  # bottom
    ], dtype=np.float32)

    mar = calculate_mar(mouth)
    result = detect_yawn(mouth)
    print(f"  MAR (closed): {mar:.2f}, Yawning: {result['is_yawning']}")
    print("✓ Detection works")


def test_module_4_temporal():
    """Test Module 4: Temporal Analysis"""
    print("\n[TEST] Module 4: Temporal Analysis")
    analyzer = TemporalAnalyzer(window_seconds=10.0)

    # Add some events
    analyzer.add_event(DetectionEvent("eye_closure", 0.0, True))
    analyzer.add_event(DetectionEvent("yawn", 1.0, False))
    analyzer.add_event(DetectionEvent("phone_use", 2.0, True))

    metrics = analyzer.evaluate(3.0)
    assert 0 <= metrics.duration_score <= 1
    assert 0 <= metrics.severity_score <= 1
    assert 0 <= metrics.frequency_score <= 1
    print(f"  Duration: {metrics.duration_score:.2f}")
    print(f"  Severity: {metrics.severity_score:.2f}")
    print(f"  Frequency: {metrics.frequency_score:.2f}")
    print("✓ Temporal analysis works")


def test_module_5_risk_scoring():
    """Test Module 5: Risk Scoring"""
    print("\n[TEST] Module 5: Risk Scoring")
    scorer = RiskScorer()

    # High risk scenario
    risk_obj = scorer.score(duration=0.8, severity=0.9, frequency=0.7)
    print(f"  Risk Score: {risk_obj.score:.2f}")
    print(f"  Risk Level: {risk_obj.level.value}")
    assert risk_obj.level.value in ["low", "moderate", "high", "critical"]
    print("✓ Risk scoring works")


def test_module_6_warning():
    """Test Module 6: Warning System"""
    print("\n[TEST] Module 6: Warning System")
    import tempfile
    import os

    # Use temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test.db")
        warning_sys = WarningSystem(db_path=db_path)

        # Log events
        warning_sys.log_detection_event("eye_closure", 1.0, 0.95, session_id=1)
        warning_sys.log_detection_event("yawn", 2.0, 0.75, session_id=1)

        # Retrieve
        summary = warning_sys.get_session_summary(1)
        assert len(summary['detections']) == 2
        print(f"  Logged {len(summary['detections'])} detection events")
        print("✓ Warning system works")


def main():
    print("="*60)
    print("DRIVER RISK ASSESSMENT PIPELINE - VERIFICATION TEST")
    print("="*60)

    try:
        test_module_1_preprocessing()
        test_module_2_enhancement()
        test_module_3_detection()
        test_module_4_temporal()
        test_module_5_risk_scoring()
        test_module_6_warning()

        print("\n" + "="*60)
        print("✅ ALL MODULES VERIFIED SUCCESSFULLY")
        print("="*60)
        print("\nYou can now run:")
        print("  python src/main.py --source 0")
        print("="*60)

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
