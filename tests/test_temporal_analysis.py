"""
test_temporal_analysis.py
Unit tests for src/temporal_analysis.py (Module 4).

Run with: pytest tests/test_temporal_analysis.py -v --cov=src.temporal_analysis
"""

import pytest

from src.temporal_analysis import (
    TemporalAnalyzer,
    DetectionEvent,
    TemporalMetrics,
    DEFAULT_WINDOW_SECONDS,
)


class TestInit:
    def test_default_window(self):
        analyzer = TemporalAnalyzer()
        assert analyzer.window_seconds == DEFAULT_WINDOW_SECONDS

    def test_custom_window(self):
        analyzer = TemporalAnalyzer(window_seconds=10.0)
        assert analyzer.window_seconds == 10.0

    def test_zero_window_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TemporalAnalyzer(window_seconds=0)

    def test_negative_window_raises(self):
        with pytest.raises(ValueError, match="positive"):
            TemporalAnalyzer(window_seconds=-5)


class TestAddEvent:
    def test_add_valid_event(self):
        analyzer = TemporalAnalyzer(window_seconds=30)
        analyzer.add_event(DetectionEvent("eye_closure", timestamp=1.0, is_active=True))
        assert len(analyzer._events) == 1

    def test_none_event_raises(self):
        analyzer = TemporalAnalyzer()
        with pytest.raises(ValueError, match="cannot be None"):
            analyzer.add_event(None)

    def test_unknown_event_type_raises(self):
        analyzer = TemporalAnalyzer()
        with pytest.raises(ValueError, match="unrecognized"):
            analyzer.add_event(DetectionEvent("unknown_type", timestamp=1.0, is_active=True))

    def test_old_events_pruned(self):
        analyzer = TemporalAnalyzer(window_seconds=5.0)
        analyzer.add_event(DetectionEvent("yawn", timestamp=0.0, is_active=True))
        analyzer.add_event(DetectionEvent("yawn", timestamp=10.0, is_active=True))
        # first event (t=0) is now 10s old, outside the 5s window
        assert len(analyzer._events) == 1
        assert analyzer._events[0].timestamp == 10.0


class TestDurationScore:
    def test_no_active_events_zero_duration(self):
        analyzer = TemporalAnalyzer(window_seconds=30)
        analyzer.add_event(DetectionEvent("yawn", timestamp=1.0, is_active=False))
        assert analyzer.get_duration_score(current_time=1.0) == 0.0

    def test_duration_scales_with_time_elapsed(self):
        analyzer = TemporalAnalyzer(window_seconds=30)
        analyzer.add_event(DetectionEvent("eye_closure", timestamp=0.0, is_active=True))
        score = analyzer.get_duration_score(current_time=15.0)
        assert score == pytest.approx(0.5, abs=0.01)

    def test_duration_caps_at_one(self):
        analyzer = TemporalAnalyzer(window_seconds=10)
        analyzer.add_event(DetectionEvent("eye_closure", timestamp=0.0, is_active=True))
        score = analyzer.get_duration_score(current_time=100.0)
        assert score == 1.0


class TestSeverityScore:
    def test_no_active_events_zero_severity(self):
        analyzer = TemporalAnalyzer()
        analyzer.add_event(DetectionEvent("yawn", timestamp=1.0, is_active=False))
        assert analyzer.get_severity_score() == 0.0

    def test_takes_max_severity_among_active(self):
        analyzer = TemporalAnalyzer()
        analyzer.add_event(DetectionEvent("yawn", timestamp=1.0, is_active=True))       # 0.5
        analyzer.add_event(DetectionEvent("phone_use", timestamp=2.0, is_active=True))  # 1.0
        assert analyzer.get_severity_score() == 1.0

    def test_inactive_events_ignored(self):
        analyzer = TemporalAnalyzer()
        analyzer.add_event(DetectionEvent("phone_use", timestamp=1.0, is_active=False))
        analyzer.add_event(DetectionEvent("yawn", timestamp=2.0, is_active=True))
        assert analyzer.get_severity_score() == 0.5


class TestFrequencyScore:
    def test_empty_window_zero_frequency(self):
        analyzer = TemporalAnalyzer()
        assert analyzer.get_frequency_score() == 0.0

    def test_all_active_gives_frequency_one(self):
        analyzer = TemporalAnalyzer()
        analyzer.add_event(DetectionEvent("yawn", timestamp=1.0, is_active=True))
        analyzer.add_event(DetectionEvent("yawn", timestamp=2.0, is_active=True))
        assert analyzer.get_frequency_score() == 1.0

    def test_mixed_active_gives_fraction(self):
        analyzer = TemporalAnalyzer()
        analyzer.add_event(DetectionEvent("yawn", timestamp=1.0, is_active=True))
        analyzer.add_event(DetectionEvent("yawn", timestamp=2.0, is_active=False))
        assert analyzer.get_frequency_score() == 0.5


class TestEvaluate:
    def test_evaluate_returns_temporal_metrics(self):
        analyzer = TemporalAnalyzer(window_seconds=30)
        analyzer.add_event(DetectionEvent("eye_closure", timestamp=0.0, is_active=True))
        metrics = analyzer.evaluate(current_time=15.0)
        assert isinstance(metrics, TemporalMetrics)
        assert 0.0 <= metrics.duration_score <= 1.0
        assert 0.0 <= metrics.severity_score <= 1.0
        assert 0.0 <= metrics.frequency_score <= 1.0


class TestReset:
    def test_reset_clears_events(self):
        analyzer = TemporalAnalyzer()
        analyzer.add_event(DetectionEvent("yawn", timestamp=1.0, is_active=True))
        analyzer.reset()
        assert len(analyzer._events) == 0
        assert analyzer.get_frequency_score() == 0.0
