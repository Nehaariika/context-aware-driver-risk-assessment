"""
test_warning_system.py
Unit tests for src/warning_system.py (Module 6).

Uses a temporary SQLite database for each test to avoid pollution.

Run with: pytest tests/test_warning_system.py -v --cov=src.warning_system
"""

import os
import tempfile
import pytest
import sqlite3

from src.risk_scoring import RiskLevel
from src.warning_system import (
    WarningSystem,
    AlertEvent,
    ALERT_CONFIG,
)


@pytest.fixture
def temp_db():
    """Create a temporary database file for testing."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    yield path
    if os.path.exists(path):
        os.remove(path)


@pytest.fixture
def warning_system(temp_db):
    """Create a WarningSystem instance with a temporary database."""
    return WarningSystem(db_path=temp_db)


class TestInit:
    def test_init_creates_database(self, temp_db):
        sys = WarningSystem(db_path=temp_db)
        assert os.path.exists(temp_db)

    def test_init_creates_tables(self, warning_system, temp_db):
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        assert "detection_events" in tables
        assert "alert_events" in tables

    def test_init_none_db_path_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            WarningSystem(db_path=None)

    def test_init_empty_db_path_raises(self):
        with pytest.raises(ValueError, match="cannot be None"):
            WarningSystem(db_path="")


class TestTriggerAlert:
    def test_trigger_low_alert(self, warning_system):
        event = warning_system.trigger_alert(RiskLevel.LOW, timestamp=1.0)
        assert event.risk_level == RiskLevel.LOW
        assert event.timestamp == 1.0

    def test_trigger_critical_alert(self, warning_system):
        event = warning_system.trigger_alert(RiskLevel.CRITICAL, timestamp=2.0)
        assert event.risk_level == RiskLevel.CRITICAL

    def test_invalid_risk_level_raises(self, warning_system):
        with pytest.raises(ValueError, match="RiskLevel"):
            warning_system.trigger_alert("not_a_level", timestamp=1.0)

    def test_alert_escalation_triggers_sound(self, warning_system):
        # First alert at LOW
        warning_system.trigger_alert(RiskLevel.LOW, timestamp=1.0)
        # Escalate to HIGH - sound should trigger
        event = warning_system.trigger_alert(RiskLevel.HIGH, timestamp=2.0)
        assert event.risk_level == RiskLevel.HIGH

    def test_same_level_doesnt_re_alert(self, warning_system):
        warning_system.trigger_alert(RiskLevel.MODERATE, timestamp=1.0)
        # Same level again
        event = warning_system.trigger_alert(RiskLevel.MODERATE, timestamp=2.0)
        assert event.risk_level == RiskLevel.MODERATE


class TestLogDetectionEvent:
    def test_log_detection_event(self, warning_system, temp_db):
        row_id = warning_system.log_detection_event("eye_closure", timestamp=1.0, confidence=0.95)
        assert row_id > 0

        # Verify it was written
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT event_type, confidence FROM detection_events WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        conn.close()
        assert row[0] == "eye_closure"
        assert row[1] == 0.95

    def test_log_with_session_id(self, warning_system, temp_db):
        row_id = warning_system.log_detection_event("yawn", timestamp=2.0, confidence=0.8, session_id=1)
        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT session_id FROM detection_events WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        conn.close()
        assert row[0] == 1

    def test_empty_event_type_raises(self, warning_system):
        with pytest.raises(ValueError, match="empty"):
            warning_system.log_detection_event("", timestamp=1.0)

    def test_confidence_out_of_range_raises(self, warning_system):
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            warning_system.log_detection_event("eye_closure", timestamp=1.0, confidence=1.5)


class TestLogAlertEvent:
    def test_log_alert_event(self, warning_system, temp_db):
        row_id = warning_system.log_alert_event(RiskLevel.HIGH, timestamp=3.0, response_time_ms=150)
        assert row_id > 0

        conn = sqlite3.connect(temp_db)
        cursor = conn.cursor()
        cursor.execute("SELECT risk_level, response_time_ms FROM alert_events WHERE id = ?", (row_id,))
        row = cursor.fetchone()
        conn.close()
        assert row[0] == "high"
        assert row[1] == 150

    def test_invalid_risk_level_raises(self, warning_system):
        with pytest.raises(ValueError, match="RiskLevel"):
            warning_system.log_alert_event("not_a_level", timestamp=1.0)

    def test_negative_response_time_raises(self, warning_system):
        with pytest.raises(ValueError, match="negative"):
            warning_system.log_alert_event(RiskLevel.MODERATE, timestamp=1.0, response_time_ms=-100)


class TestGetSessionSummary:
    def test_empty_session_summary(self, warning_system):
        summary = warning_system.get_session_summary(session_id=999)
        assert summary["detections"] == []
        assert summary["alerts"] == []

    def test_session_summary_with_events(self, warning_system):
        warning_system.log_detection_event("eye_closure", timestamp=1.0, session_id=1)
        warning_system.log_detection_event("yawn", timestamp=2.0, session_id=1)
        warning_system.log_alert_event(RiskLevel.MODERATE, timestamp=2.5, session_id=1)

        summary = warning_system.get_session_summary(session_id=1)
        assert len(summary["detections"]) == 2
        assert len(summary["alerts"]) == 1
        assert summary["detections"][0]["event_type"] == "eye_closure"
        assert summary["alerts"][0]["risk_level"] == "moderate"

    def test_session_summary_filters_by_session_id(self, warning_system):
        warning_system.log_detection_event("eye_closure", timestamp=1.0, session_id=1)
        warning_system.log_detection_event("yawn", timestamp=2.0, session_id=2)

        summary = warning_system.get_session_summary(session_id=1)
        assert len(summary["detections"]) == 1
        assert summary["detections"][0]["event_type"] == "eye_closure"


class TestAlertConfig:
    def test_all_risk_levels_have_config(self):
        for level in RiskLevel:
            assert level in ALERT_CONFIG
            config = ALERT_CONFIG[level]
            assert "sound" in config
            assert "visual" in config
            assert "vibration_ms" in config

    def test_low_level_no_sound(self):
        config = ALERT_CONFIG[RiskLevel.LOW]
        assert config["sound"] is None

    def test_critical_level_maximum_vibration(self):
        config = ALERT_CONFIG[RiskLevel.CRITICAL]
        assert config["vibration_ms"] >= ALERT_CONFIG[RiskLevel.HIGH]["vibration_ms"]
