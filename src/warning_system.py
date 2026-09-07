"""
warning_system.py
Module 6 of the Context-Aware Driver Risk Assessment pipeline.

Generates adaptive alerts (sound, visual, haptic) based on risk level
and logs all detection/warning events to SQLite for post-session review.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from src.risk_scoring import RiskLevel


DEFAULT_DB_PATH = "driver_monitoring.db"

# Alert properties per risk level: (sound_file, visual_text, vibration_duration_ms)
ALERT_CONFIG = {
    RiskLevel.LOW: {
        "sound": None,
        "visual": "⚠️ Stay Alert",
        "vibration_ms": 0,
    },
    RiskLevel.MODERATE: {
        "sound": "alert_moderate.wav",
        "visual": "⚠️ WARNING: Drowsiness Detected",
        "vibration_ms": 200,
    },
    RiskLevel.HIGH: {
        "sound": "alert_high.wav",
        "visual": "🔴 CRITICAL: Immediate Action Required",
        "vibration_ms": 500,
    },
    RiskLevel.CRITICAL: {
        "sound": "alert_critical.wav",
        "visual": "🔴🔴 CRITICAL: PULL OVER NOW",
        "vibration_ms": 1000,
    },
}


@dataclass
class AlertEvent:
    """A single alert triggered by the warning system."""
    timestamp: float
    risk_level: RiskLevel
    response_time_ms: int  # Time from alert trigger to driver response


class WarningSystem:
    """
    Generates adaptive alerts and logs events to a SQLite database.
    The actual sound/vibration playback is delegated to thin methods
    that can be stubbed or mocked in unit tests.
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """
        Args:
            db_path: Path to the SQLite database file.

        Raises:
            ValueError: If db_path is None or empty.
        """
        if not db_path:
            raise ValueError("db_path cannot be None or empty")
        self.db_path = db_path
        self._initialize_db()
        self._last_alert_level: Optional[RiskLevel] = None

    def _initialize_db(self) -> None:
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS detection_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                event_type TEXT NOT NULL,
                confidence REAL,
                session_id INTEGER
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS alert_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                risk_level TEXT NOT NULL,
                response_time_ms INTEGER,
                session_id INTEGER
            )
        """)

        conn.commit()
        conn.close()

    def trigger_alert(self, risk_level: RiskLevel, timestamp: float) -> AlertEvent:
        """
        Trigger an adaptive alert based on the risk level. Avoids
        alert fatigue by not re-alerting if the level hasn't changed.

        Args:
            risk_level: The current RiskLevel.
            timestamp: The timestamp of the alert.

        Returns:
            AlertEvent describing the triggered alert.

        Raises:
            ValueError: If risk_level is not a RiskLevel enum value.
        """
        if not isinstance(risk_level, RiskLevel):
            raise ValueError(f"risk_level must be a RiskLevel, got {type(risk_level)}")

        config = ALERT_CONFIG[risk_level]

        # Only trigger sound/vibration if level escalated
        if risk_level.value > (self._last_alert_level.value if self._last_alert_level else ""):
            self._play_sound(config["sound"])
            self._trigger_vibration(config["vibration_ms"])

        self._show_visual_alert(config["visual"])
        self._last_alert_level = risk_level

        return AlertEvent(
            timestamp=timestamp,
            risk_level=risk_level,
            response_time_ms=0,  # Placeholder; actual response time measured post-alert
        )

    def log_detection_event(self, event_type: str, timestamp: float,
                             confidence: float = 0.0, session_id: Optional[int] = None) -> int:
        """
        Log a detection event (eye closure, yawn, etc.) to the database.

        Args:
            event_type: Type of detection ("eye_closure", "yawn", "phone_use", etc.).
            timestamp: Timestamp of the event.
            confidence: Confidence score of the detection [0, 1].
            session_id: Optional session ID for grouping events.

        Returns:
            The inserted row ID.

        Raises:
            ValueError: If event_type is empty or confidence is out of range.
        """
        if not event_type:
            raise ValueError("event_type cannot be empty")
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"confidence must be in [0, 1], got {confidence}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO detection_events (timestamp, event_type, confidence, session_id)
            VALUES (?, ?, ?, ?)
        """, (timestamp, event_type, confidence, session_id))

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return row_id

    def log_alert_event(self, risk_level: RiskLevel, timestamp: float,
                        response_time_ms: int = 0, session_id: Optional[int] = None) -> int:
        """
        Log an alert event (warning triggered) to the database.

        Args:
            risk_level: The RiskLevel that triggered the alert.
            timestamp: Timestamp of the alert.
            response_time_ms: Time until driver responded (0 if not yet responded).
            session_id: Optional session ID.

        Returns:
            The inserted row ID.

        Raises:
            ValueError: If inputs are invalid.
        """
        if not isinstance(risk_level, RiskLevel):
            raise ValueError(f"risk_level must be a RiskLevel, got {type(risk_level)}")
        if response_time_ms < 0:
            raise ValueError(f"response_time_ms cannot be negative, got {response_time_ms}")

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            INSERT INTO alert_events (timestamp, risk_level, response_time_ms, session_id)
            VALUES (?, ?, ?, ?)
        """, (timestamp, risk_level.value, response_time_ms, session_id))

        row_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return row_id

    def get_session_summary(self, session_id: int) -> dict:
        """
        Retrieve a summary of all detection and alert events for a session.

        Args:
            session_id: The session ID to query.

        Returns:
            dict with keys "detections" (list) and "alerts" (list).
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM detection_events WHERE session_id = ?", (session_id,))
        detections = [dict(row) for row in cursor.fetchall()]

        cursor.execute("SELECT * FROM alert_events WHERE session_id = ?", (session_id,))
        alerts = [dict(row) for row in cursor.fetchall()]

        conn.close()

        return {"detections": detections, "alerts": alerts}

    def _play_sound(self, sound_file: Optional[str]) -> None:
        """
        Play an alert sound. Stubbed here for testing; real implementation
        would use pydub, playsound, or similar library.
        """
        if sound_file:
            pass  # In production: actually load and play the audio file

    def _trigger_vibration(self, duration_ms: int) -> None:
        """
        Trigger haptic feedback for duration_ms. Stubbed for testing;
        real implementation would use pydub's simpleaudio or a hardware interface.
        """
        if duration_ms > 0:
            pass  # In production: actually trigger vibration hardware

    def _show_visual_alert(self, visual_text: str) -> None:
        """
        Display a visual alert. Stubbed for testing; real implementation
        would update the dashboard UI or overlay on the video stream.
        """
        pass  # In production: render visual_text in the UI
