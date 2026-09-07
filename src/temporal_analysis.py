"""
temporal_analysis.py
Module 4 of the Context-Aware Driver Risk Assessment pipeline.

Tracks detection events over a sliding time window and derives three
metrics feeding the risk scorer: Duration (how long a risky state has
persisted), Severity (how intense the current state is), and Frequency
(how often risky events recur in the window).
"""

from collections import deque
from dataclasses import dataclass, field


DEFAULT_WINDOW_SECONDS = 30.0

# Severity weights per event type — higher means a more dangerous cue.
EVENT_SEVERITY = {
    "eye_closure": 0.9,
    "yawn": 0.5,
    "head_distraction": 0.6,
    "phone_use": 1.0,
}


@dataclass
class DetectionEvent:
    """A single timestamped detection result fed into the window."""
    event_type: str
    timestamp: float
    is_active: bool  # True if the risky condition is present in this frame


@dataclass
class TemporalMetrics:
    """Output of one window evaluation, consumed by the risk scorer."""
    duration_score: float
    severity_score: float
    frequency_score: float


class TemporalAnalyzer:
    """
    Maintains a sliding window of DetectionEvents and computes
    normalized Duration / Severity / Frequency scores, each in [0, 1].
    """

    def __init__(self, window_seconds: float = DEFAULT_WINDOW_SECONDS):
        """
        Args:
            window_seconds: Size of the sliding time window in seconds.

        Raises:
            ValueError: If window_seconds is not positive.
        """
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self.window_seconds = window_seconds
        self._events: deque = deque()

    def add_event(self, event: DetectionEvent) -> None:
        """
        Add a new detection event and drop events that have fallen
        outside the sliding window.

        Args:
            event: The DetectionEvent to add.

        Raises:
            ValueError: If event is None or event_type is unrecognized.
        """
        if event is None:
            raise ValueError("event cannot be None")
        if event.event_type not in EVENT_SEVERITY:
            raise ValueError(f"unrecognized event_type: {event.event_type}")

        self._events.append(event)
        self._prune(event.timestamp)

    def _prune(self, current_time: float) -> None:
        """Remove events older than window_seconds relative to current_time."""
        cutoff = current_time - self.window_seconds
        while self._events and self._events[0].timestamp < cutoff:
            self._events.popleft()

    def get_duration_score(self, current_time: float) -> float:
        """
        Compute how long the most recent unbroken run of active events
        (of any type) has persisted, normalized against the window size.

        Args:
            current_time: The current timestamp to measure duration up to.

        Returns:
            Duration score in [0, 1], where 1.0 means active for the
            full window.
        """
        active_events = [e for e in self._events if e.is_active]
        if not active_events:
            return 0.0

        earliest_active = min(e.timestamp for e in active_events)
        duration = current_time - earliest_active
        return min(duration / self.window_seconds, 1.0)

    def get_severity_score(self) -> float:
        """
        Compute the severity of currently active events as the max
        severity weight among active event types in the window.

        Returns:
            Severity score in [0, 1]. 0.0 if no active events.
        """
        active_types = {e.event_type for e in self._events if e.is_active}
        if not active_types:
            return 0.0
        return max(EVENT_SEVERITY[t] for t in active_types)

    def get_frequency_score(self) -> float:
        """
        Compute how often risky (active) events recur within the window,
        as the fraction of all events in the window that are active.

        Returns:
            Frequency score in [0, 1]. 0.0 if the window is empty.
        """
        if not self._events:
            return 0.0
        active_count = sum(1 for e in self._events if e.is_active)
        return active_count / len(self._events)

    def evaluate(self, current_time: float) -> TemporalMetrics:
        """
        Compute all three metrics at once. This is the entry point
        MainPipeline calls after each detection result is added.

        Args:
            current_time: The current timestamp.

        Returns:
            TemporalMetrics with duration_score, severity_score, frequency_score.
        """
        return TemporalMetrics(
            duration_score=self.get_duration_score(current_time),
            severity_score=self.get_severity_score(),
            frequency_score=self.get_frequency_score(),
        )

    def reset(self) -> None:
        """Clear all events, e.g. at the start of a new driving session."""
        self._events.clear()
