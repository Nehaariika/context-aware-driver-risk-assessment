"""
risk_scorer.py
Module 5 of the Context-Aware Driver Risk Assessment pipeline.

Computes a driver's overall risk score from Duration/Severity/Frequency
metrics using a weighted formula, then maps the score to a risk level
(Low/Moderate/High/Critical) that drives the adaptive warning system.
"""

from enum import Enum
from dataclasses import dataclass


# Default weights for the risk formula: w₁·Duration + w₂·Severity + w₃·Frequency
# These sum to 1.0 and can be empirically calibrated against ground truth
DEFAULT_WEIGHT_DURATION = 0.3
DEFAULT_WEIGHT_SEVERITY = 0.5
DEFAULT_WEIGHT_FREQUENCY = 0.2

# Risk level thresholds — a score in [0, 1] maps to one of four levels
THRESHOLD_LOW = 0.0
THRESHOLD_MODERATE = 0.3
THRESHOLD_HIGH = 0.6
THRESHOLD_CRITICAL = 0.8


class RiskLevel(Enum):
    """Four-tier driver risk classification."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskScore:
    """Output of the risk computation, consumed by the warning system."""
    score: float  # Overall score in [0, 1]
    level: RiskLevel
    duration_component: float
    severity_component: float
    frequency_component: float


def compute_risk_score(duration: float, severity: float, frequency: float,
                       w_duration: float = DEFAULT_WEIGHT_DURATION,
                       w_severity: float = DEFAULT_WEIGHT_SEVERITY,
                       w_frequency: float = DEFAULT_WEIGHT_FREQUENCY) -> float:
    """
    Compute a weighted risk score from D/S/F metrics.

    Args:
        duration: Duration score in [0, 1].
        severity: Severity score in [0, 1].
        frequency: Frequency score in [0, 1].
        w_duration: Weight for duration component.
        w_severity: Weight for severity component.
        w_frequency: Weight for frequency component.

    Returns:
        Overall risk score in [0, 1].

    Raises:
        ValueError: If any score is outside [0, 1] or weights don't sum to ~1.0.
    """
    _validate_metrics(duration, severity, frequency)
    _validate_weights(w_duration, w_severity, w_frequency)

    return (w_duration * duration +
            w_severity * severity +
            w_frequency * frequency)


def classify_risk_level(score: float) -> RiskLevel:
    """
    Map a numerical risk score to a risk level using fixed thresholds.

    Args:
        score: Risk score in [0, 1].

    Returns:
        One of RiskLevel.LOW, MODERATE, HIGH, CRITICAL.

    Raises:
        ValueError: If score is outside [0, 1].
    """
    if not (0.0 <= score <= 1.0):
        raise ValueError(f"score must be in [0, 1], got {score}")

    if score >= THRESHOLD_CRITICAL:
        return RiskLevel.CRITICAL
    elif score >= THRESHOLD_HIGH:
        return RiskLevel.HIGH
    elif score >= THRESHOLD_MODERATE:
        return RiskLevel.MODERATE
    else:
        return RiskLevel.LOW


class RiskScorer:
    """
    Stateful scorer that maintains configurable weights and produces
    RiskScore objects ready for the warning system to consume.
    """

    def __init__(self, w_duration: float = DEFAULT_WEIGHT_DURATION,
                 w_severity: float = DEFAULT_WEIGHT_SEVERITY,
                 w_frequency: float = DEFAULT_WEIGHT_FREQUENCY):
        """
        Args:
            w_duration: Weight for the duration component.
            w_severity: Weight for the severity component.
            w_frequency: Weight for the frequency component.

        Raises:
            ValueError: If weights are invalid.
        """
        _validate_weights(w_duration, w_severity, w_frequency)
        self.w_duration = w_duration
        self.w_severity = w_severity
        self.w_frequency = w_frequency

    def score(self, duration: float, severity: float, frequency: float) -> RiskScore:
        """
        Compute risk and return a complete RiskScore object.

        Args:
            duration: Duration metric in [0, 1].
            severity: Severity metric in [0, 1].
            frequency: Frequency metric in [0, 1].

        Returns:
            RiskScore with score, level, and component breakdowns.

        Raises:
            ValueError: If any metric is outside [0, 1].
        """
        _validate_metrics(duration, severity, frequency)

        duration_component = self.w_duration * duration
        severity_component = self.w_severity * severity
        frequency_component = self.w_frequency * frequency

        overall_score = (duration_component +
                         severity_component +
                         frequency_component)

        level = classify_risk_level(overall_score)

        return RiskScore(
            score=overall_score,
            level=level,
            duration_component=duration_component,
            severity_component=severity_component,
            frequency_component=frequency_component,
        )

    def update_weights(self, w_duration: float, w_severity: float,
                       w_frequency: float) -> None:
        """
        Recalibrate the risk formula weights (e.g. after collecting real
        ground-truth data).

        Args:
            w_duration: New duration weight.
            w_severity: New severity weight.
            w_frequency: New frequency weight.

        Raises:
            ValueError: If new weights are invalid.
        """
        _validate_weights(w_duration, w_severity, w_frequency)
        self.w_duration = w_duration
        self.w_severity = w_severity
        self.w_frequency = w_frequency


def _validate_metrics(duration: float, severity: float, frequency: float) -> None:
    """Shared validation for D/S/F metric arrays."""
    for name, value in [("duration", duration), ("severity", severity), ("frequency", frequency)]:
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}")


def _validate_weights(w_duration: float, w_severity: float, w_frequency: float) -> None:
    """Shared validation for weight coefficients."""
    for name, w in [("w_duration", w_duration), ("w_severity", w_severity), ("w_frequency", w_frequency)]:
        if w < 0.0 or w > 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {w}")

    total = w_duration + w_severity + w_frequency
    if not (0.99 <= total <= 1.01):
        raise ValueError(f"weights must sum to 1.0, got {total}")
