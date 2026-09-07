"""
test_risk_scoring.py
Unit tests for src/risk_scoring.py (Module 5).

Run with: pytest tests/test_risk_scoring.py -v --cov=src.risk_scoring
"""

import pytest

from src.risk_scoring import (
    compute_risk_score,
    classify_risk_level,
    RiskScorer,
    RiskLevel,
    DEFAULT_WEIGHT_DURATION,
    DEFAULT_WEIGHT_SEVERITY,
    DEFAULT_WEIGHT_FREQUENCY,
    THRESHOLD_LOW,
    THRESHOLD_MODERATE,
    THRESHOLD_HIGH,
    THRESHOLD_CRITICAL,
)


class TestComputeRiskScore:
    def test_all_zeros_gives_zero(self):
        score = compute_risk_score(0.0, 0.0, 0.0)
        assert score == 0.0

    def test_all_ones_gives_one(self):
        score = compute_risk_score(1.0, 1.0, 1.0)
        assert score == 1.0

    def test_weighted_combination(self):
        score = compute_risk_score(1.0, 0.0, 0.0)
        expected = DEFAULT_WEIGHT_DURATION * 1.0
        assert score == pytest.approx(expected)

    def test_severity_dominates_with_default_weights(self):
        score_high_severity = compute_risk_score(0.0, 1.0, 0.0)
        score_high_duration = compute_risk_score(1.0, 0.0, 0.0)
        assert score_high_severity > score_high_duration

    def test_custom_weights(self):
        score = compute_risk_score(1.0, 1.0, 1.0,
                                   w_duration=0.5, w_severity=0.3, w_frequency=0.2)
        assert score == pytest.approx(1.0)

    def test_duration_out_of_range_raises(self):
        with pytest.raises(ValueError, match="duration"):
            compute_risk_score(1.5, 0.0, 0.0)

    def test_weights_dont_sum_raises(self):
        with pytest.raises(ValueError, match="sum to 1.0"):
            compute_risk_score(0.5, 0.5, 0.5, w_duration=0.2, w_severity=0.2, w_frequency=0.2)


class TestClassifyRiskLevel:
    def test_low_level(self):
        level = classify_risk_level(THRESHOLD_LOW)
        assert level == RiskLevel.LOW

    def test_moderate_level(self):
        level = classify_risk_level(THRESHOLD_MODERATE)
        assert level == RiskLevel.MODERATE

    def test_high_level(self):
        level = classify_risk_level(THRESHOLD_HIGH)
        assert level == RiskLevel.HIGH

    def test_critical_level(self):
        level = classify_risk_level(THRESHOLD_CRITICAL)
        assert level == RiskLevel.CRITICAL

    def test_score_below_threshold_is_low(self):
        level = classify_risk_level(0.1)
        assert level == RiskLevel.LOW

    def test_score_between_moderate_and_high(self):
        level = classify_risk_level(0.45)
        assert level == RiskLevel.MODERATE

    def test_score_out_of_range_raises(self):
        with pytest.raises(ValueError, match="\\[0, 1\\]"):
            classify_risk_level(1.5)


class TestRiskScorer:
    def test_init_default_weights(self):
        scorer = RiskScorer()
        assert scorer.w_severity == DEFAULT_WEIGHT_SEVERITY

    def test_init_custom_weights(self):
        scorer = RiskScorer(w_duration=0.4, w_severity=0.4, w_frequency=0.2)
        assert scorer.w_duration == 0.4

    def test_init_invalid_weights_raises(self):
        with pytest.raises(ValueError):
            RiskScorer(w_duration=0.5, w_severity=0.3, w_frequency=0.1)

    def test_score_returns_risk_score_object(self):
        scorer = RiskScorer()
        result = scorer.score(duration=0.5, severity=0.5, frequency=0.5)
        assert result.score >= 0.0
        assert result.score <= 1.0
        assert isinstance(result.level, RiskLevel)

    def test_score_includes_component_breakdown(self):
        scorer = RiskScorer()
        result = scorer.score(1.0, 1.0, 1.0)
        assert result.duration_component == pytest.approx(DEFAULT_WEIGHT_DURATION)
        assert result.severity_component == pytest.approx(DEFAULT_WEIGHT_SEVERITY)
        assert result.frequency_component == pytest.approx(DEFAULT_WEIGHT_FREQUENCY)

    def test_score_high_severity_with_custom_weights_yields_high_level(self):
        # With custom weights emphasizing severity, high severity alone gives HIGH
        scorer = RiskScorer(w_duration=0.1, w_severity=0.8, w_frequency=0.1)
        result = scorer.score(0.0, 1.0, 0.0)
        assert result.level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_update_weights(self):
        scorer = RiskScorer()
        scorer.update_weights(0.6, 0.2, 0.2)
        assert scorer.w_duration == 0.6
        result = scorer.score(1.0, 0.0, 0.0)
        assert result.score == pytest.approx(0.6)

    def test_update_weights_invalid_raises(self):
        scorer = RiskScorer()
        with pytest.raises(ValueError):
            scorer.update_weights(0.5, 0.3, 0.1)


class TestRiskLevelEnum:
    def test_all_risk_levels_defined(self):
        assert hasattr(RiskLevel, 'LOW')
        assert hasattr(RiskLevel, 'MODERATE')
        assert hasattr(RiskLevel, 'HIGH')
        assert hasattr(RiskLevel, 'CRITICAL')

    def test_risk_level_values(self):
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.CRITICAL.value == "critical"
