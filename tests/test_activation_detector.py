"""
Unit tests for phasestop.activation_detector — build stage T1.
Each test verifies one behavioural contract of bayesian_cp().
"""

import pytest
from phasestop.config import Signal
from phasestop.activation_detector import bayesian_cp


# ---------------------------------------------------------------------------
# Test windows
# ---------------------------------------------------------------------------

# Baseline ≈ 0.503 (std≈0.005), recent ≈ 0.780 → ratio≈48 — clear regime shift upward
BCP_RISING_WINDOW  = [0.50, 0.50, 0.51, 0.72, 0.79, 0.83]
# Baseline ≈ 0.833 (std≈0.005), recent ≈ 0.837 → ratio≈0.58 — within noise envelope
BCP_PLATEAU_WINDOW = [0.83, 0.84, 0.83, 0.84, 0.83, 0.84]
# Baseline ≈ 0.780 (std≈0.005), recent ≈ 0.503 → ratio≈48 — clear regime shift downward
BCP_FALLING_WINDOW = [0.72, 0.79, 0.83, 0.51, 0.50, 0.50]
BCP_SHORT_WINDOW   = [0.80, 0.83, 0.85]
# Baseline all identical (std=0) — tests the epsilon guard against division-by-zero
BCP_FLAT_BASELINE  = [0.50, 0.50, 0.50, 0.82, 0.83, 0.84]


def test_bayesian_cp_rising_returns_improving():
    """Clear upward regime shift (ratio≈48) is detected as IMPROVING."""
    result = bayesian_cp(BCP_RISING_WINDOW)
    assert result.signal == Signal.IMPROVING


def test_bayesian_cp_plateau_returns_stabilized():
    """Scores within the baseline noise envelope (ratio≈0.58) are not flagged as a shift."""
    result = bayesian_cp(BCP_PLATEAU_WINDOW)
    assert result.signal == Signal.STABILIZED


def test_bayesian_cp_falling_returns_declining():
    """Downward regime shift is caught — feeds the REGRESSED path in the state machine."""
    result = bayesian_cp(BCP_FALLING_WINDOW)
    assert result.signal == Signal.DECLINING


def test_bayesian_cp_short_window_returns_insufficient():
    """Window shorter than WINDOW_K must not crash or return a misleading signal."""
    result = bayesian_cp(BCP_SHORT_WINDOW)
    assert result.signal == Signal.INSUFFICIENT


def test_bayesian_cp_flat_baseline_does_not_crash():
    """All-identical baseline (std=0) must not divide by zero — epsilon guard must fire."""
    result = bayesian_cp(BCP_FLAT_BASELINE)
    assert result.signal == Signal.IMPROVING


def test_bayesian_cp_metric_is_ratio():
    """metric field must report the stability ratio, not any other statistic."""
    result = bayesian_cp(BCP_RISING_WINDOW)
    assert result.metric.startswith("ratio=")


def test_bayesian_cp_confidence_between_zero_and_one():
    """Confidence field must stay in [0.0, 1.0] for all valid signal types."""
    for window in (BCP_RISING_WINDOW, BCP_PLATEAU_WINDOW, BCP_FALLING_WINDOW):
        result = bayesian_cp(window)
        assert 0.0 <= result.confidence <= 1.0


def test_bayesian_cp_insufficient_has_zero_confidence():
    """No false confidence when there is not enough data to split the window."""
    result = bayesian_cp(BCP_SHORT_WINDOW)
    assert result.confidence == 0.0
