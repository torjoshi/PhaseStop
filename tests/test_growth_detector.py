"""
Unit tests for phasestop.growth_detector — build stage T1.
Each test verifies one behavioural contract of the detector.
"""

import pytest
from phasestop.config import Signal
from phasestop.growth_detector import mann_kendall, moving_avg


# ---------------------------------------------------------------------------
# mann_kendall
# ---------------------------------------------------------------------------

RISING_WINDOW  = [0.50, 0.60, 0.70, 0.80, 0.85, 0.88]   # S=15, p=0.009
PLATEAU_WINDOW = [0.84, 0.84, 0.83, 0.84, 0.84, 0.83]   # S=-4, p=0.573
FALLING_WINDOW = [0.88, 0.85, 0.80, 0.70, 0.60, 0.50]   # S=-15, p=0.009
SHORT_WINDOW   = [0.80, 0.83, 0.85]                       # n=3 < WINDOW_K


def test_mann_kendall_rising_returns_improving():
    """Core happy path — a clear upward trend (S=15, p=0.009) is detected as IMPROVING."""
    result = mann_kendall(RISING_WINDOW)
    assert result.signal == Signal.IMPROVING


def test_mann_kendall_plateau_returns_stabilized():
    """Noisy flat line (S=-4, p=0.573) must not be misclassified as DECLINING."""
    result = mann_kendall(PLATEAU_WINDOW)
    assert result.signal == Signal.STABILIZED


def test_mann_kendall_falling_returns_declining():
    """Downward trend (S=-15, p=0.009) is caught — feeds the REGRESSED path in the state machine."""
    result = mann_kendall(FALLING_WINDOW)
    assert result.signal == Signal.DECLINING


def test_mann_kendall_short_window_returns_insufficient():
    """Window shorter than WINDOW_K must not crash or return a misleading signal."""
    result = mann_kendall(SHORT_WINDOW)
    assert result.signal == Signal.INSUFFICIENT


def test_mann_kendall_metric_is_p_value():
    """Confirms the spec change from 'tau=...' to 'p=...' is honoured in the metric field."""
    result = mann_kendall(RISING_WINDOW)
    assert result.metric.startswith("p=")


def test_mann_kendall_confidence_between_zero_and_one():
    """Confidence field must stay in [0.0, 1.0] for all valid signal types."""
    for window in (RISING_WINDOW, PLATEAU_WINDOW, FALLING_WINDOW):
        result = mann_kendall(window)
        assert 0.0 <= result.confidence <= 1.0


def test_mann_kendall_insufficient_has_zero_confidence():
    """No false confidence when there is not enough data to compute a statistic."""
    result = mann_kendall(SHORT_WINDOW)
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# moving_avg
# ---------------------------------------------------------------------------

# prior mean ≈ 0.650, recent mean ≈ 0.823 → shift = +0.173 (well above 0.01 threshold)
MA_RISING_WINDOW  = [0.60, 0.65, 0.70, 0.78, 0.83, 0.86]
# prior mean ≈ 0.833, recent mean ≈ 0.837 → shift = +0.003 (below threshold)
MA_PLATEAU_WINDOW = [0.83, 0.84, 0.83, 0.84, 0.83, 0.84]
# prior mean ≈ 0.823, recent mean ≈ 0.650 → shift = -0.173 (below -0.01 threshold)
MA_FALLING_WINDOW = [0.86, 0.83, 0.78, 0.70, 0.65, 0.60]
MA_SHORT_WINDOW   = [0.80, 0.83, 0.85]


def test_moving_avg_rising_returns_improving():
    """Sustained upward shift across half-windows is detected as IMPROVING."""
    result = moving_avg(MA_RISING_WINDOW)
    assert result.signal == Signal.IMPROVING


def test_moving_avg_plateau_returns_stabilized():
    """Negligible mean shift (0.003) must not be classified as IMPROVING."""
    result = moving_avg(MA_PLATEAU_WINDOW)
    assert result.signal == Signal.STABILIZED


def test_moving_avg_falling_returns_declining():
    """Sustained downward shift is caught — feeds the REGRESSED path in the state machine."""
    result = moving_avg(MA_FALLING_WINDOW)
    assert result.signal == Signal.DECLINING


def test_moving_avg_short_window_returns_insufficient():
    """Window shorter than WINDOW_K must not crash or return a misleading signal."""
    result = moving_avg(MA_SHORT_WINDOW)
    assert result.signal == Signal.INSUFFICIENT


def test_moving_avg_metric_is_shift():
    """metric field must report the mean shift, not any other statistic."""
    result = moving_avg(MA_RISING_WINDOW)
    assert result.metric.startswith("shift=")


def test_moving_avg_confidence_between_zero_and_one():
    """Confidence field must stay in [0.0, 1.0] for all valid signal types."""
    for window in (MA_RISING_WINDOW, MA_PLATEAU_WINDOW, MA_FALLING_WINDOW):
        result = moving_avg(window)
        assert 0.0 <= result.confidence <= 1.0


def test_moving_avg_insufficient_has_zero_confidence():
    """No false confidence when there is not enough data to split the window."""
    result = moving_avg(MA_SHORT_WINDOW)
    assert result.confidence == 0.0


# ---------------------------------------------------------------------------
# Disagreement cases — MA and MK capture different signals
# ---------------------------------------------------------------------------

# Scores jump to a higher level but oscillate there (not monotonic).
# MA sees the level shift (+0.080) → IMPROVING.
# MK sees no consistent rank order (p≈0.199) → STABILIZED.
OSCILLATING_HIGH = [0.70, 0.80, 0.70, 0.82, 0.78, 0.84]

# Scores rise slowly but perfectly (every step is higher than the last).
# MK sees a perfectly monotonic sequence (S=15, p≈0.004) → IMPROVING.
# MA sees a tiny mean shift (+0.006 < threshold 0.01) → STABILIZED.
SLOW_MONOTONIC_RISE = [0.800, 0.802, 0.804, 0.806, 0.808, 0.810]


def test_ma_detects_level_shift_that_mk_misses():
    """MA catches a jump to a higher oscillating level; MK does not.

    The scores bounce up and down at the top, so there is no consistent
    rank order for MK to detect. MA only cares that the recent mean is
    higher than the prior mean — which it clearly is.
    """
    assert moving_avg(OSCILLATING_HIGH).signal == Signal.IMPROVING
    assert mann_kendall(OSCILLATING_HIGH).signal == Signal.STABILIZED


def test_mk_detects_slow_trend_that_ma_misses():
    """MK catches a perfectly monotonic but tiny rise; MA does not.

    Every score is higher than the last, so MK's rank-based test fires
    with high confidence. But the absolute mean shift between the two
    half-windows is only 0.006 — below MA's 0.01 threshold — so MA
    reports no meaningful level change.
    """
    assert mann_kendall(SLOW_MONOTONIC_RISE).signal == Signal.IMPROVING
    assert moving_avg(SLOW_MONOTONIC_RISE).signal == Signal.STABILIZED