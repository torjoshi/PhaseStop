"""
Unit tests for phasestop.saturation_detector — build stage T1.
Each test verifies one behavioural contract of the detector.
"""

import pytest
from phasestop.config import Signal
from phasestop.saturation_detector import ewma_stable


# ---------------------------------------------------------------------------
# ewma_stable
# ---------------------------------------------------------------------------

# EWMA travels from 0.60 to ~0.767 → delta ≈ +0.167 (well above epsilon 0.005)
EWMA_RISING_WINDOW  = [0.60, 0.65, 0.70, 0.78, 0.83, 0.86]
# EWMA barely moves → delta ≈ -0.001 (within epsilon 0.005)
EWMA_PLATEAU_WINDOW = [0.84, 0.84, 0.83, 0.84, 0.84, 0.84]
# EWMA travels from 0.86 to ~0.704 → delta ≈ -0.156 (well below -epsilon)
EWMA_FALLING_WINDOW = [0.86, 0.83, 0.78, 0.70, 0.65, 0.60]
EWMA_SHORT_WINDOW   = [0.80, 0.83, 0.85]


def test_ewma_stable_rising_returns_improving():
    """EWMA climbs significantly — trajectory is still in growth, not yet saturated."""
    result = ewma_stable(EWMA_RISING_WINDOW)
    assert result.signal == Signal.IMPROVING


def test_ewma_stable_plateau_returns_stabilized():
    """EWMA barely moves (delta ≈ -0.001) — smoothed value is flat, saturation confirmed."""
    result = ewma_stable(EWMA_PLATEAU_WINDOW)
    assert result.signal == Signal.STABILIZED


def test_ewma_stable_falling_returns_declining():
    """EWMA drops significantly — feeds the REGRESSED path in the state machine."""
    result = ewma_stable(EWMA_FALLING_WINDOW)
    assert result.signal == Signal.DECLINING


def test_ewma_stable_short_window_returns_insufficient():
    """Window shorter than WINDOW_K must not crash or return a misleading signal."""
    result = ewma_stable(EWMA_SHORT_WINDOW)
    assert result.signal == Signal.INSUFFICIENT


def test_ewma_stable_metric_is_delta():
    """metric field must report the EWMA delta, not any other statistic."""
    result = ewma_stable(EWMA_RISING_WINDOW)
    assert result.metric.startswith("delta=")


def test_ewma_stable_confidence_between_zero_and_one():
    """Confidence field must stay in [0.0, 1.0] for all valid signal types."""
    for window in (EWMA_RISING_WINDOW, EWMA_PLATEAU_WINDOW, EWMA_FALLING_WINDOW):
        result = ewma_stable(window)
        assert 0.0 <= result.confidence <= 1.0


def test_ewma_stable_insufficient_has_zero_confidence():
    """No false confidence when there is not enough data to compute EWMA."""
    result = ewma_stable(EWMA_SHORT_WINDOW)
    assert result.confidence == 0.0
