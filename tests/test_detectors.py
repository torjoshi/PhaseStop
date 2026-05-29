"""
Unit tests for phasestop.detectors — build stage T1.
Each test verifies one behavioural contract of the detector.
"""

import pytest
from phasestop.config import Signal
from phasestop.detectors import mann_kendall


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