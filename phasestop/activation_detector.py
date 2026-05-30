"""
PhaseStop — Activation phase detector (P1).
Implements bayesian_cp() — build stage D5 (see CLAUDE.md).

S-curve position: fires first. Detects when scores leave their initial
baseline and enter a new distributional regime, confirming the trajectory
has exited Activation and entered Growth (P1 → P2).
"""

import math

from phasestop.config import (
    DetectorResult,
    Signal,
    WINDOW_K,
)

_STABILITY_RATIO_THRESHOLD: float = 1.0
"""Recent mean must be this many baseline std-devs away to count as a regime shift."""

_STD_EPSILON: float = 0.001
"""Added to baseline std before dividing — prevents division-by-zero on a flat baseline."""


def bayesian_cp(window: list[float]) -> DetectorResult:
    """Bayesian change point — regime shift detector — Section 3.1.

    S-curve position: active at Activation (P1). Used alongside moving_avg()
    to confirm the P1→P2 transition: BCP detects that scores have *left* the
    baseline distribution; MA confirms the new level is *higher*.

    Splits the window into baseline (first half) and recent (second half).
    Computes the stability ratio:

        ratio = |mean(recent) - mean(baseline)| / (std(baseline) + epsilon)

    A ratio > 1.0 means the recent scores are more than one baseline
    standard deviation away from the baseline mean — a new distributional
    regime has been entered.

    Returns:
        IMPROVING    ratio > threshold AND recent mean > baseline mean (upward shift)
        DECLINING    ratio > threshold AND recent mean < baseline mean (downward shift)
        STABILIZED   ratio <= threshold (still within baseline noise envelope)
        INSUFFICIENT fewer than WINDOW_K points in window
    """
    if len(window) < WINDOW_K:
        return DetectorResult(
            name="bayesian_cp",
            signal=Signal.INSUFFICIENT,
            confidence=0.0,
            metric=f"n={len(window)}",
            note=f"Need {WINDOW_K} points, have {len(window)}",
        )

    half = len(window) // 2
    baseline = window[:half]
    recent = window[half:]

    baseline_mean = sum(baseline) / len(baseline)
    recent_mean = sum(recent) / len(recent)

    baseline_variance = sum((x - baseline_mean) ** 2 for x in baseline) / len(baseline)
    baseline_std = math.sqrt(baseline_variance)

    ratio = abs(recent_mean - baseline_mean) / (baseline_std + _STD_EPSILON)
    confidence = round(min(ratio / _STABILITY_RATIO_THRESHOLD, 1.0), 3)

    if ratio > _STABILITY_RATIO_THRESHOLD:
        if recent_mean > baseline_mean:
            signal = Signal.IMPROVING
            note = (f"ratio={ratio:.3f} > {_STABILITY_RATIO_THRESHOLD}: "
                    f"recent mean {recent_mean:.3f} > baseline mean {baseline_mean:.3f} "
                    f"(baseline std={baseline_std:.4f}) — regime shift upward")
        else:
            signal = Signal.DECLINING
            note = (f"ratio={ratio:.3f} > {_STABILITY_RATIO_THRESHOLD}: "
                    f"recent mean {recent_mean:.3f} < baseline mean {baseline_mean:.3f} "
                    f"(baseline std={baseline_std:.4f}) — regime shift downward")
    else:
        signal = Signal.STABILIZED
        note = (f"ratio={ratio:.3f} <= {_STABILITY_RATIO_THRESHOLD}: "
                f"recent mean {recent_mean:.3f} within baseline noise envelope "
                f"(baseline mean={baseline_mean:.3f}, std={baseline_std:.4f})")

    return DetectorResult(
        name="bayesian_cp",
        signal=signal,
        confidence=confidence,
        metric=f"ratio={ratio:.3f}",
        note=note,
    )
