"""
PhaseStop — Saturation phase detector (P3).
Implements ewma_stable() and linear_reg() — build stages D3, D4 (see CLAUDE.md).

S-curve position: active in Saturation (P3). Both detectors must agree
(conjunction, not vote) before STABILIZED is returned. ewma_stable() confirms
the smoothed value has stopped moving; linear_reg() confirms the OLS slope
is statistically indistinguishable from zero.
"""

import math

from scipy import stats

from phasestop.config import (
    DetectorResult,
    EWMA_ALPHA,
    EWMA_EPSILON,
    LR_P_THRESHOLD,
    Signal,
    WINDOW_K,
)


def ewma_stable(window: list[float]) -> DetectorResult:
    """Exponentially weighted moving average stability detector — Section 3.4.

    S-curve position: active in Saturation (P3). Must agree with linear_reg()
    (conjunction) before STABILIZED is returned by the state machine.

    Runs EWMA across the window with smoothing factor EWMA_ALPHA, then measures
    the delta between the final smoothed value and the window mean. A small delta
    means the smoothed trajectory is near the window's center of gravity — genuine
    plateau. Using the window mean (not window[0]) as baseline avoids false IMPROVING
    signals when the oldest score happens to be a low outlier.

    Returns:
        IMPROVING    delta > +EWMA_EPSILON  (smoothed value still rising)
        DECLINING    delta < -EWMA_EPSILON  (smoothed value falling)
        STABILIZED   |delta| <= EWMA_EPSILON (smoothed value flat)
        INSUFFICIENT fewer than WINDOW_K points in window
    """
    if len(window) < WINDOW_K:
        return DetectorResult(
            name="ewma_stable",
            signal=Signal.INSUFFICIENT,
            confidence=0.0,
            metric=f"n={len(window)}",
            note=f"Need {WINDOW_K} points, have {len(window)}",
        )

    window_mean = sum(window) / len(window)
    ewma = window[0]
    for x in window[1:]:
        ewma = EWMA_ALPHA * x + (1 - EWMA_ALPHA) * ewma

    # Delta is measured against the window mean, not window[0]. Using window[0]
    # as baseline causes an outlier at index 0 to inflate delta and produce a
    # false IMPROVING signal on an otherwise flat plateau.
    delta = ewma - window_mean
    confidence = round(min(abs(delta) / EWMA_EPSILON, 1.0), 3)

    if delta > EWMA_EPSILON:
        signal = Signal.IMPROVING
        note = (f"EWMA={ewma:.3f} > window mean {window_mean:.3f} "
                f"(delta={delta:+.4f} > epsilon {EWMA_EPSILON}) — still rising")
    elif delta < -EWMA_EPSILON:
        signal = Signal.DECLINING
        note = (f"EWMA={ewma:.3f} < window mean {window_mean:.3f} "
                f"(delta={delta:+.4f} < -epsilon {EWMA_EPSILON}) — falling")
    else:
        signal = Signal.STABILIZED
        note = (f"|EWMA delta| {abs(delta):.4f} <= epsilon {EWMA_EPSILON} "
                f"— EWMA {ewma:.3f} near window mean {window_mean:.3f}")

    return DetectorResult(
        name="ewma_stable",
        signal=signal,
        confidence=confidence,
        metric=f"delta={delta:+.4f}",
        note=note,
    )


def linear_reg(window: list[float]) -> DetectorResult:
    """OLS linear regression slope test — Section 3.5.

    S-curve position: active in Saturation (P3). Must agree with ewma_stable()
    (conjunction) before STABILIZED is returned by the state machine.

    Fits a straight line to the window using ordinary least squares, where x is
    the run index and y is the composite score. Tests whether the slope is
    statistically distinguishable from zero using a two-tailed t-test.

    p > LR_P_THRESHOLD (0.10) → slope not significant → STABILIZED.
    0.10 is used (not 0.05) because gradual plateaus produce small but
    non-negligible slopes that 0.05 would incorrectly treat as a real trend.

    Returns:
        IMPROVING    p <= threshold AND slope > 0  (significant positive trend)
        DECLINING    p <= threshold AND slope < 0  (significant negative trend)
        STABILIZED   p > threshold                 (slope indistinguishable from zero)
        INSUFFICIENT fewer than WINDOW_K points in window
    """
    if len(window) < WINDOW_K:
        return DetectorResult(
            name="linear_reg",
            signal=Signal.INSUFFICIENT,
            confidence=0.0,
            metric=f"n={len(window)}",
            note=f"Need {WINDOW_K} points, have {len(window)}",
        )

    n = len(window)
    x = list(range(n))

    x_mean = sum(x) / n
    y_mean = sum(window) / n

    ss_xx = sum((xi - x_mean) ** 2 for xi in x)
    ss_xy = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, window))

    slope = ss_xy / ss_xx
    intercept = y_mean - slope * x_mean

    residuals = [yi - (intercept + slope * xi) for xi, yi in zip(x, window)]
    ss_res = sum(r ** 2 for r in residuals)

    # Guard against perfect fit (ss_res == 0).
    # slope == 0 means all values are identical — definitively flat, route to STABILIZED.
    # slope != 0 means a perfect non-zero line — definitively trending, p = 0.0.
    if ss_res == 0:
        p_value = 1.0 if slope == 0 else 0.0
    else:
        se_slope = math.sqrt(ss_res / (n - 2)) / math.sqrt(ss_xx)
        t_stat = slope / se_slope
        p_value = float(2 * stats.t.sf(abs(t_stat), df=n - 2))

    confidence = round(1.0 - p_value, 3)

    if p_value > LR_P_THRESHOLD:
        signal = Signal.STABILIZED
        note = (f"p={p_value:.3f} > {LR_P_THRESHOLD}: slope={slope:+.4f} "
                f"not significant — plateau confirmed")
    elif slope > 0:
        signal = Signal.IMPROVING
        note = (f"p={p_value:.3f} <= {LR_P_THRESHOLD}: slope={slope:+.4f} "
                f"significant positive trend")
    else:
        signal = Signal.DECLINING
        note = (f"p={p_value:.3f} <= {LR_P_THRESHOLD}: slope={slope:+.4f} "
                f"significant negative trend")

    return DetectorResult(
        name="linear_reg",
        signal=signal,
        confidence=confidence,
        metric=f"slope={slope:+.4f}",
        note=note,
    )