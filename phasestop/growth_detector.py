"""
PhaseStop — Growth phase detectors (P2).

moving_avg()  — Did scores shift to a higher level?
    Compares the mean of the oldest half of the window to the newest half.
    Used at the P1→P2 boundary (with BCP) to confirm Growth has started,
    and as a guard inside Growth to catch reversals (REGRESSED).

mann_kendall() — Is the upward trend still statistically significant?
    Rank-based significance test (p-value). Used at the P2→P3 boundary
    to detect when the trend is fading (move to SATURATION_CHECK),
    and in SATURATION_CHECK to detect if growth has resumed (revert to P2).
"""

import math

from phasestop.config import (
    DetectorResult,
    MA_SLOPE_THRESHOLD,
    MK_P_THRESHOLD,
    Signal,
    WINDOW_K,
)


def mann_kendall(window: list[float]) -> DetectorResult:
    """Mann-Kendall significance test for monotonic trend — Section 3.3.

    S-curve position: active in the Growth phase (P2). A fading p-value
    (p rising above threshold) signals the transition to Saturation (P3).
    Also re-checked in SATURATION_CHECK: a newly significant p signals
    the trajectory has resumed growth and the state reverts to P2.

    Counts concordant (+1) and discordant (-1) pairs across all (i, j)
    combinations where i < j to produce the test statistic S. Converts S
    to a Z score using the known variance formula, then derives a two-tailed
    p-value via the standard normal CDF.

    p < MK_P_THRESHOLD  → significant trend; direction set by tau sign.
    p >= MK_P_THRESHOLD → no significant trend → plateau candidate.

    Using p-value instead of raw tau is statistically principled: the p-value
    accounts for window size, eliminating the need for an arbitrary tau cutoff.

    Returns:
        IMPROVING    p < threshold AND tau > 0  (significant positive trend)
        DECLINING    p < threshold AND tau < 0  (significant negative trend)
        STABILIZED   p >= threshold             (no significant trend)
        INSUFFICIENT fewer than WINDOW_K points in window
    """
    if len(window) < WINDOW_K:
        return DetectorResult(
            name="mann_kendall",
            signal=Signal.INSUFFICIENT,
            confidence=0.0,
            metric=f"n={len(window)}",
            note=f"Need {WINDOW_K} points, have {len(window)}",
        )

    n = len(window)
    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = window[j] - window[i]
            if diff > 0:
                s += 1
            elif diff < 0:
                s -= 1

    # Variance of S under the null hypothesis of no trend
    var_s = n * (n - 1) * (2 * n + 5) / 18

    # Z statistic with continuity correction
    if s > 0:
        z_stat = (s - 1) / math.sqrt(var_s)
    elif s < 0:
        z_stat = (s + 1) / math.sqrt(var_s)
    else:
        z_stat = 0.0

    # Two-tailed p-value: p = erfc(|z| / sqrt(2))
    p_value = math.erfc(abs(z_stat) / math.sqrt(2))

    # Tau still computed — used only for direction and note text
    tau = s / (n * (n - 1) / 2)

    if p_value < MK_P_THRESHOLD:
        if tau > 0:
            signal = Signal.IMPROVING
            note = (f"p={p_value:.3f} < {MK_P_THRESHOLD}: "
                    f"significant positive trend (tau={tau:.2f})")
        else:
            signal = Signal.DECLINING
            note = (f"p={p_value:.3f} < {MK_P_THRESHOLD}: "
                    f"significant negative trend (tau={tau:.2f})")
    else:
        signal = Signal.STABILIZED
        note = (f"p={p_value:.3f} >= {MK_P_THRESHOLD}: "
                f"no significant trend — plateau candidate (tau={tau:.2f})")

    return DetectorResult(
        name="mann_kendall",
        signal=signal,
        confidence=round(1.0 - p_value, 3),
        metric=f"p={p_value:.3f}",
        note=note,
    )



def moving_avg(window: list[float]) -> DetectorResult:
    """Moving average mean-shift detector — Section 3.2.

    S-curve position: active in Activation (P1) alongside bayesian_cp().
    Both must agree before the state machine advances to Growth (P2).
    Also guards the backward path in Growth: a DECLINING signal here
    triggers an immediate REGRESSED decision.

    Splits the window into two equal halves — prior (oldest) and recent
    (newest) — and compares their means. A sustained shift of at least
    MA_SLOPE_THRESHOLD confirms the trajectory is still moving, not spiking.

    Returns:
        IMPROVING    recent mean > prior mean by >= MA_SLOPE_THRESHOLD
        DECLINING    recent mean < prior mean by >= MA_SLOPE_THRESHOLD
        STABILIZED   |shift| < MA_SLOPE_THRESHOLD — means are approximately equal
        INSUFFICIENT fewer than WINDOW_K points in window
    """
    if len(window) < WINDOW_K:
        return DetectorResult(
            name="moving_avg",
            signal=Signal.INSUFFICIENT,
            confidence=0.0,
            metric=f"n={len(window)}",
            note=f"Need {WINDOW_K} points, have {len(window)}",
        )

    half = len(window) // 2
    prior = window[:half]
    recent = window[half:]

    prior_mean = sum(prior) / len(prior)
    recent_mean = sum(recent) / len(recent)
    shift = recent_mean - prior_mean

    confidence = round(min(abs(shift) / MA_SLOPE_THRESHOLD, 1.0), 3)

    if shift >= MA_SLOPE_THRESHOLD:
        signal = Signal.IMPROVING
        note = (f"recent mean {recent_mean:.3f} > prior mean {prior_mean:.3f} "
                f"(shift={shift:+.3f} >= threshold {MA_SLOPE_THRESHOLD})")
    elif shift <= -MA_SLOPE_THRESHOLD:
        signal = Signal.DECLINING
        note = (f"recent mean {recent_mean:.3f} < prior mean {prior_mean:.3f} "
                f"(shift={shift:+.3f} <= -{MA_SLOPE_THRESHOLD})")
    else:
        signal = Signal.STABILIZED
        note = (f"|shift| {abs(shift):.3f} < threshold {MA_SLOPE_THRESHOLD} "
                f"— means approximately equal (prior={prior_mean:.3f}, recent={recent_mean:.3f})")

    return DetectorResult(
        name="moving_avg",
        signal=signal,
        confidence=confidence,
        metric=f"shift={shift:+.3f}",
        note=note,
    )