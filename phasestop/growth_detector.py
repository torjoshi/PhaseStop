"""
PhaseStop — Growth phase detector (P2 → P3 boundary).
Implements mann_kendall() and moving_avg() — build stages D1, D2 (see CLAUDE.md).

S-curve position: active during Growth (P2). Detects when the upward trend
is fading and the trajectory is approaching Saturation (P3). Also guards the
backward path: if growth resumes from SATURATION_CHECK, the state reverts to P2.
"""

import math

from phasestop.config import (
    DetectorResult,
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

# moving_avg() — added in D2