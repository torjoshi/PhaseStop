"""
PhaseStop detectors — domain-blind trajectory analysis functions.
Each function accepts a window of floats (oldest first, newest last)
and returns a DetectorResult. No detector knows what the numbers mean.
Build stages D1–D5 (see CLAUDE.md).
"""

from phasestop.config import (
    DetectorResult,
    MK_TAU_THRESHOLD,
    Signal,
    WINDOW_K,
)


def mann_kendall(window: list[float]) -> DetectorResult:
    """Kendall rank-correlation trend test — Section 3.3 of the paper.

    For every pair of points (i, j) where i < j, scores +1 if the later
    point is higher, -1 if lower, 0 if equal. Sums all scores to get S,
    then normalises by the total number of pairs to produce Kendall tau.

    Used in the GROWTH phase to detect when the monotonic upward trend
    is fading — the signal that saturation is approaching.

    Returns:
        IMPROVING    tau >  MK_TAU_THRESHOLD  (trend sustaining)
        STABILIZED   |tau| <= MK_TAU_THRESHOLD (trend fading)
        DECLINING    tau < -MK_TAU_THRESHOLD  (negative trend)
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

    num_pairs = n * (n - 1) / 2
    tau = s / num_pairs

    if tau > MK_TAU_THRESHOLD:
        signal = Signal.IMPROVING
        note = f"Kendall tau={tau:.2f} indicates positive monotonic trend"
    elif tau < -MK_TAU_THRESHOLD:
        signal = Signal.DECLINING
        note = f"Kendall tau={tau:.2f} indicates negative monotonic trend"
    else:
        signal = Signal.STABILIZED
        note = f"Kendall tau={tau:.2f} — trend fading, saturation likely"

    return DetectorResult(
        name="mann_kendall",
        signal=signal,
        confidence=abs(tau),
        metric=f"tau={tau:.2f}",
        note=note,
    )