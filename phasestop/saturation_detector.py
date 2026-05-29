"""
PhaseStop — Saturation phase detector (P3).
Implements ewma_stable() and linear_reg() — build stages D3, D4 (see CLAUDE.md).

S-curve position: active in Saturation (P3). Both detectors must agree
(conjunction, not vote) before STABILIZED is returned. ewma_stable() confirms
the smoothed value has stopped moving; linear_reg() confirms the OLS slope
is statistically indistinguishable from zero.
"""

from phasestop.config import (
    DetectorResult,
    EWMA_ALPHA,
    LR_P_THRESHOLD,
    Signal,
    WINDOW_K,
)

# ewma_stable() — added in D3
# linear_reg()  — added in D4