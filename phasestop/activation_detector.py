"""
PhaseStop — Activation phase detector (P1).
Implements bayesian_cp() — build stage D5 (see CLAUDE.md).

S-curve position: fires first. Detects when scores leave their initial
baseline and enter a new distributional regime, confirming the trajectory
has exited Activation and entered Growth (P1 → P2).
"""

from phasestop.config import (
    DetectorResult,
    Signal,
    WINDOW_K,
)

# bayesian_cp() — added in D5