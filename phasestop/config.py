"""
PhaseStop configuration — enums, dataclasses, and hyperparameters.
Build stages C1–C4 (see CLAUDE.md).
"""

from dataclasses import dataclass
from enum import Enum

# ---------------------------------------------------------------------------
# Hyperparameters — all tuning knobs live here; never hardcode elsewhere
# ---------------------------------------------------------------------------

WINDOW_K: int = 6
"""Sliding window length passed to every detector.

Minimum 5 points required for Mann-Kendall statistical power; 6 adds one buffer point.
Too small: detectors react to noise. Too large: phase transitions are detected late.
"""

LR_P_THRESHOLD: float = 0.10
"""P-value cutoff for the linear regression slope test (Section 3.5).

When p > threshold the slope is statistically indistinguishable from zero — saturation confirmed.
Set to 0.10 rather than the conventional 0.05 because gradual plateaus produce small but
non-negligible slopes that 0.05 would incorrectly reject as significant.
"""

EWMA_ALPHA: float = 0.3
"""Smoothing factor for the exponentially weighted moving average (Roberts, 1959).

Each update: ewma = alpha * latest + (1 - alpha) * ewma_prev.
At 0.3 the newest score contributes 30% and accumulated history 70%.
Too high (→ 1.0): no smoothing, EWMA tracks raw scores. Too low (→ 0.0): EWMA barely moves.
"""

ROLLBACK_MARGIN: float = 0.05
"""Composite drop below best_composite that triggers an immediate REGRESSED decision.

A 5% fall is large enough to signal a real regression yet tolerates normal run-to-run noise.
Too tight (e.g. 0.01): noise causes false regressions. Too loose (e.g. 0.20): real collapses ignored.
"""

QUALITY_FLOOR: float = 0.75
"""Minimum composite score required before the state machine is entered (Section 3.5).

Below 0.75 the system is still in early exploration; trajectory analysis is not yet meaningful.
Calibrated from RAGAS production-grade RAG baselines on BEIR benchmarks.
"""

MK_TAU_THRESHOLD: float = 0.3
"""Kendall tau boundary that separates an improving trend from a fading one.

tau > threshold  → IMPROVING (roughly 65% of pairs are concordant).
|tau| <= threshold → STABILIZED (ordering too weak to call a trend).
tau < -threshold → DECLINING.
Set to 0.3 rather than the commonly cited 0.2 so that noisy plateau sequences
(small alternating dips) are classified as STABILIZED rather than DECLINING.
"""

STORAGE_FORMAT: str = "json"
STORAGE_PATH: str = "results/run_history.json"




class Signal(Enum):
    """What a single detector reports about the trajectory it examined."""

    IMPROVING    = "IMPROVING"
    STABILIZED   = "STABILIZED"
    DECLINING    = "DECLINING"
    INSUFFICIENT = "INSUFFICIENT"


class Decision(Enum):
    """What the PhaseStop state machine returns to the caller after each run."""

    ITERATE    = "ITERATE"
    STABILIZED = "STABILIZED"
    REGRESSED  = "REGRESSED"


@dataclass
class DetectorResult:
    """The full output of one detector for one window evaluation."""

    name:       str
    signal:     Signal
    confidence: float
    metric:     str
    note:       str





# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RunResult:
    """Everything recorded for one call to add_run() — stored as one row on disk."""

    run_id:           int
    timestamp:        str
    composite:        float
    phase_state:      str
    detector_results: list[DetectorResult]
    decision:         Decision
    notes:            list[str]
