"""
PhaseStop — Retrospective trace analysis utility.

Slides a window of size WINDOW_K across a composite history and runs all five
detectors at every position. Diagnostic only — no state machine, no decisions,
no storage. Use this to see what each detector was signalling throughout a full
trajectory.
"""

from dataclasses import dataclass

from phasestop.activation_detector import bayesian_cp
from phasestop.config import DetectorResult, WINDOW_K
from phasestop.growth_detector import mann_kendall, moving_avg
from phasestop.saturation_detector import ewma_stable, linear_reg


@dataclass
class TracePoint:
    """One window position in a trajectory trace.

    Attributes:
        position:         1-indexed run number the window ends at.
        window:           The WINDOW_K composite scores in this window.
        detector_results: All five detectors run on this window.
    """
    position: int
    window: list[float]
    detector_results: list[DetectorResult]


def trace(history: list[float]) -> list[TracePoint]:
    """Slide a WINDOW_K window across history and run all five detectors — Section 3.

    Returns one TracePoint per valid window position (positions 1 through
    len(history)). For positions before the window fills (fewer than WINDOW_K
    scores available), all detectors return INSUFFICIENT — those points are
    still included so position numbers align with run_ids.

    Args:
        history: Complete list of composite scores, oldest first, newest last.

    Returns:
        List of TracePoint objects, one per position in history.
    """
    points: list[TracePoint] = []

    for i in range(len(history)):
        position = i + 1
        # Use a partial window until WINDOW_K scores are available.
        window = history[max(0, i + 1 - WINDOW_K): i + 1]

        bcp = bayesian_cp(window)
        mk = mann_kendall(window)
        ma = moving_avg(window)
        ewma = ewma_stable(window)
        lr = linear_reg(window)

        points.append(TracePoint(
            position=position,
            window=list(window),
            detector_results=[bcp, mk, ma, ewma, lr],
        ))

    return points


def print_trace(points: list[TracePoint]) -> None:
    """Print a formatted table of all detector signals across a trace.

    Columns: position, composite (last value in window), then one column
    per detector showing signal + confidence.
    """
    _DETECTOR_ORDER = ["bayesian_cp", "mann_kendall", "moving_avg", "ewma_stable", "linear_reg"]
    _ABBREV = {
        "bayesian_cp":  "BCP",
        "mann_kendall": "MK ",
        "moving_avg":   "MA ",
        "ewma_stable":  "EWMA",
        "linear_reg":   "LR  ",
    }
    _SIG_ABBREV = {
        "IMPROVING":    "IMP ",
        "STABILIZED":   "STAB",
        "DECLINING":    "DECL",
        "INSUFFICIENT": "INSF",
    }

    header = f"{'pos':>3}  {'composite':>9}  " + "  ".join(
        f"{_ABBREV[d]:>4}" for d in _DETECTOR_ORDER
    )
    print(header)
    print("-" * len(header))

    for pt in points:
        composite = pt.window[-1] if pt.window else float("nan")
        sig_by_name = {dr.name: dr.signal.value for dr in pt.detector_results}
        cells = "  ".join(
            f"{_SIG_ABBREV.get(sig_by_name.get(d, 'INSUFFICIENT'), '????'):>4}"
            for d in _DETECTOR_ORDER
        )
        print(f"{pt.position:>3}  {composite:>9.4f}  {cells}")
