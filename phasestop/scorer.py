"""
PhaseStop — State machine and run orchestrator.
Build stages S1–S4 (see CLAUDE.md).
"""

import csv
import dataclasses
import datetime
import json
import os
from enum import Enum

from phasestop.activation_detector import bayesian_cp
from phasestop.config import (
    Decision,
    DetectorResult,
    QUALITY_FLOOR,
    ROLLBACK_MARGIN,
    RunResult,
    Signal,
    STORAGE_FORMAT,
    STORAGE_PATH,
    WINDOW_K,
)
from phasestop.growth_detector import mann_kendall, moving_avg
from phasestop.saturation_detector import ewma_stable, linear_reg

_ACTIVATION = "ACTIVATION"
_GROWTH = "GROWTH"
_SATURATION_CHECK = "SATURATION_CHECK"


class _EnumEncoder(json.JSONEncoder):
    """Serialises Enum values to their string value for JSON storage."""
    def default(self, obj: object) -> object:
        if isinstance(obj, Enum):
            return obj.value
        return super().default(obj)


class PhaseStop:
    """Phase-conditioned detector orchestrator — Section 3.

    Receives one composite float per run via add_run(). Maintains a
    sliding window of recent scores, activates the right detectors for
    the current phase state, and returns a Decision each run.

    Usage:
        ps = PhaseStop()
        decision = ps.add_run(run_id=1, composite=0.812)
    """

    def __init__(
        self,
        storage_path: str = STORAGE_PATH,
        storage_format: str = STORAGE_FORMAT,
    ) -> None:
        self.state: str = _ACTIVATION
        self.history: list[float] = []
        self.best_composite: float = 0.0
        self._storage_path = storage_path
        self._storage_format = storage_format

    def __repr__(self) -> str:
        return (
            f"PhaseStop(state={self.state}, "
            f"runs={len(self.history)}, "
            f"best={self.best_composite:.3f})"
        )

    def add_run(self, run_id: int, composite: float) -> Decision:
        """Process one run and return a Decision — Section 3.6.

        Applies Layer 0 composite gates first, then the phase state machine.
        Every call appends one record to the configured storage file.
        """
        notes: list[str] = []
        detector_results: list[DetectorResult] = []

        # --- Layer 0: quality floor -------------------------------------------
        # Composite below floor means the system is still in early exploration.
        # Return ITERATE without touching history or the state machine.
        if composite < QUALITY_FLOOR:
            notes.append(
                f"composite {composite:.3f} below quality floor {QUALITY_FLOOR}"
            )
            self._store(RunResult(
                run_id=run_id,
                timestamp=_now(),
                composite=composite,
                phase_state=self.state,
                detector_results=[],
                decision=Decision.ITERATE,
                notes=notes,
            ))
            return Decision.ITERATE

        # --- Layer 0: rollback margin -----------------------------------------
        # A drop of more than ROLLBACK_MARGIN below the best composite seen
        # signals a real regression, not just noise.
        if composite < self.best_composite - ROLLBACK_MARGIN:
            notes.append(
                f"composite {composite:.3f} < best {self.best_composite:.3f} "
                f"- margin {ROLLBACK_MARGIN} = "
                f"{self.best_composite - ROLLBACK_MARGIN:.3f}"
            )
            self._store(RunResult(
                run_id=run_id,
                timestamp=_now(),
                composite=composite,
                phase_state=self.state,
                detector_results=[],
                decision=Decision.REGRESSED,
                notes=notes,
            ))
            return Decision.REGRESSED

        # Update tracking state now that Layer 0 passed.
        if composite > self.best_composite:
            self.best_composite = composite
        self.history.append(composite)
        window = self.history[-WINDOW_K:]

        # Capture the phase this run was evaluated in BEFORE _advance() may
        # mutate self.state. The stored record reflects what the run saw, not
        # the state it transitioned to.
        phase_at_evaluation = self.state

        # --- Layer 1: state machine -------------------------------------------
        decision = self._advance(window, detector_results, notes)

        self._store(RunResult(
            run_id=run_id,
            timestamp=_now(),
            composite=composite,
            phase_state=phase_at_evaluation,
            detector_results=detector_results,
            decision=decision,
            notes=notes,
        ))
        return decision

    # -------------------------------------------------------------------------
    # Internal helpers
    # -------------------------------------------------------------------------

    def _advance(
        self,
        window: list[float],
        detector_results: list[DetectorResult],
        notes: list[str],
    ) -> Decision:
        """Run the phase-conditioned detectors and advance state if warranted."""

        if self.state == _ACTIVATION:
            bcp = bayesian_cp(window)
            ma = moving_avg(window)
            detector_results.extend([bcp, ma])
            if bcp.signal == Signal.IMPROVING and ma.signal == Signal.IMPROVING:
                self.state = _GROWTH
                notes.append("BCP + MA both IMPROVING → ACTIVATION → GROWTH")
            return Decision.ITERATE

        if self.state == _GROWTH:
            mk = mann_kendall(window)
            ma = moving_avg(window)
            detector_results.extend([mk, ma])
            if mk.signal == Signal.STABILIZED:
                self.state = _SATURATION_CHECK
                notes.append("MK trend fading → GROWTH → SATURATION_CHECK")
            elif ma.signal == Signal.DECLINING:
                notes.append("MA declining in GROWTH → REGRESSED")
                return Decision.REGRESSED
            return Decision.ITERATE

        if self.state == _SATURATION_CHECK:
            lr = linear_reg(window)
            ewma = ewma_stable(window)
            mk = mann_kendall(window)
            detector_results.extend([lr, ewma, mk])
            if lr.signal == Signal.STABILIZED and ewma.signal == Signal.STABILIZED:
                notes.append("LR + EWMA both STABILIZED → STABILIZED")
                return Decision.STABILIZED
            if mk.signal == Signal.IMPROVING:
                self.state = _GROWTH
                notes.append("MK strongly increasing → SATURATION_CHECK → GROWTH")
            return Decision.ITERATE

        return Decision.ITERATE

    def _store(self, result: RunResult) -> None:
        """Append one RunResult to disk — Section 3.7. Never rewrites past records."""
        dir_name = os.path.dirname(self._storage_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        if self._storage_format == "json":
            record = dataclasses.asdict(result)
            with open(self._storage_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, cls=_EnumEncoder) + "\n")

        elif self._storage_format == "csv":
            fieldnames = [
                "run_id", "timestamp", "composite",
                "phase_state", "decision", "notes",
            ]
            file_exists = os.path.exists(self._storage_path)
            with open(self._storage_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({
                    "run_id": result.run_id,
                    "timestamp": result.timestamp,
                    "composite": result.composite,
                    "phase_state": result.phase_state,
                    "decision": result.decision.value,
                    "notes": "; ".join(result.notes),
                })


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")
