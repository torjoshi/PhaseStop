"""
PhaseStop — Synthetic trajectory generator.
Build stages Y1–Y3 (see CLAUDE.md, Section 5).

Generates composite score trajectories for testing and paper figures.
Each trajectory type encodes a distinct pattern that exercises a specific
part of the PhaseStop state machine.
"""

import math
import random
from dataclasses import dataclass


@dataclass
class Trajectory:
    """A synthetic composite score trajectory with known phase structure.

    Attributes:
        composites:        Composite score at each run, oldest first, newest last.
        phase_boundaries:  Approximate run numbers (1-indexed) marking phase ends.
                           {"activation_end": int, "growth_end": int}
        trajectory_type:   Human-readable label for the trajectory pattern.
    """
    composites: list[float]
    phase_boundaries: dict[str, int]
    trajectory_type: str


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sigmoid(x: float, steepness: float, midpoint: float) -> float:
    """Standard sigmoid (logistic) function — maps any real to (0, 1).

    Clamped to avoid math.exp overflow on extreme inputs.
    """
    exponent = -steepness * (x - midpoint)
    exponent = max(-500.0, min(500.0, exponent))
    return 1.0 / (1.0 + math.exp(exponent))


def _scurve_scores(
    n_runs: int,
    floor_score: float,
    ceiling_score: float,
    steepness: float,
    midpoint_frac: float,
) -> list[float]:
    """Shared sigmoid scaffold used by clean_scurve and noisy_scurve.

    Produces scores in [floor_score, ceiling_score] following a sigmoid
    curve whose steepest point is at (midpoint_frac * n_runs).
    """
    midpoint = midpoint_frac * n_runs
    scores: list[float] = []
    for i in range(n_runs):
        t = i + 1  # 1-indexed to align with run_id
        raw = _sigmoid(t, steepness, midpoint)
        score = floor_score + (ceiling_score - floor_score) * raw
        scores.append(round(score, 4))
    return scores


# ---------------------------------------------------------------------------
# Trajectory generator — Y1 base class
# ---------------------------------------------------------------------------

class TrajectoryGenerator:
    """Generates synthetic composite score trajectories — Section 5.

    All trajectory types are deterministic given the same seed and parameters.
    Trajectory types that add Gaussian noise (noisy_scurve, oscillating, etc.)
    consume the seed via random.Random for reproducibility.

    Usage:
        gen = TrajectoryGenerator(seed=42)
        traj = gen.clean_scurve(n_runs=20)
        print(traj.composites)
    """

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def clean_scurve(self, n_runs: int = 20) -> Trajectory:
        """Clean sigmoid S-curve with no noise — Section 5.1.

        Designed to drive the state machine through all three phases:
          Activation  first ~30% of runs — gentle rise from floor
          Growth      middle ~40% of runs — steep sigmoid rise
          Saturation  final ~30% of runs — plateau near ceiling

        Args:
            n_runs: Number of simulated runs to generate (default 20).

        Returns:
            Trajectory with composites and approximate phase boundaries.
        """
        scores = _scurve_scores(
            n_runs=n_runs,
            floor_score=0.76,
            ceiling_score=0.92,
            steepness=0.5,
            midpoint_frac=0.40,
        )

        # Approximate phase boundary run numbers derived from sigmoid shape.
        # T3 (test_synthetic.py) verifies these align with actual state machine
        # transitions when the trajectory is fed into PhaseStop.
        activation_end = max(1, round(n_runs * 0.30))
        growth_end = max(activation_end + 1, round(n_runs * 0.70))

        return Trajectory(
            composites=scores,
            phase_boundaries={
                "activation_end": activation_end,
                "growth_end": growth_end,
            },
            trajectory_type="clean_scurve",
        )
