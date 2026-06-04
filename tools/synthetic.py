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
        """Clean S-curve: sigmoid rise then hard plateau — Section 5.1.

        A pure sigmoid asymptotically approaches ceiling but never truly
        plateaus within a practical run count, so MK always sees a small
        positive trend and GROWTH never transitions to SATURATION_CHECK.
        The hard tail (_TAIL flat runs at 0.92) gives LR and EWMA a clean
        window of zero-slope data to confirm saturation.

        Args:
            n_runs: Total simulated runs including the hard tail (default 20).

        Returns:
            Trajectory with composites and approximate phase boundaries.
        """
        _TAIL = 8  # must be > WINDOW_K so LR+EWMA see a fully flat window
        rise_runs = n_runs - _TAIL
        rise = _scurve_scores(
            n_runs=rise_runs,
            floor_score=0.76,
            ceiling_score=0.92,
            steepness=0.5,
            midpoint_frac=0.40,
        )
        scores = rise + [0.92] * _TAIL

        activation_end = max(1, round(rise_runs * 0.30))
        growth_end = rise_runs  # plateau begins immediately after sigmoid rise

        return Trajectory(
            composites=scores,
            phase_boundaries={
                "activation_end": activation_end,
                "growth_end": growth_end,
            },
            trajectory_type="clean_scurve",
        )

    # -------------------------------------------------------------------------
    # Y2 trajectory types
    # -------------------------------------------------------------------------

    def noisy_scurve(self, n_runs: int = 20, noise_std: float = 0.01) -> Trajectory:
        """Sigmoid S-curve with Gaussian noise added at every run — Section 5.2.

        Same underlying shape as clean_scurve but each score is perturbed by
        gauss(0, noise_std). Scores are clamped to [floor, ceiling] after noise
        so values stay realistic. Tests whether PhaseStop detects real phase
        patterns through measurement noise.

        Args:
            n_runs:     Number of simulated runs to generate (default 20).
            noise_std:  Standard deviation of Gaussian noise (default 0.01).
        """
        rng = random.Random(self._seed)
        base = _scurve_scores(
            n_runs=n_runs,
            floor_score=0.76,
            ceiling_score=0.92,
            steepness=0.5,
            midpoint_frac=0.40,
        )
        composites = [
            round(max(0.76, min(0.92, s + rng.gauss(0, noise_std))), 4)
            for s in base
        ]
        activation_end = max(1, round(n_runs * 0.30))
        growth_end = max(activation_end + 1, round(n_runs * 0.70))
        return Trajectory(
            composites=composites,
            phase_boundaries={"activation_end": activation_end, "growth_end": growth_end},
            trajectory_type="noisy_scurve",
        )

    def oscillating(self, n_runs: int = 20) -> Trajectory:
        """Scores that alternate between 0.84 and 0.80 on every run — Section 5.3.

        The 0.04 swing is deliberately within the rollback margin (0.05), so
        Layer 0 never fires REGRESSED. BCP cannot confirm a regime shift because
        the baseline half and recent half of any window have the same mean.
        The state machine stays in ACTIVATION for the entire trajectory.

        No phase_boundaries — no phase transitions are expected.
        """
        composites = [0.84 if i % 2 == 0 else 0.80 for i in range(n_runs)]
        return Trajectory(
            composites=composites,
            phase_boundaries={},
            trajectory_type="oscillating",
        )

    def premature_saturation(self, n_runs: int = 24) -> Trajectory:
        """Two-plateau trajectory that exercises backward regression — Section 5.4.

        Structure (4 equal segments of n_runs // 4 runs each):
            Segment 1  rise from 0.76 to 0.83  (activation + early growth)
            Segment 2  flat at 0.83             (premature plateau — triggers SATURATION_CHECK)
            Segment 3  rise from 0.83 to 0.92  (resumed growth — MK IMPROVING → back to GROWTH)
            Segment 4  flat at 0.92             (real plateau — LR + EWMA → STABILIZED)

        Exercises: ACTIVATION → GROWTH → SATURATION_CHECK → GROWTH → SATURATION_CHECK → STABILIZED.
        Any remainder runs after the 4 equal segments are appended to segment 4.
        """
        seg = n_runs // 4
        composites = []

        for i in range(n_runs):
            t = i + 1
            if t <= seg:
                # Segment 1: rise from 0.76 to 0.83
                raw = _sigmoid(t, 0.8, seg * 0.6)
                score = 0.76 + 0.07 * raw
            elif t <= seg * 2:
                # Segment 2: flat plateau at 0.83
                score = 0.83
            elif t <= seg * 3:
                # Segment 3: rise from 0.83 to 0.92
                local_t = t - seg * 2
                raw = _sigmoid(local_t, 1.0, seg * 0.5)
                score = 0.83 + 0.09 * raw
            else:
                # Segment 4: flat plateau at 0.92 (includes any remainder)
                score = 0.92
            composites.append(round(score, 4))

        return Trajectory(
            composites=composites,
            phase_boundaries={
                "activation_end": seg,
                "growth_end": seg * 3,
            },
            trajectory_type="premature_saturation",
        )

    # -------------------------------------------------------------------------
    # Y3 trajectory types
    # -------------------------------------------------------------------------

    def regression_post_saturation(self, n_runs: int = 28) -> Trajectory:
        """S-curve that drops sharply after reaching stable saturation — Section 5.5.

        Structure (fixed segment sizes, n_runs=28 default):
            Runs 1..16    clean sigmoid rise from 0.76 to 0.92
            Runs 17..24   hard plateau at 0.92 — 8 flat runs guarantee LR+EWMA confirm
                          saturation (STABILIZED fires here)
            Runs 25..28   drop to 0.84 — below 0.92 - 0.05 = 0.87 (REGRESSED)

        Plateau must be >= WINDOW_K + 2 so the window is fully flat when STABILIZED
        fires. With 4 plateau runs (one full WINDOW_K) the sigmoid tail bleeds in and
        LR/EWMA never both agree — STABILIZED never fires before the drop.

        Tests: Layer 0 rollback margin fires even after STABILIZED has been returned,
        catching post-saturation degradation (overfitting, data drift, regression).
        """
        _RISE = 16
        _PLATEAU = 8   # > WINDOW_K: ensures window is fully flat before drop
        _DROP_VAL = 0.84  # 0.84 < 0.92 - 0.05 = 0.87 → triggers REGRESSED
        _DROP = n_runs - _RISE - _PLATEAU

        rise = _scurve_scores(
            n_runs=_RISE,
            floor_score=0.76,
            ceiling_score=0.92,
            steepness=0.5,
            midpoint_frac=0.40,
        )
        plateau = [0.92] * _PLATEAU
        drop = [_DROP_VAL] * _DROP

        return Trajectory(
            composites=rise + plateau + drop,
            phase_boundaries={
                "activation_end": max(1, round(_RISE * 0.30)),
                "growth_end": _RISE,
                "regression_start": _RISE + _PLATEAU + 1,
            },
            trajectory_type="regression_post_saturation",
        )

    def fast_convergence(self, n_runs: int = 15) -> Trajectory:
        """S-curve that reaches saturation very quickly — Section 5.6.

        High steepness (1.5) with early midpoint (25% of runs). Scores shoot
        up in the first few runs and plateau well before the halfway point.
        Tests whether WINDOW_K=6 is sufficient to detect regime shift and
        confirm saturation with minimal run history.
        """
        scores = _scurve_scores(
            n_runs=n_runs,
            floor_score=0.76,
            ceiling_score=0.92,
            steepness=1.5,
            midpoint_frac=0.25,
        )
        activation_end = max(1, round(n_runs * 0.20))
        growth_end = max(activation_end + 1, round(n_runs * 0.45))
        return Trajectory(
            composites=scores,
            phase_boundaries={"activation_end": activation_end, "growth_end": growth_end},
            trajectory_type="fast_convergence",
        )

    def slow_convergence(self, n_runs: int = 38) -> Trajectory:
        """S-curve that rises very gradually, with a hard plateau at the end — Section 5.7.

        Structure: sigmoid rise over (n_runs - 10) runs at low steepness (0.3),
        followed by 10 hard plateau runs at 0.92. The hard plateau is necessary
        because a low-steepness pure sigmoid approaches the ceiling so slowly
        that LR/EWMA never see a statistically flat window within reasonable run counts.

        Early runs (1..~12) are nearly flat just above the quality floor — the state
        machine must wait in ACTIVATION until the window spans the genuine rise.
        Counterpart to fast_convergence: together they bracket WINDOW_K=6 sensitivity.
        """
        _TAIL = 10  # hard plateau runs appended to the sigmoid
        rise_runs = n_runs - _TAIL
        rise = _scurve_scores(
            n_runs=rise_runs,
            floor_score=0.76,
            ceiling_score=0.92,
            steepness=0.3,
            midpoint_frac=0.60,
        )
        scores = rise + [0.92] * _TAIL

        activation_end = max(1, round(rise_runs * 0.40))
        growth_end = max(activation_end + 1, rise_runs)
        return Trajectory(
            composites=scores,
            phase_boundaries={"activation_end": activation_end, "growth_end": growth_end},
            trajectory_type="slow_convergence",
        )
