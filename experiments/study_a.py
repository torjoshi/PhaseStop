"""
Study A — Stopping accuracy across all seven synthetic trajectory types.

Feeds each trajectory into a fresh PhaseStop instance (no disk writes) and
measures: stopping run, terminal decision, stop error vs. ground-truth
growth_end, and whether REGRESSED fires after STABILIZED.

Run:
    python -m experiments.study_a
"""

import os
from dataclasses import dataclass

from phasestop.config import Decision
from phasestop.scorer import PhaseStop
from tools.synthetic import Trajectory, TrajectoryGenerator


@dataclass
class StudyAResult:
    """Per-trajectory result for Study A.

    stop_error is positive when PhaseStop stops late (conservative),
    negative when it stops early (aggressive), None when no terminal fired
    or there is no ground-truth growth_end.
    """
    trajectory_type: str
    n_runs: int
    composites: list[float]
    decisions: list[Decision]
    first_terminal_run: int | None      # 1-indexed run where STABILIZED or REGRESSED first fired
    first_terminal_decision: Decision | None
    regressed_at: int | None            # run where REGRESSED fired if it came after first_terminal
    ground_truth_growth_end: int | None # from trajectory.phase_boundaries["growth_end"]
    stop_error: int | None              # first_terminal_run - ground_truth_growth_end


def _run_trajectory(traj: Trajectory) -> StudyAResult:
    """Feed all composites into a fresh PhaseStop and collect per-run decisions.

    Uses os.devnull as storage_path so no files are written to disk.
    Continues feeding runs even after the first terminal decision fires —
    this lets us detect a second REGRESSED in regression_post_saturation.
    """
    ps = PhaseStop(storage_path=os.devnull, storage_format="json")
    decisions: list[Decision] = []
    first_terminal_run: int | None = None
    first_terminal_decision: Decision | None = None
    regressed_at: int | None = None

    for i, composite in enumerate(traj.composites):
        run_id = i + 1
        decision = ps.add_run(run_id=run_id, composite=composite)
        decisions.append(decision)

        is_terminal = decision in (Decision.STABILIZED, Decision.REGRESSED)

        if first_terminal_run is None and is_terminal:
            first_terminal_run = run_id
            first_terminal_decision = decision

        # Track REGRESSED even after a prior terminal (regression_post_saturation)
        if decision == Decision.REGRESSED and run_id != first_terminal_run:
            if regressed_at is None:
                regressed_at = run_id

    growth_end = traj.phase_boundaries.get("growth_end")
    stop_error = (
        (first_terminal_run - growth_end)
        if first_terminal_run is not None and growth_end is not None
        else None
    )

    return StudyAResult(
        trajectory_type=traj.trajectory_type,
        n_runs=len(traj.composites),
        composites=traj.composites,
        decisions=decisions,
        first_terminal_run=first_terminal_run,
        first_terminal_decision=first_terminal_decision,
        regressed_at=regressed_at,
        ground_truth_growth_end=growth_end,
        stop_error=stop_error,
    )


def run_study_a(seed: int = 42) -> list[StudyAResult]:
    """Run all seven trajectory types and return one StudyAResult per trajectory."""
    gen = TrajectoryGenerator(seed=seed)
    trajectories = [
        gen.clean_scurve(),
        gen.noisy_scurve(),
        gen.oscillating(),
        gen.premature_saturation(),
        gen.regression_post_saturation(),
        gen.fast_convergence(),
        gen.slow_convergence(),
    ]
    return [_run_trajectory(t) for t in trajectories]


def print_study_a(results: list[StudyAResult]) -> None:
    """Print a summary table of Study A results — one row per trajectory."""
    col = "{:<32}  {:>3}  {:>6}  {:<12}  {:>6}  {:>6}  {:>12}"
    header = col.format(
        "Trajectory", "N", "Stop@", "Decision", "GT end", "Error", "Regressed@"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        stop_str = str(r.first_terminal_run) if r.first_terminal_run is not None else "—"
        dec_str  = r.first_terminal_decision.value if r.first_terminal_decision is not None else "ITERATE"
        gt_str   = str(r.ground_truth_growth_end) if r.ground_truth_growth_end is not None else "—"
        err_str  = f"{r.stop_error:+d}" if r.stop_error is not None else "—"
        reg_str  = str(r.regressed_at) if r.regressed_at is not None else "—"
        print(col.format(r.trajectory_type, r.n_runs, stop_str, dec_str, gt_str, err_str, reg_str))


if __name__ == "__main__":
    results = run_study_a()
    print_study_a(results)
