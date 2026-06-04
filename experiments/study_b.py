"""
Study B — PhaseStop vs. naive stopping baselines.

Compares PhaseStop against two baselines on each trajectory:
  FixedBudget(FIXED_BUDGET): stop unconditionally at run FIXED_BUDGET
  Threshold(THRESHOLD):      stop at the first run where composite >= THRESHOLD

Key questions this study answers:
  1. Does the threshold fire during growth (premature stop)?
  2. Does fixed budget stop too early or too late?
  3. Does PhaseStop correctly abstain on oscillating?
  4. Which method captures the highest quality at stop?

Run:
    python -m experiments.study_b
"""

from dataclasses import dataclass

from experiments.study_a import run_study_a

FIXED_BUDGET: int = 15
THRESHOLD: float = 0.90


@dataclass
class StudyBRow:
    """Per-trajectory comparison across PhaseStop and two baselines."""

    trajectory_type: str
    n_runs: int

    # PhaseStop (from Study A)
    ps_stop: int | None         # run where first terminal decision fired
    ps_decision: str            # STABILIZED, REGRESSED, or ITERATE (no stop)
    ps_composite: float | None  # composite at ps_stop

    # Fixed-budget baseline
    fb_stop: int                # always fires at min(FIXED_BUDGET, n_runs)
    fb_composite: float

    # Threshold baseline
    th_stop: int | None         # first run where composite >= THRESHOLD
    th_composite: float | None


def _threshold_stop(
    composites: list[float], threshold: float
) -> tuple[int | None, float | None]:
    """Return (run, composite) for the first run at or above threshold, else (None, None)."""
    for i, c in enumerate(composites):
        if c >= threshold:
            return i + 1, round(c, 4)
    return None, None


def run_study_b(seed: int = 42) -> list[StudyBRow]:
    """Build one StudyBRow per trajectory using Study A results and baseline logic."""
    rows: list[StudyBRow] = []

    for a in run_study_a(seed=seed):
        fb_run = min(FIXED_BUDGET, a.n_runs)
        fb_comp = round(a.composites[fb_run - 1], 4)

        th_stop, th_comp = _threshold_stop(a.composites, THRESHOLD)

        ps_comp = (
            round(a.composites[a.first_terminal_run - 1], 4)
            if a.first_terminal_run is not None
            else None
        )
        ps_decision = (
            a.first_terminal_decision.value
            if a.first_terminal_decision is not None
            else "ITERATE"
        )

        rows.append(StudyBRow(
            trajectory_type=a.trajectory_type,
            n_runs=a.n_runs,
            ps_stop=a.first_terminal_run,
            ps_decision=ps_decision,
            ps_composite=ps_comp,
            fb_stop=fb_run,
            fb_composite=fb_comp,
            th_stop=th_stop,
            th_composite=th_comp,
        ))

    return rows


def print_study_b(rows: list[StudyBRow]) -> None:
    """Print a side-by-side comparison table."""
    print(f"Baselines:  FixedBudget(n={FIXED_BUDGET})   Threshold(t={THRESHOLD})\n")

    col = "{:<32}  {:>3}  {:>5}  {:>8}  {:>5}  {:>8}  {:>5}  {:>8}"
    hdr = col.format(
        "Trajectory", "N",
        "PS@", "PS qual",
        "FB@", "FB qual",
        "TH@", "TH qual",
    )
    print(hdr)
    print("-" * len(hdr))

    for r in rows:
        ps_s = str(r.ps_stop) if r.ps_stop is not None else "—"
        ps_c = f"{r.ps_composite:.4f}" if r.ps_composite is not None else "—"
        th_s = str(r.th_stop) if r.th_stop is not None else "—"
        th_c = f"{r.th_composite:.4f}" if r.th_composite is not None else "—"

        print(col.format(
            r.trajectory_type, r.n_runs,
            ps_s, ps_c,
            str(r.fb_stop), f"{r.fb_composite:.4f}",
            th_s, th_c,
        ))


if __name__ == "__main__":
    rows = run_study_b()
    print_study_b(rows)
