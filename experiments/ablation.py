"""
Ablation study — contribution of each detector to PhaseStop's decisions.

Each variant disables one detector by forcing it to return the signal that
keeps the state machine advancing, as if the detector were absent. All other
logic (Layer 0 gates, state transitions, conjunction rules) is identical to
the full system.

Variants:
  full       — baseline: all five detectors active
  no_bcp     — BCP forced IMPROVING in ACTIVATION; GROWTH requires MA alone
  no_ma_act  — MA forced IMPROVING in ACTIVATION; GROWTH requires BCP alone
  no_mk      — MK forced STABILIZED in GROWTH; SATURATION_CHECK entered immediately
  no_lr      — LR forced STABILIZED in SATURATION_CHECK; STABILIZED requires EWMA alone
  no_ewma    — EWMA forced STABILIZED in SATURATION_CHECK; STABILIZED requires LR alone

Run:
    python -m experiments.ablation
"""

from dataclasses import dataclass

from phasestop.activation_detector import bayesian_cp
from phasestop.config import (
    Decision,
    QUALITY_FLOOR,
    ROLLBACK_MARGIN,
    Signal,
    WINDOW_K,
)
from phasestop.growth_detector import mann_kendall, moving_avg
from phasestop.saturation_detector import ewma_stable, linear_reg
from tools.synthetic import TrajectoryGenerator

VARIANTS: list[str] = ["full", "no_bcp", "no_ma_act", "no_mk", "no_lr", "no_ewma"]

VARIANT_LABELS: dict[str, str] = {
    "full":      "Full",
    "no_bcp":    "No-BCP",
    "no_ma_act": "No-MA(act)",
    "no_mk":     "No-MK",
    "no_lr":     "No-LR",
    "no_ewma":   "No-EWMA",
}


@dataclass
class AblationResult:
    """Stopping outcome for one variant on one trajectory."""
    variant: str
    trajectory_type: str
    stopping_run: int | None
    stopping_decision: str          # STABILIZED, REGRESSED, or ITERATE (no stop)
    composite_at_stop: float | None


def _run_variant(composites: list[float], variant: str) -> tuple[int | None, str, float | None]:
    """Run the ablated state machine and return (stopping_run, decision_str, composite)."""
    state = "ACTIVATION"
    history: list[float] = []
    best: float = 0.0
    stopping_run: int | None = None
    stopping_decision: Decision | None = None

    for i, c in enumerate(composites):
        run_id = i + 1

        # Layer 0: identical across all variants
        if c < QUALITY_FLOOR:
            continue
        if c < best - ROLLBACK_MARGIN:
            if stopping_run is None:
                stopping_run = run_id
                stopping_decision = Decision.REGRESSED
            continue

        if c > best:
            best = c
        history.append(c)
        window = history[-WINDOW_K:]

        decision = Decision.ITERATE

        if state == "ACTIVATION":
            # Disabled detector is treated as IMPROVING (transparent pass-through)
            bcp_sig = (
                Signal.IMPROVING if variant == "no_bcp"
                else bayesian_cp(window).signal
            )
            ma_sig = (
                Signal.IMPROVING if variant == "no_ma_act"
                else moving_avg(window).signal
            )
            if bcp_sig == Signal.IMPROVING and ma_sig == Signal.IMPROVING:
                state = "GROWTH"

        elif state == "GROWTH":
            # Disabled MK is treated as STABILIZED → immediate entry to SATURATION_CHECK
            mk_sig = (
                Signal.STABILIZED if variant == "no_mk"
                else mann_kendall(window).signal
            )
            # MA always runs in GROWTH as the regression guard (never ablated here)
            ma_sig = moving_avg(window).signal
            if mk_sig == Signal.STABILIZED:
                state = "SATURATION_CHECK"
            elif ma_sig == Signal.DECLINING:
                decision = Decision.REGRESSED

        elif state == "SATURATION_CHECK":
            # Disabled detector is treated as STABILIZED (one side of conjunction always met)
            lr_sig = (
                Signal.STABILIZED if variant == "no_lr"
                else linear_reg(window).signal
            )
            ewma_sig = (
                Signal.STABILIZED if variant == "no_ewma"
                else ewma_stable(window).signal
            )
            # MK backward regression always runs in SATURATION_CHECK
            mk_sig = mann_kendall(window).signal
            if lr_sig == Signal.STABILIZED and ewma_sig == Signal.STABILIZED:
                decision = Decision.STABILIZED
            elif mk_sig == Signal.IMPROVING:
                state = "GROWTH"

        if decision in (Decision.STABILIZED, Decision.REGRESSED) and stopping_run is None:
            stopping_run = run_id
            stopping_decision = decision

    dec_str = stopping_decision.value if stopping_decision is not None else "ITERATE"
    comp = (
        round(composites[stopping_run - 1], 4)
        if stopping_run is not None
        else None
    )
    return stopping_run, dec_str, comp


def run_ablation(seed: int = 42) -> list[AblationResult]:
    """Run all variants against all seven trajectory types."""
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

    results: list[AblationResult] = []
    for traj in trajectories:
        for variant in VARIANTS:
            stop, dec, comp = _run_variant(traj.composites, variant)
            results.append(AblationResult(
                variant=variant,
                trajectory_type=traj.trajectory_type,
                stopping_run=stop,
                stopping_decision=dec,
                composite_at_stop=comp,
            ))
    return results


def print_ablation(results: list[AblationResult]) -> None:
    """Print two matrices: stopping run and composite quality, variant × trajectory."""
    # Collect unique trajectory types in insertion order
    traj_types = list(dict.fromkeys(r.trajectory_type for r in results))

    # Short labels for column headers
    short = {
        "clean_scurve":              "clean",
        "noisy_scurve":              "noisy",
        "oscillating":               "osc",
        "premature_saturation":      "premature",
        "regression_post_saturation":"regress",
        "fast_convergence":          "fast",
        "slow_convergence":          "slow",
    }

    # Index results: (variant, trajectory_type) → AblationResult
    idx: dict[tuple[str, str], AblationResult] = {
        (r.variant, r.trajectory_type): r for r in results
    }

    col_w = 12  # width per trajectory column
    label_w = 12

    def _row(label: str, values: list[str]) -> str:
        return f"{label:<{label_w}}" + "".join(f"{v:>{col_w}}" for v in values)

    headers = [short.get(t, t[:col_w]) for t in traj_types]

    print("Stopping run  (— = never stopped,  * = stopped on oscillating = false positive)\n")
    print(_row("Variant", headers))
    print("-" * (label_w + col_w * len(traj_types)))
    for variant in VARIANTS:
        values = []
        for traj in traj_types:
            r = idx[(variant, traj)]
            if r.stopping_run is None:
                cell = "—"
            else:
                cell = str(r.stopping_run)
                if traj == "oscillating":
                    cell += "*"  # flag false positive
            values.append(cell)
        print(_row(VARIANT_LABELS[variant], values))

    print()
    print("Composite at stop  (— = never stopped)\n")
    print(_row("Variant", headers))
    print("-" * (label_w + col_w * len(traj_types)))
    for variant in VARIANTS:
        values = []
        for traj in traj_types:
            r = idx[(variant, traj)]
            cell = f"{r.composite_at_stop:.4f}" if r.composite_at_stop is not None else "—"
            values.append(cell)
        print(_row(VARIANT_LABELS[variant], values))


if __name__ == "__main__":
    results = run_ablation()
    print_ablation(results)
