"""
Tests for synthetic trajectory generation — build stage T3.

Each test feeds a trajectory into a real PhaseStop instance and asserts
the state machine produces the expected decision pattern. Ground truth
was verified empirically by running all trajectories before writing
assertions.

Key n_runs notes:
  clean_scurve n=35: pure sigmoid needs 35 runs for the tail to flatten enough
                     for LR + EWMA to both confirm saturation.
  All other trajectories use their default n_runs.
"""

import pytest

from phasestop.config import Decision
from phasestop.scorer import PhaseStop
from tools.synthetic import TrajectoryGenerator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture
def gen() -> TrajectoryGenerator:
    return TrajectoryGenerator(seed=42)


def _run(traj, tmp_path):
    """Feed a trajectory into a fresh PhaseStop and return (scorer, decisions)."""
    ps = PhaseStop(storage_path=str(tmp_path / "h.json"))
    decisions = [ps.add_run(i + 1, c) for i, c in enumerate(traj.composites)]
    return ps, decisions


# ---------------------------------------------------------------------------
# clean_scurve — pure sigmoid, n_runs=35 required for STABILIZED
# ---------------------------------------------------------------------------

def test_clean_scurve_reaches_stabilized(gen, tmp_path):
    """n_runs=35 ensures the sigmoid tail is flat enough for LR+EWMA confirmation."""
    traj = gen.clean_scurve(n_runs=35)
    _, decisions = _run(traj, tmp_path)
    assert Decision.STABILIZED in decisions


def test_clean_scurve_no_regressed(gen, tmp_path):
    """A monotonically rising S-curve must never trigger the rollback margin."""
    traj = gen.clean_scurve(n_runs=35)
    _, decisions = _run(traj, tmp_path)
    assert Decision.REGRESSED not in decisions


def test_clean_scurve_advances_to_growth(gen, tmp_path):
    """n_runs=20 confirms the state machine transitions out of ACTIVATION."""
    traj = gen.clean_scurve(n_runs=20)
    ps, _ = _run(traj, tmp_path)
    assert ps.state in ("GROWTH", "SATURATION_CHECK")


# ---------------------------------------------------------------------------
# noisy_scurve — sigmoid + small Gaussian noise
# ---------------------------------------------------------------------------

def test_noisy_scurve_reaches_stabilized(gen, tmp_path):
    """Small noise (std=0.01) must not prevent STABILIZED from firing."""
    traj = gen.noisy_scurve()
    _, decisions = _run(traj, tmp_path)
    assert Decision.STABILIZED in decisions


def test_noisy_scurve_no_regressed(gen, tmp_path):
    """Noise std=0.01 is far below rollback margin (0.05) — REGRESSED must not fire."""
    traj = gen.noisy_scurve()
    _, decisions = _run(traj, tmp_path)
    assert Decision.REGRESSED not in decisions


def test_noisy_scurve_stabilizes_in_second_half(gen, tmp_path):
    """STABILIZED must not fire before the trajectory is past its midpoint."""
    traj = gen.noisy_scurve()
    _, decisions = _run(traj, tmp_path)
    first_stab = next(i + 1 for i, d in enumerate(decisions) if d == Decision.STABILIZED)
    assert first_stab > len(traj.composites) // 2


# ---------------------------------------------------------------------------
# oscillating — alternates between 0.84 and 0.80
# ---------------------------------------------------------------------------

def test_oscillating_all_iterate(gen, tmp_path):
    """BCP cannot confirm a regime shift in an oscillating window — all ITERATE."""
    traj = gen.oscillating()
    _, decisions = _run(traj, tmp_path)
    assert all(d == Decision.ITERATE for d in decisions)


def test_oscillating_never_stabilized(gen, tmp_path):
    """Oscillation never satisfies LR+EWMA conjunction — STABILIZED must not fire."""
    traj = gen.oscillating()
    _, decisions = _run(traj, tmp_path)
    assert Decision.STABILIZED not in decisions


def test_oscillating_state_stays_activation(gen, tmp_path):
    """Without a confirmed regime shift, the state machine must not leave ACTIVATION."""
    traj = gen.oscillating()
    ps, _ = _run(traj, tmp_path)
    assert ps.state == "ACTIVATION"


# ---------------------------------------------------------------------------
# premature_saturation — two-plateau trajectory
# ---------------------------------------------------------------------------

def test_premature_saturation_fires_stabilized(gen, tmp_path):
    """The first hard plateau must trigger STABILIZED (LR+EWMA see a flat window)."""
    traj = gen.premature_saturation()
    _, decisions = _run(traj, tmp_path)
    assert Decision.STABILIZED in decisions


def test_premature_saturation_no_regressed(gen, tmp_path):
    """Plateau scores never drop below rollback threshold — REGRESSED must not fire."""
    traj = gen.premature_saturation()
    _, decisions = _run(traj, tmp_path)
    assert Decision.REGRESSED not in decisions


def test_premature_saturation_stabilizes_at_first_plateau(gen, tmp_path):
    """STABILIZED must fire during or before the midpoint (first plateau, not second)."""
    traj = gen.premature_saturation()
    _, decisions = _run(traj, tmp_path)
    first_stab = next(i + 1 for i, d in enumerate(decisions) if d == Decision.STABILIZED)
    assert first_stab <= len(traj.composites) // 2


# ---------------------------------------------------------------------------
# regression_post_saturation — S-curve then sharp drop
# ---------------------------------------------------------------------------

def test_regression_post_saturation_has_stabilized(gen, tmp_path):
    """The plateau phase (runs 17-24) must produce at least one STABILIZED decision."""
    traj = gen.regression_post_saturation()
    _, decisions = _run(traj, tmp_path)
    assert Decision.STABILIZED in decisions


def test_regression_post_saturation_has_regressed(gen, tmp_path):
    """The drop to 0.84 is below rollback threshold (0.92-0.05=0.87) — REGRESSED must fire."""
    traj = gen.regression_post_saturation()
    _, decisions = _run(traj, tmp_path)
    assert Decision.REGRESSED in decisions


def test_regression_post_saturation_order(gen, tmp_path):
    """STABILIZED must appear before REGRESSED — saturation confirmed before the drop."""
    traj = gen.regression_post_saturation()
    _, decisions = _run(traj, tmp_path)
    first_stab = next(i for i, d in enumerate(decisions) if d == Decision.STABILIZED)
    first_reg  = next(i for i, d in enumerate(decisions) if d == Decision.REGRESSED)
    assert first_stab < first_reg


def test_regression_post_saturation_regressed_at_boundary(gen, tmp_path):
    """REGRESSED must first fire at or after the regression_start boundary."""
    traj = gen.regression_post_saturation()
    _, decisions = _run(traj, tmp_path)
    reg_start = traj.phase_boundaries["regression_start"]
    first_reg = next(i + 1 for i, d in enumerate(decisions) if d == Decision.REGRESSED)
    assert first_reg >= reg_start


# ---------------------------------------------------------------------------
# fast_convergence — high-steepness sigmoid
# ---------------------------------------------------------------------------

def test_fast_convergence_reaches_stabilized(gen, tmp_path):
    """High-steepness sigmoid plateaus quickly — STABILIZED must fire in 15 runs."""
    traj = gen.fast_convergence()
    _, decisions = _run(traj, tmp_path)
    assert Decision.STABILIZED in decisions


def test_fast_convergence_no_regressed(gen, tmp_path):
    traj = gen.fast_convergence()
    _, decisions = _run(traj, tmp_path)
    assert Decision.REGRESSED not in decisions


# ---------------------------------------------------------------------------
# slow_convergence — gradual rise + hard plateau
# ---------------------------------------------------------------------------

def test_slow_convergence_reaches_stabilized(gen, tmp_path):
    """Gradual rise + 10-run hard plateau must eventually confirm saturation."""
    traj = gen.slow_convergence()
    _, decisions = _run(traj, tmp_path)
    assert Decision.STABILIZED in decisions


def test_slow_convergence_stabilizes_late(gen, tmp_path):
    """STABILIZED must fire in the second half of the trajectory — the rise is slow."""
    traj = gen.slow_convergence()
    _, decisions = _run(traj, tmp_path)
    first_stab = next(i + 1 for i, d in enumerate(decisions) if d == Decision.STABILIZED)
    assert first_stab > len(traj.composites) // 2


def test_slow_convergence_no_regressed(gen, tmp_path):
    traj = gen.slow_convergence()
    _, decisions = _run(traj, tmp_path)
    assert Decision.REGRESSED not in decisions


# ---------------------------------------------------------------------------
# Reproducibility — same seed always produces same trajectories
# ---------------------------------------------------------------------------

def test_same_seed_produces_identical_noisy_scurve():
    """seed=42 must produce the same noisy trajectory on every call."""
    g1 = TrajectoryGenerator(seed=42)
    g2 = TrajectoryGenerator(seed=42)
    assert g1.noisy_scurve().composites == g2.noisy_scurve().composites


def test_different_seeds_produce_different_noisy_scurve():
    """Different seeds must produce different noisy trajectories."""
    g1 = TrajectoryGenerator(seed=42)
    g2 = TrajectoryGenerator(seed=99)
    assert g1.noisy_scurve().composites != g2.noisy_scurve().composites
