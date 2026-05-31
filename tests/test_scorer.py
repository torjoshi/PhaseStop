"""
Unit tests for phasestop.scorer — build stage T2.
Tests verify state machine transitions and Layer 0 gates.
"""

import json
import pytest
from phasestop.config import Decision
from phasestop.scorer import PhaseStop


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def run_sequence(composites: list[float], tmp_path) -> list[Decision]:
    """Feed composites into a fresh PhaseStop and collect all decisions."""
    ps = PhaseStop(storage_path=str(tmp_path / "history.json"))
    return [ps.add_run(run_id=i + 1, composite=c) for i, c in enumerate(composites)]


# ---------------------------------------------------------------------------
# S1 — Class shell
# ---------------------------------------------------------------------------

def test_repr_no_error():
    """PhaseStop() shell must be constructable and printable with no errors."""
    ps = PhaseStop()
    text = repr(ps)
    assert "ACTIVATION" in text
    assert "runs=0" in text


def test_initial_state_is_activation():
    ps = PhaseStop()
    assert ps.state == "ACTIVATION"


# ---------------------------------------------------------------------------
# S2 — Layer 0: composite gates
# ---------------------------------------------------------------------------

def test_below_quality_floor_returns_iterate(tmp_path):
    """Composite below 0.75 must return ITERATE without touching history."""
    ps = PhaseStop(storage_path=str(tmp_path / "h.json"))
    decision = ps.add_run(run_id=1, composite=0.70)
    assert decision == Decision.ITERATE
    assert len(ps.history) == 0


def test_above_quality_floor_enters_history(tmp_path):
    """Composite at or above quality floor must be appended to history."""
    ps = PhaseStop(storage_path=str(tmp_path / "h.json"))
    ps.add_run(run_id=1, composite=0.80)
    assert len(ps.history) == 1


def test_rollback_margin_triggers_regressed(tmp_path):
    """A drop of more than 5% below best composite must return REGRESSED."""
    ps = PhaseStop(storage_path=str(tmp_path / "h.json"))
    ps.add_run(run_id=1, composite=0.88)  # best = 0.88
    decision = ps.add_run(run_id=2, composite=0.80)  # 0.80 < 0.88 - 0.05 = 0.83
    assert decision == Decision.REGRESSED


def test_small_drop_does_not_trigger_regressed(tmp_path):
    """A drop within the rollback margin must not return REGRESSED."""
    ps = PhaseStop(storage_path=str(tmp_path / "h.json"))
    ps.add_run(run_id=1, composite=0.88)
    decision = ps.add_run(run_id=2, composite=0.84)  # 0.84 > 0.83 threshold
    assert decision != Decision.REGRESSED


# ---------------------------------------------------------------------------
# S3 / S4 — Scenario 1: clean S-curve reaches STABILIZED
# ---------------------------------------------------------------------------

# Trajectory designed to drive all three phase transitions:
#   Runs 1–5:  rising above quality floor, building history (window not full yet)
#   Run 6:     BCP + MA both fire → ACTIVATION → GROWTH (still returns ITERATE)
#   Runs 7–8:  in GROWTH, trend still significant
#   Run 9:     plateau window causes MK to fade → GROWTH → SATURATION_CHECK
#   Run 10:    LR + EWMA both flat → STABILIZED
CLEAN_SCURVE = [0.76, 0.79, 0.82, 0.85, 0.87, 0.88, 0.88, 0.88, 0.88, 0.88]


def test_clean_scurve_reaches_stabilized(tmp_path):
    """A clean S-curve must eventually return STABILIZED."""
    decisions = run_sequence(CLEAN_SCURVE, tmp_path)
    assert Decision.STABILIZED in decisions


def test_clean_scurve_stabilized_is_last_decision(tmp_path):
    """STABILIZED must appear at the end of a clean S-curve, not mid-sequence."""
    decisions = run_sequence(CLEAN_SCURVE, tmp_path)
    assert decisions[-1] == Decision.STABILIZED


def test_clean_scurve_no_regressed(tmp_path):
    """A monotonically rising S-curve must never trigger REGRESSED."""
    decisions = run_sequence(CLEAN_SCURVE, tmp_path)
    assert Decision.REGRESSED not in decisions


def test_clean_scurve_state_advances_to_growth(tmp_path):
    """State must reach GROWTH before STABILIZED is possible."""
    ps = PhaseStop(storage_path=str(tmp_path / "h.json"))
    for i, c in enumerate(CLEAN_SCURVE):
        ps.add_run(run_id=i + 1, composite=c)
    # After a complete S-curve the state machine ends in SATURATION_CHECK
    # (it returned STABILIZED, so state was not advanced further).
    assert ps.state in ("GROWTH", "SATURATION_CHECK")


# ---------------------------------------------------------------------------
# Scenario 2: oscillating window — never exits ACTIVATION
# ---------------------------------------------------------------------------

# Scores oscillate between 0.80 and 0.84. The 0.04 swing is within the
# rollback margin (0.05), so Layer 0 never triggers REGRESSED. BCP's
# stability ratio stays below 1.0 because the recent half mean is only
# marginally different from the baseline half mean. State stays ACTIVATION.
OSCILLATING = [0.80, 0.84, 0.80, 0.84, 0.80, 0.84, 0.80, 0.84, 0.80, 0.84]


def test_oscillating_stays_iterate(tmp_path):
    """Oscillating scores must never leave ITERATE — BCP cannot confirm a regime shift."""
    decisions = run_sequence(OSCILLATING, tmp_path)
    assert all(d == Decision.ITERATE for d in decisions)


def test_oscillating_never_leaves_activation(tmp_path):
    """State must remain ACTIVATION throughout an oscillating sequence."""
    ps = PhaseStop(storage_path=str(tmp_path / "h.json"))
    for i, c in enumerate(OSCILLATING):
        ps.add_run(run_id=i + 1, composite=c)
    assert ps.state == "ACTIVATION"


# ---------------------------------------------------------------------------
# Scenario 3: regression — Layer 0 rollback fires
# ---------------------------------------------------------------------------

# Rises cleanly then drops 0.08 below best (above the 0.05 margin).
REGRESSION_SEQUENCE = [0.76, 0.79, 0.82, 0.85, 0.87, 0.88, 0.80]


def test_regression_returns_regressed(tmp_path):
    """A significant score drop must return REGRESSED."""
    decisions = run_sequence(REGRESSION_SEQUENCE, tmp_path)
    assert decisions[-1] == Decision.REGRESSED


def test_regression_only_fires_at_drop(tmp_path):
    """REGRESSED must not appear before the actual drop."""
    decisions = run_sequence(REGRESSION_SEQUENCE, tmp_path)
    assert all(d != Decision.REGRESSED for d in decisions[:-1])


# ---------------------------------------------------------------------------
# S4 — Storage writer
# ---------------------------------------------------------------------------

def test_storage_writes_one_line_per_run(tmp_path):
    """JSON storage must append exactly one record per add_run() call."""
    path = str(tmp_path / "history.json")
    ps = PhaseStop(storage_path=path)
    for i, c in enumerate([0.80, 0.82, 0.84]):
        ps.add_run(run_id=i + 1, composite=c)
    lines = [l for l in open(path).read().splitlines() if l.strip()]
    assert len(lines) == 3


def test_storage_records_are_valid_json(tmp_path):
    """Each line in the storage file must be parseable JSON."""
    path = str(tmp_path / "history.json")
    ps = PhaseStop(storage_path=path)
    ps.add_run(run_id=1, composite=0.80)
    line = open(path).read().strip()
    record = json.loads(line)
    assert record["run_id"] == 1
    assert record["composite"] == 0.80


def test_storage_records_decision_as_string(tmp_path):
    """Decision must be stored as a plain string, not an Enum repr."""
    path = str(tmp_path / "history.json")
    ps = PhaseStop(storage_path=path)
    ps.add_run(run_id=1, composite=0.80)
    record = json.loads(open(path).read().strip())
    assert record["decision"] == "ITERATE"
