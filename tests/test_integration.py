"""
Integration tests for the PhaseStop end-to-end pipeline — build stage T2.

Unit tests verify individual components in isolation. These tests verify
the full pipeline working together: correct detectors fired per phase,
exact phase-boundary run numbers, complete storage round-trip, and
consistent signal propagation from detectors through to stored records.

Canonical trajectory used throughout:
    CANONICAL = [0.76, 0.79, 0.82, 0.85, 0.87, 0.88, 0.88, 0.88, 0.88, 0.88]

Verified ground truth for this trajectory:
    Runs 1–6:  phase=ACTIVATION, detectors=[bayesian_cp, moving_avg]
    Runs 7–9:  phase=GROWTH,     detectors=[mann_kendall, moving_avg]
    Run 10:    phase=SATURATION_CHECK, detectors=[linear_reg, ewma_stable, mann_kendall]
    STABILIZED fires at exactly run 10.
"""

import json
import pytest
from phasestop.config import Decision, Signal
from phasestop.scorer import PhaseStop

# ---------------------------------------------------------------------------
# Canonical trajectory and ground truth tables
# ---------------------------------------------------------------------------

CANONICAL = [0.76, 0.79, 0.82, 0.85, 0.87, 0.88, 0.88, 0.88, 0.88, 0.88]

# phase_state records the phase each run was EVALUATED IN (before any transition).
GROUND_TRUTH_PHASE = {
    1: "ACTIVATION", 2: "ACTIVATION", 3: "ACTIVATION",
    4: "ACTIVATION", 5: "ACTIVATION", 6: "ACTIVATION",
    7: "GROWTH",     8: "GROWTH",     9: "GROWTH",
    10: "SATURATION_CHECK",
}

GROUND_TRUTH_DECISION = {
    1: "ITERATE", 2: "ITERATE", 3: "ITERATE",
    4: "ITERATE", 5: "ITERATE", 6: "ITERATE",
    7: "ITERATE", 8: "ITERATE", 9: "ITERATE",
    10: "STABILIZED",
}

# Detectors the state machine must call for each phase.
DETECTORS_PER_PHASE = {
    "ACTIVATION":       {"bayesian_cp", "moving_avg"},
    "GROWTH":           {"mann_kendall", "moving_avg"},
    "SATURATION_CHECK": {"linear_reg", "ewma_stable", "mann_kendall"},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_and_store(composites: list[float], tmp_path) -> tuple[list[Decision], list[dict]]:
    """Run PhaseStop and return (decisions, stored_records)."""
    path = str(tmp_path / "history.json")
    ps = PhaseStop(storage_path=path)
    decisions = [ps.add_run(run_id=i + 1, composite=c) for i, c in enumerate(composites)]
    records = [json.loads(line) for line in open(path) if line.strip()]
    return decisions, records


# ---------------------------------------------------------------------------
# 1. Exact phase-boundary run numbers
# ---------------------------------------------------------------------------

def test_exact_phase_per_run(tmp_path):
    """Stored phase_state must match ground truth at every run."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    for r in records:
        run_id = r["run_id"]
        assert r["phase_state"] == GROUND_TRUTH_PHASE[run_id], (
            f"run {run_id}: expected phase {GROUND_TRUTH_PHASE[run_id]}, "
            f"got {r['phase_state']}"
        )


def test_exact_decision_per_run(tmp_path):
    """Decision must match ground truth at every run."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    for r in records:
        run_id = r["run_id"]
        assert r["decision"] == GROUND_TRUTH_DECISION[run_id], (
            f"run {run_id}: expected {GROUND_TRUTH_DECISION[run_id]}, "
            f"got {r['decision']}"
        )


def test_activation_spans_runs_1_to_6(tmp_path):
    """ACTIVATION phase must cover exactly runs 1–6."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    activation_runs = {r["run_id"] for r in records if r["phase_state"] == "ACTIVATION"}
    assert activation_runs == {1, 2, 3, 4, 5, 6}


def test_growth_spans_runs_7_to_9(tmp_path):
    """GROWTH phase must cover exactly runs 7–9."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    growth_runs = {r["run_id"] for r in records if r["phase_state"] == "GROWTH"}
    assert growth_runs == {7, 8, 9}


def test_stabilized_fires_at_run_10(tmp_path):
    """STABILIZED must fire at exactly run 10, not earlier or later."""
    decisions, _ = _run_and_store(CANONICAL, tmp_path)
    # decisions list is 0-indexed; run 10 is index 9
    assert decisions[9] == Decision.STABILIZED
    assert all(d != Decision.STABILIZED for d in decisions[:9])


# ---------------------------------------------------------------------------
# 2. Detector identity per phase
# ---------------------------------------------------------------------------

def test_activation_calls_bcp_and_ma_only(tmp_path):
    """In ACTIVATION, only bayesian_cp and moving_avg must be called."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    for r in records:
        if r["phase_state"] == "ACTIVATION":
            names = {d["name"] for d in r["detector_results"]}
            assert names == DETECTORS_PER_PHASE["ACTIVATION"], (
                f"run {r['run_id']}: unexpected detectors {names}"
            )


def test_growth_calls_mk_and_ma_only(tmp_path):
    """In GROWTH, only mann_kendall and moving_avg must be called."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    for r in records:
        if r["phase_state"] == "GROWTH":
            names = {d["name"] for d in r["detector_results"]}
            assert names == DETECTORS_PER_PHASE["GROWTH"], (
                f"run {r['run_id']}: unexpected detectors {names}"
            )


def test_saturation_check_calls_lr_ewma_mk(tmp_path):
    """In SATURATION_CHECK, linear_reg, ewma_stable, and mann_kendall must all be called."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    for r in records:
        if r["phase_state"] == "SATURATION_CHECK":
            names = {d["name"] for d in r["detector_results"]}
            assert names == DETECTORS_PER_PHASE["SATURATION_CHECK"], (
                f"run {r['run_id']}: unexpected detectors {names}"
            )


# ---------------------------------------------------------------------------
# 3. Signal propagation — detectors produce valid, sensible signals
# ---------------------------------------------------------------------------

def test_all_detector_signals_are_valid_enum_values(tmp_path):
    """Every stored detector signal must be a valid Signal enum value."""
    valid = {s.value for s in Signal}
    _, records = _run_and_store(CANONICAL, tmp_path)
    for r in records:
        for dr in r["detector_results"]:
            assert dr["signal"] in valid, (
                f"run {r['run_id']}, {dr['name']}: invalid signal {dr['signal']!r}"
            )


def test_all_detector_confidences_in_range(tmp_path):
    """Every stored detector confidence must be in [0.0, 1.0]."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    for r in records:
        for dr in r["detector_results"]:
            assert 0.0 <= dr["confidence"] <= 1.0, (
                f"run {r['run_id']}, {dr['name']}: confidence {dr['confidence']} out of range"
            )


def test_short_window_detectors_return_insufficient(tmp_path):
    """Runs 1–5 have fewer than WINDOW_K points — all detectors must report INSUFFICIENT."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    short_window_records = [r for r in records if r["run_id"] <= 5]
    for r in short_window_records:
        for dr in r["detector_results"]:
            assert dr["signal"] == "INSUFFICIENT", (
                f"run {r['run_id']}, {dr['name']}: expected INSUFFICIENT, got {dr['signal']}"
            )


def test_full_window_activation_detectors_fire_improving(tmp_path):
    """At run 6 (first full window in ACTIVATION), BCP and MA must both signal IMPROVING."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    run6 = next(r for r in records if r["run_id"] == 6)
    signals = {d["name"]: d["signal"] for d in run6["detector_results"]}
    assert signals["bayesian_cp"] == "IMPROVING"
    assert signals["moving_avg"] == "IMPROVING"


def test_saturation_detectors_both_stabilized_at_run_10(tmp_path):
    """At run 10 (STABILIZED), LR and EWMA must both signal STABILIZED."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    run10 = next(r for r in records if r["run_id"] == 10)
    signals = {d["name"]: d["signal"] for d in run10["detector_results"]}
    assert signals["linear_reg"] == "STABILIZED"
    assert signals["ewma_stable"] == "STABILIZED"


# ---------------------------------------------------------------------------
# 4. Storage round-trip — every field survives serialization
# ---------------------------------------------------------------------------

def test_storage_round_trip_all_fields_present(tmp_path):
    """Every stored record must contain all required top-level fields."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    required = {"run_id", "timestamp", "composite", "phase_state",
                "detector_results", "decision", "notes"}
    for r in records:
        assert required <= r.keys(), (
            f"run {r.get('run_id')}: missing fields {required - r.keys()}"
        )


def test_storage_round_trip_detector_fields_present(tmp_path):
    """Every stored DetectorResult must contain all five required fields."""
    required = {"name", "signal", "confidence", "metric", "note"}
    _, records = _run_and_store(CANONICAL, tmp_path)
    for r in records:
        for dr in r["detector_results"]:
            assert required <= dr.keys(), (
                f"run {r['run_id']}, {dr.get('name')}: missing fields {required - dr.keys()}"
            )


def test_storage_composite_values_preserved(tmp_path):
    """Stored composite values must exactly match the input sequence."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    stored = [r["composite"] for r in records]
    assert stored == CANONICAL


def test_storage_run_ids_are_sequential(tmp_path):
    """Stored run_ids must be 1-indexed and sequential."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    assert [r["run_id"] for r in records] == list(range(1, len(CANONICAL) + 1))


def test_storage_timestamps_are_iso_format(tmp_path):
    """Stored timestamps must be parseable ISO-8601 strings."""
    import datetime
    _, records = _run_and_store(CANONICAL, tmp_path)
    for r in records:
        # fromisoformat raises ValueError if format is wrong
        datetime.datetime.fromisoformat(r["timestamp"])


def test_storage_record_count_matches_run_count(tmp_path):
    """Exactly one record must be written per add_run() call."""
    _, records = _run_and_store(CANONICAL, tmp_path)
    assert len(records) == len(CANONICAL)
