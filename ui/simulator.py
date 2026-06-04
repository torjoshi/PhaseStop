"""
PhaseStop — Value Simulator UI (build stages V2-a through V2-c).

Feed composite scores into PhaseStop interactively — type them manually
or select a synthetic trajectory — and watch the state machine respond
run by run.

Run with:
    python -m streamlit run ui/simulator.py
"""

import json
import os
import tempfile

import pandas as pd
import streamlit as st

from phasestop import PhaseStop
from tools.synthetic import TrajectoryGenerator

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DECISION_COLORS = {
    "ITERATE":    "#9e9e9e",
    "STABILIZED": "#4caf50",
    "REGRESSED":  "#f44336",
}

_SIGNAL_COLORS = {
    "IMPROVING":    "#1976d2",
    "STABILIZED":   "#4caf50",
    "DECLINING":    "#f44336",
    "INSUFFICIENT": "#9e9e9e",
}

_DETECTOR_ABBREV = {
    "bayesian_cp":  "BCP",
    "mann_kendall": "MK",
    "moving_avg":   "MA",
    "ewma_stable":  "EWMA",
    "linear_reg":   "LR",
}

_SIGNAL_ABBREV = {
    "IMPROVING":    "IMP",
    "STABILIZED":   "STAB",
    "DECLINING":    "DECL",
    "INSUFFICIENT": "INSF",
}

_CARDS_PER_ROW = 4


# ---------------------------------------------------------------------------
# V2-b helpers — HTML badge and signal row builders
# ---------------------------------------------------------------------------

def _badge(text: str, color: str) -> str:
    """Inline HTML pill badge — colored border + tinted background."""
    return (
        f'<span style="background:{color}22;color:{color};'
        f'border:1px solid {color}66;padding:1px 7px;'
        f'border-radius:4px;font-size:11px;font-weight:600;white-space:nowrap;">'
        f"{text}</span>"
    )


def _signal_row(detector_results: list[dict]) -> str:
    """Compact HTML row: detector name + colored signal abbreviation per detector."""
    parts = []
    for dr in detector_results:
        name   = _DETECTOR_ABBREV.get(dr["name"], dr["name"])
        sig    = dr["signal"]
        color  = _SIGNAL_COLORS.get(sig, "#000")
        abbrev = _SIGNAL_ABBREV.get(sig, sig[:4])
        parts.append(
            f'<span style="font-size:10px;">'
            f'<span style="color:#888;">{name}</span>'
            f'&nbsp;<span style="color:{color};font-weight:600;">{abbrev}</span>'
            f"</span>"
        )
    divider = '<span style="color:#e0e0e0;font-size:10px;">│</span>'
    inner = f"&nbsp;{divider}&nbsp;".join(parts)
    return f'<div style="margin-top:4px;line-height:1.8;">{inner}</div>'

_DEFAULT_INPUT = "0.76, 0.79, 0.82, 0.85, 0.87, 0.88, 0.88, 0.88, 0.88, 0.88"

# Trajectory types exposed in the selectbox.
# "manual" is a sentinel meaning "don't auto-populate — user types freely".
_TRAJECTORY_OPTIONS = [
    "manual",
    "clean_scurve",
    "noisy_scurve",
    "oscillating",
    "premature_saturation",
    "regression_post_saturation",
    "fast_convergence",
    "slow_convergence",
]


# ---------------------------------------------------------------------------
# V2-c — Trajectory selector callback
# ---------------------------------------------------------------------------

def _load_trajectory() -> None:
    """on_change callback for the selectbox and seed input.

    Fires before the script rerenders, so the updated session_state value
    is already in place when st.text_area reads it.
    """
    traj_type = st.session_state.get("traj_selector", "manual")
    seed      = int(st.session_state.get("seed_input", 42))
    if traj_type != "manual":
        gen  = TrajectoryGenerator(seed=seed)
        traj = getattr(gen, traj_type)()
        st.session_state["input_text"] = ", ".join(str(c) for c in traj.composites)


# ---------------------------------------------------------------------------
# V2-a — App shell and manual input
# ---------------------------------------------------------------------------

st.set_page_config(page_title="PhaseStop Simulator", layout="wide")
st.title("PhaseStop — Value Simulator")
st.caption(
    "Select a synthetic trajectory or enter your own scores, then click "
    "**Run simulation** to feed them into a fresh PhaseStop instance."
)

# Initialise text area value in session_state so the keyed widget can update it.
if "input_text" not in st.session_state:
    st.session_state["input_text"] = _DEFAULT_INPUT

# V2-c — Trajectory selector and seed input.
col_sel, col_seed = st.columns([4, 1])
with col_sel:
    st.selectbox(
        label="Synthetic trajectory",
        options=_TRAJECTORY_OPTIONS,
        key="traj_selector",
        on_change=_load_trajectory,
        help="Selecting a trajectory auto-populates the scores below.",
    )
with col_seed:
    st.number_input(
        label="Seed",
        min_value=0,
        value=42,
        step=1,
        key="seed_input",
        on_change=_load_trajectory,
        help="Random seed for noisy trajectories. Changing it reloads the scores.",
    )

# Text area is keyed so _load_trajectory can update it via session_state.
# Do NOT pass value= here — the value lives in st.session_state["input_text"].
st.text_area(
    label="Composite scores",
    key="input_text",
    height=80,
    help="Floats between 0 and 1, comma-separated. Each value is one add_run() call.",
)

run_btn = st.button("Run simulation", type="primary")

# ---------------------------------------------------------------------------
# Simulation logic — only executes on button click
# ---------------------------------------------------------------------------

if run_btn:
    # Parse and validate input — read from session_state, not a local variable.
    raw = st.session_state.get("input_text", "")
    try:
        composites = [float(x.strip()) for x in raw.split(",") if x.strip()]
    except ValueError as exc:
        st.error(f"Could not parse scores: {exc}")
        st.stop()

    if not composites:
        st.warning("Enter at least one composite score.")
        st.stop()

    # Run — fresh PhaseStop per UI Standard #12. Write to a temp file so the
    # viewer's results/run_history.json is never touched.
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "sim.json")
        ps = PhaseStop(storage_path=path)
        for i, c in enumerate(composites):
            ps.add_run(run_id=i + 1, composite=c)
        records = [json.loads(line) for line in open(path) if line.strip()]

    # Persist records so results survive rerenders (e.g. user scrolls).
    st.session_state["sim_records"] = records

# ---------------------------------------------------------------------------
# Results table — shown whenever records exist in session state
# ---------------------------------------------------------------------------

if st.session_state.get("sim_records"):
    records = st.session_state["sim_records"]

    df = pd.DataFrame([
        {
            "run_id":      r["run_id"],
            "composite":   r["composite"],
            "phase_state": r["phase_state"],
            "decision":    r["decision"],
        }
        for r in records
    ])

    st.subheader(f"Results — {len(df)} runs")
    st.dataframe(df, width="stretch", hide_index=True)

    # -----------------------------------------------------------------------
    # V2-b — Per-run cards
    # -----------------------------------------------------------------------

    st.subheader("Run Cards")

    for row_start in range(0, len(records), _CARDS_PER_ROW):
        row_slice = records[row_start: row_start + _CARDS_PER_ROW]
        cols = st.columns(_CARDS_PER_ROW)
        for col, r in zip(cols, row_slice):
            decision_color = _DECISION_COLORS.get(r["decision"], "#333")
            with col:
                # Composite score as a metric — large, prominent number.
                st.metric(
                    label=f"Run {r['run_id']}",
                    value=f"{r['composite']:.4f}",
                )
                # Phase state + decision as colored inline badges.
                phase_badge    = _badge(r["phase_state"], "#546e7a")
                decision_badge = _badge(r["decision"], decision_color)
                st.markdown(
                    f"{phase_badge}&nbsp;{decision_badge}",
                    unsafe_allow_html=True,
                )
                # Compact detector signal row (empty for Layer 0 gated runs).
                if r["detector_results"]:
                    st.markdown(
                        _signal_row(r["detector_results"]),
                        unsafe_allow_html=True,
                    )
