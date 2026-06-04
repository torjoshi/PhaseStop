"""
PhaseStop — Trajectory Review UI (build stages V1-a through V1-c).

Read-only viewer for a completed run history stored in results/run_history.json.
Displays the composite trajectory, phase boundaries, decision outcomes, and
per-run detector breakdowns.

Run with:
    streamlit run ui/viewer.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")  # must be called before importing pyplot — sets headless backend

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HISTORY_PATH = "results/run_history.json"

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

# ---------------------------------------------------------------------------
# V1-a — App shell and data loader
# ---------------------------------------------------------------------------

st.set_page_config(page_title="PhaseStop Viewer", layout="wide")
st.title("PhaseStop — Trajectory Viewer")

if not os.path.exists(_HISTORY_PATH):
    st.info(
        f"No run history found at `{_HISTORY_PATH}`.  \n"
        "Run PhaseStop on a trajectory first to generate data."
    )
    st.stop()

# Load JSONL — one JSON object per line.
records: list[dict] = []
with open(_HISTORY_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

if not records:
    st.warning(f"`{_HISTORY_PATH}` exists but contains no records.")
    st.stop()

# Build summary DataFrame for the top table.
summary_df = pd.DataFrame([
    {
        "run_id":      r["run_id"],
        "composite":   r["composite"],
        "phase_state": r["phase_state"],
        "decision":    r["decision"],
    }
    for r in records
])

st.subheader(f"Run Summary — {len(summary_df)} runs")
st.dataframe(summary_df, width="stretch", hide_index=True)

# ---------------------------------------------------------------------------
# V1-b — Composite trajectory chart
# ---------------------------------------------------------------------------

st.subheader("Composite Trajectory")

run_ids    = summary_df["run_id"].tolist()
composites = summary_df["composite"].tolist()
y_min = min(composites) - 0.005
y_max = max(composites) + 0.005

fig, ax = plt.subplots(figsize=(12, 3.5))

# Base line connecting all runs.
ax.plot(run_ids, composites, color="#455a64", linewidth=1.5, zorder=1)

# Decision markers — one scatter series per decision type so each color
# appears in the legend automatically.
for decision, color in _DECISION_COLORS.items():
    mask = summary_df["decision"] == decision
    if mask.any():
        ax.scatter(
            summary_df.loc[mask, "run_id"],
            summary_df.loc[mask, "composite"],
            color=color, s=70, zorder=2, label=decision,
        )

# Vertical dashed lines at phase state transitions.
# phase_state records the phase a run was evaluated IN (before transition),
# so a change at run N means the new phase started at run N.
prev_phase = None
for r in records:
    current_phase = r["phase_state"]
    if prev_phase is not None and current_phase != prev_phase:
        x = r["run_id"] - 0.5          # draw between the two runs
        ax.axvline(x=x, color="#90a4ae", linestyle="--", linewidth=0.9, alpha=0.8)
        ax.text(
            x + 0.1, y_max - 0.001,
            current_phase,
            fontsize=7, color="#546e7a", ha="left", va="top",
        )
    prev_phase = current_phase

ax.set_xlabel("Run ID", fontsize=9)
ax.set_ylabel("Composite Score", fontsize=9)
ax.set_xlim(0.5, max(run_ids) + 0.5)
ax.set_ylim(y_min, y_max + 0.012)      # headroom for phase labels
ax.legend(loc="lower right", fontsize=8, framealpha=0.7)
ax.grid(True, alpha=0.25, linewidth=0.5)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(labelsize=8)

st.pyplot(fig)
plt.close(fig)                          # release memory — Streamlit re-runs the full script

# ---------------------------------------------------------------------------
# V1-c — Per-run detector breakdown
# ---------------------------------------------------------------------------

st.subheader("Per-run Detector Breakdown")

def _signal_style(val: str) -> str:
    """CSS for a signal cell — colored text so it works on light and dark themes."""
    color = _SIGNAL_COLORS.get(val, "#000000")
    return f"color: {color}; font-weight: 600"


for r in records:
    # Expander header is plain text: run number, score, phase, decision.
    header = (
        f"Run {r['run_id']:>3}  ·  "
        f"composite={r['composite']:.4f}  ·  "
        f"{r['phase_state']}  ·  "
        f"{r['decision']}"
    )
    with st.expander(header):
        # State machine transition notes (e.g. "BCP + MA both IMPROVING → GROWTH").
        if r.get("notes"):
            for note in r["notes"]:
                st.caption(note)

        if not r["detector_results"]:
            st.caption("No detectors ran — run was caught by a Layer 0 gate.")
        else:
            det_df = pd.DataFrame([
                {
                    "detector":   dr["name"],
                    "signal":     dr["signal"],
                    "confidence": dr["confidence"],
                    "metric":     dr["metric"],
                    "note":       dr["note"],
                }
                for dr in r["detector_results"]
            ])
            # Color the signal column; leave everything else unstyled.
            # hide_index not passed with Styler — Streamlit ignores it and warns
            styled = det_df.style.hide(axis="index").map(_signal_style, subset=["signal"])
            st.dataframe(styled, width="stretch")
