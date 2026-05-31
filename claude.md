# PhaseStop — Project Context for Claude Code

## What this is

PhaseStop is a research framework for **trajectory-aware iteration
termination in iterative AI system optimization**. It addresses a gap
common to all iterative AI optimization processes — model fine-tuning,
prompt tuning, component selection, hyperparameter search, agentic
configuration — none of which currently have a principled,
trajectory-aware stopping criterion.

The framework is domain-agnostic. The RAG domain is the demonstration
case study because labeled public benchmarks (BEIR) enable objective
validation.

**Paper:** PhaseStop: Phase-Conditioned Detector Orchestration for
Iterative AI System Optimization (Draft v0.10, AI Cohort capstone)
**Author:** Rajesh Joshi
**Target:** EMNLP 2026 Industry Track or RAG Workshop

---

## How to Use This File

Rajesh is learning every piece of code as it is written.

**Rules for every increment:**
1. Write only what the current increment specifies — nothing more
2. Before writing code, explain in plain English what it does and why
3. After writing code, explain each block line by line
4. End every increment with a verify step Rajesh can run himself
5. Wait for "understood, next" before proceeding to the next increment
6. Never skip ahead or add future functionality early

---

## Core Concept

AI optimization trajectories follow a three-phase pattern:

```
Activation → Growth → Saturation
    P1           P2        P3
```

Each phase has a distinct statistical signature. PhaseStop assigns
one or two detectors to each phase and activates them in sequence
via a state machine.

**Central contribution:** phase-conditioned orchestration of
heterogeneous detectors — not any individual detector.

---

## System Boundary

PhaseStop receives **one float per run**. Nothing else.

```python
decision = phasestop.add_run(run_id=9, composite=0.836)
```

**PhaseStop never:**
- Sees raw metric scores
- Knows metric names or weights
- Knows the aggregation method
- Computes the composite
- Checks per-metric floor thresholds

The caller owns all of that. PhaseStop owns trajectory analysis only.

**Why not track individual metrics inside PhaseStop:**
Each metric stabilises at a different run. Multiple trajectories
produce conflicting decisions. The composite collapses all signals
into one trajectory that PhaseStop watches cleanly.

---

## Five Detectors and Phase Assignments

| Detector | Phase | What it detects |
|---|---|---|
| Bayesian Change Point (BCP) | Activation (P1) | Distributional regime shift — scores left a new baseline |
| Moving Average Slope (MA) | Growth (P2) | Mean shift sustained across windows — not a spike |
| Mann-Kendall (MK) | Growth → Saturation (P2→P3) | Trend significance via p-value — p < 0.05 means trend is real; p >= 0.05 means no significant trend (plateau) |
| Linear Regression (LR) | Saturation (P3) | Slope statistically indistinguishable from zero |
| EWMA | Saturation (P3) | Smoothed value stopped moving — plateau is real |

**STABILIZED requires LR AND EWMA to agree — conjunction, not vote.**

---

## Detector Input

Every detector takes one argument:

```python
window: list[float]
```

The last k composite scores, oldest first, newest last.

```python
window = [0.731, 0.769, 0.798, 0.821, 0.833, 0.836]
#          run4   run5   run6   run7   run8   run9
```

Detectors are domain-blind — numbers only. This makes them
testable with a single list of floats. No mocks needed.

---

## State Machine

```
States: ACTIVATION → GROWTH → SATURATION_CHECK → STABILIZED | REGRESSED

for each add_run(composite):

    # Layer 0 — composite gates only
    if composite < quality_floor:
        return ITERATE
    if composite < best_composite - rollback_margin:
        return REGRESSED
    update best_composite
    append composite to history
    window = last k composites

    # Layer 1 — state machine
    if state == ACTIVATION:
        if BCP.regime_shifted AND MA.rising:
            state = GROWTH
        return ITERATE

    if state == GROWTH:
        if MK.trend_fading:
            state = SATURATION_CHECK
        elif MA.declining:
            return REGRESSED
        return ITERATE

    if state == SATURATION_CHECK:
        if LR.slope_near_zero AND EWMA.stable:
            return STABILIZED
        elif MK.strongly_increasing:
            state = GROWTH
        return ITERATE
```

**Layer 0 checks composite only — never individual metrics.**
**States advance on positive confirmation only — never by timeout.**
**Backward regression permitted: SATURATION_CHECK → GROWTH.**

---

## Formal Phase Definitions

σ(S_base) = std of first 3 runs. μ(W) = mean of window W.
β(W) = OLS slope of window W.

| Phase | Criterion |
|---|---|
| P1 Activation | Δs > 2σ(S_base) for at least one run |
| P2 Growth | β(W_recent) > 0 AND μ(W_recent) > μ(W_prior) for two consecutive windows |
| P3 Saturation | β(W_recent) not significant (p > 0.10) AND EWMA delta < ε for k runs |

---

## Hyperparameter Defaults

All in config.py — no magic numbers anywhere else.

| Parameter | Default | Why |
|---|---|---|
| Window k | 6 | Minimum for Mann-Kendall statistical power |
| MK p-value | 0.05 | Standard significance threshold — trend is real if p < 0.05 |
| LR p-value | 0.10 | Conservative — 0.05 misses gradual saturation |
| EWMA alpha | 0.3 | Standard process-control default |
| Rollback margin | 0.05 | 5% composite drop triggers REGRESSED |
| Quality floor | 0.75 | Composite must exceed this to be eligible |
| Storage format | "json" | Configurable — "csv" also supported |
| Storage path | "results/run_history.json" | Configurable |

**Why MK uses p-value not raw tau:**
Raw tau has no universal threshold — its interpretation depends on
window size and data characteristics. The p-value from the
Mann-Kendall significance test is statistically principled:
p < 0.05 = significant trend exists (IMPROVING or DECLINING).
p >= 0.05 = no significant trend (STABILIZED).
This eliminates the arbitrary tau > 0.2 heuristic and is
consistent with how LR slope significance is handled.

---

## Storage

Append-only. One record per run. Never modifies past records.

```
run_id | timestamp           | composite | phase_state      | decision   | notes
1      | 2026-05-01T09:00:00 | 0.612     | ACTIVATION       | ITERATE    | [...]
9      | 2026-05-21T16:40:00 | 0.836     | SATURATION_CHECK | STABILIZED | [...]
```

---

## Data Structures (defined in config.py)

```python
# What a detector returns
@dataclass
class DetectorResult:
    name:       str
    signal:     Signal    # IMPROVING | STABILIZED | DECLINING | INSUFFICIENT
    confidence: float     # 0.0–1.0
    metric:     str       # the key number e.g. "p=0.031" for MK, "slope=+0.012" for LR
    note:       str       # plain English explanation

# What the state machine returns per run
class Decision(Enum):
    ITERATE     = "ITERATE"
    STABILIZED  = "STABILIZED"
    REGRESSED   = "REGRESSED"

# Full record stored per run
@dataclass
class RunResult:
    run_id:           int
    timestamp:        str
    composite:        float
    phase_state:      str
    detector_results: list[DetectorResult]
    decision:         Decision
    notes:            list[str]
```

---

## Project Structure

```
PhaseStop/
├── CLAUDE.md
├── README.md
├── requirements.txt
├── phasestop/
│   ├── __init__.py
│   ├── config.py               ← Session C1–C4
│   ├── activation_detector.py  ← Session D5
│   ├── growth_detector.py      ← Session D1–D2
│   ├── saturation_detector.py  ← Session D3–D4
│   ├── scorer.py               ← Session S1–S4
│   └── synthetic.py            ← Session Y1–Y3
├── experiments/
│   ├── __init__.py
│   ├── study_a.py
│   ├── study_b.py
│   └── ablation.py
├── tests/
│   ├── test_activation_detector.py  ← Session T1
│   ├── test_growth_detector.py      ← Session T1
│   ├── test_saturation_detector.py  ← Session T1
│   ├── test_scorer.py               ← Session T2
│   └── test_synthetic.py            ← Session T3
├── ui/
│   ├── __init__.py
│   ├── viewer.py     ← Session V1: trajectory review
│   └── simulator.py  ← Session V2: value simulator
├── results/
│   └── .gitkeep
└── paper/
    └── EvalReady_paper_outline_v10.pdf
```

---

## Build Stages — Incremental

Each increment: explain → write → verify → wait for "understood, next"

### config.py

**C1 — Enums**
Write Signal enum and Decision enum only.
Explain: what an enum is, why we use it instead of strings,
what each value means in the context of PhaseStop.
Verify: `python -c "from phasestop.config import Signal, Decision; print(Signal.IMPROVING)"`

**C2 — DetectorResult dataclass**
Add DetectorResult only.
Explain: what a dataclass is, what each field represents,
why confidence is a float not a bool.
Verify: create one DetectorResult in the Python shell and print it.

**C3 — RunResult dataclass**
Add RunResult only.
Explain: difference between DetectorResult (one detector's view)
and RunResult (the full picture for one run).
Verify: create one RunResult in the Python shell and print it.

**C4 — Hyperparameters and storage config**
Add all constants: WINDOW_K, LR_P_THRESHOLD, EWMA_ALPHA,
ROLLBACK_MARGIN, QUALITY_FLOOR, STORAGE_FORMAT, STORAGE_PATH.
Explain: why each value was chosen, what happens if it is too
high or too low (refer to paper Section 3.5).
Verify: `python -c "from phasestop.config import WINDOW_K; print(WINDOW_K)"`

---

### detectors.py

**D1 — mann_kendall() only**
Write file scaffold (imports, docstring) and mann_kendall() only.
Explain: what Mann-Kendall detects, why it is non-parametric,
what Kendall tau measures, why we use the p-value not raw tau,
what p < 0.05 means (significant trend), what p >= 0.05 means
(no significant trend — plateau candidate).
Verify:
```python
from phasestop.detectors import mann_kendall
print(mann_kendall([0.5, 0.6, 0.7, 0.8, 0.85, 0.88]))
print(mann_kendall([0.84, 0.84, 0.83, 0.84, 0.84, 0.83]))
```

**D2 — moving_avg()**
Add moving_avg() only.
Explain: what window mean shift means, why two windows are
compared, what slope_threshold controls.
Verify:
```python
from phasestop.detectors import moving_avg
print(moving_avg([0.60, 0.65, 0.70, 0.78, 0.83, 0.86]))
print(moving_avg([0.83, 0.84, 0.83, 0.84, 0.83, 0.84]))
```

**D3 — ewma_stable()**
Add ewma_stable() only.
Explain: what exponential weighting does, what alpha controls,
why EWMA is better than simple moving average for noisy data.
Verify:
```python
from phasestop.detectors import ewma_stable
print(ewma_stable([0.60, 0.65, 0.70, 0.78, 0.83, 0.86]))
print(ewma_stable([0.84, 0.84, 0.83, 0.84, 0.84, 0.84]))
```

**D4 — linear_reg()**
Add linear_reg() only.
Explain: what OLS slope means, what the t-statistic measures,
why p > 0.10 means "slope is effectively zero".
Verify:
```python
from phasestop.detectors import linear_reg
print(linear_reg([0.60, 0.65, 0.72, 0.79, 0.84, 0.88]))
print(linear_reg([0.84, 0.83, 0.84, 0.83, 0.84, 0.84]))
```

**D5 — bayesian_cp()**
Add bayesian_cp() only.
Explain: what a regime change means, what the stability ratio
measures, why this detector fires first (at activation).
Verify:
```python
from phasestop.detectors import bayesian_cp
print(bayesian_cp([0.50, 0.50, 0.51, 0.72, 0.79, 0.83]))
print(bayesian_cp([0.83, 0.84, 0.83, 0.84, 0.83, 0.84]))
```

---

### scorer.py

**S1 — Class shell and add_run() signature**
Write PhaseStop class with __init__ and add_run() signature only.
No logic yet — just the structure.
Explain: what __init__ sets up, what add_run() will do,
what instance variables are needed and why.
Verify: `ps = PhaseStop(); print(ps)` — no errors.

**S2 — Layer 0: composite gates**
Add the composite gate logic inside add_run().
No state machine yet — just the two checks:
quality floor and rollback margin.
Explain: why these are composite-level only, what each gate
catches, what happens to best_composite tracking.
Verify: call add_run() with values below and above the floor.
Confirm correct Decision returned each time.

**S3 — Layer 1: ACTIVATION and GROWTH states**
Add the ACTIVATION and GROWTH state handling.
Explain: what triggers the transition from ACTIVATION to GROWTH
(BCP + MA), what triggers the transition from GROWTH to
SATURATION_CHECK (MK fading), what triggers early REGRESSED.
Verify: feed a rising sequence and confirm state advances
from ACTIVATION → GROWTH across runs.

**S4 — SATURATION_CHECK + storage writer**
Add SATURATION_CHECK state and the JSON/CSV storage writer.
Explain: why STABILIZED requires LR AND EWMA conjunction,
what backward regression means, how the storage writer works
(append-only, ISO timestamp).
Verify: run a full clean S-curve sequence end to end.
Confirm STABILIZED is returned. Open run_history.json and
read the records.

---

### synthetic.py

**Y1 — Base generator + clean_scurve**
Write the base trajectory generator class and clean_scurve only.
Explain: what a sigmoid function produces, how phase boundaries
are embedded as ground truth, why seed=42 default.
Verify: generate 3 clean_scurve trajectories and plot/print them.
Confirm they look like S-curves.

**Y2 — noisy_scurve, oscillating, premature_saturation**
Add three more trajectory types.
Explain: what Gaussian noise does to a trajectory, what
oscillation looks like statistically, what premature saturation
means for the state machine.
Verify: generate one of each and print. Confirm they are
visually distinct from clean_scurve.

**Y3 — regression_post_saturation, fast_convergence,
slow_convergence**
Add the final three trajectory types.
Explain: why regression after saturation is a critical test
case, what fast/slow convergence tests about window size.
Verify: generate one of each. Confirm regression trajectory
shows a clear drop after plateau.

---

### tests

**T1 — test_detectors.py**
One test per detector. Each test uses a clean_scurve window
(should return IMPROVING) and a plateau window (STABILIZED).
Explain: what assert does, why we test with synthetic data,
how to read pytest output.
Verify: `pytest tests/test_detectors.py -v`
All tests pass.

**T2 — test_scorer.py**
Test state machine transitions.
Three scenarios: clean S-curve reaches STABILIZED, oscillating
stays ITERATE, regression returns REGRESSED.
Verify: `pytest tests/test_scorer.py -v`
All tests pass.

**T3 — test_synthetic.py**
Verify each trajectory type has correct phase boundaries.
Verify: `pytest tests/test_synthetic.py -v`
All tests pass.

---

### ui/viewer.py — Trajectory Review UI

Purpose: load a completed run history from `results/run_history.json`
and let the user explore how PhaseStop progressed run by run — which
phase it was in, what each detector reported, and when decisions fired.
Read-only. No input. Depends on S4 (storage writer) being complete.

Run with: `streamlit run ui/viewer.py`

**V1-a — App shell + data loader**
Create `ui/viewer.py` with Streamlit scaffold. Load
`results/run_history.json`. Display raw run records as a table
(run_id, composite, phase_state, decision columns).
Explain: what Streamlit is, how `st.title / st.dataframe` work,
how to load JSON into a pandas DataFrame.
Verify: `streamlit run ui/viewer.py` — table appears in browser,
one row per run.

**V1-b — Composite trajectory chart**
Add a line chart of composite score over run_id.
Annotate phase state transitions with vertical dashed lines
(ACTIVATION→GROWTH, GROWTH→SATURATION_CHECK).
Color-code decision markers: ITERATE=gray, STABILIZED=green,
REGRESSED=red.
Explain: what `st.line_chart` vs `st.pyplot` offers, why we use
matplotlib here for annotation control.
Verify: chart appears with visible phase boundaries and colored
decision points.

**V1-c — Per-run detector breakdown**
Add an expander for each run that shows a table of detector
results: name, signal, confidence, metric, note.
Signals color-coded: IMPROVING=blue, STABILIZED=green,
DECLINING=red, INSUFFICIENT=gray.
Explain: what `st.expander` does, how to render a colored
DataFrame with `st.dataframe(styler)`.
Verify: click any run row — detector table expands with
correct signals and notes.

---

### ui/simulator.py — Value Simulator UI

Purpose: let the user feed composite scores into PhaseStop
interactively — either by typing them manually or selecting a
synthetic trajectory — and watch the state machine respond in
real time. Depends on scorer.py (S1–S4) and synthetic.py (Y1–Y3)
being complete.

Run with: `streamlit run ui/simulator.py`

**V2-a — App shell + manual input**
Create `ui/simulator.py` with Streamlit scaffold.
Text area accepting comma-separated composite floats
(e.g. `0.61, 0.70, 0.75, 0.79, 0.83, 0.85, 0.84, 0.84`).
"Run simulation" button feeds each value into a fresh
`PhaseStop()` instance via `add_run()` and collects results.
Display results as a table (run_id, composite, phase_state,
decision).
Explain: how `st.text_area` and `st.button` work, how to
instantiate PhaseStop and call add_run() in a loop.
Verify: paste 8 values, click Run — table shows ITERATE/STABILIZED
decisions matching expected state machine behaviour.

**V2-b — Step-by-step run cards**
Replace or augment the table with per-run cards showing:
phase state badge (colored), decision badge (colored),
and a compact detector signal row (icons or colored pills).
Explain: how to use `st.columns` and `st.metric` to lay out
per-run cards side by side.
Verify: run a clean S-curve input — cards show phase advancing
ACTIVATION → GROWTH → SATURATION_CHECK → STABILIZED with
correct colors at each step.

**V2-c — Synthetic trajectory selector**
Add a selectbox listing all trajectory types from synthetic.py
(clean_scurve, noisy_scurve, oscillating, premature_saturation,
regression_post_saturation, fast_convergence, slow_convergence).
Selecting one auto-populates the text area with that trajectory's
composite scores. "Run simulation" button still works the same way.
Add a seed input (default 42) for reproducibility.
Explain: how synthetic.py generates trajectories, why exposing
seed matters for reproducible demos and paper figures.
Verify: select `regression_post_saturation` — text area fills
with a plateau-then-drop sequence; simulation returns REGRESSED
at the correct run.

---

## Coding Standards

1. **Explain before writing** — plain English first, code second.
2. **No magic numbers** — everything from config.py.
3. **Type hints on all signatures.**
4. **Docstring on every function** — names paper section it implements.
5. **DetectorResult always returned** from detector functions.
6. **Reproducibility** — random operations take seed param, default 42.
7. **No external ML deps** — scipy, numpy, statistics only.
8. **Storage append-only** — never overwrite past records.

### UI Standards (ui/ only)

9. **Streamlit only** — no Flask, no Dash. `streamlit run ui/<file>.py` is
   the single launch command for both UIs.
10. **No business logic in ui/** — viewer.py and simulator.py only call
    into `phasestop/` modules. No detector math, no state machine logic
    lives in the UI layer.
11. **Read-only data access in viewer.py** — never writes to
    `results/run_history.json`. The viewer is a lens, not an editor.
12. **Fresh PhaseStop instance per simulation** — simulator.py must
    create a new `PhaseStop()` each time "Run simulation" is clicked.
    Never reuse state across runs.
13. **pandas for table display** — load JSON into a DataFrame before
    passing to `st.dataframe`. Keeps display logic clean.
14. **Add `streamlit` and `pandas` to requirements.txt** when V1-a
    is implemented.

---

## Key Design Decisions (locked)

1. PhaseStop receives one float per run. Nothing else.
2. Per-sample composite first, then aggregate across batch.
3. Floor gates (per-metric) belong to the caller, not PhaseStop.
4. Do not track metrics separately — composite is the correct input.
5. STABILIZED = LR AND EWMA conjunction. Not a vote.
6. Weighted query aggregation is future work — not in this version.

---

## Key Related Papers

- RAGAS (Es et al., 2023) — arXiv:2309.15217
- Maiorano (2026) — quality gate — arXiv:2603.15676
- Stop-RAG (Park et al., 2025) — arXiv:2510.14337
- BEIR (Thakur et al., 2021) — NeurIPS
- Mann (1945) — Econometrica
- Adams & MacKay (2007) — Bayesian change point detection
- Roberts (1959) — EWMA — Technometrics

---

## First Prompt for Claude Code

> "Read CLAUDE.md fully. We are building PhaseStop incrementally.
> Rajesh must understand every line before we move on.
> Start with increment C1: write Signal enum and Decision enum
> in phasestop/config.py only. Before writing the code, explain
> in plain English what an enum is and why we use it here.
> After writing, explain each line. End with the verify command.
> Wait for Rajesh to say 'understood, next' before C2."

---

*PhaseStop — Rajesh Joshi — AI Cohort capstone*
*Paper: v0.10 | CLAUDE.md: v5 | Build: D5 complete, S1 next*