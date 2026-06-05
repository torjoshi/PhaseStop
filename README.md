# PhaseStop

Every iterative AI optimization process produces a recurring
decision after each run: has the system improved enough to keep
going, or is further iteration wasting effort? 

PhaseStop is Phase-Conditioned Detector Orchestration
for Iterative AI System Optimization. It is a Python library for deciding when an iterative system has likely stopped improving.

You feed it one composite score per run. It tracks the recent trajectory, applies phase-specific detectors, and returns one of three decisions:

- `ITERATE`: keep going
- `STABILIZED`: the score has plateaued
- `REGRESSED`: quality dropped enough to count as a regression

The repository also includes synthetic trajectory generators, experiment scripts, tests, and two Streamlit apps for simulation and history review.

## Paper

**PhaseStop: Phase-Conditioned Detector Orchestration for Iterative AI System Optimization**
Rajesh Joshi — AI Cohort Capstone, 2026

Published on Zenodo: [https://zenodo.org/records/20560596](https://zenodo.org/records/20560596)

## What It Does

PhaseStop models optimization as a three-phase process:

- `ACTIVATION`: has the system actually moved off its baseline yet?
- `GROWTH`: is there still statistically meaningful improvement?
- `SATURATION_CHECK`: has the curve flattened enough to stop?

Internally it uses five detectors:

- `bayesian_cp`
- `moving_avg`
- `mann_kendall`
- `ewma_stable`
- `linear_reg`

All thresholds and tuning knobs live in [phasestop/config.py](/Users/rjoshi/ai_projects/PhaseStop/phasestop/config.py).

## Requirements

- Python 3.11+ recommended
- `pip`

Dependencies are listed in [requirements.txt](/Users/rjoshi/ai_projects/PhaseStop/requirements.txt):

- `scipy`
- `matplotlib`
- `streamlit`
- `pandas`

## Quick Start

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then try a minimal example:

```python
from phasestop import PhaseStop

scores = [0.76, 0.79, 0.82, 0.85, 0.87, 0.88, 0.88, 0.88, 0.88, 0.88]

ps = PhaseStop()

for run_id, composite in enumerate(scores, start=1):
    decision = ps.add_run(run_id=run_id, composite=composite)
    print(run_id, composite, decision.value)
```

Expected behavior for that canonical trajectory: `STABILIZED` fires at run 10.

## How To Use The Library

The main public API is exported from [phasestop/__init__.py](/Users/rjoshi/ai_projects/PhaseStop/phasestop/__init__.py):

```python
from phasestop import PhaseStop, Decision, DetectorResult, RunResult, Signal
```

The most important method is:

```python
decision = ps.add_run(run_id=7, composite=0.884)
```

Notes:

- `run_id` is an integer identifier for the run you are recording.
- `composite` should be a float score, typically normalized to `0..1`.
- Each call appends a record to disk.
- By default, run history is written to `results/run_history.json` as JSON Lines.

You can override storage:

```python
ps = PhaseStop(
    storage_path="results/my_run.json",
    storage_format="json",  # or "csv"
)
```

## Decision Semantics

- `ITERATE` means no stopping condition has been confirmed yet.
- `STABILIZED` means the recent window looks flat enough to stop.
- `REGRESSED` means the latest score dropped below the best seen score by more than the rollback margin.

Two layer-0 gates run before the phase machine:

- Scores below the quality floor return `ITERATE` immediately.
- Large drops below the best prior score return `REGRESSED` immediately.

## First Things To Try

### 1. Run the tests

```bash
pytest -q
```

The current repo passes: `97 passed`.

### 2. Run a study script

```bash
python -m experiments.study_a
python -m experiments.study_b
python -m experiments.ablation
```

These scripts compare stopping behavior across synthetic trajectories and ablated detector variants.

### 3. Open the simulator UI

```bash
streamlit run ui/simulator.py
```

This app lets you:

- paste your own composite scores
- load built-in synthetic trajectories
- step a fresh `PhaseStop` instance through the sequence
- inspect the per-run phase and detector outputs

### 4. Open the history viewer

```bash
streamlit run ui/viewer.py
```

This app reads `results/run_history.json` and shows:

- a run summary table
- a trajectory plot
- phase transitions
- per-run detector breakdowns

If you have not generated any run history yet, the viewer will tell you that the file is missing.

## Synthetic Trajectories

[tools/synthetic.py](/Users/rjoshi/ai_projects/PhaseStop/tools/synthetic.py) provides reusable trajectory generators such as:

- `clean_scurve`
- `noisy_scurve`
- `oscillating`
- `premature_saturation`
- `regression_post_saturation`
- `fast_convergence`
- `slow_convergence`

Example:

```python
from tools.synthetic import TrajectoryGenerator

gen = TrajectoryGenerator(seed=42)
traj = gen.noisy_scurve()

print(traj.composites)
print(traj.phase_boundaries)
```

## Trace Utility

If you want detector-by-detector diagnostics without the state machine, use [tools/trace.py](/Users/rjoshi/ai_projects/PhaseStop/tools/trace.py):

```python
from tools.synthetic import TrajectoryGenerator
from tools.trace import print_trace, trace

history = TrajectoryGenerator(seed=42).clean_scurve().composites
points = trace(history)
print_trace(points)
```

This is useful for understanding what each detector is signaling across a full trajectory.

## Project Layout

- [phasestop](/Users/rjoshi/ai_projects/PhaseStop/phasestop): core library and state machine
- [tools](/Users/rjoshi/ai_projects/PhaseStop/tools): synthetic data and trace utilities
- [experiments](/Users/rjoshi/ai_projects/PhaseStop/experiments): reproducible study scripts
- [ui](/Users/rjoshi/ai_projects/PhaseStop/ui): Streamlit apps
- [tests](/Users/rjoshi/ai_projects/PhaseStop/tests): unit and integration tests

## Notes For First-Time Users

- Run commands from the repository root so imports like `from phasestop import PhaseStop` work cleanly.
- The default output file is append-only. If you re-run examples many times, `results/run_history.json` will keep growing.
- The Streamlit simulator uses a temporary file internally, so it does not overwrite your main run history.
- If you want a clean viewer session, point `PhaseStop` at a fresh storage path before generating runs.

## Example Workflow

1. Install dependencies.
2. Run `pytest -q` to verify the environment.
3. Try `streamlit run ui/simulator.py` with a built-in trajectory.
4. Integrate `PhaseStop.add_run()` into your own evaluation loop.
5. Review saved run history with `streamlit run ui/viewer.py`.

## License / Status

Paper published at [https://zenodo.org/records/20560596](https://zenodo.org/records/20560596).

**Non-commercial use** (research, academic, personal) is freely permitted with attribution.
**Commercial use** requires a separate written agreement — contact torjoshi@gmail.com.

See [LICENSE](LICENSE) for full terms.
