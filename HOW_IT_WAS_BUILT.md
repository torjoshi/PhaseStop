# How PhaseStop Was Built

PhaseStop was not built by prompting an AI and accepting whatever came back.
It was built using **agentic engineering** — a disciplined approach to
AI-assisted development in which the human defines the system, the AI
implements it, and verification gates every step.

This document describes that process.

---

## The Distinction That Matters

In May 2026, Addy Osmani, Shubham Saboo, and Sokratis Kartakis published
*The New SDLC With Vibe Coding: From Ad-Hoc Prompting to Agentic Engineering*.
The paper defines a spectrum:

> "The key differentiator is not whether you use AI. It's how much structure,
> verification, and human judgment surrounds the AI's output."

At one end is **vibe coding** — casual prompts, minimal verification, code the
developer may not fully understand. At the other end is **agentic engineering** —
formal specifications, automated test suites, guardrails, and human oversight of
every architectural decision.

PhaseStop was built at the agentic end.

---

## The Harness: claude.md

Before a single line of code was written, a `claude.md` file defined the full
system contract. The paper describes this class of artifact as the harness —
the scaffolding that surrounds the AI model and governs its behavior.

The `claude.md` for PhaseStop specifies:

- **System boundary** — PhaseStop receives one composite float per run. Nothing else. The rationale for this constraint is documented explicitly.
- **State machine logic** — the full phase transition pseudocode, written out before implementation began.
- **Detector assignments** — which detector runs in which phase, and why.
- **Hyperparameter defaults** — every threshold, with the reasoning behind each value.
- **Locked design decisions** — six decisions that cannot be changed without revisiting the paper's argument.
- **Coding standards** — no magic numbers, type hints on all signatures, docstrings referencing the paper section each function implements.
- **Build stages** — 20+ named increments (C1–C4, D1–D5, S1–S4, and so on), each with an explain → write → verify → wait cycle.

This file is the harness. It is what separates the build process from vibe coding.

---

## The Build Process

The factory model described in the paper states:

> "A factory manager does not assemble every widget by hand.
> They design the assembly line and ensure quality control."

That is the role taken here. The assembly line was designed first.
Claude Code implemented each increment. No increment was accepted
without running the verification step and confirming the output
matched the expected behavior.

The process for every increment:

1. Claude explained the increment in plain English before writing code.
2. Claude wrote only what the increment specified — nothing more.
3. A verify command was run and the output was checked.
4. The next increment did not begin until the current one was understood.

This is the conductor mode described in the paper — hands-on, real-time
direction, with the developer guiding every movement.

---

## Verification

The paper draws a hard line:

> "Without both tests and evals, the practice is always vibe coding,
> regardless of how sophisticated the prompts are."

PhaseStop has 97 passing tests covering:

- Unit tests for every detector (clean window → IMPROVING, plateau window → STABILIZED)
- State machine transition tests (clean S-curve reaches STABILIZED, oscillating stays ITERATE, regression returns REGRESSED)
- Integration tests asserting exact phase boundaries on the canonical trajectory
- Storage round-trip tests verifying all fields survive serialization
- Synthetic trajectory tests verifying correct phase boundaries per trajectory type

Tests were written as part of the build, not added afterward.

---

## Separation of Concerns

The codebase enforces a strict dependency rule:

- `phasestop/` — core library, no imports from tools or UI
- `tools/` — research utilities, imports from core only
- `ui/` — Streamlit apps, no business logic, calls into core only
- `experiments/` — reproducible study scripts
- `tests/` — may import from any layer

The core can be packaged and deployed independently.
The UI contains zero detector math and zero state machine logic.
This is the harness boundary the paper describes.

---

## What This Means for the Code

Anyone reviewing this repository can apply the paper's field test:
ask the author to explain the logic line by line.

Every design decision in PhaseStop has a documented reason:
- Why Mann-Kendall uses p-values instead of raw tau
- Why STABILIZED requires LR AND EWMA conjunction, not a vote
- Why the composite floor gate belongs to the caller, not to PhaseStop
- Why window size k=6 is the minimum for Mann-Kendall statistical power

The code was not accepted because it ran. It was accepted because it
was understood.

---

## Reference

Osmani, A., Saboo, S., and Kartakis, S. (2026).
*The New SDLC With Vibe Coding: From Ad-Hoc Prompting to Agentic Engineering.*
Published May 2026.

---

*PhaseStop — Rajesh Joshi — AI Cohort Capstone, 2026*
