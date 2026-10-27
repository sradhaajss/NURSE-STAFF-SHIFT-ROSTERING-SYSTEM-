# SmartRoster

**CSP-Based Nurse/Staff Shift Rostering — Min-Conflicts Local Search + Warm-Start Repair**

A Principles of Artificial Intelligence academic project: a working,
end-to-end Streamlit application that formulates hospital shift rostering as
a Constraint Satisfaction Problem (CSP), solves it with a genuine
**Min-Conflicts local search** algorithm, and demonstrates real-time
**warm-start repair** when live disruptions (sick calls, leave, transfers)
occur — benchmarked honestly against a full re-solve and a recursive
backtracking baseline.

**Technique:** Constraint Satisfaction Problem · Min-Conflicts Local Search
· Warm-Start Repair

**Dataset:** INRC-II n012w8 — Week 0 (real benchmark), plus a self-contained
synthetic generator

---

## Problem Statement

Hospitals must roster staff across shifts and days while respecting hard
rules (skills, availability, minimum coverage, maximum workload, legal
shift-sequencing rules) and soft goals (fairness, preferences, rest
patterns). When a nurse calls in sick mid-week, re-solving the entire roster
from scratch is slow and disruptive to staff who were already scheduled.
This project asks: **can we repair just the affected part of an existing
roster, instead of starting over — and how much faster is that, measured
honestly?**

## Objectives

1. Formulate nurse rostering as a CSP (variables, domains, hard/soft
   constraints, cost function).
2. Implement a genuine backtracking baseline solver.
3. Implement Min-Conflicts local search as the core solver.
4. Load and correctly use the real INRC-II n012w8 benchmark data, including
   its history file.
5. Simulate live availability disruptions.
6. Implement warm-start repair and measure it against a full re-solve.
7. Track fairness (night/weekend/total shift balance) throughout.
8. Present all of this transparently in a professional, faculty-demo-ready
   dashboard — never overstating what the code actually does.

## Features

- **Real INRC-II benchmark adapter** — loads the actual n012w8 Second
  International Nurse Rostering Competition scenario, week-0 demand, *and*
  history files (see "Dataset Pipeline" below).
- **Synthetic data generator** — deterministic, seeded, ~45 staff / 3 wards
  / 7 days / 3 shifts, with multi-skilled float staff — runs immediately,
  no external files required.
- **Backtracking baseline** — a real recursive CSP search, bounded by a node
  limit and timeout; auto-downsizes and says so honestly (with the exact
  caveat shown in the UI) when the full instance is too large to search
  exhaustively.
- **Min-Conflicts local search** — the core algorithm: greedy initial fill,
  iterative conflict minimization with random restarts, full iteration
  logging for genuine convergence charts. Its local scoring understands
  every hard constraint the evaluator checks (see "Implemented Constraints"
  below) — nothing is checked at evaluation time that the solver can't see.
- **Live Availability Simulator** — synthetic sick-call / leave / transfer
  events with a real-time event feed (actual wall-clock timestamps). Not
  connected to any live hospital system.
- **Warm-Start Repair** — preserves unaffected assignments, re-optimizes
  only the impacted portion of the roster, and is benchmarked against a full
  re-solve on every disruption (measured, never fabricated, speed
  improvement — including the possibility of a negative number if warm-start
  is ever slower on a given run).
- **Fairness module** — std. dev. of night shifts, weekend shifts, and total
  shifts per staff member; before/after comparisons.
- **Algorithm Comparison & Evaluation pages** — multi-seed controlled
  experiments; every number shown comes from an actual execution.
- **CSP Inspector** — dynamically lists only the constraints actually
  active for the loaded dataset, shows the real configured cost-function
  weights, and displays live Min-Conflicts convergence data.
- **Dataset Transparency card + Model Coverage checklist** — states plainly
  what INRC-II data is loaded and how each field is used (Loaded / Enforced
  / Approximated / Not modelled) — see below.
- **Run Full Demo** button — executes the entire pipeline (load INRC-II data
  → solve → disrupt → repair → compare) in a few seconds for a faculty
  walkthrough.

## Architecture

```
smart_roster/
├── app.py                     # Streamlit UI, all pages + session state
├── requirements.txt
├── README.md
├── data/
│   └── inrc_n012w8/            # Real INRC-II benchmark files (n012w8)
│       ├── Sc-n012w8.json       # scenario: nurses, skills, shift types,
│       │                        # forbidden successions, contracts
│       ├── WD-n012w8-0.json     # week-0 demand + shift-off requests
│       └── H0-n012w8-0.json     # history: each nurse's state entering week 0
├── core/
│   ├── models.py               # Staff, HistoricalState, HospitalData, Roster, CSP vars/domains
│   ├── constraints.py          # Hard (H1-H11) & soft (S1-S8) constraints, cost function
│   ├── fairness.py             # Night/weekend/total shift std-dev tracking
│   ├── backtracking.py         # Recursive backtracking baseline
│   ├── min_conflicts.py        # Core Min-Conflicts local search solver
│   ├── repair.py               # Warm-start repair + full re-solve comparison
│   ├── events.py               # Disruption event simulation
│   └── evaluation.py           # Multi-seed controlled experiments
├── ui/
│   ├── styles.py                # Dark "AI operations dashboard" CSS
│   ├── charts.py                # Plotly chart builders (dark theme)
│   └── roster_view.py           # Roster → DataFrame / CSV helpers
└── utils/
    └── data_generator.py        # Synthetic generator + INRC-II 3-file adapter
```

## Dataset Pipeline

```
Sc-n012w8.json  (scenario)
        +
WD-n012w8-0.json  (week 0 demand)
        +
H0-n012w8-0.json  (history)
        ↓
utils.data_generator.load_inrc_benchmark(scenario_path, week_data_path, history_path)
        ↓
HospitalData  (internal CSP model, incl. HistoricalState per nurse)
        ↓
Min-Conflicts
        ↓
Roster
```

The application identifies the loaded instance in the UI as:

> **INRC-II n012w8 — Week 0**
> Single-week prototype using the official INRC-II benchmark data.

This is **not** a complete implementation of the full 8-week INRC-II
competition specification — see "Model Coverage" below for exactly what is
and isn't modelled, shown live in the app's Dataset Transparency card.

### How H0 (history) is used

`H0-n012w8-0.json` gives each nurse's state entering week 0: last assigned
shift type, current consecutive-working-days run, consecutive-days-off run,
consecutive same-shift-type run, and working-weekends count so far. This is
parsed into a `HistoricalState` per nurse (`core/models.py`) and used as
**carry-in** when evaluating:

- H8 forbidden shift succession at the week-1 boundary (was the last shift
  before Monday one that forbids Monday's assignment?)
- H9 maximum consecutive working days (a run that started before week 0
  still counts)
- H10 maximum consecutive same-shift-type assignments (same idea, per shift
  type)
- H11 maximum working weekends (adds to the count already accrued)

## Implemented Constraints

**Hard constraints (H1-H11):**

| # | Constraint | Data source |
|---|---|---|
| H1 | No double booking (two shifts same day) | universal |
| H2 | No assignment while unavailable | universal |
| H3 | Minimum ward/shift/day staffing | week demand file |
| H4 | Required skill for the requirement | scenario skills |
| H5 | Maximum total assignments | contract (weekly-approximated, see below) |
| H6 | Ward/eligibility | universal |
| H7 | Disruption/unavailability (same mechanism as H2) | live events |
| H8 | Forbidden shift-type succession | scenario `forbiddenShiftTypeSuccessions` + H0 |
| H9 | Maximum consecutive working days | contract + H0 carry-in |
| H10 | Maximum consecutive same-shift-type assignments | scenario `shiftTypes` + H0 carry-in |
| H11 | Maximum working weekends | contract + H0 carry-in |

H8-H11 are **inactive (no-ops) for the synthetic generator**, which doesn't
define INRC-II scenario/contract data — they only fire when the INRC-II
benchmark is loaded.

**Soft constraints (S1-S8):** S1-S3 balanced total/night/weekend shifts
(fairness module), S4 shift preferences, S5 reaching the minimum total
assignments, S6 minimum consecutive working-day stretch, S7 minimum
consecutive rest stretch, S8 complete weekends (work both Sat & Sun, or
neither). S5-S8 are modelled as **soft** preferences, not hard constraints —
see "Known Approximations" below for why.

Min-Conflicts' local conflict-scoring function (`conflicts_for_variable` in
`core/constraints.py`) checks H1, H2/H7, H4, H5, H6, H8, H9, H10 and H11
directly when scoring a candidate assignment, so the solver understands
every hard constraint the evaluator checks — nothing is evaluated after the
fact that the solver was blind to during search.

## Min-Conflicts Algorithm

1. Build a greedy initial assignment (intentionally left slightly
   under-filled so the local search has genuine repair work to do).
2. Repeat: find conflicted variables (local hard-constraint violations —
   including H8-H11 where applicable — or staff who could fill an
   understaffed slot), pick one at random, and move it to the domain value
   that minimizes conflicts (ties broken randomly).
3. Restart from a fresh initial assignment (up to a configurable number of
   times) if stuck, keeping the best roster found.
4. Every iteration's conflict count, hard-violation count and fairness score
   are logged for the convergence chart — nothing is precomputed or faked.

Min-Conflicts is a **heuristic local search**. The UI reports
**"Feasible Roster Generated"** or **"Best Feasible Roster Found Within
Search Budget"** — it never claims a mathematically proven optimum.

## Warm-Start Repair

When a disruption event fires:
1. The affected `(staff, day, shift)` slots are identified and released.
2. Every other assignment is left untouched.
3. Min-Conflicts is re-run, scoped to the released slots plus any variable
   that develops a new conflict as a side effect — not the whole roster.

This is benchmarked on every disruption against **Approach A: Full
Re-Solve** (discard the roster, solve from scratch), with solve time,
violations, fairness and changed-assignment counts measured directly:

```
Speed Improvement = ((full_resolve_time − warm_start_time) / full_resolve_time) × 100
```

If warm-start is ever slower on a given run, the UI shows that actual
(possibly negative) number — it is not hard-coded to always look faster.

## Fairness Metric

For each staff member: total shifts, night shifts, weekend shifts. The
project tracks the **standard deviation** of night and weekend shifts across
staff — **lower std. dev. = more balanced/fair distribution.** A non-zero or
below-threshold score is never described as "fair" outright; the UI always
frames it as "lower deviation indicates a more balanced distribution."

## Model Coverage (INRC-II)

Shown live in the app for the currently loaded dataset. For the INRC-II
benchmark:

| Field | Status |
|---|---|
| Scenario data (skills, shift types, contracts) | ✓ Loaded |
| Nurse skills | ✓ Loaded |
| Contracts | ✓ Loaded |
| Weekly requirements | ✓ Loaded |
| Shift-off requests | ✓ Loaded |
| Historical state (H0) | ✓ Loaded |
| Forbidden shift successions | ✓ Enforced (H8) |
| Max consecutive working days | ✓ Enforced (H9) |
| Max consecutive same-shift-type assignments | ✓ Enforced (H10) |
| Max working weekends | ✓ Enforced (H11) |
| Min consecutive working days | ◐ Soft preference (S6), not hard |
| Min consecutive days off | ◐ Soft preference (S7), not hard |
| Complete weekends | ◐ Soft preference (S8), not hard |
| Min/Max total assignments | ◐ Approximated as a weekly-scaled cap from the 8-week contract |
| Multi-week rotation / full 8-week competition scoring | ✗ Not modelled (single-week prototype) |

## Known Approximations / Limitations

- **Single planning week.** Only week 0 is loaded and solved. The real
  INRC-II competition schedules all 8 weeks together with rules that span
  the full horizon; this prototype only sees week 0 plus the H0 carry-in
  state, not weeks 2-8.
- **Weekly-scaled min/max assignments.** The scenario's
  `minimumNumberOfAssignments` / `maximumNumberOfAssignments` are defined
  over 8 weeks. This prototype divides them by 8 as an approximation
  (`utils/data_generator.py`, `_week_cap` / `_week_floor`) — this is *not*
  the real multi-week rule, and is documented as such in the code and the
  Model Coverage card.
- **Min-side sequence rules are soft, not hard.** Minimum consecutive
  working days, minimum consecutive days off, and complete-weekends are
  enforced as soft preferences (S6-S8), not hard constraints, because
  correctly enforcing a *minimum* run length inside a single-slot local
  search — without look-ahead into the following week — risks either false
  violations at the week boundary or an intractable global constraint. Runs
  touching either edge of the week are deliberately left unchecked rather
  than risk a false positive.
- **Backtracking baseline doesn't check H9-H11.** The recursive backtracking
  baseline checks H1-H8 during search (H8 via a cheap single-day lookback)
  but not the multi-day consecutive-run constraints H9-H11, for tractability
  — those are still caught in the final `evaluate_hard_constraints()`
  report, same as it already does for the global H3 minimum-staffing check.
- **The backtracking baseline auto-downsizes** to a smaller staff subset on
  large instances and states this in the UI: *"Backtracking baseline uses a
  reduced benchmark slice for tractability; results are not directly
  equivalent to the full-instance Min-Conflicts run."*
- **Min-Conflicts / warm-start repair find feasible, near-optimal
  solutions, not provably globally optimal ones.**
- **Custom CSV/JSON upload** is scaffolded in the UI but not yet wired to a
  full parser — the architecture (`utils/data_generator.py`) is written so
  an additional adapter can be dropped in later.
- This project never claims to be connected to a real, live hospital
  system — all disruption events and the synthetic dataset are explicitly
  labelled as simulated.

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. No Docker, database, API keys or paid
services required.

## Demo Instructions

1. Open the app — you'll land on **Overview**.
2. Click **▶ RUN FULL DEMO** for an automatic end-to-end walkthrough using
   the real INRC-II benchmark (falls back to the synthetic generator if the
   benchmark files aren't found), or:
3. Go to **Roster Generator** → load **INRC-II n012w8 — Week 0** (or the
   synthetic dataset) → **Generate Roster**.
4. Go to **Live Updates** → pick or randomize a disruption → **Apply
   Event** → **Repair Roster**.
5. Go to **Warm-Start Repair** to see the full re-solve vs. warm-start
   timing comparison and the before/after roster grids.
6. Check **Fairness Analytics**, **Algorithm Comparison** and
   **Evaluation** for the supporting charts and multi-seed experiment data.
7. Use **CSP Inspector** to walk faculty through the formulation itself —
   it lists only the constraints actually active for the loaded dataset.

## Evaluation Metrics

Solve time, node/iteration counts, hard-violation counts, fairness score,
changed-assignment counts, and warm-start speed improvement — all computed
live on the Evaluation page across several random seeds, for the currently
loaded dataset (INRC-II or synthetic).

## Future Enhancements

- MRV/degree-ordering heuristics for the backtracking baseline.
- A full custom CSV/JSON schema adapter for arbitrary hospital data.
- True multi-week rostering across all 8 INRC-II weeks with correctly
  carried-forward history between weeks (rather than a single week + H0
  carry-in).
- Enforcing S6-S8 as hard constraints once a look-ahead/look-back window
  spanning adjacent weeks is available.
- Additional soft constraints (seniority-based weighting, shift-sequence
  preferences beyond forbidden successions).
