"""
core/repair.py

Warm-start repair — the main novelty of this project — plus the full
re-solve comparison used to benchmark it.

Warm-start repair preserves unaffected assignments and re-optimizes only
the impacted portion of the roster:

  1. Preserve the existing valid roster.
  2. Apply the availability change (core/events.py already mutated staff
     availability before this is called).
  3. Identify affected assignments (the variables the event directly
     touches).
  4. Remove/release only affected assignments (set them to OFF / re-open
     them for reassignment).
  5. Keep every other assignment fixed.
  6. Run Min-Conflicts, but seeded with the *existing* roster (not a fresh
     random one) and restricted so it only touches the affected variables
     and any variables that become newly conflicted as a side-effect.

This is contrasted against Approach A: FULL RE-SOLVE, which throws away
the existing roster entirely and runs Min-Conflicts from scratch.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from core.models import HospitalData, Roster, OFF, all_variables, domain_for
from core.constraints import evaluate_hard_constraints, conflicts_for_variable
from core.fairness import compute_fairness
from core.min_conflicts import min_conflicts_solve, _understaffed_slots, _score_value


@dataclass
class RepairResult:
    roster: Roster
    affected_variables: List[Tuple[str, str, str]]
    variables_repaired: int
    assignments_preserved: int
    assignments_changed: int
    repair_iterations: int
    time_seconds: float
    hard_violations: int
    fairness_score: float


@dataclass
class ComparisonResult:
    full_resolve: "FullResolveResult"
    warm_start: RepairResult
    speed_improvement_pct: float


@dataclass
class FullResolveResult:
    roster: Roster
    time_seconds: float
    iterations: int
    hard_violations: int
    fairness_score: float


def warm_start_repair(
    original_roster: Roster,
    affected_vars: List[Tuple[str, str, str]],
    data: HospitalData,
    max_iterations: int = 800,
    seed: int = 11,
    timeout_seconds: float = 8.0,
) -> RepairResult:
    rng = random.Random(seed)
    start = time.time()

    roster = dict(original_roster)
    preserved_before = sum(1 for v in roster.values() if v != OFF)

    # Step: release affected assignments so they can be repaired
    affected_set = set(affected_vars)
    for v in affected_vars:
        roster[v] = OFF

    # Repair loop: only touch affected variables + anything that newly
    # conflicts as a result (e.g. an understaffed slot the release created,
    # or a replacement causing a new double-booking elsewhere).
    frontier = set(affected_set)
    iterations = 0
    repaired_vars = set()

    for it in range(max_iterations):
        if time.time() - start > timeout_seconds:
            break
        understaffed = _understaffed_slots(roster, data)

        # Build the working conflict frontier: affected vars still OFF that
        # could fill a gap, plus any assigned var with a live hard conflict.
        candidates = []
        for v in frontier:
            candidates.append(v)
        for v, val in roster.items():
            if val != OFF and conflicts_for_variable(v, val, roster, data) > 0:
                candidates.append(v)

        candidates = list(dict.fromkeys(candidates))  # dedupe, preserve order
        if not candidates:
            break

        var = rng.choice(candidates)
        iterations += 1
        domain = domain_for(var, data)
        scored = [(_score_value(var, v, roster, data, understaffed), v) for v in domain]
        min_score = min(s for s, _ in scored)
        best_values = [v for s, v in scored if s == min_score]
        chosen = rng.choice(best_values)
        if roster[var] != chosen:
            roster[var] = chosen
            repaired_vars.add(var)
        # Always drop from the frontier once considered — any new conflict
        # this change creates elsewhere (e.g. a double-booking) is picked
        # up automatically next loop by the general conflict scan below,
        # so we don't need to pre-seed sibling slots (which previously
        # caused the frontier to never empty).
        frontier.discard(var)

    elapsed = time.time() - start
    hard = evaluate_hard_constraints(roster, data)
    fairness = compute_fairness(roster, data)

    changed = sum(1 for v in all_variables(data) if roster.get(v, OFF) != original_roster.get(v, OFF))
    preserved = sum(
        1 for v in all_variables(data)
        if roster.get(v, OFF) == original_roster.get(v, OFF) and original_roster.get(v, OFF) != OFF
    )

    return RepairResult(
        roster=roster,
        affected_variables=affected_vars,
        variables_repaired=len(repaired_vars),
        assignments_preserved=preserved,
        assignments_changed=changed,
        repair_iterations=iterations,
        time_seconds=elapsed,
        hard_violations=hard.total,
        fairness_score=fairness["combined_fairness_score"],
    )


def full_resolve(data: HospitalData, max_iterations: int = 3000, restarts: int = 2, seed: int = 21) -> FullResolveResult:
    result = min_conflicts_solve(data, max_iterations=max_iterations, restarts=restarts, seed=seed)
    return FullResolveResult(
        roster=result.roster,
        time_seconds=result.time_seconds,
        iterations=result.iterations_used,
        hard_violations=result.hard_violations,
        fairness_score=result.fairness_score,
    )


def compare_full_resolve_vs_warm_start(
    original_roster: Roster,
    affected_vars: List[Tuple[str, str, str]],
    data: HospitalData,
) -> ComparisonResult:
    full = full_resolve(data)
    warm = warm_start_repair(original_roster, affected_vars, data)

    if full.time_seconds > 0:
        speed_improvement = ((full.time_seconds - warm.time_seconds) / full.time_seconds) * 100
    else:
        speed_improvement = 0.0

    return ComparisonResult(
        full_resolve=full,
        warm_start=warm,
        speed_improvement_pct=speed_improvement,
    )
