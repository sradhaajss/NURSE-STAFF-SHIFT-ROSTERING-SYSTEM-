"""
core/evaluation.py

Runs controlled experiments comparing Backtracking, Min-Conflicts, and
Warm-Start Repair vs Full Re-Solve, across a set of random seeds. Every
number is produced by an actual execution — nothing here is hardcoded.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

from core.models import HospitalData
from core.backtracking import backtracking_solve
from core.min_conflicts import min_conflicts_solve
from core.events import generate_demo_events, apply_event, affected_variables
from core.repair import compare_full_resolve_vs_warm_start
import copy


@dataclass
class EvaluationRun:
    seed: int
    backtracking_time: float
    backtracking_violations: int
    backtracking_nodes: int
    min_conflicts_time: float
    min_conflicts_violations: int
    min_conflicts_iterations: int
    min_conflicts_fairness: float
    full_resolve_time: float
    warm_start_time: float
    speed_improvement_pct: float
    changed_assignments: int


def run_evaluation(data: HospitalData, seeds: List[int]) -> List[EvaluationRun]:
    runs = []
    for seed in seeds:
        bt = backtracking_solve(data, max_nodes=8000, timeout_seconds=5.0)
        mc = min_conflicts_solve(data, max_iterations=1500, restarts=2, seed=seed)

        # Apply one disruption event to a fresh copy of the data/roster for
        # the full-resolve vs warm-start comparison.
        data_copy = copy.deepcopy(data)
        events = generate_demo_events(data_copy, seed=seed, count=1)
        event = events[0]
        mc2 = min_conflicts_solve(data_copy, max_iterations=1500, restarts=2, seed=seed)
        affected = affected_variables(event, data_copy)
        apply_event(event, data_copy)
        comparison = compare_full_resolve_vs_warm_start(mc2.roster, affected, data_copy)

        runs.append(EvaluationRun(
            seed=seed,
            backtracking_time=bt.time_seconds,
            backtracking_violations=bt.hard_violations,
            backtracking_nodes=bt.nodes_explored,
            min_conflicts_time=mc.time_seconds,
            min_conflicts_violations=mc.hard_violations,
            min_conflicts_iterations=mc.iterations_used,
            min_conflicts_fairness=mc.fairness_score,
            full_resolve_time=comparison.full_resolve.time_seconds,
            warm_start_time=comparison.warm_start.time_seconds,
            speed_improvement_pct=comparison.speed_improvement_pct,
            changed_assignments=comparison.warm_start.assignments_changed,
        ))
    return runs
