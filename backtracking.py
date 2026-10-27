"""
core/backtracking.py

A genuine recursive backtracking CSP solver used as the academic baseline.
Because a 40-60 staff hospital makes naive backtracking expensive, this
solver is bounded by a node limit AND a wall-clock timeout, and is intended
to be run on a smaller benchmark slice (a subset of staff/days) when the
full instance is too large — this is surfaced honestly in the UI, never
faked.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.models import HospitalData, Roster, OFF, all_variables, domain_for
from core.constraints import evaluate_hard_constraints
from core.fairness import compute_fairness


@dataclass
class BacktrackingResult:
    roster: Roster
    success: bool
    nodes_explored: int
    time_seconds: float
    hard_violations: int
    fairness_score: float
    downsized: bool
    note: str = ""


def _order_variables(variables: List[Tuple[str, str, str]], data: HospitalData) -> List[Tuple[str, str, str]]:
    # Order chronologically (staff, then day-in-week-order, then shift-order)
    # rather than alphabetically — plain sorted() would scramble day order
    # (e.g. "Fri" < "Mon" alphabetically), which breaks any adjacency-based
    # reasoning like the forbidden-succession check below.
    day_index = {d: i for i, d in enumerate(data.days)}
    shift_index = {s: i for i, s in enumerate(data.shifts)}
    return sorted(variables, key=lambda v: (v[0], day_index.get(v[1], 0), shift_index.get(v[2], 0)))


def _consistent(var, value, assignment: Dict, data: HospitalData) -> bool:
    staff_id, day, shift = var
    staff = data.staff[staff_id]
    if value == OFF:
        return True
    if day in staff.unavailable_days or (day, shift) in staff.unavailable_slots:
        return False
    if not staff.is_eligible_for(value):
        return False
    reqs = data.requirement_for(value, day, shift)
    if reqs:
        required_skills = {r.required_skill for r in reqs if r.required_skill}
        if required_skills and not (set(staff.skills) & required_skills):
            return False
    # H1: no double booking same day
    for sh in data.shifts:
        if sh != shift and assignment.get((staff_id, day, sh), OFF) != OFF:
            return False
    # H5: max shifts
    count = sum(1 for (s, d, sh2), v in assignment.items() if s == staff_id and v != OFF)
    if count + 1 > staff.max_shifts:
        return False
    # H8: forbidden shift-type succession (cheap single-lookback check; the
    # variable ordering above guarantees the previous day for this staff
    # member — if any — has already been decided). Consecutive-run
    # constraints (H9-H11) are NOT enforced here — see the module docstring
    # and README limitations: a plain static-order backtracking baseline
    # checking multi-day runs at every node is prohibitively slow, so this
    # baseline only re-affirms H8 and leaves H9-H11 to be caught by the
    # final evaluate_hard_constraints() report, same as it does for H3.
    if data.forbidden_successions:
        day_idx = data.days.index(day)
        if day_idx == 0:
            hist = data.history.get(staff_id)
            prev_shift = hist.last_assigned_shift_type if (hist and hist.last_assigned_shift_type not in (None, "None")) else None
        else:
            prev_day = data.days[day_idx - 1]
            prev_shift = None
            for sh in data.shifts:
                if assignment.get((staff_id, prev_day, sh), OFF) != OFF:
                    prev_shift = sh
                    break
        if prev_shift and shift in data.forbidden_successions.get(prev_shift, set()):
            return False
    return True


def backtracking_solve(
    data: HospitalData,
    max_nodes: int = 20000,
    timeout_seconds: float = 8.0,
    max_staff_for_full_search: int = 14,
) -> BacktrackingResult:
    """
    Runs true recursive backtracking. If the full instance has more staff
    than `max_staff_for_full_search`, it is automatically run on a smaller
    benchmark instance (first N staff, all days/shifts) and this is stated
    clearly in the result note — never silently faked as the full solve.
    """
    downsized = False
    note = ""
    work_data = data
    if len(data.staff) > max_staff_for_full_search:
        downsized = True
        subset_ids = data.staff_ids()[:max_staff_for_full_search]
        sub_staff = {sid: data.staff[sid] for sid in subset_ids}
        work_data = HospitalData(
            staff=sub_staff, wards=data.wards, shifts=data.shifts,
            days=data.days, requirements=data.requirements, skills=data.skills,
            source=data.source, seed=data.seed,
            shift_type_rules=data.shift_type_rules,
            forbidden_successions=data.forbidden_successions,
            history=data.history,
            model_coverage=data.model_coverage,
            dataset_label=data.dataset_label,
        )
        note = (f"Full backtracking on {len(data.staff)} staff exceeds the configured "
                f"search budget, so this run used a smaller benchmark instance of "
                f"{max_staff_for_full_search} staff to keep the UI responsive. "
                f"Backtracking baseline uses a reduced benchmark slice for tractability; "
                f"results are not directly equivalent to the full-instance Min-Conflicts run.")

    variables = _order_variables(all_variables(work_data), work_data)
    assignment: Dict[Tuple[str, str, str], str] = {}
    nodes = [0]
    start = time.time()
    result_holder = {"assignment": None}

    def backtrack(idx: int) -> bool:
        if time.time() - start > timeout_seconds:
            return False
        if nodes[0] > max_nodes:
            return False
        if idx == len(variables):
            result_holder["assignment"] = dict(assignment)
            return True
        var = variables[idx]
        staff_id, day, shift = var
        # Try ward assignments before OFF: this makes the baseline actually
        # attempt to staff shifts (rather than trivially satisfying local
        # constraints by leaving everyone unassigned), which is the fair
        # comparison point against Min-Conflicts.
        domain = [v for v in domain_for(var, work_data) if v != OFF] + [OFF]
        for value in domain:
            nodes[0] += 1
            if nodes[0] > max_nodes or time.time() - start > timeout_seconds:
                return False
            if _consistent(var, value, assignment, work_data):
                assignment[var] = value
                if backtrack(idx + 1):
                    return True
                del assignment[var]
        return False

    success = backtrack(0)
    elapsed = time.time() - start

    if success:
        final_roster = result_holder["assignment"]
    else:
        # Partial assignment stands in as the "best effort" roster so the UI
        # can still show violations rather than nothing at all.
        final_roster = {v: assignment.get(v, OFF) for v in variables}
        if not note:
            note = ("No fully feasible solution found within the configured search "
                    "budget (node limit or timeout reached).")

    hard = evaluate_hard_constraints(final_roster, work_data)
    fairness = compute_fairness(final_roster, work_data)

    return BacktrackingResult(
        roster=final_roster,
        success=success,
        nodes_explored=nodes[0],
        time_seconds=elapsed,
        hard_violations=hard.total,
        fairness_score=fairness["combined_fairness_score"],
        downsized=downsized,
        note=note,
    )
