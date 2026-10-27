"""
core/fairness.py

Fairness tracking: total / night / weekend shift distribution per staff
member and their standard deviations. LOWER standard deviation = more
balanced/fair distribution across staff.
"""
from __future__ import annotations

from typing import Dict

from core.models import HospitalData, Roster, OFF, WEEKEND_DAYS


def _std(values):
    values = list(values)
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var ** 0.5


def compute_fairness(roster: Roster, data: HospitalData) -> Dict:
    night_shift = data.shifts[-1] if data.shifts else None

    total_shifts = {s: 0 for s in data.staff_ids()}
    night_shifts = {s: 0 for s in data.staff_ids()}
    weekend_shifts = {s: 0 for s in data.staff_ids()}

    for (staff_id, day, shift), ward in roster.items():
        if ward == OFF:
            continue
        total_shifts[staff_id] += 1
        if shift == night_shift:
            night_shifts[staff_id] += 1
        if day in WEEKEND_DAYS:
            weekend_shifts[staff_id] += 1

    night_std = _std(night_shifts.values())
    weekend_std = _std(weekend_shifts.values())
    total_std = _std(total_shifts.values())

    # Combined fairness score: lower is better. Simple weighted sum of the
    # two metrics the project brief calls out explicitly (night & weekend
    # std dev), plus a small term for overall shift-count balance.
    combined = (2.0 * night_std) + (2.0 * weekend_std) + (1.0 * total_std)

    return {
        "total_shifts": total_shifts,
        "night_shifts": night_shifts,
        "weekend_shifts": weekend_shifts,
        "total_shifts_std": total_std,
        "night_shifts_std": night_std,
        "weekend_shifts_std": weekend_std,
        "combined_fairness_score": combined,
    }
