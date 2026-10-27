"""
core/constraints.py

Hard-constraint (feasibility) and soft-constraint (quality/fairness)
evaluation for a roster. These functions are pure and reusable by every
solver (backtracking, min-conflicts, repair) and by the UI.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

from core.models import HospitalData, Roster, OFF, WEEKEND_DAYS


@dataclass
class ViolationReport:
    total: int = 0
    by_category: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    details: List[str] = field(default_factory=list)

    def add(self, category: str, message: str, weight: int = 1):
        self.total += weight
        self.by_category[category] += weight
        if len(self.details) < 500:  # cap detail list to keep UI responsive
            self.details.append(message)


def evaluate_hard_constraints(roster: Roster, data: HospitalData) -> ViolationReport:
    """
    Checks H1-H11:
      H1  no two shifts same day
      H2  not assigned when unavailable
      H3  minimum ward/shift staffing met
      H4  staff has required skill for the ward/skill requirement
      H5  max shifts not exceeded
      H6  staff assigned only to eligible wards
      H7  staff on leave/sick must not remain assigned (handled by
          unavailable_slots / unavailable_days, same check as H2)
      H8  forbidden shift-type succession (INRC-II scenario data; e.g. a
          Night shift cannot be immediately followed by an Early shift)
      H9  maximum consecutive working days (INRC-II contract), using H0
          history as carry-in for the start of the planning week
      H10 maximum consecutive assignments of the same shift type (INRC-II
          shift-type rule), using H0 history as carry-in
      H11 maximum working weekends (INRC-II contract), using H0 history as
          carry-in

      H8-H11 only fire when the loaded dataset actually supplies the
      relevant INRC-II data (forbidden_successions / shift_type_rules /
      contract fields) — they are silently inactive for the synthetic
      generator, which doesn't model these rules.
    """
    report = ViolationReport()

    # H1: no two shifts on the same day
    for staff_id in data.staff_ids():
        for day in data.days:
            worked = [sh for sh in data.shifts if roster.get((staff_id, day, sh), OFF) != OFF]
            if len(worked) > 1:
                report.add("H1_double_booking",
                            f"{staff_id} assigned to {len(worked)} shifts on {day} ({', '.join(worked)})",
                            weight=len(worked) - 1)

    # H2 / H7: unavailability (covers sick/leave/disruption too)
    for (staff_id, day, shift), ward in roster.items():
        if ward == OFF:
            continue
        staff = data.staff[staff_id]
        if day in staff.unavailable_days or (day, shift) in staff.unavailable_slots:
            report.add("H2_unavailable_assigned",
                        f"{staff_id} assigned on {day}/{shift} while unavailable")

    # H4 & H6: skill eligibility and ward eligibility
    for (staff_id, day, shift), ward in roster.items():
        if ward == OFF:
            continue
        staff = data.staff[staff_id]
        if not staff.is_eligible_for(ward):
            report.add("H6_ward_ineligible",
                        f"{staff_id} assigned to {ward} on {day}/{shift} but not eligible for that ward")
        reqs = data.requirement_for(ward, day, shift)
        if reqs:
            required_skills = {r.required_skill for r in reqs if r.required_skill}
            if required_skills and not (set(staff.skills) & required_skills):
                report.add("H4_skill_mismatch",
                            f"{staff_id} lacks required skill for {ward} {day}/{shift}")

    # H5: max shifts
    shift_counts = defaultdict(int)
    for (staff_id, day, shift), ward in roster.items():
        if ward != OFF:
            shift_counts[staff_id] += 1
    for staff_id, staff in data.staff.items():
        if shift_counts[staff_id] > staff.max_shifts:
            report.add("H5_max_shifts_exceeded",
                        f"{staff_id} assigned {shift_counts[staff_id]} shifts, exceeds max {staff.max_shifts}",
                        weight=shift_counts[staff_id] - staff.max_shifts)

    # H3: minimum staffing per ward/day/shift/skill
    for ward in data.wards:
        for day in data.days:
            for shift in data.shifts:
                reqs = data.requirement_for(ward, day, shift)
                if not reqs:
                    continue
                assigned_staff_ids = [
                    s for s in data.staff_ids()
                    if roster.get((s, day, shift), OFF) == ward
                ]
                for req in reqs:
                    if req.required_skill:
                        count = sum(1 for s in assigned_staff_ids if req.required_skill in data.staff[s].skills)
                    else:
                        count = len(assigned_staff_ids)
                    if count < req.min_staff:
                        report.add("H3_understaffed",
                                    f"{ward} {day}/{shift} skill={req.required_skill or 'any'}: "
                                    f"{count}/{req.min_staff} minimum staff",
                                    weight=(req.min_staff - count))

    # H8-H11: INRC-II sequence/history constraints (no-ops when the dataset
    # doesn't supply the relevant scenario/contract data).
    _check_forbidden_successions(roster, data, report)
    _check_max_consecutive_working_days(roster, data, report)
    _check_max_consecutive_shift_type(roster, data, report)
    _check_max_working_weekends(roster, data, report)

    return report


def _shift_worked_on(staff_id: str, day: str, data: HospitalData, roster: Roster):
    """Returns the shift type worked by staff_id on `day`, or None if OFF."""
    for sh in data.shifts:
        if roster.get((staff_id, day, sh), OFF) != OFF:
            return sh
    return None


def _check_forbidden_successions(roster: Roster, data: HospitalData, report: ViolationReport):
    if not data.forbidden_successions:
        return  # dataset doesn't define any (e.g. synthetic generator)
    for staff_id in data.staff_ids():
        hist = data.history.get(staff_id)
        prev_shift = None
        if hist and hist.last_assigned_shift_type not in (None, "None"):
            prev_shift = hist.last_assigned_shift_type
        for day in data.days:
            today_shift = _shift_worked_on(staff_id, day, data, roster)
            if prev_shift and today_shift:
                if today_shift in data.forbidden_successions.get(prev_shift, set()):
                    report.add("H8_forbidden_succession",
                                f"{staff_id}: {prev_shift} → {today_shift} into {day} is a forbidden succession")
            prev_shift = today_shift  # a day OFF resets the chain (rest was taken)


def _check_max_consecutive_working_days(roster: Roster, data: HospitalData, report: ViolationReport):
    for staff_id, staff in data.staff.items():
        max_days = staff.max_consecutive_working_days
        if not max_days:
            continue  # rule not defined for this dataset (e.g. synthetic)
        hist = data.history.get(staff_id)
        run = hist.number_of_consecutive_working_days if hist else 0
        for day in data.days:
            working = _shift_worked_on(staff_id, day, data, roster) is not None
            if working:
                run += 1
                if run > max_days:
                    report.add("H9_max_consecutive_working_days",
                                f"{staff_id} exceeds max {max_days} consecutive working days "
                                f"(through {day}, run={run})")
            else:
                run = 0


def _check_max_consecutive_shift_type(roster: Roster, data: HospitalData, report: ViolationReport):
    if not data.shift_type_rules:
        return  # rule not defined for this dataset
    for staff_id, staff in data.staff.items():
        hist = data.history.get(staff_id)
        run_shift = hist.last_assigned_shift_type if hist else "None"
        run_len = hist.number_of_consecutive_assignments if hist else 0
        for day in data.days:
            today_shift = _shift_worked_on(staff_id, day, data, roster)
            if today_shift is None:
                run_shift, run_len = "None", 0
                continue
            if today_shift == run_shift:
                run_len += 1
            else:
                run_shift, run_len = today_shift, 1
            max_consec = data.shift_type_rules.get(today_shift, (0, 10**6))[1]
            if run_len > max_consec:
                report.add("H10_max_consecutive_shift_type",
                            f"{staff_id} exceeds max {max_consec} consecutive {today_shift} shifts "
                            f"(through {day}, run={run_len})")


def _check_max_working_weekends(roster: Roster, data: HospitalData, report: ViolationReport):
    for staff_id, staff in data.staff.items():
        if not staff.max_working_weekends:
            continue  # rule not defined for this dataset
        hist = data.history.get(staff_id)
        count = hist.number_of_working_weekends if hist else 0
        worked_this_week = any(
            roster.get((staff_id, d, sh), OFF) != OFF
            for d in data.days if d in WEEKEND_DAYS for sh in data.shifts
        )
        if worked_this_week:
            count += 1
        if count > staff.max_working_weekends:
            report.add("H11_max_working_weekends",
                        f"{staff_id} exceeds max {staff.max_working_weekends} working weekends (count={count})")


def evaluate_soft_constraints(roster: Roster, data: HospitalData) -> ViolationReport:
    """
    S1 balanced total shifts, S2 balanced night shifts, S3 balanced weekend
    shifts, S4 preference satisfaction — these feed the fairness module.

    S5-S8 are additional INRC-II-derived soft preferences that this
    single-week prototype does NOT enforce as hard constraints (see
    HospitalData.model_coverage for why), but does still track and penalize
    softly so the solver is nudged toward respecting them where possible:
      S5 reaching the (weekly-approximated) minimum total assignments
      S6 minimum consecutive working days per work stretch
      S7 minimum consecutive days off per rest stretch
      S8 complete weekends (work both Sat & Sun, or neither)

    Returns a report whose 'total' is a soft-violation *count*; fairness
    itself is scored separately by core/fairness.py.
    """
    report = ViolationReport()

    for staff_id, staff in data.staff.items():
        if staff.preferred_shifts:
            for day in data.days:
                for sh in data.shifts:
                    if roster.get((staff_id, day, sh), OFF) != OFF and sh not in staff.preferred_shifts:
                        report.add("S4_preference_miss",
                                    f"{staff_id} worked non-preferred shift {sh} on {day}", weight=1)

    # S5: minimum total assignments (only meaningful where min_shifts > 0)
    for staff_id, staff in data.staff.items():
        if staff.min_shifts and staff.min_shifts > 0:
            worked = sum(1 for day in data.days for sh in data.shifts
                         if roster.get((staff_id, day, sh), OFF) != OFF)
            if worked < staff.min_shifts:
                report.add("S5_below_min_assignments",
                            f"{staff_id} worked {worked} shifts, below target minimum {staff.min_shifts}",
                            weight=(staff.min_shifts - worked))

    # S6/S7: minimum consecutive working-day / days-off run length. Only
    # checked for runs fully bounded by the other state within this planning
    # week (a run touching day 1 or day 7 may continue into the adjacent
    # week, which this single-week prototype cannot see, so those edge runs
    # are left unchecked rather than risk a false positive).
    for staff_id, staff in data.staff.items():
        if not (staff.min_consecutive_working_days or staff.min_consecutive_days_off):
            continue
        working_flags = [_shift_worked_on(staff_id, day, data, roster) is not None for day in data.days]
        # scan runs, skipping ones touching either edge of the week
        i = 0
        n = len(working_flags)
        while i < n:
            j = i
            while j < n and working_flags[j] == working_flags[i]:
                j += 1
            run_len = j - i
            touches_edge = (i == 0) or (j == n)
            if not touches_edge:
                if working_flags[i] and staff.min_consecutive_working_days and run_len < staff.min_consecutive_working_days:
                    report.add("S6_short_working_stretch",
                                f"{staff_id} has a {run_len}-day working stretch (days "
                                f"{data.days[i]}-{data.days[j-1]}), below preferred minimum "
                                f"{staff.min_consecutive_working_days}")
                if not working_flags[i] and staff.min_consecutive_days_off and run_len < staff.min_consecutive_days_off:
                    report.add("S7_short_rest_stretch",
                                f"{staff_id} has a {run_len}-day rest stretch (days "
                                f"{data.days[i]}-{data.days[j-1]}), below preferred minimum "
                                f"{staff.min_consecutive_days_off}")
            i = j

    # S8: complete weekends (work both Saturday & Sunday, or neither)
    for staff_id, staff in data.staff.items():
        if not staff.complete_weekends:
            continue
        if "Sat" not in data.days or "Sun" not in data.days:
            continue
        works_sat = _shift_worked_on(staff_id, "Sat", data, roster) is not None
        works_sun = _shift_worked_on(staff_id, "Sun", data, roster) is not None
        if works_sat != works_sun:
            report.add("S8_incomplete_weekend",
                        f"{staff_id} works only one weekend day (Sat={works_sat}, Sun={works_sun})")

    return report


def total_shift_counts(roster: Roster, data: HospitalData) -> Dict[str, int]:
    counts = {s: 0 for s in data.staff_ids()}
    for (staff_id, day, shift), ward in roster.items():
        if ward != OFF:
            counts[staff_id] += 1
    return counts


def compute_cost(
    roster: Roster,
    data: HospitalData,
    hard_weight: int = 1000,
    soft_weight: int = 10,
    fairness_weight: int = 5,
) -> Dict[str, float]:
    """
    TOTAL COST = hard_weight * hard_violations
               + soft_weight * soft_violations
               + fairness_weight * fairness_penalty
    Returns a breakdown dict so the UI can show each component separately.
    """
    from core.fairness import compute_fairness  # local import avoids cycle

    hard = evaluate_hard_constraints(roster, data)
    soft = evaluate_soft_constraints(roster, data)
    fairness = compute_fairness(roster, data)

    hard_penalty = hard_weight * hard.total
    soft_penalty = soft_weight * soft.total
    fairness_penalty = fairness_weight * fairness["combined_fairness_score"]

    return {
        "hard_violations": hard.total,
        "soft_violations": soft.total,
        "hard_penalty": hard_penalty,
        "soft_penalty": soft_penalty,
        "fairness_penalty": fairness_penalty,
        "total_cost": hard_penalty + soft_penalty + fairness_penalty,
        "hard_report": hard,
        "soft_report": soft,
        "fairness": fairness,
    }


def conflicts_for_variable(var: Tuple[str, str, str], value: str, roster: Roster, data: HospitalData) -> int:
    """
    Fast, local conflict count used by Min-Conflicts: how many hard
    constraints would be violated if `var` were set to `value`, given the
    rest of the roster held fixed. This is a cheap approximation (checks
    only constraints touching this variable) so it can be called thousands
    of times per solve.

    Covers H1, H2/H7, H4, H5, H6 always, plus H8/H9/H10/H11 whenever the
    loaded dataset actually supplies the relevant INRC-II scenario/contract
    data (they are no-ops otherwise, e.g. for the synthetic generator) —
    this keeps the solver's local scoring consistent with what
    evaluate_hard_constraints checks globally.
    """
    staff_id, day, shift = var
    staff = data.staff[staff_id]
    conflicts = 0

    if value != OFF:
        # H2/H7 unavailability
        if day in staff.unavailable_days or (day, shift) in staff.unavailable_slots:
            conflicts += 1
        # H6 ward eligibility
        if not staff.is_eligible_for(value):
            conflicts += 1
        # H4 skill match
        reqs = data.requirement_for(value, day, shift)
        if reqs:
            required_skills = {r.required_skill for r in reqs if r.required_skill}
            if required_skills and not (set(staff.skills) & required_skills):
                conflicts += 1
        # H1 double booking: any other shift this day already filled?
        for sh in data.shifts:
            if sh != shift and roster.get((staff_id, day, sh), OFF) != OFF:
                conflicts += 1
        # H5 max shifts: would this push staff over the cap?
        current_count = sum(
            1 for d in data.days for sh in data.shifts
            if (d, sh) != (day, shift) and roster.get((staff_id, d, sh), OFF) != OFF
        )
        if current_count + 1 > staff.max_shifts:
            conflicts += 1

        day_idx = data.days.index(day)

        # H8 forbidden succession — check both the day before and the day
        # after this assignment (a local move can create a bad transition
        # on either side).
        if data.forbidden_successions:
            prev_shift = _local_shift_before(staff_id, day_idx, data, roster)
            if prev_shift and shift in data.forbidden_successions.get(prev_shift, set()):
                conflicts += 1
            next_shift = _local_shift_after(staff_id, day_idx, data, roster)
            if next_shift and next_shift in data.forbidden_successions.get(shift, set()):
                conflicts += 1

        # H9 max consecutive working days — count the run through this day
        # in both directions plus H0 carry-in if the run reaches day 1.
        if staff.max_consecutive_working_days:
            run = 1
            d = day_idx - 1
            hit_start = False
            while d >= 0:
                if _shift_worked_on(staff_id, data.days[d], data, roster) is not None:
                    run += 1
                    d -= 1
                else:
                    break
            else:
                hit_start = True
            if hit_start:
                hist = data.history.get(staff_id)
                if hist:
                    run += hist.number_of_consecutive_working_days
            d = day_idx + 1
            while d < len(data.days) and _shift_worked_on(staff_id, data.days[d], data, roster) is not None:
                run += 1
                d += 1
            if run > staff.max_consecutive_working_days:
                conflicts += 1

        # H10 max consecutive same-shift-type run (backward + H0 carry-in;
        # forward direction skipped for simplicity/performance since same-
        # shift-type runs are rarer to extend forward in local search).
        if shift in data.shift_type_rules:
            max_consec = data.shift_type_rules[shift][1]
            run = 1
            d = day_idx - 1
            hit_start = False
            while d >= 0:
                if roster.get((staff_id, data.days[d], shift), OFF) != OFF:
                    run += 1
                    d -= 1
                else:
                    break
            else:
                hit_start = True
            if hit_start:
                hist = data.history.get(staff_id)
                if hist and hist.last_assigned_shift_type == shift:
                    run += hist.number_of_consecutive_assignments
            if run > max_consec:
                conflicts += 1

        # H11 max working weekends
        if staff.max_working_weekends and day in WEEKEND_DAYS:
            other_day = "Sun" if day == "Sat" else "Sat"
            already_working_other = (
                other_day in data.days
                and _shift_worked_on(staff_id, other_day, data, roster) is not None
            )
            if not already_working_other:
                hist = data.history.get(staff_id)
                carry = hist.number_of_working_weekends if hist else 0
                if carry + 1 > staff.max_working_weekends:
                    conflicts += 1

    # H3 understaffing is a *global* constraint; Min-Conflicts handles it by
    # scoring candidate values on how much they help fill understaffed
    # slots (see min_conflicts.py), rather than penalizing here per-variable.
    return conflicts


def _local_shift_before(staff_id: str, day_idx: int, data: HospitalData, roster: Roster):
    """Shift worked the calendar day before day_idx, or H0 history's last
    shift if day_idx is the first planning day."""
    if day_idx == 0:
        hist = data.history.get(staff_id)
        if hist and hist.last_assigned_shift_type not in (None, "None"):
            return hist.last_assigned_shift_type
        return None
    return _shift_worked_on(staff_id, data.days[day_idx - 1], data, roster)


def _local_shift_after(staff_id: str, day_idx: int, data: HospitalData, roster: Roster):
    """Shift worked the calendar day after day_idx, or None past the
    planning horizon (this single-week prototype has no visibility into
    the following week)."""
    if day_idx >= len(data.days) - 1:
        return None
    return _shift_worked_on(staff_id, data.days[day_idx + 1], data, roster)
