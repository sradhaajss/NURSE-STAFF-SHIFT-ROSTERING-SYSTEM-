"""
utils/data_generator.py

Two ways to obtain a HospitalData instance:

1. generate_synthetic_hospital(...) — a self-contained, deterministic random
   generator that builds a ~45-staff / 3-ward / 7-day / 3-shift demo hospital.
   This lets the app run immediately with no external files.

2. load_inrc_benchmark(...) — an adapter for the real INRC-II (Second
   International Nurse Rostering Competition) benchmark files the user
   supplied (Sc-n012w8.json, WD-n012w8-0.json, H0-n012w8-0.json). INRC-II
   instances have no ward concept, so all nurses are placed in a single
   ward ("General Ward") and the shift types are the authentic
   Early/Day/Late/Night INRC labels. This is clearly labelled in the UI as
   real benchmark data, not synthetic.
"""
from __future__ import annotations

import json
import random
from typing import Dict, List, Optional

from core.models import HospitalData, Staff, WardRequirement, HistoricalState, DAYS

WARDS = ["Emergency", "General", "ICU"]
SHIFTS = ["Morning", "Evening", "Night"]
SKILLS = ["General", "Emergency", "ICU", "Pediatric", "Critical Care"]

FIRST_NAMES = [
    "Aarav", "Meera", "Rohan", "Ananya", "Kabir", "Diya", "Vihaan", "Isha",
    "Arjun", "Kavya", "Dev", "Priya", "Sanjay", "Nisha", "Rahul", "Tara",
    "Vikram", "Neha", "Aditya", "Pooja", "Karan", "Ritu", "Amit", "Sneha",
    "Rajesh", "Divya", "Manoj", "Swati", "Ajay", "Lakshmi", "Suresh", "Anita",
    "Gopal", "Radha", "Naveen", "Shreya", "Harish", "Deepa", "Vinod", "Kiran",
    "Ashok", "Meenal", "Ravi", "Nandini", "Sameer",
]


def _std(values: List[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var ** 0.5


def generate_synthetic_hospital(
    num_staff: int = 45,
    num_wards: int = 3,
    planning_days: int = 7,
    shifts_per_day: int = 3,
    seed: int = 42,
) -> HospitalData:
    rng = random.Random(seed)

    wards = WARDS[:num_wards] if num_wards <= len(WARDS) else WARDS + [f"Ward{i}" for i in range(len(WARDS), num_wards)]
    shifts = SHIFTS[:shifts_per_day] if shifts_per_day <= len(SHIFTS) else SHIFTS + [f"Shift{i}" for i in range(len(SHIFTS), shifts_per_day)]
    days = DAYS[:planning_days] if planning_days <= len(DAYS) else (DAYS * ((planning_days // 7) + 1))[:planning_days]

    ward_required_skill = {
        "Emergency": "Emergency",
        "General": "General",
        "ICU": "ICU",
    }

    staff: Dict[str, Staff] = {}
    for i in range(num_staff):
        sid = f"N{i+1:03d}"
        name = FIRST_NAMES[i % len(FIRST_NAMES)]
        home_ward = wards[i % len(wards)]

        # ~35% of staff are multi-skilled and can float between wards
        multi_skilled = rng.random() < 0.35
        skills = [ward_required_skill.get(home_ward, "General")]
        eligible_wards = [home_ward]
        if multi_skilled:
            extra_ward = rng.choice([w for w in wards if w != home_ward])
            skills.append(ward_required_skill.get(extra_ward, "General"))
            eligible_wards.append(extra_ward)
        if rng.random() < 0.15:
            skills.append(rng.choice(["Pediatric", "Critical Care"]))

        role = "HeadNurse" if rng.random() < 0.2 else "Nurse"
        preferred = rng.sample(shifts, k=rng.randint(1, min(2, len(shifts))))
        max_shifts = rng.choice([4, 4, 5, 5])  # tight enough for Min-Conflicts to have real repair work

        # Random unavailability: ~18% chance of 1 unavailable day (keeps the
        # instance tight enough that Min-Conflicts has genuine repair work
        # to do, rather than solving in a single greedy pass)
        unavailable_days = set()
        if rng.random() < 0.12:
            unavailable_days.add(rng.choice(days))

        staff[sid] = Staff(
            staff_id=sid,
            name=name,
            role=role,
            home_ward=home_ward,
            skills=skills,
            preferred_shifts=preferred,
            max_shifts=max_shifts,
            min_shifts=max(2, max_shifts - 3),
            unavailable_days=unavailable_days,
            eligible_wards=eligible_wards,
        )

    # Minimum staffing requirements per ward/shift/day
    requirements: List[WardRequirement] = []
    base_min = {"Morning": 3, "Evening": 3, "Night": 2}
    base_opt = {"Morning": 4, "Evening": 4, "Night": 3}
    for ward in wards:
        for day in days:
            weekend_factor = 0 if day in ("Sat", "Sun") else 1
            for shift in shifts:
                mn = max(1, base_min.get(shift, 2) - (0 if weekend_factor else 1))
                op = max(mn, base_opt.get(shift, 3) - (0 if weekend_factor else 1))
                requirements.append(
                    WardRequirement(
                        ward=ward, shift=shift, day=day,
                        min_staff=mn, optimal_staff=op,
                        required_skill=ward_required_skill.get(ward),
                    )
                )

    return HospitalData(
        staff=staff, wards=wards, shifts=shifts, days=days,
        requirements=requirements, skills=SKILLS,
        source="Simulated Hospital Dataset", seed=seed,
        dataset_label="Simulated Hospital Dataset",
        model_coverage={
            "Scenario data": "N/A — synthetic generator, no scenario file",
            "Historical state": "N/A — synthetic generator has no prior-week history",
            "Forbidden shift successions": "N/A — not defined for the synthetic generator",
            "Contract consecutive-day/weekend rules": "N/A — synthetic staff use a simple max/min shift count only",
        },
    )


# ---------------------------------------------------------------------------
# INRC-II real benchmark adapter
# ---------------------------------------------------------------------------

_INRC_DAY_MAP = {
    "Monday": "Mon", "Tuesday": "Tue", "Wednesday": "Wed", "Thursday": "Thu",
    "Friday": "Fri", "Saturday": "Sat", "Sunday": "Sun",
}
_INRC_SINGLE_WARD = "General Ward (INRC-II)"


def load_inrc_benchmark(scenario_path: str, week_data_path: str,
                         history_path: str) -> HospitalData:
    """
    Load a real INRC-II (Second International Nurse Rostering Competition)
    instance from all three official file types:

      Sc-n012w8.json (scenario: nurses, skills, shift types, forbidden shift
                      successions, contracts)
      WD-n012w8-0.json (week 0 demand: min/optimal staffing, shift-off
                        requests)
      H0-n012w8-0.json (history: each nurse's state entering week 0 —
                        consecutive working days/off, last shift type,
                        working-weekend count so far)

    INRC-II has no ward dimension, so all nurses/requirements are placed
    into one virtual ward. Only week 0 (7 days) is loaded as the planning
    horizon; this is a SINGLE-WEEK prototype, not a full 8-week competition
    solve — see HospitalData.model_coverage for exactly what is and isn't
    modelled.
    """
    with open(scenario_path, "r") as f:
        scenario = json.load(f)
    with open(week_data_path, "r") as f:
        week = json.load(f)
    with open(history_path, "r") as f:
        history_raw = json.load(f)

    shift_types = [st["id"] for st in scenario["shiftTypes"]]
    skills = scenario["skills"]
    contracts = {c["id"]: c for c in scenario["contracts"]}

    # Per-shift-type consecutive-assignment rule, straight from the scenario.
    shift_type_rules = {
        st["id"]: (st["minimumNumberOfConsecutiveAssignments"],
                    st["maximumNumberOfConsecutiveAssignments"])
        for st in scenario["shiftTypes"]
    }

    # Forbidden successions: precedingShiftType -> shift types that may NOT
    # follow it the next day (e.g. Night cannot be followed by Early/Day/Late
    # in this scenario — not enough rest). Taken verbatim from the scenario,
    # never hard-coded.
    forbidden_successions = {
        row["precedingShiftType"]: set(row["succeedingShiftTypes"])
        for row in scenario["forbiddenShiftTypeSuccessions"]
    }

    # History: each nurse's carried-in state before week 0.
    history: Dict[str, HistoricalState] = {}
    for h in history_raw.get("nurseHistory", []):
        history[h["nurse"]] = HistoricalState(
            staff_id=h["nurse"],
            number_of_assignments=h.get("numberOfAssignments", 0),
            number_of_working_weekends=h.get("numberOfWorkingWeekends", 0),
            last_assigned_shift_type=h.get("lastAssignedShiftType", "None"),
            number_of_consecutive_assignments=h.get("numberOfConsecutiveAssignments", 0),
            number_of_consecutive_working_days=h.get("numberOfConsecutiveWorkingDays", 0),
            number_of_consecutive_days_off=h.get("numberOfConsecutiveDaysOff", 0),
        )

    staff: Dict[str, Staff] = {}
    for n in scenario["nurses"]:
        contract = contracts.get(n["contract"], {})
        staff[n["id"]] = Staff(
            staff_id=n["id"],
            name=n["id"],
            role="HeadNurse" if "HeadNurse" in n["skills"] else "Nurse",
            home_ward=_INRC_SINGLE_WARD,
            skills=list(n["skills"]),
            preferred_shifts=[],
            # NOTE: minimumNumberOfAssignments / maximumNumberOfAssignments in
            # the scenario are defined over the FULL 8-week competition
            # horizon, not one week. This single-week prototype cannot
            # enforce the real multi-week total directly (it has no
            # visibility into the other 7 weeks), so it approximates a
            # weekly cap by dividing the 8-week bound by 8. This is a
            # deliberate, documented approximation — NOT the real INRC-II
            # rule — and the UI/README state this explicitly.
            max_shifts=_week_cap(contract),
            min_shifts=_week_floor(contract),
            eligible_wards=[_INRC_SINGLE_WARD],
            contract_id=n["contract"],
            min_consecutive_working_days=contract.get("minimumNumberOfConsecutiveWorkingDays"),
            max_consecutive_working_days=contract.get("maximumNumberOfConsecutiveWorkingDays"),
            min_consecutive_days_off=contract.get("minimumNumberOfConsecutiveDaysOff"),
            max_consecutive_days_off=contract.get("maximumNumberOfConsecutiveDaysOff"),
            max_working_weekends=contract.get("maximumNumberOfWorkingWeekends"),
            complete_weekends=bool(contract.get("completeWeekends", 0)),
        )

    # Shift-off requests -> per-slot unavailability
    for req in week.get("shiftOffRequests", []):
        nurse = req["nurse"]
        day = _INRC_DAY_MAP.get(req["day"], req["day"])
        shift = req["shiftType"]
        if nurse not in staff:
            continue
        if shift == "Any":
            staff[nurse].unavailable_days.add(day)
        else:
            staff[nurse].unavailable_slots.add((day, shift))

    # Requirements -> WardRequirement rows in the single virtual ward
    requirements: List[WardRequirement] = []
    for row in week.get("requirements", []):
        shift = row["shiftType"]
        skill = row["skill"]
        for full_day, short_day in _INRC_DAY_MAP.items():
            key = f"requirementOn{full_day}"
            if key not in row:
                continue
            mn = row[key]["minimum"]
            op = row[key]["optimal"]
            requirements.append(
                WardRequirement(
                    ward=_INRC_SINGLE_WARD, shift=shift, day=short_day,
                    min_staff=mn, optimal_staff=op, required_skill=skill,
                )
            )

    model_coverage = {
        "Scenario data (skills, shift types, contracts)": "Loaded",
        "Nurse skills": "Loaded",
        "Contracts": "Loaded",
        "Weekly requirements (min/optimal staffing)": "Loaded",
        "Shift-off requests": "Loaded",
        "Historical state (H0)": "Loaded",
        "Forbidden shift successions (e.g. Night → Early)": "Enforced as a hard constraint (H8), using H0 for the week-1 boundary",
        "Max consecutive working days": "Enforced as a hard constraint (H9), using H0 as carry-in",
        "Max consecutive same-shift-type assignments": "Enforced as a hard constraint (H10), using H0 as carry-in",
        "Max working weekends": "Enforced as a hard constraint (H11), using H0 as carry-in",
        "Min consecutive working days": "Modelled as a soft preference (S6) — not enforced as hard, see limitations",
        "Min consecutive days off": "Modelled as a soft preference (S7) — not enforced as hard, see limitations",
        "Complete weekends": "Modelled as a soft preference (S8) — not enforced as hard, see limitations",
        "Min/Max total assignments per nurse": "Approximated as a weekly-scaled cap from the 8-week contract bound (see code comment) — not the real multi-week rule",
        "Multi-week rotation / full 8-week competition scoring": "Not modelled — this is a single planning-week prototype",
    }

    data = HospitalData(
        staff=staff,
        wards=[_INRC_SINGLE_WARD],
        shifts=shift_types,
        days=DAYS,  # Mon..Sun
        requirements=requirements,
        skills=skills,
        source=f"Real INRC-II Benchmark ({scenario.get('id', 'unknown')}) — Week {history_raw.get('week', 0)}, single planning week",
        seed=None,
        shift_type_rules=shift_type_rules,
        forbidden_successions=forbidden_successions,
        history=history,
        model_coverage=model_coverage,
        dataset_label=f"INRC-II {scenario.get('id', 'unknown')} — Week {history_raw.get('week', 0)}",
    )
    return data


def _week_cap(contract: dict) -> int:
    """Convert an 8-week contract cap into an approximate 1-week cap."""
    total = contract.get("maximumNumberOfAssignments", 40)
    return max(3, round(total / 8))


def _week_floor(contract: dict) -> int:
    total = contract.get("minimumNumberOfAssignments", 14)
    return max(1, round(total / 8))
