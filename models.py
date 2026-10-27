"""
core/models.py
Core data structures for the SmartRoster AI CSP rostering system.

These are plain dataclasses / lightweight containers used across the
CSP formulation, solvers, repair engine, fairness module and UI.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

DAYS: List[str] = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
WEEKEND_DAYS: Set[str] = {"Sat", "Sun"}

OFF = "OFF"  # sentinel value meaning "not working" in a variable's domain


@dataclass
class Staff:
    staff_id: str
    name: str
    role: str                      # e.g. "HeadNurse" / "Nurse"
    home_ward: str
    skills: List[str] = field(default_factory=list)
    preferred_shifts: List[str] = field(default_factory=list)
    max_shifts: int = 5
    min_shifts: int = 0
    # availability[day] = False means staff CANNOT work that day at all
    unavailable_days: Set[str] = field(default_factory=set)
    # unavailable_slots = set of (day, shift) the staff cannot work (finer grain
    # than unavailable_days; used for shift-off requests and disruptions)
    unavailable_slots: Set[Tuple[str, str]] = field(default_factory=set)
    eligible_wards: List[str] = field(default_factory=list)

    # --- INRC-II contract rules (None/0 for the synthetic generator, where
    # these rules simply don't apply and the related constraints are
    # skipped) ---
    contract_id: Optional[str] = None
    min_consecutive_working_days: Optional[int] = None
    max_consecutive_working_days: Optional[int] = None
    min_consecutive_days_off: Optional[int] = None
    max_consecutive_days_off: Optional[int] = None
    max_working_weekends: Optional[int] = None
    complete_weekends: bool = False

    def is_eligible_for(self, ward: str) -> bool:
        return ward in self.eligible_wards

    def has_skill_for(self, required_skill: str) -> bool:
        return required_skill in self.skills


@dataclass
class HistoricalState:
    """
    A staff member's state carried in from before the planning week —
    parsed from an INRC-II history file (e.g. H0-n012w8-0.json). Used to
    correctly evaluate constraints that span the week boundary, such as a
    forbidden shift succession or a consecutive-working-days run that
    started before day 1 of this planning week.
    """
    staff_id: str
    number_of_assignments: int = 0
    number_of_working_weekends: int = 0
    last_assigned_shift_type: str = "None"
    number_of_consecutive_assignments: int = 0
    number_of_consecutive_working_days: int = 0
    number_of_consecutive_days_off: int = 0


@dataclass
class WardRequirement:
    ward: str
    shift: str
    day: str
    min_staff: int
    optimal_staff: int
    required_skill: Optional[str] = None  # None => any skill acceptable


@dataclass
class HospitalData:
    """Container for one complete problem instance."""
    staff: Dict[str, Staff]
    wards: List[str]
    shifts: List[str]          # e.g. ["Morning", "Evening", "Night"]
    days: List[str]            # e.g. DAYS (7 planning days)
    requirements: List[WardRequirement]
    skills: List[str]
    source: str = "Simulated Hospital Dataset"
    seed: Optional[int] = None

    # --- INRC-II-specific model (empty/neutral for the synthetic generator) ---
    # shift -> (min_consecutive_assignments, max_consecutive_assignments)
    shift_type_rules: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    # preceding shift -> set of shift types forbidden to follow it
    forbidden_successions: Dict[str, Set[str]] = field(default_factory=dict)
    # staff_id -> HistoricalState carried in from before the planning week
    history: Dict[str, "HistoricalState"] = field(default_factory=dict)
    # Human-readable transparency record: which INRC-II fields are actually
    # used by this model and how (Loaded / Enforced / Approximated / Not
    # modelled). Populated by the data loader, shown verbatim in the UI —
    # never silently claim more coverage than the code implements.
    model_coverage: Dict[str, str] = field(default_factory=dict)
    dataset_label: str = ""    # e.g. "INRC-II n012w8 — Week 0"

    def staff_ids(self) -> List[str]:
        return list(self.staff.keys())

    def requirement_for(self, ward: str, day: str, shift: str) -> List[WardRequirement]:
        return [r for r in self.requirements if r.ward == ward and r.day == day and r.shift == shift]


@dataclass
class DisruptionEvent:
    event_id: str
    timestamp: str
    staff_id: str
    event_type: str           # "sick_call" | "leave_request" | "transfer"
    day: str
    shift: Optional[str]      # None for whole-day / transfer events
    ward: Optional[str]
    reason: str
    status: str = "pending"   # pending | applied | resolved


# A roster assignment is represented as a dict:
#   { (staff_id, day, shift): ward_or_OFF }
Roster = Dict[Tuple[str, str, str], str]


def all_variables(data: HospitalData) -> List[Tuple[str, str, str]]:
    """Every (staff, day, shift) variable in the CSP."""
    return [
        (s, d, sh)
        for s in data.staff_ids()
        for d in data.days
        for sh in data.shifts
    ]


def empty_roster(data: HospitalData) -> Roster:
    return {v: OFF for v in all_variables(data)}


def domain_for(var: Tuple[str, str, str], data: HospitalData) -> List[str]:
    """Legal values (wards or OFF) for a given (staff, day, shift) variable."""
    staff_id, day, shift = var
    staff = data.staff[staff_id]
    values = [OFF]
    if day in staff.unavailable_days:
        return values
    if (day, shift) in staff.unavailable_slots:
        return values
    for ward in staff.eligible_wards:
        values.append(ward)
    return values
