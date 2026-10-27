"""
core/events.py

Live availability / disruption event simulation: sick calls, approved
leave, and sudden transfers. Timestamps are generated at creation/apply
time (real wall-clock times), never hardcoded.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

from core.models import HospitalData, DisruptionEvent


def _now() -> str:
    return datetime.now().strftime("%H:%M:%S")


def generate_demo_events(data: HospitalData, seed: int = 3, count: int = 3) -> List[DisruptionEvent]:
    rng = random.Random(seed)
    staff_ids = data.staff_ids()
    events = []

    event_specs = [
        ("sick_call", "Sudden sick call reported by staff member"),
        ("leave_request", "Approved personal leave request"),
        ("transfer", "Sudden inter-ward transfer directive"),
    ]

    for i in range(count):
        event_type, reason = event_specs[i % len(event_specs)]
        staff_id = rng.choice(staff_ids)
        day = rng.choice(data.days)
        shift = rng.choice(data.shifts) if event_type != "leave_request" else None
        ward = None
        if event_type == "transfer":
            eligible = data.staff[staff_id].eligible_wards
            others = [w for w in data.wards if w not in eligible] or data.wards
            ward = rng.choice(others)

        events.append(DisruptionEvent(
            event_id=f"EVT{i+1:03d}",
            timestamp=_now(),
            staff_id=staff_id,
            event_type=event_type,
            day=day,
            shift=shift,
            ward=ward,
            reason=reason,
            status="pending",
        ))
    return events


def apply_event(event: DisruptionEvent, data: HospitalData) -> HospitalData:
    """
    Mutates a *copy* of the staff's availability to reflect the disruption.
    Returns the same HospitalData object with staff availability updated
    in place (staff dict is shared by reference intentionally — callers
    should have already snapshotted the roster before calling this).
    """
    staff = data.staff.get(event.staff_id)
    if staff is None:
        return data

    if event.event_type == "sick_call":
        if event.shift:
            staff.unavailable_slots.add((event.day, event.shift))
        else:
            staff.unavailable_days.add(event.day)
    elif event.event_type == "leave_request":
        staff.unavailable_days.add(event.day)
    elif event.event_type == "transfer":
        if event.ward and event.ward not in staff.eligible_wards:
            staff.eligible_wards = [event.ward]  # now only eligible for new ward
            staff.home_ward = event.ward

    event.status = "applied"
    return data


def affected_variables(event: DisruptionEvent, data: HospitalData):
    """Which (staff, day, shift) roster slots are directly touched by this event."""
    variables = []
    if event.event_type in ("sick_call",):
        if event.shift:
            variables.append((event.staff_id, event.day, event.shift))
        else:
            for sh in data.shifts:
                variables.append((event.staff_id, event.day, sh))
    elif event.event_type == "leave_request":
        for sh in data.shifts:
            variables.append((event.staff_id, event.day, sh))
    elif event.event_type == "transfer":
        for day in data.days:
            for sh in data.shifts:
                variables.append((event.staff_id, day, sh))
    return variables
