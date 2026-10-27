"""ui/roster_view.py — turn a Roster dict into a display-friendly DataFrame."""
import pandas as pd
from core.models import HospitalData, Roster, OFF


def roster_to_dataframe(roster: Roster, data: HospitalData) -> pd.DataFrame:
    """
    Builds a Staff x Day grid. When the dataset has a single (virtual) ward
    — as with the INRC-II adapter — cells show the shift type directly
    (Early / Day / Late / Night / OFF). When there are multiple real wards
    — as with the synthetic generator — cells show "shift-initial:ward"
    since the ward assignment itself is meaningful there.
    """
    single_ward = len(data.wards) <= 1
    rows = []
    for staff_id in data.staff_ids():
        staff = data.staff[staff_id]
        row = {"Staff": staff_id, "Name": staff.name, "Role": staff.role}
        if not single_ward:
            row["Home Ward"] = staff.home_ward
        for day in data.days:
            cell_parts = []
            for shift in data.shifts:
                ward = roster.get((staff_id, day, shift), OFF)
                if ward != OFF:
                    if single_ward:
                        cell_parts.append(shift)
                    else:
                        ward_abbrev = "".join(w[0] for w in ward.split()[:2])
                        cell_parts.append(f"{shift[:1]}:{ward_abbrev}")
            row[day] = " / ".join(cell_parts) if cell_parts else "OFF"
        rows.append(row)
    return pd.DataFrame(rows)


def roster_to_csv_bytes(roster: Roster, data: HospitalData) -> bytes:
    rows = []
    for (staff_id, day, shift), ward in sorted(roster.items()):
        if ward != OFF:
            rows.append({"staff_id": staff_id, "day": day, "shift": shift, "ward": ward})
    df = pd.DataFrame(rows)
    return df.to_csv(index=False).encode("utf-8")
