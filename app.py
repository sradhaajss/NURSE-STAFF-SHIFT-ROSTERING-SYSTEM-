"""
app.py — SmartRoster AI
CSP-Based Nurse & Staff Shift Rostering with Min-Conflicts Local Search
and real-time Warm-Start Repair.

Run with:  streamlit run app.py
"""
import copy
import os
import time
from datetime import datetime

import streamlit as st
import pandas as pd

from core.models import HospitalData, OFF, WEEKEND_DAYS
from utils.data_generator import generate_synthetic_hospital, load_inrc_benchmark
from core.constraints import evaluate_hard_constraints, evaluate_soft_constraints, compute_cost
from core.fairness import compute_fairness
from core.backtracking import backtracking_solve
from core.min_conflicts import min_conflicts_solve
from core.events import generate_demo_events, apply_event, affected_variables
from core.repair import warm_start_repair, full_resolve, compare_full_resolve_vs_warm_start
from core.evaluation import run_evaluation
from ui.styles import CUSTOM_CSS
from ui.charts import (bar_shifts_per_staff, before_after_bar, convergence_chart,
                        comparison_bar, fairness_trend_chart)
from ui.roster_view import roster_to_dataframe, roster_to_csv_bytes

st.set_page_config(page_title="SmartRoster", layout="wide", page_icon="🩺")
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

INRC_SC = "data/inrc_n012w8/Sc-n012w8.json"
INRC_WD = "data/inrc_n012w8/WD-n012w8-0.json"
INRC_H0 = "data/inrc_n012w8/H0-n012w8-0.json"

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "data": None,
        "roster": None,
        "pre_event_roster": None,
        "dataset_source": None,
        "mc_result": None,
        "bt_result": None,
        "events": [],
        "current_event": None,
        "event_feed": [],
        "repair_result": None,
        "full_resolve_result": None,
        "comparison_result": None,
        "evaluation_runs": None,
        "settings": {
            "hard_weight": 1000,
            "soft_weight": 10,
            "fairness_weight": 5,
            "mc_iterations": 2000,
            "mc_restarts": 2,
            "mc_seed": 7,
            "bt_max_nodes": 8000,
            "bt_timeout": 5.0,
        },
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


def log_event(message: str):
    st.session_state.event_feed.append((datetime.now().strftime("%H:%M:%S"), message))


def roster_status():
    if st.session_state.roster is None:
        return "NO ROSTER", "sr-badge-warning"
    hard = evaluate_hard_constraints(st.session_state.roster, st.session_state.data)
    if hard.total == 0:
        return "VALID", "sr-badge-valid"
    return "NEEDS REPAIR", "sr-badge-invalid"


def safe_action(fn, *args, error_prefix="Something went wrong", **kwargs):
    """
    Run a solver/action function and show a friendly Streamlit error instead
    of a raw traceback if it fails — the app should never crash outright
    because of a missing dataset, an infeasible instance, or a malformed
    upload (see README limitations).
    """
    try:
        return fn(*args, **kwargs)
    except Exception as exc:  # noqa: BLE001 — deliberately broad: this is a UI safety net
        st.error(f"{error_prefix}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    '<div style="line-height:1.15;">'
    '<div style="font-size:20px;font-weight:800;letter-spacing:1px;">SMARTROSTER</div>'
    '<div style="font-size:11px;color:#94A3B8;letter-spacing:1.5px;text-transform:uppercase;">'
    'AI Rostering Platform</div></div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("")
page = st.sidebar.radio("Navigate", [
    "Overview",
    "Roster Generator",
    "Live Updates",
    "Warm-Start Repair",
    "Fairness Analytics",
    "Algorithm Comparison",
    "Evaluation",
    "CSP Inspector",
], label_visibility="collapsed")

st.sidebar.markdown("---")
with st.sidebar.expander("⚙️ Advanced Settings"):
    s = st.session_state.settings
    s["hard_weight"] = st.number_input("Hard violation weight", 100, 5000, s["hard_weight"], 100)
    s["soft_weight"] = st.number_input("Soft preference weight", 1, 200, s["soft_weight"], 1)
    s["fairness_weight"] = st.number_input("Fairness weight", 1, 100, s["fairness_weight"], 1)
    s["mc_iterations"] = st.number_input("Min-Conflicts max iterations", 200, 10000, s["mc_iterations"], 100)
    s["mc_restarts"] = st.number_input("Min-Conflicts restarts", 0, 10, s["mc_restarts"], 1)
    s["mc_seed"] = st.number_input("Random seed", 0, 9999, s["mc_seed"], 1)
    s["bt_max_nodes"] = st.number_input("Backtracking max nodes", 500, 50000, s["bt_max_nodes"], 500)
    s["bt_timeout"] = st.number_input("Backtracking timeout (s)", 1.0, 30.0, s["bt_timeout"], 1.0)

_dataset_label = st.session_state.data.dataset_label if st.session_state.data else "None loaded"
_status_label, _ = ("Ready", None) if st.session_state.data else ("Awaiting dataset", None)
st.sidebar.markdown(
    f'<div class="sr-sidebar-footer">'
    f'<div class="k">Dataset</div><div class="v">{_dataset_label}</div>'
    f'<div class="k">System Status</div>'
    f'<div class="v">{"🟢" if st.session_state.data else "🟡"} {_status_label}</div>'
    f'</div>',
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Shared solve helpers
# ---------------------------------------------------------------------------

def run_min_conflicts():
    s = st.session_state.settings
    result = min_conflicts_solve(
        st.session_state.data,
        max_iterations=int(s["mc_iterations"]),
        restarts=int(s["mc_restarts"]),
        seed=int(s["mc_seed"]),
    )
    st.session_state.mc_result = result
    st.session_state.roster = result.roster
    return result


def run_backtracking():
    s = st.session_state.settings
    result = backtracking_solve(
        st.session_state.data,
        max_nodes=int(s["bt_max_nodes"]),
        timeout_seconds=float(s["bt_timeout"]),
    )
    st.session_state.bt_result = result
    return result


# ---------------------------------------------------------------------------
# PAGE: Overview
# ---------------------------------------------------------------------------

def page_overview():
    st.markdown(
        """<div class="sr-header">
        <div class="sr-eyebrow">AI-Powered Nurse & Staff Rostering</div>
        <h1>SmartRoster</h1>
        <p>CSP-based scheduling with Min-Conflicts local search and warm-start repair
        under dynamic availability.</p>
        </div>""",
        unsafe_allow_html=True,
    )

    data = st.session_state.data
    roster = st.session_state.roster

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Staff", len(data.staff) if data else "—")
    c2.metric("Wards", len(data.wards) if data else "—")
    c3.metric("Shifts/Day", len(data.shifts) if data else "—")
    if roster is not None and data is not None:
        hard = evaluate_hard_constraints(roster, data)
        fairness = compute_fairness(roster, data)
        c4.metric("Current Violations", hard.total)
        c5.metric("Fairness Score", round(fairness["combined_fairness_score"], 2))
    else:
        c4.metric("Current Violations", "—")
        c5.metric("Fairness Score", "—")
    last_time = None
    if st.session_state.mc_result:
        last_time = st.session_state.mc_result.time_seconds
    c6.metric("Last Solve Time", f"{last_time:.2f}s" if last_time is not None else "—")

    status_label, status_class = roster_status()
    st.markdown(
        f'<span class="sr-badge {status_class}">● Roster Status: {status_label}</span>',
        unsafe_allow_html=True,
    )
    if data is not None:
        st.caption(f"Data source: {data.source}")

    if data is not None:
        st.markdown(
            f'<div class="sr-card"><div class="sr-card-title">Current Dataset</div>'
            f'{data.dataset_label or data.source} · {len(data.staff)} nurses · '
            f'{len(data.days)} days · {len(data.shifts)} shift types'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("#### System Flow")
    st.markdown(
        """<div class="sr-flow">
        <div class="sr-flow-step">DATA</div><div class="sr-flow-arrow">→</div>
        <div class="sr-flow-step">CSP</div><div class="sr-flow-arrow">→</div>
        <div class="sr-flow-step">MIN-CONFLICTS</div><div class="sr-flow-arrow">→</div>
        <div class="sr-flow-step">LIVE EVENT</div><div class="sr-flow-arrow">→</div>
        <div class="sr-flow-step">WARM-START REPAIR</div><div class="sr-flow-arrow">→</div>
        <div class="sr-flow-step">FAIRNESS</div><div class="sr-flow-arrow">→</div>
        <div class="sr-flow-step">EVALUATION</div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.markdown("---")
    st.markdown("#### 🎬 Demonstration Mode")
    st.write("Runs the complete pipeline end-to-end: generate data → solve → disrupt → repair → compare → evaluate.")
    if st.button("▶ RUN FULL DEMO", type="primary"):
        run_full_demo()

    if st.session_state.get("demo_summary"):
        render_demo_summary(st.session_state["demo_summary"])


def run_full_demo():
    try:
        progress = st.progress(0, text="Step 1/10 — Loading INRC-II n012w8 benchmark data...")
        if os.path.exists(INRC_SC) and os.path.exists(INRC_WD) and os.path.exists(INRC_H0):
            st.session_state.data = load_inrc_benchmark(INRC_SC, INRC_WD, INRC_H0)
            st.session_state.dataset_source = "inrc"
        else:
            st.session_state.data = generate_synthetic_hospital(num_staff=45, seed=42)
            st.session_state.dataset_source = "demo"
        progress.progress(10, text="Step 2/10 — Running Min-Conflicts to generate initial roster...")
        mc = run_min_conflicts()
        progress.progress(30, text="Step 3/10 — Validating roster...")
        hard_before = evaluate_hard_constraints(mc.roster, st.session_state.data)
        fair_before = compute_fairness(mc.roster, st.session_state.data)
        progress.progress(40, text="Step 4/10 — Triggering simulated sick-call event...")
        events = generate_demo_events(st.session_state.data, seed=st.session_state.settings["mc_seed"], count=1)
        event = events[0]
        st.session_state.events = events
        st.session_state.current_event = event
        affected = affected_variables(event, st.session_state.data)
        original_roster = copy.deepcopy(mc.roster)
        apply_event(event, st.session_state.data)
        progress.progress(50, text="Step 5/10 — Showing constraint violation...")
        disrupted_roster = dict(original_roster)
        for v in affected:
            disrupted_roster[v] = OFF
        hard_disrupted = evaluate_hard_constraints(disrupted_roster, st.session_state.data)
        progress.progress(65, text="Step 6/10 — Running full re-solve (Approach A)...")
        comparison = compare_full_resolve_vs_warm_start(original_roster, affected, st.session_state.data)
        progress.progress(80, text="Step 7/10 — Restoring original roster for comparison...")
        progress.progress(85, text="Step 8/10 — Running warm-start repair (Approach B)...")
        progress.progress(95, text="Step 9/10 — Comparing both methods...")
        st.session_state.roster = comparison.warm_start.roster
        st.session_state.repair_result = comparison.warm_start
        st.session_state.full_resolve_result = comparison.full_resolve
        st.session_state.comparison_result = comparison
        progress.progress(100, text="Step 10/10 — Done.")
        time.sleep(0.2)
        progress.empty()

        st.session_state["demo_summary"] = {
            "mc": mc, "hard_before": hard_before, "fair_before": fair_before,
            "event": event, "hard_disrupted": hard_disrupted, "comparison": comparison,
        }
    except Exception as exc:  # noqa: BLE001
        st.error(f"Demo run failed: {exc}")


def render_demo_summary(summary):
    st.success("Full demo complete — results below.")
    mc = summary["mc"]
    comparison = summary["comparison"]
    event = summary["event"]

    st.markdown("##### Step 2-3: Initial Roster (Min-Conflicts)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Hard Violations", summary["hard_before"].total)
    c2.metric("Fairness Score", round(summary["fair_before"]["combined_fairness_score"], 2))
    c3.metric("Solve Time", f"{mc.time_seconds:.2f}s")
    c4.metric("Iterations", mc.iterations_used)

    st.markdown(f"##### Step 4-5: Disruption — `{event.event_type}` for **{event.staff_id}** on {event.day}"
                + (f"/{event.shift}" if event.shift else ""))
    st.markdown(
        f'<span class="sr-badge sr-badge-invalid">● {summary["hard_disrupted"].total} constraint violation(s) introduced</span>',
        unsafe_allow_html=True,
    )

    st.markdown("##### Step 6-9: Full Re-Solve vs Warm-Start Repair")
    c1, c2, c3 = st.columns(3)
    c1.metric("Full Re-Solve Time", f"{comparison.full_resolve.time_seconds*1000:.1f} ms")
    c2.metric("Warm-Start Repair Time", f"{comparison.warm_start.time_seconds*1000:.1f} ms")
    c3.metric("Speed Improvement", f"{comparison.speed_improvement_pct:.1f}%")
    st.caption(f"Assignments preserved: {comparison.warm_start.assignments_preserved} · "
               f"Assignments changed: {comparison.warm_start.assignments_changed}")


# ---------------------------------------------------------------------------
# PAGE: Roster Generator
# ---------------------------------------------------------------------------

def render_dataset_transparency_card(data):
    """Section 13/24: dataset transparency card + model coverage checklist."""
    is_inrc = bool(data.model_coverage) and any("Loaded" in v or "Enforced" in v for v in data.model_coverage.values())
    st.markdown('<div class="sr-dataset-card">', unsafe_allow_html=True)
    st.markdown('<div class="sr-card-title">Dataset</div>', unsafe_allow_html=True)
    rows = [
        ("Label", data.dataset_label or data.source),
        ("Nurses / Staff", str(len(data.staff))),
        ("Planning Horizon", f"{len(data.days)} days"),
        ("Shift Types", f"{len(data.shifts)} ({', '.join(data.shifts)})"),
        ("Requirements", "Loaded" if data.requirements else "Not loaded"),
        ("Historical Data (H0)", "Loaded" if data.history else "Not applicable"),
        ("Source", data.source),
    ]
    table_html = "<table>" + "".join(
        f'<tr><td class="label">{k}</td><td>{v}</td></tr>' for k, v in rows
    ) + "</table>"
    st.markdown(table_html, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if data.model_coverage:
        with st.expander("📋 Model Coverage — what this prototype actually implements", expanded=False):
            for field, status in data.model_coverage.items():
                if status.startswith("Loaded") or status.startswith("Enforced"):
                    cls, mark = "sr-coverage-yes", "✓"
                elif status.startswith("Approximated") or status.startswith("Modelled as a soft"):
                    cls, mark = "sr-coverage-partial", "◐ Partial"
                elif status.startswith("N/A"):
                    cls, mark = "sr-coverage-no", "—"
                else:
                    cls, mark = "sr-coverage-no", "—"
                st.markdown(f'<div style="margin-bottom:4px;"><span class="{cls}">{mark}</span> '
                            f'&nbsp;<b>{field}:</b> {status}</div>', unsafe_allow_html=True)


def page_roster_generator():
    st.header("Roster Generator")

    with st.expander("📁 Dataset", expanded=st.session_state.data is None):
        source = st.radio("Dataset source", ["INRC-II n012w8 — Week 0 (Real Benchmark)",
                                              "Simulated Demo Dataset",
                                              "Upload CSV/JSON"], horizontal=False)

        if source == "Simulated Demo Dataset":
            col1, col2, col3, col4 = st.columns(4)
            num_staff = col1.number_input("Number of staff", 10, 100, 45, 5)
            num_wards = col2.number_input("Number of wards", 1, 3, 3, 1)
            planning_days = col3.number_input("Planning days", 5, 14, 7, 1)
            shifts_per_day = col4.number_input("Shifts/day", 2, 3, 3, 1)
            seed = st.number_input("Generator seed", 0, 9999, 42, 1)
            if st.button("Generate Demo Data"):
                result = safe_action(
                    generate_synthetic_hospital,
                    num_staff=int(num_staff), num_wards=int(num_wards),
                    planning_days=int(planning_days), shifts_per_day=int(shifts_per_day),
                    seed=int(seed),
                    error_prefix="Could not generate the synthetic dataset",
                )
                if result is not None:
                    st.session_state.data = result
                    st.session_state.dataset_source = "demo"
                    st.session_state.roster = None
                    st.success(f"Generated {num_staff} staff across {num_wards} wards.")

        elif source == "INRC-II n012w8 — Week 0 (Real Benchmark)":
            st.markdown(
                '<div class="sr-section-note">Single-week prototype using the official INRC-II '
                '(Second International Nurse Rostering Competition) benchmark data — the real '
                'Sc-n012w8.json scenario, WD-n012w8-0.json week-0 demand, and H0-n012w8-0.json '
                'history files. This is NOT a complete implementation of the full 8-week INRC-II '
                'competition specification — see Model Coverage below for exactly what is and '
                "isn't modelled.</div>",
                unsafe_allow_html=True,
            )
            if st.button("Load INRC-II Benchmark", type="primary"):
                if os.path.exists(INRC_SC) and os.path.exists(INRC_WD) and os.path.exists(INRC_H0):
                    result = safe_action(
                        load_inrc_benchmark, INRC_SC, INRC_WD, INRC_H0,
                        error_prefix="Could not load the INRC-II benchmark",
                    )
                    if result is not None:
                        st.session_state.data = result
                        st.session_state.dataset_source = "inrc"
                        st.session_state.roster = None
                        st.success("Loaded INRC-II n012w8 — Week 0 Benchmark.")
                else:
                    st.error("INRC-II benchmark files not found in data/inrc_n012w8/ "
                             "(expected Sc-n012w8.json, WD-n012w8-0.json, H0-n012w8-0.json).")

        else:
            uploaded = st.file_uploader("Upload a CSV or JSON dataset", type=["csv", "json"])
            st.caption("Upload support is limited in this prototype — falls back to the demo "
                       "dataset generator if the file doesn't match the expected schema. The "
                       "architecture is modular so a full custom adapter can be added later.")
            if uploaded is not None:
                st.info("Custom dataset adapters are not yet wired up in this prototype build — "
                        "use the Simulated Demo Dataset or the INRC-II Benchmark option above.")

    if st.session_state.data is None:
        st.warning("No dataset loaded yet. Choose a source above.")
        return

    data = st.session_state.data
    render_dataset_transparency_card(data)

    st.markdown("#### Solver Controls")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🚀 Generate Roster", type="primary"):
            with st.spinner("Building CSP and running Min-Conflicts..."):
                result = safe_action(run_min_conflicts, error_prefix="The solver failed to run")
            if result is not None:
                if result.hard_violations == 0:
                    st.success("Feasible Roster Generated — 0 hard constraint violations.")
                else:
                    st.warning(f"Best Feasible Roster Found Within Search Budget — "
                               f"{result.hard_violations} hard violation(s) remain.")
    with col2:
        if st.button("Reset"):
            st.session_state.roster = None
            st.session_state.mc_result = None
            st.rerun()

    if st.session_state.roster is not None:
        result = st.session_state.mc_result
        cost = compute_cost(st.session_state.roster, data,
                             hard_weight=st.session_state.settings["hard_weight"],
                             soft_weight=st.session_state.settings["soft_weight"],
                             fairness_weight=st.session_state.settings["fairness_weight"])
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Hard Violations", cost["hard_violations"])
        c2.metric("Soft Violations", cost["soft_violations"])
        c3.metric("Fairness Score", round(cost["fairness"]["combined_fairness_score"], 2))
        c4.metric("Solve Time", f"{result.time_seconds:.2f}s" if result else "—")
        c5.metric("Iterations", result.iterations_used if result else "—")

        with st.expander("Cost function breakdown"):
            st.write(f"**Hard penalty:** {st.session_state.settings['hard_weight']} × "
                     f"{cost['hard_violations']} = {cost['hard_penalty']:.0f}")
            st.write(f"**Soft (preference) penalty:** {st.session_state.settings['soft_weight']} × "
                     f"{cost['soft_violations']} = {cost['soft_penalty']:.0f}")
            st.write(f"**Fairness penalty:** {st.session_state.settings['fairness_weight']} × "
                     f"{cost['fairness']['combined_fairness_score']:.2f} = {cost['fairness_penalty']:.1f}")
            st.write(f"**Total cost:** {cost['total_cost']:.1f}")

        st.markdown("#### Roster Grid")
        df = roster_to_dataframe(st.session_state.roster, data)
        st.dataframe(df, use_container_width=True, height=420)

        st.download_button("⬇ Download Roster CSV",
                            data=roster_to_csv_bytes(st.session_state.roster, data),
                            file_name="roster.csv", mime="text/csv")

        if result and result.history:
            st.plotly_chart(convergence_chart(result.history), use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Live Updates
# ---------------------------------------------------------------------------

def page_live_updates():
    st.header("Live Availability Simulator")
    st.markdown(
        '<div class="sr-section-note">Simulated events demonstrate how the roster responds to '
        'dynamic availability changes. This is not connected to a live hospital system.</div>',
        unsafe_allow_html=True,
    )

    data = st.session_state.data
    if data is None or st.session_state.roster is None:
        st.warning("Generate a roster first on the Roster Generator page.")
        return

    col1, col2 = st.columns([2, 1])
    with col2:
        if st.button("🎲 Randomly Simulate Disruption"):
            events = generate_demo_events(data, seed=int(time.time()) % 10000, count=1)
            st.session_state.events = events
            st.session_state.current_event = events[0]
            log_event(f"New event generated: {events[0].event_type} for {events[0].staff_id}")
        if st.button("Reset Scenario"):
            st.session_state.current_event = None
            st.session_state.pre_event_roster = None
            st.session_state.repair_result = None
            st.session_state.event_feed = []
            st.rerun()

    with col1:
        if not st.session_state.events:
            st.session_state.events = generate_demo_events(data, seed=st.session_state.settings["mc_seed"], count=3)
        options = {f"{e.event_id}: {e.event_type} — {e.staff_id} on {e.day}"
                   + (f"/{e.shift}" if e.shift else ""): e for e in st.session_state.events}
        choice = st.selectbox("Select an event", list(options.keys()))
        st.session_state.current_event = options[choice]

    event = st.session_state.current_event
    if event is None:
        return

    c1, c2 = st.columns(2)
    c1.markdown(f"**Current event:** `{event.event_type}` — {event.staff_id}")
    c2.markdown(f"**Affected slot:** {event.day}" + (f" / {event.shift}" if event.shift else " (whole day)")
                + (f" / {event.ward}" if event.ward else ""))

    b1, b2 = st.columns(2)
    with b1:
        apply_clicked = st.button("⚡ Apply Event", type="primary")
    with b2:
        repair_clicked = st.button("🔧 Repair Roster")

    if apply_clicked:
        try:
            st.session_state.pre_event_roster = copy.deepcopy(st.session_state.roster)
            log_event(f"{event.staff_id} — {event.event_type.replace('_',' ')} reported")
            affected = affected_variables(event, data)
            apply_event(event, data)
            log_event("Conflict detected")
            for v in affected:
                st.session_state.roster[v] = OFF
            log_event("Affected slot(s) isolated")
            hard = evaluate_hard_constraints(st.session_state.roster, data)
            st.session_state["last_affected"] = affected
            st.session_state["disruption_violations"] = hard.total
            if hard.total > 0:
                st.warning(f"Violation: {hard.total} constraint(s) now broken — unavailable staff assigned "
                           f"or ward left understaffed.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not apply the event: {exc}")

    if repair_clicked and st.session_state.pre_event_roster is not None:
        try:
            log_event("Warm-start repair started")
            affected = st.session_state.get("last_affected", affected_variables(event, data))
            result = warm_start_repair(st.session_state.pre_event_roster, affected, data)
            st.session_state.repair_result = result
            st.session_state.roster = result.roster
            log_event("Replacement assigned")
            hard = evaluate_hard_constraints(result.roster, data)
            if hard.total == 0:
                log_event("Roster valid")
            else:
                log_event(f"Roster has {hard.total} remaining violation(s)")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Repair failed: {exc}")

    if st.session_state.event_feed:
        st.markdown("#### Event Feed")
        log_html = "<div class='sr-event-log'>" + "".join(
            f"<div><span class='ts'>{ts}</span>{msg}</div>" for ts, msg in st.session_state.event_feed[-20:]
        ) + "</div>"
        st.markdown(log_html, unsafe_allow_html=True)

    if st.session_state.pre_event_roster is not None:
        st.markdown("#### Before / After")
        colA, colB = st.columns(2)
        with colA:
            st.markdown("**BEFORE DISRUPTION**")
            hard_before = evaluate_hard_constraints(st.session_state.pre_event_roster, data)
            fair_before = compute_fairness(st.session_state.pre_event_roster, data)
            st.metric("Violations", hard_before.total)
            st.metric("Fairness", round(fair_before["combined_fairness_score"], 2))
        with colB:
            st.markdown("**CURRENT / AFTER REPAIR**")
            hard_after = evaluate_hard_constraints(st.session_state.roster, data)
            fair_after = compute_fairness(st.session_state.roster, data)
            st.metric("Violations", hard_after.total)
            st.metric("Fairness", round(fair_after["combined_fairness_score"], 2))


# ---------------------------------------------------------------------------
# PAGE: Warm-Start Repair
# ---------------------------------------------------------------------------

def page_warm_start_repair():
    st.header("Warm-Start Repair")
    st.markdown(
        '<div class="sr-section-note">Warm-start repair preserves unaffected assignments and '
        're-optimizes only the impacted portion of the roster — unlike a full re-solve, which '
        'throws away the entire roster and starts over.</div>',
        unsafe_allow_html=True,
    )

    data = st.session_state.data
    if data is None or st.session_state.roster is None:
        st.warning("Generate a roster first on the Roster Generator page, then trigger a disruption "
                   "on the Live Updates page.")
        return

    event = st.session_state.current_event
    if event is None or st.session_state.pre_event_roster is None:
        st.info("No disruption applied yet. Go to Live Updates, select or randomize an event, and "
                "click Apply Event first.")
        return

    if st.button("Run Full Re-Solve vs Warm-Start Comparison", type="primary"):
        affected = st.session_state.get("last_affected", affected_variables(event, data))
        with st.spinner("Running both approaches..."):
            comparison = safe_action(
                compare_full_resolve_vs_warm_start, st.session_state.pre_event_roster, affected, data,
                error_prefix="Comparison failed to run",
            )
        if comparison is not None:
            st.session_state.comparison_result = comparison
            st.session_state.repair_result = comparison.warm_start
            st.session_state.full_resolve_result = comparison.full_resolve
            st.session_state.roster = comparison.warm_start.roster

    comparison = st.session_state.comparison_result
    if comparison is None:
        return

    st.markdown("#### Approach A: Full Re-Solve  vs  Approach B: Warm-Start Repair")
    c1, c2, c3 = st.columns(3)
    c1.metric("Full Re-Solve Time", f"{comparison.full_resolve.time_seconds*1000:.1f} ms")
    c2.metric("Warm-Start Repair Time", f"{comparison.warm_start.time_seconds*1000:.1f} ms")
    c3.metric("Speed Improvement", f"{comparison.speed_improvement_pct:.1f}%")

    c4, c5, c6, c7 = st.columns(4)
    c4.metric("Affected Variables", len(comparison.warm_start.affected_variables))
    c5.metric("Variables Repaired", comparison.warm_start.variables_repaired)
    c6.metric("Assignments Preserved", comparison.warm_start.assignments_preserved)
    c7.metric("Assignments Changed", comparison.warm_start.assignments_changed)

    total_slots = comparison.warm_start.assignments_preserved + comparison.warm_start.assignments_changed
    if total_slots > 0:
        pct_preserved = 100 * comparison.warm_start.assignments_preserved / total_slots
        pct_changed = 100 * comparison.warm_start.assignments_changed / total_slots
        st.caption(f"Assignments preserved: {pct_preserved:.1f}% · Assignments changed: {pct_changed:.1f}%")

    st.plotly_chart(
        comparison_bar(
            ["Full Re-Solve", "Warm-Start Repair"],
            [comparison.full_resolve.time_seconds * 1000, comparison.warm_start.time_seconds * 1000],
            "Solve Time (ms)",
            colors=["#9aa7b8", "#0f8b8d"],
        ),
        use_container_width=True,
    )

    st.markdown("#### Roster Comparison")
    colA, colB = st.columns(2)
    with colA:
        st.markdown("**BEFORE (pre-disruption roster)**")
        st.dataframe(roster_to_dataframe(st.session_state.pre_event_roster, data), use_container_width=True, height=320)
    with colB:
        st.markdown("**AFTER WARM-START REPAIR**")
        st.dataframe(roster_to_dataframe(comparison.warm_start.roster, data), use_container_width=True, height=320)


# ---------------------------------------------------------------------------
# PAGE: Fairness Analytics
# ---------------------------------------------------------------------------

def page_fairness():
    st.header("Fairness Analytics")
    data = st.session_state.data
    if data is None or st.session_state.roster is None:
        st.warning("Generate a roster first.")
        return

    fairness = compute_fairness(st.session_state.roster, data)
    c1, c2, c3 = st.columns(3)
    c1.metric("Std Dev — Night Shifts", round(fairness["night_shifts_std"], 3))
    c2.metric("Std Dev — Weekend Shifts", round(fairness["weekend_shifts_std"], 3))
    c3.metric("Combined Fairness Score", round(fairness["combined_fairness_score"], 3))
    st.caption("Lower standard deviation = more balanced/fair distribution across staff.")

    st.plotly_chart(bar_shifts_per_staff(fairness["total_shifts"], "Total Shifts per Staff"), use_container_width=True)
    st.plotly_chart(bar_shifts_per_staff(fairness["night_shifts"], "Night Shifts per Staff", color="#b8860b"),
                     use_container_width=True)
    st.plotly_chart(bar_shifts_per_staff(fairness["weekend_shifts"], "Weekend Shifts per Staff", color="#c0392b"),
                     use_container_width=True)

    if st.session_state.pre_event_roster is not None:
        st.markdown("#### Before vs After Repair")
        fair_before = compute_fairness(st.session_state.pre_event_roster, data)
        st.plotly_chart(fairness_trend_chart(fair_before, fairness), use_container_width=True)
        st.plotly_chart(before_after_bar(fair_before["total_shifts"], fairness["total_shifts"],
                                          "Total Shifts: Before vs After"), use_container_width=True)


# ---------------------------------------------------------------------------
# PAGE: Algorithm Comparison
# ---------------------------------------------------------------------------

def page_algorithm_comparison():
    st.header("Algorithm Comparison")
    st.markdown(
        "- **Backtracking** = baseline CSP search (recursive, exhaustive, exact)\n"
        "- **Min-Conflicts** = local search solver (fast, scalable, heuristic — finds a feasible, "
        "high-quality solution, not a mathematically proven optimum)\n"
        "- **Warm-Start Repair** = incremental repair using an existing solution after a disruption"
    )

    data = st.session_state.data
    if data is None:
        st.warning("Generate a dataset first.")
        return

    if st.button("Run Backtracking Baseline"):
        with st.spinner("Running backtracking..."):
            safe_action(run_backtracking, error_prefix="Backtracking failed to run")

    bt = st.session_state.bt_result
    mc = st.session_state.mc_result
    comparison = st.session_state.comparison_result

    rows = []
    if bt:
        rows.append({"Metric": "Solve Time (s)", "Backtracking": round(bt.time_seconds, 3),
                     "Min-Conflicts": round(mc.time_seconds, 3) if mc else "—",
                     "Warm-Start Repair": round(comparison.warm_start.time_seconds, 3) if comparison else "—"})
        rows.append({"Metric": "Hard Violations", "Backtracking": bt.hard_violations,
                     "Min-Conflicts": mc.hard_violations if mc else "—",
                     "Warm-Start Repair": comparison.warm_start.hard_violations if comparison else "—"})
        rows.append({"Metric": "Fairness Score", "Backtracking": round(bt.fairness_score, 2),
                     "Min-Conflicts": round(mc.fairness_score, 2) if mc else "—",
                     "Warm-Start Repair": round(comparison.warm_start.fairness_score, 2) if comparison else "—"})
        rows.append({"Metric": "Nodes / Iterations", "Backtracking": bt.nodes_explored,
                     "Min-Conflicts": mc.iterations_used if mc else "—",
                     "Warm-Start Repair": comparison.warm_start.repair_iterations if comparison else "—"})
        if bt.downsized:
            st.caption(f"⚠️ {bt.note}")

        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.plotly_chart(
            comparison_bar(
                ["Backtracking", "Min-Conflicts"] + (["Warm-Start Repair"] if comparison else []),
                [bt.time_seconds, mc.time_seconds if mc else 0] + ([comparison.warm_start.time_seconds] if comparison else []),
                "Solve Time Comparison (s)",
                colors=["#9aa7b8", "#0f8b8d", "#1e9e5a"] if comparison else ["#9aa7b8", "#0f8b8d"],
            ),
            use_container_width=True,
        )
        st.plotly_chart(
            comparison_bar(
                ["Backtracking", "Min-Conflicts"] + (["Warm-Start Repair"] if comparison else []),
                [bt.hard_violations, mc.hard_violations if mc else 0] + ([comparison.warm_start.hard_violations] if comparison else []),
                "Hard Violations Comparison",
                colors=["#c0392b", "#b8860b", "#1e9e5a"] if comparison else ["#c0392b", "#b8860b"],
            ),
            use_container_width=True,
        )
    else:
        st.info("Click 'Run Backtracking Baseline' above to populate the comparison.")


# ---------------------------------------------------------------------------
# PAGE: Evaluation
# ---------------------------------------------------------------------------

def page_evaluation():
    st.header("Evaluation")
    st.write("Runs multiple experiments using controlled random seeds. All values below come from "
             "actual executions — nothing is hardcoded.")

    data = st.session_state.data
    if data is None:
        st.warning("Generate a dataset first.")
        return

    render_dataset_transparency_card(data)

    num_seeds = st.slider("Number of experiment seeds", 2, 8, 4)
    if st.button("Run Evaluation Suite", type="primary"):
        seeds = list(range(1, num_seeds + 1))
        with st.spinner(f"Running {num_seeds} controlled experiments..."):
            runs = safe_action(run_evaluation, data, seeds, error_prefix="Evaluation suite failed")
        if runs is not None:
            st.session_state.evaluation_runs = runs

    runs = st.session_state.evaluation_runs
    if not runs:
        st.info("Click 'Run Evaluation Suite' to generate results.")
        return

    df = pd.DataFrame([r.__dict__ for r in runs])
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.plotly_chart(comparison_bar(df["seed"].astype(str), df["backtracking_time"],
                                    "Backtracking Solve Time by Seed (s)"), use_container_width=True)
    st.plotly_chart(comparison_bar(df["seed"].astype(str), df["min_conflicts_time"],
                                    "Min-Conflicts Solve Time by Seed (s)", colors="#0f8b8d"),
                     use_container_width=True)

    fig_fr = before_after_bar(
        dict(zip(df["seed"].astype(str), df["full_resolve_time"])),
        dict(zip(df["seed"].astype(str), df["warm_start_time"])),
        "Full Re-Solve vs Warm-Start Repair Time (s)",
    )
    st.plotly_chart(fig_fr, use_container_width=True)

    st.plotly_chart(comparison_bar(df["seed"].astype(str), df["min_conflicts_fairness"],
                                    "Fairness Score by Seed", colors="#1e9e5a"), use_container_width=True)
    st.plotly_chart(comparison_bar(df["seed"].astype(str), df["speed_improvement_pct"],
                                    "Warm-Start Speed Improvement (%)", colors="#b8860b"),
                     use_container_width=True)
    st.plotly_chart(comparison_bar(df["seed"].astype(str), df["changed_assignments"],
                                    "Changed Assignments in Repair"), use_container_width=True)

    st.markdown("#### Summary")
    st.write(f"- Average Min-Conflicts solve time: **{df['min_conflicts_time'].mean():.2f}s** "
             f"over {df['min_conflicts_iterations'].mean():.0f} average iterations")
    st.write(f"- Average warm-start speed improvement over full re-solve: "
             f"**{df['speed_improvement_pct'].mean():.1f}%**")
    st.write(f"- Average fairness score achieved: **{df['min_conflicts_fairness'].mean():.2f}** "
             f"(lower is more balanced)")


# ---------------------------------------------------------------------------
# PAGE: CSP Inspector
# ---------------------------------------------------------------------------

def page_csp_inspector():
    st.header("CSP Inspector")
    data = st.session_state.data
    if data is None:
        st.warning("Generate a dataset first.")
        return

    st.markdown("#### Variables")
    st.code("X(nurse, day, shift)  →  ward assigned, or OFF", language="text")
    st.write(f"Total variables in this instance: **{len(data.staff) * len(data.days) * len(data.shifts)}**")

    st.markdown("#### Domain")
    domain_values = ["OFF"] + list(data.shifts) if len(data.wards) <= 1 else None
    if domain_values:
        st.code(" / ".join(domain_values), language="text")
    else:
        st.write("Eligible staff assignments — for each variable, the domain is `OFF` plus every ward "
                 "the staff member is eligible for.")
        sample_staff = next(iter(data.staff.values()))
        st.write(f"Example — **{sample_staff.staff_id}** is eligible for: {sample_staff.eligible_wards} "
                 f"→ domain = {['OFF'] + sample_staff.eligible_wards}")

    st.markdown("#### Hard Constraints (actually implemented)")
    hard_lines = [
        "**H1** No double booking — no staff member works two shifts on the same day",
        "**H2** Availability — no assignment while unavailable",
        "**H3** Minimum staffing — every ward/shift/day meets its minimum required staff",
        "**H4** Required skills — staff must hold the skill a requirement calls for",
        "**H5** Maximum assignments — staff cannot exceed their max shift count",
        "**H6** Eligibility — staff can only be assigned to wards they're eligible for",
        "**H7** Disruption/unavailability — staff on leave/sick must not remain assigned",
    ]
    if data.forbidden_successions:
        hard_lines.append("**H8** Forbidden shift succession — e.g. a Night shift cannot be immediately "
                           "followed by an Early/Day/Late shift (from the INRC-II scenario data)")
    if any(s.max_consecutive_working_days for s in data.staff.values()):
        hard_lines.append("**H9** Maximum consecutive working days (INRC-II contract + H0 history carry-in)")
    if data.shift_type_rules:
        hard_lines.append("**H10** Maximum consecutive assignments of the same shift type "
                           "(INRC-II shift-type rule + H0 history carry-in)")
    if any(s.max_working_weekends for s in data.staff.values()):
        hard_lines.append("**H11** Maximum working weekends (INRC-II contract + H0 history carry-in)")
    st.markdown("\n".join(f"- {line}" for line in hard_lines))

    st.markdown("#### Soft Constraints (actually implemented)")
    soft_lines = [
        "**S1-S3** Balanced total / night / weekend shift distribution (fairness module)",
        "**S4** Respect staff shift preferences where possible",
    ]
    if any(s.min_shifts for s in data.staff.values()):
        soft_lines.append("**S5** Reach the (weekly-approximated) minimum total assignments")
    if any(s.min_consecutive_working_days for s in data.staff.values()):
        soft_lines.append("**S6** Minimum consecutive working days per work stretch")
    if any(s.min_consecutive_days_off for s in data.staff.values()):
        soft_lines.append("**S7** Minimum consecutive days off per rest stretch")
    if any(s.complete_weekends for s in data.staff.values()):
        soft_lines.append("**S8** Complete weekends — work both Saturday & Sunday, or neither")
    st.markdown("\n".join(f"- {line}" for line in soft_lines))
    if not data.forbidden_successions and not data.shift_type_rules:
        st.caption("This dataset (synthetic generator) doesn't define INRC-II sequence/contract rules, "
                   "so H8-H11 and S5-S8 are not applicable here — load the INRC-II benchmark to see them.")

    st.markdown("#### Objective")
    s = st.session_state.settings
    st.code(
        "TOTAL COST = Hard Penalty + Soft Penalty + Fairness Penalty\n\n"
        f"Hard Penalty     = {s['hard_weight']} × hard_violations\n"
        f"Soft Penalty     = {s['soft_weight']} × soft_violations\n"
        f"Fairness Penalty = {s['fairness_weight']} × combined_fairness_score",
        language="text",
    )

    if data.model_coverage:
        with st.expander("📋 INRC-II Data Coverage"):
            for field, status in data.model_coverage.items():
                st.write(f"- **{field}:** {status}")

    if st.session_state.mc_result and st.session_state.mc_result.history:
        st.markdown("#### Min-Conflicts Process (last run)")
        hist = st.session_state.mc_result.history
        last = hist[-1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Iterations logged", len(hist))
        c2.metric("Final conflicts", last["conflicts"])
        c3.metric("Final hard violations", last["hard_violations"])
        st.plotly_chart(convergence_chart(hist), use_container_width=True)
    else:
        st.info("Run Min-Conflicts on the Roster Generator page to see live convergence data here.")


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

PAGES = {
    "Overview": page_overview,
    "Roster Generator": page_roster_generator,
    "Live Updates": page_live_updates,
    "Warm-Start Repair": page_warm_start_repair,
    "Fairness Analytics": page_fairness,
    "Algorithm Comparison": page_algorithm_comparison,
    "Evaluation": page_evaluation,
    "CSP Inspector": page_csp_inspector,
}

if st.session_state.data is None and page not in ("Overview", "Roster Generator"):
    # auto-provision a default demo dataset so other pages aren't dead ends
    st.session_state.data = generate_synthetic_hospital(num_staff=45, seed=42)
    st.session_state.dataset_source = "demo"

PAGES[page]()
