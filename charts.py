"""ui/charts.py — Plotly chart builders used across the dashboard (dark theme)."""
import plotly.graph_objects as go
import pandas as pd

BG = "#111722"
GRID = "#273244"
TEXT = "#F8FAFC"
TEXT_MUTED = "#94A3B8"

PRIMARY = "#7DD3FC"
SECONDARY = "#60A5FA"
SUCCESS = "#34D399"
WARNING = "#FBBF24"
DANGER = "#F87171"
MUTED = "#475569"


def _apply_dark_layout(fig, title):
    fig.update_layout(
        title=title,
        height=340,
        margin=dict(t=48, l=10, r=10, b=10),
        plot_bgcolor=BG,
        paper_bgcolor=BG,
        font=dict(color=TEXT),
        title_font=dict(color=TEXT, size=15),
        legend=dict(font=dict(color=TEXT_MUTED)),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT_MUTED),
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, color=TEXT_MUTED),
    )
    return fig


def bar_shifts_per_staff(counts: dict, title: str, color=PRIMARY):
    staff = list(counts.keys())
    values = list(counts.values())
    fig = go.Figure(go.Bar(x=staff, y=values, marker_color=color))
    _apply_dark_layout(fig, title)
    fig.update_layout(xaxis_title="Staff", yaxis_title="Shifts")
    return fig


def before_after_bar(before: dict, after: dict, title: str):
    staff = list(before.keys())
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=staff, y=[before[s] for s in staff], marker_color=MUTED))
    fig.add_trace(go.Bar(name="After", x=staff, y=[after.get(s, 0) for s in staff], marker_color=PRIMARY))
    _apply_dark_layout(fig, title)
    fig.update_layout(barmode="group", height=360)
    return fig


def convergence_chart(history: list):
    if not history:
        return go.Figure()
    df = pd.DataFrame(history)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["iteration"], y=df["conflicts"], mode="lines",
                              name="Conflicted variables", line=dict(color=DANGER)))
    fig.add_trace(go.Scatter(x=df["iteration"], y=df["hard_violations"], mode="lines",
                              name="Hard violations", line=dict(color=WARNING)))
    _apply_dark_layout(fig, "Min-Conflicts Convergence")
    fig.update_layout(height=360, xaxis_title="Iteration", yaxis_title="Count")
    return fig


def comparison_bar(labels, values, title, colors=None):
    fig = go.Figure(go.Bar(x=list(labels), y=list(values), marker_color=colors or PRIMARY))
    _apply_dark_layout(fig, title)
    return fig


def fairness_trend_chart(before_fair: dict, after_fair: dict):
    metrics = ["night_shifts_std", "weekend_shifts_std", "total_shifts_std"]
    labels = ["Night Std Dev", "Weekend Std Dev", "Total Std Dev"]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="Before", x=labels, y=[before_fair[m] for m in metrics], marker_color=MUTED))
    fig.add_trace(go.Bar(name="After", x=labels, y=[after_fair[m] for m in metrics], marker_color=SUCCESS))
    _apply_dark_layout(fig, "Fairness Before vs After (lower = fairer)")
    fig.update_layout(barmode="group", height=340)
    return fig
