"""ui/styles.py — dark professional "AI operations dashboard" theme."""

CUSTOM_CSS = """
<style>
:root {
    --bg: #080B12;
    --surface: #111722;
    --surface-2: #171E2B;
    --border: #273244;
    --primary: #7DD3FC;
    --secondary: #60A5FA;
    --success: #34D399;
    --warning: #FBBF24;
    --danger: #F87171;
    --text: #F8FAFC;
    --text-muted: #94A3B8;
}

.stApp { background-color: var(--bg); color: var(--text); }
html, body { background-color: var(--bg); }
[data-testid="stHeader"], [data-testid="stToolbar"], [data-testid="stDecoration"] {
    background-color: var(--bg) !important;
}

section[data-testid="stSidebar"] {
    background-color: var(--surface);
    border-right: 1px solid var(--border);
}

h1, h2, h3, h4, h5, h6 { color: var(--text) !important; font-family: "Helvetica Neue", Arial, sans-serif; letter-spacing: 0.2px; }
p, span, label, div { color: var(--text); }
.stCaption, [data-testid="stCaptionContainer"] { color: var(--text-muted) !important; }

/* Progress bar label text ("Step X/10 — ...") */
[data-testid="stProgress"] p { color: var(--text) !important; }

/* Metrics */
[data-testid="stMetric"] {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 12px 16px;
}
[data-testid="stMetricLabel"] { color: var(--text-muted) !important; }
[data-testid="stMetricValue"] { color: var(--text) !important; }

/* Header */
.sr-header {
    background: linear-gradient(135deg, #0c1220 0%, #111722 55%, #10202150 130%);
    border: 1px solid var(--border);
    padding: 26px 30px;
    border-radius: 14px;
    margin-bottom: 20px;
}
.sr-header .sr-eyebrow { color: var(--primary); font-size: 12px; font-weight: 700; letter-spacing: 2px; text-transform: uppercase; margin-bottom: 6px; }
.sr-header h1 { color: var(--text); margin: 0; font-size: 30px; }
.sr-header p { color: var(--text-muted); margin: 8px 0 0 0; font-size: 15px; max-width: 760px; }

/* Generic card */
.sr-card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 16px 18px;
    margin-bottom: 12px;
}
.sr-card-title { color: var(--text-muted); font-size: 12px; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 8px; }

/* Badges */
.sr-badge { display: inline-block; padding: 5px 14px; border-radius: 999px; font-weight: 600; font-size: 13px; border: 1px solid transparent; }
.sr-badge-valid { background: rgba(52,211,153,0.12); color: var(--success); border-color: rgba(52,211,153,0.3); }
.sr-badge-warning { background: rgba(251,191,36,0.12); color: var(--warning); border-color: rgba(251,191,36,0.3); }
.sr-badge-invalid { background: rgba(248,113,113,0.12); color: var(--danger); border-color: rgba(248,113,113,0.3); }
.sr-badge-info { background: rgba(96,165,250,0.12); color: var(--secondary); border-color: rgba(96,165,250,0.3); }

/* Workflow chips */
.sr-flow { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin: 8px 0 4px 0; }
.sr-flow-step { background: var(--surface-2); color: var(--text); padding: 7px 14px; border-radius: 8px; font-size: 13px; font-weight: 600; border: 1px solid var(--border); }
.sr-flow-arrow { color: var(--text-muted); font-weight: 700; }

/* Event log */
.sr-event-log {
    font-family: "SFMono-Regular", Consolas, monospace;
    font-size: 13px;
    background: #05070c;
    color: #cdeee8;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 14px 16px;
    max-height: 260px;
    overflow-y: auto;
}
.sr-event-log .ts { color: var(--primary); margin-right: 8px; }

.sr-section-note {
    background: var(--surface-2);
    border-left: 3px solid var(--secondary);
    padding: 10px 14px;
    border-radius: 6px;
    font-size: 14px;
    color: var(--text-muted);
    margin-bottom: 12px;
}
.sr-section-note-warn { border-left-color: var(--warning); }

/* Dataset transparency card */
.sr-dataset-card { border: 1px solid var(--border); border-radius: 12px; padding: 16px 20px; background: var(--surface); margin-bottom: 12px; }
.sr-dataset-card table { width: 100%; border-collapse: collapse; }
.sr-dataset-card td { padding: 4px 6px; font-size: 14px; color: var(--text); border-bottom: 1px solid var(--border); }
.sr-dataset-card td.label { color: var(--text-muted); width: 55%; }

.sr-coverage-yes { color: var(--success); font-weight: 600; }
.sr-coverage-partial { color: var(--warning); font-weight: 600; }
.sr-coverage-no { color: var(--text-muted); }

/* Buttons */
.stButton>button {
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid var(--border);
    background: var(--surface-2);
    color: var(--text);
}
.stButton>button[kind="primary"] {
    background: var(--primary);
    color: #071019;
    border: 1px solid var(--primary);
}

/* Download buttons use a different component to plain buttons — style them
   to match, so they're visible by default instead of blending into the
   dark background until clicked/focused. */
[data-testid="stDownloadButton"] button {
    border-radius: 8px;
    font-weight: 600;
    border: 1px solid var(--border) !important;
    background: var(--surface-2) !important;
    color: var(--text) !important;
}
[data-testid="stDownloadButton"] button:hover {
    border-color: var(--primary) !important;
    color: var(--primary) !important;
}

/* Select boxes / dropdowns (BaseWeb) — force dark fill + light text so
   options are readable before/without being highlighted */
[data-baseweb="select"] > div {
    background-color: var(--surface-2) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
}
[data-baseweb="popover"] { background-color: var(--surface-2) !important; }
[data-baseweb="popover"] li,
ul[role="listbox"] li,
li[role="option"] {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
}
li[role="option"]:hover, li[role="option"][aria-selected="true"] {
    background-color: var(--surface) !important;
    color: var(--primary) !important;
}
[data-baseweb="select"] input { color: var(--text) !important; }

/* Code blocks (st.code) */
.stCodeBlock, .stCodeBlock pre, .stCodeBlock code, pre, code {
    background-color: var(--surface-2) !important;
    color: var(--text) !important;
}

/* Dataframes */
[data-testid="stDataFrame"] { border: 1px solid var(--border); border-radius: 10px; overflow: hidden; }

/* Sidebar nav caption blocks */
.sr-sidebar-footer { border-top: 1px solid var(--border); margin-top: 14px; padding-top: 12px; }
.sr-sidebar-footer .k { color: var(--text-muted); font-size: 11px; text-transform: uppercase; letter-spacing: 1px; }
.sr-sidebar-footer .v { color: var(--text); font-size: 13px; font-weight: 600; margin-bottom: 8px; }

.sr-cell-off { color: var(--text-muted); }

/* shift-type chips inside roster tables */
.sr-shift-Early, .sr-shift-Morning { color: var(--secondary); font-weight: 600; }
.sr-shift-Day { color: var(--primary); font-weight: 600; }
.sr-shift-Late, .sr-shift-Evening { color: var(--warning); font-weight: 600; }
.sr-shift-Night { color: #C084FC; font-weight: 600; }
</style>
"""
