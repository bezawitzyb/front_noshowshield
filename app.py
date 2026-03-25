"""
NoShowShield — Premium Streamlit Dashboard
AI-Powered Hotel Revenue Protection Against Cancellations

Connects to the live FastAPI on Google Cloud Run.
Run with:  streamlit run app.py
"""

import os
import time
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

# ══════════════════════════════════════════════════════════════════════
# SECTION 1 — PAGE CONFIG & CUSTOM STYLES
# ══════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="NoShowShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Brand palette ────────────────────────────────────────────────────
# Light theme — soft lavender background, deep indigo accent, white cards
BRAND_CSS = """
<style>
/* ── Import fonts ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ── Root variables ───────────────────────────────────────── */
:root {
    --bg-primary:    #F4F7FE;
    --bg-secondary:  #FFFFFF;
    --bg-card:       #FFFFFF;
    --bg-card-hover: #F0F3FF;
    --accent:        #4318FF;
    --accent-light:  #7551FF;
    --accent-soft:   #E9E3FF;
    --accent-glow:   rgba(67,24,255,0.10);
    --text-primary:  #1B2559;
    --text-secondary:#A3AED0;
    --text-muted:    #B0BBD5;
    --success:       #01B574;
    --warning:       #FFB547;
    --danger:        #E31A1A;
    --border:        #E9EDF7;
    --border-strong: #D6DCE9;
    --radius:        16px;
    --radius-sm:     10px;
    --shadow:        0 4px 18px rgba(27,37,89,0.06);
    --shadow-hover:  0 6px 24px rgba(67,24,255,0.10);
    --font-body:     'DM Sans', sans-serif;
    --font-mono:     'JetBrains Mono', monospace;
}

/* ══════════════════════════════════════════════════════════
   GLOBAL
   ══════════════════════════════════════════════════════════ */
html, body, [data-testid="stAppViewContainer"],
[data-testid="stAppViewBlockContainer"],
[data-testid="stBottomBlockContainer"],
[data-testid="stMain"] {
    background-color: var(--bg-primary) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stHeader"] {
    background-color: var(--bg-primary) !important;
}
/* catch-all text */
.stMarkdown, .stMarkdown p, .stMarkdown span, .stMarkdown li,
.stMarkdown strong, .stMarkdown em,
[data-testid="stText"], [data-testid="stMarkdownContainer"],
[data-testid="stMarkdownContainer"] p {
    color: var(--text-primary) !important;
}
.stMarkdown a {
    color: var(--accent) !important;
}

/* ══════════════════════════════════════════════════════════
   SIDEBAR
   ══════════════════════════════════════════════════════════ */
section[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 2px 0 12px rgba(27,37,89,0.04);
}
section[data-testid="stSidebar"] * {
    font-family: var(--font-body) !important;
}
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stWidgetLabel p {
    color: var(--text-secondary) !important;
}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: var(--text-primary) !important;
}
/* Sidebar inputs */
section[data-testid="stSidebar"] .stSelectbox > div > div,
section[data-testid="stSidebar"] .stNumberInput > div > div > input,
section[data-testid="stSidebar"] .stTextInput > div > div > input {
    background-color: var(--bg-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}

/* ══════════════════════════════════════════════════════════
   ALL LABELS
   ══════════════════════════════════════════════════════════ */
.stWidgetLabel, .stWidgetLabel p, label,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p {
    color: var(--text-primary) !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

/* ══════════════════════════════════════════════════════════
   METRIC CARDS
   ══════════════════════════════════════════════════════════ */
[data-testid="stMetric"] {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 20px 22px 16px 22px;
    box-shadow: var(--shadow);
    transition: all 0.25s ease;
}
[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-hover);
    border-color: var(--accent-soft);
}
[data-testid="stMetric"] label,
[data-testid="stMetric"] [data-testid="stMetricLabel"],
[data-testid="stMetric"] [data-testid="stMetricLabel"] p {
    color: var(--text-secondary) !important;
    font-size: 0.76rem !important;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    font-weight: 500 !important;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.55rem !important;
}
[data-testid="stMetric"] [data-testid="stMetricDelta"] {
    color: var(--success) !important;
}

/* ══════════════════════════════════════════════════════════
   BUTTONS
   ══════════════════════════════════════════════════════════ */
.stButton > button[kind="primary"],
button[data-testid="stBaseButton-primary"] {
    background: linear-gradient(135deg, var(--accent), var(--accent-light)) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    font-weight: 700 !important;
    font-family: var(--font-body) !important;
    letter-spacing: 0.03em;
    padding: 0.6rem 1.4rem !important;
    box-shadow: 0 4px 14px rgba(67,24,255,0.25);
    transition: all 0.25s ease;
}
.stButton > button[kind="primary"]:hover,
button[data-testid="stBaseButton-primary"]:hover {
    box-shadow: 0 6px 22px rgba(67,24,255,0.35);
    transform: translateY(-1px);
}
/* Secondary buttons */
.stButton > button:not([kind="primary"]),
button[data-testid="stBaseButton-secondary"] {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
.stButton > button:not([kind="primary"]):hover {
    background: var(--bg-card-hover) !important;
    border-color: var(--accent) !important;
    color: var(--accent) !important;
}

/* ══════════════════════════════════════════════════════════
   TABS
   ══════════════════════════════════════════════════════════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    background: var(--bg-secondary);
    border-radius: var(--radius);
    padding: 4px;
    border: 1px solid var(--border);
    box-shadow: var(--shadow);
}
.stTabs [data-baseweb="tab"] {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
    font-weight: 500;
    border-radius: var(--radius-sm);
    padding: 10px 24px;
    transition: all 0.25s ease;
}
.stTabs [aria-selected="true"] {
    background: var(--accent) !important;
    color: #FFFFFF !important;
    font-weight: 700;
    box-shadow: 0 2px 8px rgba(67,24,255,0.25);
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ══════════════════════════════════════════════════════════
   DIVIDERS
   ══════════════════════════════════════════════════════════ */
hr {
    border-color: var(--border) !important;
    margin: 1.2rem 0 !important;
}

/* ══════════════════════════════════════════════════════════
   ALERTS
   ══════════════════════════════════════════════════════════ */
.stAlert, [data-testid="stAlert"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
.stAlert p, [data-testid="stAlert"] p,
.stAlert span, [data-testid="stAlert"] span,
.stAlert div, [data-testid="stAlert"] div {
    color: var(--text-primary) !important;
}

/* ══════════════════════════════════════════════════════════
   TABLES & DATAFRAMES
   ══════════════════════════════════════════════════════════ */
[data-testid="stTable"] table {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stTable"] th {
    background: var(--bg-primary) !important;
    color: var(--accent) !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    font-size: 0.72rem !important;
    letter-spacing: 0.06em;
    border-color: var(--border) !important;
}
[data-testid="stTable"] td {
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
    background: var(--bg-card) !important;
}
/* st.dataframe */
[data-testid="stDataFrame"],
[data-testid="stDataFrame"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}

/* ══════════════════════════════════════════════════════════
   SELECTBOX / DROPDOWN
   ══════════════════════════════════════════════════════════ */
[data-baseweb="select"] > div {
    background: var(--bg-card) !important;
    border-color: var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
[data-baseweb="select"] > div > div,
[data-baseweb="select"] span,
[data-baseweb="select"] input {
    color: var(--text-primary) !important;
}
[data-baseweb="select"] svg {
    fill: var(--text-secondary) !important;
    color: var(--text-secondary) !important;
}
/* dropdown menu */
[data-baseweb="popover"],
[data-baseweb="popover"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    box-shadow: 0 8px 32px rgba(27,37,89,0.12) !important;
    border-radius: var(--radius-sm) !important;
}
[data-baseweb="popover"] ul,
[data-baseweb="menu"] {
    background: var(--bg-card) !important;
}
[data-baseweb="popover"] li,
[data-baseweb="menu"] li,
[role="option"] {
    color: var(--text-primary) !important;
    background: var(--bg-card) !important;
}
[data-baseweb="popover"] li:hover,
[data-baseweb="menu"] li:hover,
[role="option"]:hover,
[role="option"][aria-selected="true"] {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
}

/* ══════════════════════════════════════════════════════════
   NUMBER INPUT
   ══════════════════════════════════════════════════════════ */
.stNumberInput input {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
.stNumberInput button {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border-color: var(--border) !important;
}
.stNumberInput button:hover {
    background: var(--accent-soft) !important;
    color: var(--accent) !important;
}
.stNumberInput button svg {
    fill: var(--text-secondary) !important;
    stroke: var(--text-secondary) !important;
}

/* ══════════════════════════════════════════════════════════
   TEXT INPUT
   ══════════════════════════════════════════════════════════ */
.stTextInput input, .stTextArea textarea {
    background: var(--bg-card) !important;
    color: var(--text-primary) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
}
.stTextInput input::placeholder, .stTextArea textarea::placeholder {
    color: var(--text-muted) !important;
}

/* ══════════════════════════════════════════════════════════
   SLIDER
   ══════════════════════════════════════════════════════════ */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--accent) !important;
    border-color: var(--accent) !important;
    box-shadow: 0 2px 8px rgba(67,24,255,0.3);
}
/* filled track */
.stSlider [data-baseweb="slider"] div[data-testid] > div {
    background: var(--accent-light) !important;
}
/* unfilled track */
.stSlider [data-baseweb="slider"] > div > div {
    background: var(--border) !important;
}
/* slider labels */
.stSlider [data-testid="stThumbValue"],
.stSlider div[data-testid="stTickBarMin"],
.stSlider div[data-testid="stTickBarMax"] {
    color: var(--text-secondary) !important;
}

/* ══════════════════════════════════════════════════════════
   TOOLTIP
   ══════════════════════════════════════════════════════════ */
[data-testid="stTooltipIcon"] svg {
    fill: var(--text-muted) !important;
    color: var(--text-muted) !important;
}
[data-testid="stTooltipContent"],
[role="tooltip"] {
    background: var(--text-primary) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
}
[data-testid="stTooltipContent"] p,
[role="tooltip"] p {
    color: #FFFFFF !important;
}

/* ══════════════════════════════════════════════════════════
   CAPTIONS
   ══════════════════════════════════════════════════════════ */
.stCaption, [data-testid="stCaptionContainer"],
[data-testid="stCaptionContainer"] p {
    color: var(--text-secondary) !important;
    font-size: 0.8rem !important;
}

/* ══════════════════════════════════════════════════════════
   HEADINGS
   ══════════════════════════════════════════════════════════ */
h1, h2, h3, h4, h5, h6 {
    font-family: var(--font-body) !important;
    color: var(--text-primary) !important;
}
h1 { font-weight: 700 !important; }
h2 { font-weight: 600 !important; color: var(--accent) !important; }
h3 { font-weight: 600 !important; }

/* ══════════════════════════════════════════════════════════
   EXPANDER
   ══════════════════════════════════════════════════════════ */
[data-testid="stExpander"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    box-shadow: var(--shadow);
}
[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary span,
[data-testid="stExpander"] summary p {
    color: var(--text-primary) !important;
}
[data-testid="stExpander"] svg {
    fill: var(--text-secondary) !important;
    color: var(--text-secondary) !important;
}

/* ══════════════════════════════════════════════════════════
   CHECKBOX / RADIO
   ══════════════════════════════════════════════════════════ */
.stCheckbox label span,
.stRadio label span,
[data-testid="stCheckbox"] label span {
    color: var(--text-primary) !important;
}

/* ══════════════════════════════════════════════════════════
   SPINNER
   ══════════════════════════════════════════════════════════ */
.stSpinner > div,
.stSpinner > div > span {
    color: var(--accent) !important;
}

/* ══════════════════════════════════════════════════════════
   PLOTLY MODEBAR
   ══════════════════════════════════════════════════════════ */
.js-plotly-plot .modebar-btn path {
    fill: var(--text-muted) !important;
}
.js-plotly-plot .modebar-btn:hover path {
    fill: var(--accent) !important;
}

/* ══════════════════════════════════════════════════════════
   SCROLLBAR
   ══════════════════════════════════════════════════════════ */
::-webkit-scrollbar { width: 7px; height: 7px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border-strong); border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--text-muted); }

/* ══════════════════════════════════════════════════════════
   CUSTOM UTILITY CLASSES
   ══════════════════════════════════════════════════════════ */
.hero-banner {
    background: linear-gradient(135deg, #FFFFFF 0%, #F0F3FF 100%);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px 32px 22px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
    box-shadow: var(--shadow);
}
.hero-banner::after {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 180px; height: 180px;
    background: radial-gradient(circle, rgba(67,24,255,0.06) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--text-primary);
    margin: 0 0 4px 0;
    letter-spacing: -0.02em;
}
.hero-accent {
    color: var(--accent);
}
.hero-subtitle {
    font-size: 0.92rem;
    color: var(--text-secondary);
    margin: 0;
    line-height: 1.55;
    max-width: 680px;
}
.section-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--accent);
    margin-bottom: 8px;
}
.badge-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    margin-top: 6px;
}
.badge {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 600;
    font-family: var(--font-mono);
    color: #ffffff;
}
.badge-current { background: var(--accent); }
.badge-extra   { background: var(--accent-light); }
.badge-more    { font-size: 0.78rem; color: var(--text-secondary); padding: 4px 0; }

.config-pill {
    display: inline-block;
    background: var(--accent-soft);
    border: 1px solid var(--border);
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: var(--text-primary);
    margin-right: 8px;
    margin-bottom: 4px;
    font-family: var(--font-mono);
}
.config-pill strong { color: var(--accent); font-weight: 700; }
</style>
"""

st.markdown(BRAND_CSS, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SECTION 2 — API CONFIGURATION
# ══════════════════════════════════════════════════════════════════════

if 'API_URI' in os.environ:
    BASE_URI = st.secrets[os.environ.get('API_URI')]
else:
    BASE_URI = st.secrets['cloud_api_uri']
BASE_URI = BASE_URI if BASE_URI.endswith('/') else BASE_URI + '/'

OPTIMISE_URL          = BASE_URI + 'optimise'
EXPLAIN_GLOBAL_URL    = BASE_URI + 'explain/global-by-date'
TOP_CANCELLATIONS_URL = BASE_URI + 'top-cancellations'
GROUP_PROBS_URL       = BASE_URI + 'group-probs'
TOP_BOOKINGS_URL      = BASE_URI + 'top-bookings'
EXPLAIN_LOCAL_URL     = BASE_URI + 'explain/local'


# ══════════════════════════════════════════════════════════════════════
# SECTION 3 — UTILITY FUNCTIONS
# ══════════════════════════════════════════════════════════════════════

def poisson_binomial_pmf(probs):
    """Exact Poisson-Binomial PMF via dynamic programming."""
    probs = np.asarray(probs, dtype=np.float64)
    n = len(probs)
    if n == 0:
        return np.array([1.0])
    pmf = np.zeros(n + 1)
    pmf[0] = 1.0
    for p in probs:
        new = np.empty_like(pmf)
        new[0] = pmf[0] * (1 - p)
        new[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf = new
    return pmf


def api_get(url: str, params: dict, timeout: int = 180, max_retries: int = 3):
    """GET with retries for Cloud Run cold starts."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code != 200:
                return {"error": f"API returned status {response.status_code}: {response.text}"}
            return response.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"error": "API request timed out. The service may still be waking up — try again in a minute."}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"error": "Could not connect to the API. Check that the Cloud Run service is running."}
    return {"error": "Unexpected error during API call."}


def api_post(url: str, payload: dict, timeout: int = 60, max_retries: int = 2):
    """POST with retries."""
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(url, json=payload, timeout=timeout)
            if response.status_code != 200:
                return {"error": f"API returned status {response.status_code}: {response.text}"}
            return response.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"error": "API request timed out."}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"error": "Could not connect to the API."}
    return {"error": "Unexpected error during API call."}


# ── Plotly theme helper ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Sans, sans-serif", color="#1B2559", size=12),
    margin=dict(l=10, r=10, t=30, b=10),
    showlegend=False,
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        color="#A3AED0",
        title_font=dict(size=11, color="#A3AED0"),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(163,174,208,0.15)",
        zeroline=False,
        color="#A3AED0",
        title_font=dict(size=11, color="#A3AED0"),
    ),
)


# ══════════════════════════════════════════════════════════════════════
# SECTION 4 — SIDEBAR (Controls)
# ══════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown(
        '<div style="display:flex;align-items:center;gap:10px;margin-bottom:4px;">'
        '<span style="font-size:1.7rem;">🛡️</span>'
        '<span style="font-size:1.25rem;font-weight:700;color:#1B2559;letter-spacing:-0.02em;">'
        'NoShow<span style="color:#4318FF;">Shield</span></span></div>',
        unsafe_allow_html=True,
    )
    st.caption("AI-Powered Revenue Protection")
    st.markdown("---")

    # ── Optimisation settings ────────────────────────────────────────
    st.markdown('<p class="section-label">Optimisation Settings</p>', unsafe_allow_html=True)

    relocation_cost = st.number_input(
        "Relocation cost (€)",
        min_value=0.0, max_value=1000.0, value=300.0, step=50.0,
        help="Cost of relocating a guest to another hotel when overbooked.",
    )
    max_risk = st.slider(
        "Max relocation risk",
        min_value=0.0, max_value=0.10, value=0.02, step=0.01,
        help="Maximum acceptable probability of having to relocate a guest.",
    )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Fetching predictions … (first load may take up to 2 min)"):
            results = api_get(OPTIMISE_URL, {
                "relocation_cost": relocation_cost,
                "max_risk": max_risk,
            })
        if "error" in results:
            st.error(results["error"])
        else:
            st.session_state["results"] = results
            st.session_state["relocation_cost"] = relocation_cost
            st.session_state["max_risk"] = max_risk

    # ── Filters (only show after data is loaded) ─────────────────────
    if "results" in st.session_state:
        st.markdown("---")
        st.markdown('<p class="section-label">Filters</p>', unsafe_allow_html=True)

        recs_sidebar = pd.DataFrame(st.session_state["results"]["recommendations"])
        recs_sidebar["arrival_date"] = pd.to_datetime(recs_sidebar["arrival_date"])

        available_hotels = sorted(recs_sidebar["hotel"].unique())
        selected_hotel = st.selectbox("Hotel", available_hotels)

        available_dates = sorted(
            recs_sidebar[recs_sidebar["hotel"] == selected_hotel]["arrival_date"].dt.date.unique()
        )
        selected_date = st.selectbox("Arrival date", available_dates)

        available_rooms = sorted(
            recs_sidebar[
                (recs_sidebar["hotel"] == selected_hotel)
                & (recs_sidebar["arrival_date"].dt.date == selected_date)
            ]["assigned_room_type"].unique()
        )
        selected_room = st.selectbox("Room type", available_rooms)

        # ── Model info ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<p class="section-label">Model Performance</p>', unsafe_allow_html=True)

        metrics = st.session_state["results"]["metrics"]
        model_info = st.session_state["results"]["model_info"]
        st.caption(model_info.get("model_type", "XGBoost"))

        metrics_df = pd.DataFrame(
            {"Metric": list(metrics.keys()), "Score": list(metrics.values())}
        ).set_index("Metric")
        st.table(metrics_df)


# ══════════════════════════════════════════════════════════════════════
# SECTION 5 — MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════

# ── Hero banner ──────────────────────────────────────────────────────
st.markdown(
    '<div class="hero-banner">'
    '<p class="hero-title">🛡️ NoShow<span class="hero-accent">Shield</span></p>'
    '<p class="hero-subtitle">'
    'Predict booking cancellations and optimise overbooking levels — '
    'maximising hotel revenue while keeping guest relocation risk within your threshold. '
    'Powered by XGBoost + SHAP explainability.'
    '</p></div>',
    unsafe_allow_html=True,
)

# ── Tabs ─────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["Overbooking Recommendations", "Single Booking Prediction"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — OVERBOOKING RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════

with tab1:

    if "results" not in st.session_state:
        # ── Empty state ──────────────────────────────────────────────
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        col_empty_l, col_empty_c, col_empty_r = st.columns([1, 2, 1])
        with col_empty_c:
            st.markdown(
                '<div style="text-align:center;padding:60px 20px;">'
                '<p style="font-size:3rem;margin-bottom:12px;">🛡️</p>'
                '<p style="font-size:1.1rem;font-weight:600;color:#1B2559;margin-bottom:8px;">'
                'Ready to protect revenue</p>'
                '<p style="font-size:0.88rem;color:#A3AED0;max-width:400px;margin:0 auto;">'
                'Set your relocation cost and risk threshold in the sidebar, '
                'then click <strong style="color:#4318FF;">Get Recommendations</strong> to begin.'
                '</p></div>',
                unsafe_allow_html=True,
            )
    else:
        results = st.session_state["results"]
        recs = pd.DataFrame(results["recommendations"])
        recs["arrival_date"] = pd.to_datetime(recs["arrival_date"])
        metrics = results["metrics"]

        # ── Active config pills ──────────────────────────────────────
        rc = st.session_state.get("relocation_cost", relocation_cost)
        mr = st.session_state.get("max_risk", max_risk)
        st.markdown(
            f'<div style="margin-bottom:18px;">'
            f'<span class="config-pill">Relocation <strong>€{rc:.0f}</strong></span>'
            f'<span class="config-pill">Max Risk <strong>{mr*100:.1f}%</strong></span>'
            f'<span class="config-pill">AUC <strong>{metrics.get("auc","—")}</strong></span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # ── Filter data ─────────────────────────────────────────────
        filtered = recs[
            (recs["hotel"] == selected_hotel)
            & (recs["arrival_date"].dt.date == selected_date)
            & (recs["assigned_room_type"] == selected_room)
        ]

        if filtered.empty:
            st.warning("No data available for this selection.")
        else:
            row = filtered.iloc[0]

            # ── KPI row 1: Inventory snapshot ────────────────────────
            st.markdown('<p class="section-label">Inventory Snapshot</p>', unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            k1.metric("Capacity", int(row["capacity"]))
            k2.metric("Current Bookings", int(row["total_bookings"]))
            k3.metric("Expected Show-ups", round(row["expected_show_ups"], 1))

            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            # ── KPI row 2: Action recommendation ────────────────────
            st.markdown('<p class="section-label">Recommendation</p>', unsafe_allow_html=True)
            k4, k5, k6 = st.columns(3)
            k4.metric("Extra Bookings", int(row["recommended_extra"]))
            k5.metric("Net Benefit", f"€{row['net_benefit']:.2f}")
            k6.metric("Relocation Risk", f"{row['relocation_probability'] * 100:.2f}%")

            st.divider()

            # ──────────────────────────────────────────────────────────
            # Show-up Distribution (Poisson-Binomial)
            # ──────────────────────────────────────────────────────────
            st.markdown('<p class="section-label">Show-up Distribution</p>', unsafe_allow_html=True)
            st.caption(
                f"Poisson-Binomial distribution for **{selected_room}** on "
                f"**{selected_date}** — slide to simulate adding bookings."
            )

            recommended_total = int(row["recommended_total"])
            n_current = int(row["total_bookings"])
            capacity = int(row["capacity"])

            # Fetch individual cancel probs
            probs_cache_key = f"gprobs_{selected_hotel}_{selected_date}_{selected_room}"
            if probs_cache_key not in st.session_state:
                try:
                    resp = requests.get(
                        GROUP_PROBS_URL,
                        params={
                            "hotel": selected_hotel,
                            "arrival_date": str(selected_date),
                            "room_type": selected_room,
                        },
                        timeout=3,
                    )
                    gp_result = resp.json() if resp.status_code == 200 else {}
                    if "cancel_probs" in gp_result:
                        st.session_state[probs_cache_key] = gp_result
                except Exception:
                    gp_result = {}
            else:
                gp_result = st.session_state[probs_cache_key]

            if "cancel_probs" in gp_result:
                cancel_probs_arr = np.array(gp_result["cancel_probs"], dtype=np.float64)
            else:
                mean_cp = row.get(
                    "cancel_prob_mean",
                    row["expected_cancellations"] / row["total_bookings"],
                )
                cancel_probs_arr = np.full(n_current, float(mean_cp))

            n_simulate = st.slider(
                "Total bookings to simulate",
                min_value=0, max_value=recommended_total, value=recommended_total,
                help="Drag to see how the show-up distribution shifts as bookings are added.",
            )

            # Compute PMF locally
            if n_simulate == 0:
                show_pmf = np.array([1.0])
                mean_su = std_su = reloc_prob = 0.0
                indiv_show = np.array([])
            else:
                if n_simulate <= n_current:
                    sel_cancel = cancel_probs_arr[:n_simulate]
                else:
                    mean_cancel = cancel_probs_arr.mean()
                    sel_cancel = np.concatenate([
                        cancel_probs_arr,
                        np.full(n_simulate - n_current, mean_cancel),
                    ])
                indiv_show = 1.0 - sel_cancel
                show_pmf = poisson_binomial_pmf(indiv_show)
                mean_su = float(indiv_show.sum())
                std_su = float(np.sqrt((indiv_show * (1 - indiv_show)).sum()))
                reloc_prob = float(show_pmf[capacity + 1:].sum()) if capacity + 1 <= n_simulate else 0.0

            # Stats row
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Simulated", n_simulate)
            d2.metric("E[Show-ups]", f"{mean_su:.1f}")
            d3.metric("Std Dev", f"{std_su:.2f}")
            d4.metric("Reloc. Risk", f"{reloc_prob * 100:.2f}%")

            # ── Distribution chart ───────────────────────────────────
            x_vals = list(range(len(show_pmf)))
            bar_colors = [
                "#4318FF" if k <= n_current
                else "#7551FF" if k <= capacity
                else "#E31A1A"
                for k in x_vals
            ]

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(
                x=x_vals, y=show_pmf.tolist(),
                marker_color=bar_colors,
                marker_line_width=0,
                hovertemplate="Show-ups: %{x}<br>P = %{y:.4f}<extra></extra>",
            ))
            fig_dist.add_vline(
                x=capacity + 0.5,
                line_dash="dot", line_color="#E31A1A", line_width=2,
                annotation_text=f"Capacity ({capacity})",
                annotation_position="top left",
                annotation_font=dict(color="#E31A1A", size=11),
            )
            fig_dist.update_layout(
                **PLOTLY_LAYOUT,
                height=380,
                xaxis_title="Number of Show-ups",
                yaxis_title="Probability",
                bargap=0.06,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # ── Individual show-up probability badges ────────────────
            if n_simulate > 0:
                show_pcts = (indiv_show * 100).astype(int).tolist()
                display_limit = 18
                badges = '<div class="badge-row">'
                for i, pct in enumerate(show_pcts[:display_limit]):
                    cls = "badge-current" if i < n_current else "badge-extra"
                    badges += f'<span class="badge {cls}">{pct}%</span>'
                remaining = len(show_pcts) - display_limit
                if remaining > 0:
                    badges += f'<span class="badge-more">+{remaining} more</span>'
                badges += '</div>'

                st.markdown(
                    '<p style="font-size:0.78rem;color:#A3AED0;margin-bottom:2px;">'
                    'Individual show-up probabilities '
                    '(<span style="color:#4318FF">● current</span> '
                    '<span style="color:#7551FF">● extra</span>)</p>',
                    unsafe_allow_html=True,
                )
                st.markdown(badges, unsafe_allow_html=True)

            st.divider()

            # ──────────────────────────────────────────────────────────
            # Two-column: Top Cancellations + SHAP
            # ──────────────────────────────────────────────────────────
            left_col, right_col = st.columns([3, 2])

            # ── Left: Top 3 likely cancellations ─────────────────────
            with left_col:
                st.markdown('<p class="section-label">Top 3 Likely Cancellations</p>', unsafe_allow_html=True)
                st.caption(
                    f"Highest-risk bookings for **{selected_room}** on **{selected_date}**"
                )

                top_3_result = api_get(
                    TOP_CANCELLATIONS_URL,
                    {
                        "hotel": selected_hotel,
                        "arrival_date": str(selected_date),
                        "room_type": selected_room,
                    },
                    timeout=30, max_retries=2,
                )
                if "error" in top_3_result:
                    st.warning(f"Could not load: {top_3_result['error']}")
                else:
                    top_3_data = top_3_result.get("top_3", [])
                    if not top_3_data:
                        st.info("No booking data available.")
                    else:
                        top_3_df = pd.DataFrame(top_3_data)
                        top_3_df["cancel_prob"] = (top_3_df["cancel_prob"] * 100).map("{:.1f}%".format)
                        top_3_df["adr"] = top_3_df["adr"].map("€{:.2f}".format)
                        col_mapping = {
                            "lead_time": "Lead Time",
                            "adr": "ADR",
                            "market_segment": "Segment",
                            "deposit_type": "Deposit",
                            "customer_type": "Customer",
                            "cancel_prob": "Cancel Risk",
                        }
                        top_3_df = top_3_df.rename(columns=col_mapping)
                        display_cols = [v for v in col_mapping.values() if v in top_3_df.columns]
                        st.table(top_3_df[display_cols])

            # ── Right: SHAP chart ────────────────────────────────────
            with right_col:
                st.markdown('<p class="section-label">SHAP — Risk Factors</p>', unsafe_allow_html=True)
                st.caption(
                    f"Why **{selected_room}** bookings on **{selected_date}** cancel"
                )

                selected_date_str = str(selected_date)
                cache_key = f"shap_{selected_hotel}_{selected_date_str}_{selected_room}"

                if cache_key not in st.session_state:
                    with st.spinner("Loading SHAP …"):
                        shap_result = api_get(
                            EXPLAIN_GLOBAL_URL,
                            {
                                "selected_date": selected_date_str,
                                "room_type": selected_room,
                                "hotel": selected_hotel,
                            },
                            timeout=60, max_retries=2,
                        )
                    st.session_state[cache_key] = shap_result
                else:
                    shap_result = st.session_state[cache_key]

                if "error" in shap_result:
                    st.warning(f"Could not load SHAP: {shap_result['error']}")
                elif shap_result.get("message"):
                    st.info(shap_result["message"])
                elif shap_result.get("grouped_global_shap"):
                    shap_df = pd.DataFrame(shap_result["grouped_global_shap"]).copy()
                    shap_df["mean_abs_shap"] = pd.to_numeric(shap_df["mean_abs_shap"], errors="coerce")
                    shap_df["feature"] = (
                        shap_df["feature_group"].astype(str)
                        .str.replace("cat_ordinal__", "", regex=False)
                        .str.replace("_", " ", regex=False)
                        .str.title()
                    )
                    top_shap = (
                        shap_df[["feature", "mean_abs_shap"]]
                        .dropna()
                        .sort_values("mean_abs_shap", ascending=False)
                        .head(5)
                        .reset_index(drop=True)
                    )

                    if top_shap.empty:
                        st.info("No SHAP data available for this selection.")
                    else:
                        plot_df = top_shap.iloc[::-1].reset_index(drop=True)
                        fig_shap = go.Figure(go.Bar(
                            x=plot_df["mean_abs_shap"].tolist(),
                            y=plot_df["feature"].tolist(),
                            orientation="h",
                            marker_color="#4318FF",
                            marker_line_width=0,
                        ))
                        fig_shap.update_layout(
                            **PLOTLY_LAYOUT,
                            height=380,
                            xaxis_title="Mean |SHAP|",
                        )
                        fig_shap.update_yaxes(
                            type="category",
                            categoryorder="array",
                            categoryarray=plot_df["feature"].tolist(),
                            automargin=True,
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.info("No SHAP data available for this selection.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — SINGLE BOOKING PREDICTION
# ══════════════════════════════════════════════════════════════════════

with tab2:

    if "results" not in st.session_state:
        # ── Empty state ──────────────────────────────────────────────
        st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)
        col_e_l, col_e_c, col_e_r = st.columns([1, 2, 1])
        with col_e_c:
            st.markdown(
                '<div style="text-align:center;padding:60px 20px;">'
                '<p style="font-size:3rem;margin-bottom:12px;">🔎</p>'
                '<p style="font-size:1.1rem;font-weight:600;color:#1B2559;margin-bottom:8px;">'
                'Inspect individual bookings</p>'
                '<p style="font-size:0.88rem;color:#A3AED0;max-width:420px;margin:0 auto;">'
                'Load recommendations first, then switch to this tab to explore '
                'the highest-risk bookings with per-feature SHAP explanations.'
                '</p></div>',
                unsafe_allow_html=True,
            )
    else:
        st.markdown('<p class="section-label">Single Booking Deep-Dive</p>', unsafe_allow_html=True)
        st.caption(
            "Select one of the highest-risk bookings to see the model's prediction "
            "and the SHAP values explaining it."
        )

        # Load top bookings once
        if "top_bookings_list" not in st.session_state:
            with st.spinner("Loading high-risk bookings …"):
                top_result = api_get(TOP_BOOKINGS_URL, {}, timeout=60, max_retries=2)
            if "error" in top_result:
                st.error(top_result["error"])
                st.session_state["top_bookings_list"] = []
            else:
                st.session_state["top_bookings_list"] = top_result["top_bookings"]

        top_bookings_list = st.session_state.get("top_bookings_list", [])

        placeholder = "Select booking for prediction"
        dropdown_labels = [placeholder] + [b["label"] for b in top_bookings_list]
        selected_label = st.selectbox(
            "High-risk booking",
            options=dropdown_labels,
            index=0,
            disabled=not top_bookings_list,
        )

        if selected_label != placeholder and top_bookings_list:
            selected_entry = next(b for b in top_bookings_list if b["label"] == selected_label)

            cache_key = f"explain_{selected_entry['rank']}"
            if cache_key not in st.session_state:
                with st.spinner("Running prediction & SHAP …"):
                    explain_result = api_post(EXPLAIN_LOCAL_URL, selected_entry["booking"])
                if "error" in explain_result:
                    st.error(explain_result["error"])
                else:
                    st.session_state[cache_key] = explain_result

            if cache_key in st.session_state:
                st.session_state["single_booking"] = selected_entry["booking"]
                st.session_state["single_actual"] = selected_entry["actual_outcome"]
                st.session_state["single_explain"] = st.session_state[cache_key]

        if "single_booking" not in st.session_state:
            if top_bookings_list:
                st.info("Select a booking above to inspect its prediction.")
        else:
            booking = st.session_state["single_booking"]
            actual  = st.session_state["single_actual"]
            explain = st.session_state["single_explain"]

            prob       = explain["cancellation_probability"]
            prediction = explain.get("prediction", int(prob >= 0.5))

            # ── Verdict row ──────────────────────────────────────────
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            v1, v2, v3 = st.columns(3)
            v1.metric(
                "Prediction",
                "Will Cancel" if prediction == 1 else "Won't Cancel",
            )
            v2.metric("Cancel Probability", f"{prob * 100:.1f}%")
            v3.metric(
                "Actual Outcome",
                "Cancelled" if actual == 1 else "Not Cancelled",
            )

            st.divider()

            # ── Two-column: details + SHAP ───────────────────────────
            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.markdown('<p class="section-label">Booking Details</p>', unsafe_allow_html=True)
                details = pd.DataFrame(
                    {"Field": list(booking.keys()), "Value": [str(v) for v in booking.values()]}
                ).set_index("Field")
                st.dataframe(details, use_container_width=True, height=420)

            with right_col:
                st.markdown('<p class="section-label">SHAP — Cancellation Drivers</p>', unsafe_allow_html=True)
                st.caption("Features pushing this booking toward cancellation")

                shap_df = pd.DataFrame(explain["grouped_local_shap"]).copy()
                shap_df["shap_value"] = pd.to_numeric(shap_df["shap_value"], errors="coerce")
                shap_df["feature"] = (
                    shap_df["feature_group"].astype(str)
                    .str.replace("_", " ", regex=False)
                    .str.title()
                )
                top_shap = (
                    shap_df[shap_df["shap_value"] > 0][["feature", "shap_value"]]
                    .dropna()
                    .sort_values("shap_value", ascending=False)
                    .head(5)
                    .reset_index(drop=True)
                )

                if top_shap.empty:
                    st.info("No cancellation risk factors found.")
                else:
                    plot_df = top_shap.iloc[::-1].reset_index(drop=True)
                    fig_local = go.Figure(go.Bar(
                        x=plot_df["shap_value"].tolist(),
                        y=plot_df["feature"].tolist(),
                        orientation="h",
                        marker_color="#4318FF",
                        marker_line_width=0,
                    ))
                    fig_local.update_layout(
                        **PLOTLY_LAYOUT,
                        height=380,
                        xaxis_title="SHAP Value",
                    )
                    fig_local.update_yaxes(
                        type="category",
                        categoryorder="array",
                        categoryarray=plot_df["feature"].tolist(),
                        automargin=True,
                    )
                    st.plotly_chart(fig_local, use_container_width=True)
