"""
NoShowShield — Streamlit dashboard for overbooking recommendations
Connects to the live FastAPI on Google Cloud Run.

Run with:
    streamlit run app.py
"""

import os
import streamlit as st
import pandas as pd
import requests
import time
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# API configuration
# ------------------------------------------------------------------
if 'API_URI' in os.environ:
    BASE_URI = st.secrets[os.environ.get('API_URI')]
else:
    BASE_URI = st.secrets['cloud_api_uri']
BASE_URI = BASE_URI if BASE_URI.endswith('/') else BASE_URI + '/'

OPTIMISE_URL = BASE_URI + 'optimise'
EXPLAIN_GLOBAL_URL = BASE_URI + 'explain/global-by-date'
TOP_CANCELLATIONS_URL = BASE_URI + 'top-cancellations'
GROUP_PROBS_URL = BASE_URI + 'group-probs'


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="NoShowShield", page_icon="🛡️", layout="wide")


# ------------------------------------------------------------------
# 🎨  THEME — Custom CSS Injection
# ------------------------------------------------------------------
CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=Space+Grotesk:wght@400;500;600;700&display=swap');

/* ── ROOT PALETTE ────────────────────────────────────── */
:root {
    --primary:        #6C5CE7;
    --primary-light:  #A29BFE;
    --primary-lighter: #DDD6FE;
    --primary-bg:     #F3F0FF;
    --accent:         #8B5CF6;
    --surface:        #FFFFFF;
    --surface-glass:  rgba(255, 255, 255, 0.72);
    --text-primary:   #1E1B4B;
    --text-secondary: #6B7280;
    --text-muted:     #9CA3AF;
    --border:         rgba(108, 92, 231, 0.12);
    --shadow-sm:      0 1px 3px rgba(108, 92, 231, 0.06);
    --shadow-md:      0 4px 14px rgba(108, 92, 231, 0.08);
    --shadow-lg:      0 8px 30px rgba(108, 92, 231, 0.12);
    --shadow-glow:    0 0 20px rgba(108, 92, 231, 0.15);
    --radius:         16px;
    --radius-sm:      10px;
    --radius-xs:      6px;
    --green:          #10B981;
    --red:            #EF4444;
    --amber:          #F59E0B;
}

/* ── GLOBAL RESETS ───────────────────────────────────── */
html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif !important;
    color: var(--text-primary) !important;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(145deg, #F8F7FF 0%, #F0EEFF 40%, #EDE9FE 100%) !important;
}

/* ── HEADER / TOOLBAR CLEANUP ────────────────────────── */
header[data-testid="stHeader"] {
    background: transparent !important;
}
#MainMenu, footer, header .stDeployButton {
    display: none !important;
}

/* ── SIDEBAR ─────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1E1B4B 0%, #312E81 100%) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(30, 27, 75, 0.15) !important;
}

[data-testid="stSidebar"] * {
    color: #E0E7FF !important;
}

[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #FFFFFF !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.02em !important;
}

[data-testid="stSidebar"] label {
    color: #C7D2FE !important;
    font-weight: 500 !important;
    font-size: 0.85rem !important;
}

[data-testid="stSidebar"] .stSelectbox > div > div,
[data-testid="stSidebar"] .stNumberInput > div > div > input {
    background: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.15) !important;
    border-radius: var(--radius-sm) !important;
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] .stSlider > div > div > div {
    background: var(--primary-light) !important;
}

[data-testid="stSidebar"] .stTable,
[data-testid="stSidebar"] table {
    background: rgba(255, 255, 255, 0.06) !important;
    border-radius: var(--radius-sm) !important;
    overflow: hidden;
}

[data-testid="stSidebar"] td, [data-testid="stSidebar"] th {
    border-color: rgba(255, 255, 255, 0.08) !important;
    font-size: 0.82rem !important;
}

[data-testid="stSidebar"] .stCaption, [data-testid="stSidebar"] .stCaption p {
    color: #A5B4FC !important;
}

/* ── SIDEBAR BUTTON ──────────────────────────────────── */
[data-testid="stSidebar"] .stButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-sm) !important;
    padding: 0.7rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    letter-spacing: 0.01em !important;
    box-shadow: 0 4px 14px rgba(108, 92, 231, 0.35) !important;
    transition: all 0.25s ease !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(108, 92, 231, 0.45) !important;
}

/* ── MAIN CONTENT AREA ───────────────────────────────── */
.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    max-width: 1200px !important;
}

/* ── HEADINGS ────────────────────────────────────────── */
h1 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.85rem !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.03em !important;
    margin-bottom: 0.25rem !important;
}

h2, .stSubheader {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    color: var(--text-primary) !important;
    letter-spacing: -0.02em !important;
}

h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1.05rem !important;
    color: var(--text-primary) !important;
}

/* ── METRIC CARDS ────────────────────────────────────── */
[data-testid="stMetric"] {
    background: var(--surface-glass) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 1.15rem 1.3rem !important;
    box-shadow: var(--shadow-md) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stMetric"]:hover {
    box-shadow: var(--shadow-lg) !important;
    transform: translateY(-2px);
    border-color: rgba(108, 92, 231, 0.22);
}

[data-testid="stMetric"] label {
    color: var(--text-secondary) !important;
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}

[data-testid="stMetric"] [data-testid="stMetricValue"] {
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    font-size: 1.65rem !important;
    letter-spacing: -0.02em !important;
}

/* ── TABS ────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0 !important;
    background: var(--surface-glass) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: var(--radius) !important;
    padding: 5px !important;
    border: 1px solid var(--border) !important;
    box-shadow: var(--shadow-sm) !important;
    margin-bottom: 1.5rem !important;
}

.stTabs [data-baseweb="tab"] {
    border-radius: var(--radius-sm) !important;
    padding: 0.6rem 1.5rem !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    color: var(--text-secondary) !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%) !important;
    color: white !important;
    font-weight: 600 !important;
    box-shadow: 0 2px 8px rgba(108, 92, 231, 0.3) !important;
}

.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] {
    display: none !important;
}

/* ── DIVIDERS ────────────────────────────────────────── */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent, var(--primary-lighter), transparent) !important;
    margin: 1.5rem 0 !important;
}

/* ── TABLES ──────────────────────────────────────────── */
.stTable, [data-testid="stTable"] {
    background: var(--surface-glass) !important;
    backdrop-filter: blur(8px) !important;
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid var(--border) !important;
}

table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
}

th {
    background: rgba(108, 92, 231, 0.06) !important;
    color: var(--text-secondary) !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
    padding: 0.75rem 1rem !important;
    border-bottom: 1px solid var(--border) !important;
}

td {
    padding: 0.65rem 1rem !important;
    font-size: 0.88rem !important;
    border-bottom: 1px solid rgba(108, 92, 231, 0.06) !important;
    color: var(--text-primary) !important;
}

tr:last-child td {
    border-bottom: none !important;
}

/* ── DATAFRAMES ──────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: var(--radius) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
    border: 1px solid var(--border) !important;
}

/* ── INFO / WARNING / ERROR BOXES ────────────────────── */
.stAlert {
    border-radius: var(--radius-sm) !important;
    border: none !important;
    font-size: 0.88rem !important;
}

[data-testid="stAlert"][data-baseweb] {
    border-radius: var(--radius-sm) !important;
}

/* ── SLIDER ──────────────────────────────────────────── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background: var(--primary) !important;
    border-color: var(--primary) !important;
}

/* ── SELECTBOX ───────────────────────────────────────── */
.stSelectbox > div > div {
    border-radius: var(--radius-sm) !important;
    border-color: var(--border) !important;
}

/* ── SPINNER ─────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: var(--primary) !important;
}

/* ── PLOTLY CHARTS ───────────────────────────────────── */
[data-testid="stPlotlyChart"] {
    background: var(--surface-glass) !important;
    backdrop-filter: blur(8px) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 0.75rem !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── CAPTION ─────────────────────────────────────────── */
.stCaption, .stCaption p {
    color: var(--text-muted) !important;
    font-size: 0.8rem !important;
}

/* ── SCROLLBAR ───────────────────────────────────────── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb {
    background: var(--primary-lighter);
    border-radius: 3px;
}
::-webkit-scrollbar-thumb:hover { background: var(--primary-light); }

/* ── BADGE WRAPPER (for show-up probs) ───────────────── */
.badge-strip {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin-top: 4px;
}
.badge-strip .badge {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    min-width: 36px;
    padding: 4px 10px;
    border-radius: 20px;
    font-size: 0.75rem;
    font-weight: 600;
    color: white;
}
.badge-current { background: var(--primary); }
.badge-extra   { background: var(--green); }
.badge-more    { color: var(--text-muted) !important; font-weight: 400; font-size: 0.8rem; }
</style>
"""

st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


# ------------------------------------------------------------------
# 🎨  Plotly Theme Constants
# ------------------------------------------------------------------
PLOTLY_BG = "rgba(0,0,0,0)"
PLOTLY_GRID = "rgba(108, 92, 231, 0.06)"
PLOTLY_PRIMARY = "#6C5CE7"
PLOTLY_PRIMARY_LIGHT = "#A29BFE"
PLOTLY_GREEN = "#10B981"
PLOTLY_RED = "#EF4444"
PLOTLY_FONT = dict(family="DM Sans, sans-serif", color="#1E1B4B")

PLOTLY_LAYOUT_DEFAULTS = dict(
    paper_bgcolor=PLOTLY_BG,
    plot_bgcolor=PLOTLY_BG,
    font=PLOTLY_FONT,
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(showgrid=True, gridcolor=PLOTLY_GRID, zeroline=False),
    margin=dict(l=20, r=20, t=30, b=40),
    showlegend=False,
)


# ------------------------------------------------------------------
# App header
# ------------------------------------------------------------------
st.markdown(
    """
    <div style="margin-bottom: 0.25rem;">
        <span style="font-size: 2rem; margin-right: 0.4rem;">🛡️</span>
        <span style="font-family: 'DM Sans', sans-serif; font-weight: 700;
                      font-size: 1.85rem; color: #1E1B4B; letter-spacing: -0.03em;">
            NoShowShield
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <p style="color: #6B7280; font-size: 0.92rem; line-height: 1.6; max-width: 720px; margin-bottom: 1.5rem;">
        AI-powered revenue protection against hotel cancellations.
        Select a date and room type to get overbooking recommendations
        backed by SHAP explainability.
    </p>
    """,
    unsafe_allow_html=True,
)


# ------------------------------------------------------------------
# Inline Poisson-Binomial PMF
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="text-align: center; margin-bottom: 1.5rem; padding-top: 0.5rem;">
            <div style="font-size: 2rem; margin-bottom: 0.25rem;">🛡️</div>
            <div style="font-size: 1.1rem; font-weight: 700; color: #FFFFFF !important;
                        letter-spacing: -0.02em;">NoShowShield</div>
            <div style="font-size: 0.72rem; color: #A5B4FC !important;
                        text-transform: uppercase; letter-spacing: 0.1em; margin-top: 2px;">
                Revenue Protection
            </div>
        </div>
        <hr style="border: none; height: 1px;
                    background: linear-gradient(90deg, transparent, rgba(165,180,252,0.3), transparent);
                    margin: 0.5rem 0 1.2rem 0;">
        """,
        unsafe_allow_html=True,
    )

    st.markdown("#### ⚙️ Optimization Settings")

    relocation_cost = st.number_input(
        "💰 Relocation cost (€)",
        min_value=0.0,
        max_value=1000.0,
        value=300.0,
        step=50.0,
        help="Cost of relocating a guest to another hotel when overbooked.",
    )

    max_risk = st.slider(
        "📊 Max relocation risk",
        min_value=0.0,
        max_value=0.10,
        value=0.02,
        step=0.01,
        help="Maximum acceptable probability of having to relocate a guest.",
    )


# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------
def api_get(url: str, params: dict, timeout: int = 180, max_retries: int = 3):
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
            return {"error": "API request timed out. The API may still be waking up — try again in a minute."}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"error": "Could not connect to the API. Check that the Cloud Run service is running."}
    return {"error": "Unexpected error during API call."}


def api_post(url: str, payload: dict, timeout: int = 60, max_retries: int = 2):
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


# ------------------------------------------------------------------
# Load optimisation data
# ------------------------------------------------------------------
with st.sidebar:
    if st.button("🚀  Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Fetching predictions from API …"):
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


# ==================================================================
# TABS
# ==================================================================
tab1, tab2 = st.tabs(["📋  Overbooking Recommendations", "🔍  Single Booking Prediction"])


# ==================================================================
# TAB 1
# ==================================================================
with tab1:
    if "results" not in st.session_state:
        st.markdown(
            """
            <div style="text-align: center; padding: 4rem 2rem;">
                <div style="font-size: 3rem; margin-bottom: 1rem; opacity: 0.5;">🛡️</div>
                <p style="color: #6B7280; font-size: 0.95rem; max-width: 420px; margin: 0 auto;">
                    Adjust settings in the sidebar and click
                    <strong>Get Recommendations</strong> to start.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        results = st.session_state["results"]

        # Parse API response
        recs = pd.DataFrame(results["recommendations"])
        recs["arrival_date"] = pd.to_datetime(recs["arrival_date"])
        metrics = results["metrics"]
        model_info = results["model_info"]

        # Sidebar — filters
        with st.sidebar:
            st.markdown(
                '<hr style="border:none;height:1px;background:linear-gradient(90deg,transparent,rgba(165,180,252,0.3),transparent);margin:1.2rem 0;">',
                unsafe_allow_html=True,
            )
            st.markdown("#### 🔎 Filters")

            available_hotels = sorted(recs["hotel"].unique())
            selected_hotel = st.selectbox("🏨 Select hotel", available_hotels)

            available_dates = sorted(
                recs[recs["hotel"] == selected_hotel]["arrival_date"].dt.date.unique()
            )
            selected_date = st.selectbox("📅 Select date", available_dates)

            available_rooms = sorted(
                recs[
                    (recs["hotel"] == selected_hotel)
                    & (recs["arrival_date"].dt.date == selected_date)
                ]["assigned_room_type"].unique()
            )
            selected_room = st.selectbox("🛏️ Select room type", available_rooms)

        # Sidebar — model info
        with st.sidebar:
            st.markdown(
                '<hr style="border:none;height:1px;background:linear-gradient(90deg,transparent,rgba(165,180,252,0.3),transparent);margin:1.2rem 0;">',
                unsafe_allow_html=True,
            )
            st.markdown("#### 🤖 Model Info")
            st.caption(model_info.get("model_type", "XGBoost"))
            metrics_df = pd.DataFrame(
                {"Metric": list(metrics.keys()), "Score": list(metrics.values())}
            ).set_index("Metric")
            st.table(metrics_df)

        # Status bar
        st.caption(
            f"Relocation cost = €{st.session_state.get('relocation_cost', relocation_cost):.0f}  ·  "
            f"Max risk = {st.session_state.get('max_risk', max_risk) * 100:.1f}%  ·  "
            f"Model AUC = {metrics.get('auc', '—')}"
        )

        # Filter recommendations
        filtered = recs[
            (recs["hotel"] == selected_hotel)
            & (recs["arrival_date"].dt.date == selected_date)
            & (recs["assigned_room_type"] == selected_room)
        ]

        # Display recommendation
        st.subheader("Recommendation")

        if filtered.empty:
            st.warning("No data available for this selection.")
        else:
            row = filtered.iloc[0]

            # Top metrics row
            col1, col2, col3 = st.columns(3)
            col1.metric("Capacity", int(row["capacity"]))
            col2.metric("Current Bookings", int(row["total_bookings"]))
            col3.metric("Expected Show-ups", round(row["expected_show_ups"], 1))

            st.divider()

            col4, col5, col6 = st.columns(3)
            col4.metric("Recommended Extra Bookings", int(row["recommended_extra"]))
            col5.metric("Net Benefit (€)", f"€{row['net_benefit']:.2f}")
            col6.metric("Relocation Risk", f"{row['relocation_probability'] * 100:.2f}%")

            st.divider()

            # ----------------------------------------------------------
            # Show-up Distribution
            # ----------------------------------------------------------
            st.subheader("Show-up Distribution")
            st.caption(
                f"Poisson-Binomial distribution of expected show-ups for "
                f"**{selected_room}** on **{selected_date}**. "
                f"Slide to see how adding bookings shifts the distribution."
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
                    if resp.status_code == 200:
                        gp_result = resp.json()
                        if "cancel_probs" in gp_result:
                            st.session_state[probs_cache_key] = gp_result
                    else:
                        gp_result = {}
                except Exception:
                    gp_result = {}
            else:
                gp_result = st.session_state[probs_cache_key]

            if "cancel_probs" in gp_result:
                cancel_probs_arr = np.array(gp_result["cancel_probs"], dtype=np.float64)
            else:
                mean_cp = row.get("cancel_prob_mean", row["expected_cancellations"] / row["total_bookings"])
                cancel_probs_arr = np.full(n_current, float(mean_cp))

            n_simulate = st.slider(
                "Total bookings to simulate",
                min_value=0,
                max_value=recommended_total,
                value=recommended_total,
                help="Drag to see how the show-up distribution changes as bookings are added.",
            )

            # Compute show-up PMF locally
            if n_simulate == 0:
                show_pmf = np.array([1.0])
                mean_su = 0.0
                std_su = 0.0
                reloc_prob = 0.0
                indiv_show = np.array([])
            else:
                if n_simulate <= n_current:
                    sel_cancel = cancel_probs_arr[:n_simulate]
                else:
                    mean_cancel = cancel_probs_arr.mean()
                    extra = n_simulate - n_current
                    sel_cancel = np.concatenate([
                        cancel_probs_arr,
                        np.full(extra, mean_cancel),
                    ])

                indiv_show = 1.0 - sel_cancel
                show_pmf = poisson_binomial_pmf(indiv_show)
                mean_su = float(indiv_show.sum())
                std_su = float(np.sqrt((indiv_show * (1 - indiv_show)).sum()))

                if capacity + 1 <= n_simulate:
                    reloc_prob = float(show_pmf[capacity + 1:].sum())
                else:
                    reloc_prob = 0.0

            # Stats row
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            dcol1.metric("Bookings Simulated", n_simulate)
            dcol2.metric("Expected Show-ups", f"{mean_su:.1f}")
            dcol3.metric("Std Deviation", f"{std_su:.2f}")
            dcol4.metric("Relocation Risk", f"{reloc_prob * 100:.2f}%")

            # Plotly PMF chart — themed
            x_vals = list(range(len(show_pmf)))

            bar_colors = [
                PLOTLY_PRIMARY if k <= n_current
                else PLOTLY_GREEN if k <= capacity
                else PLOTLY_RED
                for k in x_vals
            ]

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(
                x=x_vals,
                y=show_pmf.tolist(),
                marker_color=bar_colors,
                marker_line=dict(width=0),
                name="P(show-ups = k)",
                hovertemplate="Show-ups: %{x}<br>Probability: %{y:.4f}<extra></extra>",
            ))

            fig_dist.add_vline(
                x=capacity + 0.5,
                line_dash="dash",
                line_color=PLOTLY_RED,
                line_width=2,
                annotation_text=f"Capacity = {capacity}",
                annotation_position="top left",
                annotation_font_color=PLOTLY_RED,
                annotation_font_size=12,
                annotation_font=dict(family="DM Sans"),
            )

            fig_dist.update_layout(
                **PLOTLY_LAYOUT_DEFAULTS,
                height=380,
                xaxis_title="Number of Show-ups",
                yaxis_title="Probability",
                bargap=0.08,
            )

            st.plotly_chart(fig_dist, use_container_width=True)

            # Individual show-up probability badges
            if n_simulate > 0:
                show_pcts = (indiv_show * 100).astype(int).tolist()
                display_limit = 15

                badges_html = '<div class="badge-strip">'
                for i, pct in enumerate(show_pcts[:display_limit]):
                    cls = "badge badge-current" if i < n_current else "badge badge-extra"
                    badges_html += f'<span class="{cls}">{pct}%</span>'
                remaining = len(show_pcts) - display_limit
                if remaining > 0:
                    badges_html += f'<span class="badge-more">…+{remaining} more</span>'
                badges_html += '</div>'

                st.markdown(
                    f"Individual show-up probabilities "
                    f"(<span style='color:{PLOTLY_PRIMARY}'>●</span> Current "
                    f"<span style='color:{PLOTLY_GREEN}'>●</span> Extra)",
                    unsafe_allow_html=True,
                )
                st.markdown(badges_html, unsafe_allow_html=True)

            st.divider()

            # ----------------------------------------------------------
            # SHAP + Top Cancellations — two-column layout
            # ----------------------------------------------------------
            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.subheader("Top 3 Likely Cancellations")
                st.caption(
                    f"Bookings for **{selected_room}** on **{selected_date}** with highest risk"
                )

                top_3_result = api_get(
                    TOP_CANCELLATIONS_URL,
                    {
                        "hotel": selected_hotel,
                        "arrival_date": str(selected_date),
                        "room_type": selected_room,
                    },
                    timeout=30,
                    max_retries=2,
                )

                if "error" in top_3_result:
                    st.warning(f"Could not load top cancellations: {top_3_result['error']}")
                else:
                    top_3_data = top_3_result.get("top_3", [])
                    if not top_3_data:
                        st.info("No booking data available.")
                    else:
                        top_3_df = pd.DataFrame(top_3_data)
                        top_3_df["cancel_prob"] = (top_3_df["cancel_prob"] * 100).map("{:.1f}%".format)
                        top_3_df["adr"] = top_3_df["adr"].map("€{:.2f}".format)

                        col_mapping = {
                            "lead_time": "Lead Time (days)",
                            "adr": "ADR",
                            "market_segment": "Market Segment",
                            "deposit_type": "Deposit",
                            "customer_type": "Customer",
                            "cancel_prob": "Cancel Risk",
                        }
                        top_3_df = top_3_df.rename(columns=col_mapping)
                        display_cols = [v for k, v in col_mapping.items() if v in top_3_df.columns]
                        st.table(top_3_df[display_cols])

            with right_col:
                st.subheader("SHAP — Top Risk Factors")
                st.caption(
                    f"Why bookings on **{selected_date}** for room type **{selected_room}** "
                    f"are likely to cancel"
                )

                selected_date_str = str(selected_date)
                cache_key = f"shap_{selected_hotel}_{selected_date_str}_{selected_room}"

                if cache_key not in st.session_state:
                    with st.spinner("Loading SHAP explanations …"):
                        shap_result = api_get(
                            EXPLAIN_GLOBAL_URL,
                            {
                                "selected_date": selected_date_str,
                                "room_type": selected_room,
                                "hotel": selected_hotel,
                            },
                            timeout=60,
                            max_retries=2,
                        )
                    st.session_state[cache_key] = shap_result
                else:
                    shap_result = st.session_state[cache_key]

                if "error" in shap_result:
                    st.warning(f"Could not load SHAP data: {shap_result['error']}")
                elif shap_result.get("message"):
                    st.info(shap_result["message"])
                elif shap_result.get("grouped_global_shap"):
                    shap_df = pd.DataFrame(shap_result["grouped_global_shap"]).copy()
                    shap_df["mean_abs_shap"] = pd.to_numeric(shap_df["mean_abs_shap"], errors="coerce")
                    shap_df["feature"] = (
                        shap_df["feature_group"]
                        .astype(str)
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
                        st.info("No SHAP data available for this date and room type.")
                    else:
                        plot_df = top_shap.iloc[::-1].reset_index(drop=True)

                        fig = go.Figure(
                            go.Bar(
                                x=plot_df["mean_abs_shap"].tolist(),
                                y=plot_df["feature"].tolist(),
                                orientation="h",
                                marker_color=PLOTLY_PRIMARY_LIGHT,
                                marker_line=dict(width=0),
                            )
                        )

                        fig.update_layout(
                            **PLOTLY_LAYOUT_DEFAULTS,
                            height=520,
                            xaxis_title="Mean |SHAP Value|",
                            yaxis_title="",
                        )
                        fig.update_yaxes(
                            type="category",
                            categoryorder="array",
                            categoryarray=plot_df["feature"].tolist(),
                            automargin=True,
                        )
                        fig.update_xaxes(automargin=True)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No SHAP data available for this date and room type.")


# ==================================================================
# TAB 2 — Single booking prediction
# ==================================================================
TOP_BOOKINGS_URL = BASE_URI + 'top-bookings'
EXPLAIN_LOCAL_URL = BASE_URI + 'explain/local'

with tab2:
    st.subheader("Single Booking Prediction")
    st.markdown(
        '<p style="color: #6B7280; font-size: 0.88rem;">Select one of the top 3 bookings '
        'with the highest predicted cancellation risk to see the model\'s prediction '
        'and the SHAP values explaining it.</p>',
        unsafe_allow_html=True,
    )

    if "top_bookings_list" not in st.session_state and "results" in st.session_state:
        with st.spinner("Loading top high-risk bookings …"):
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
        "Select a high-risk booking",
        options=dropdown_labels,
        index=0,
        disabled=not top_bookings_list,
    )

    if selected_label != placeholder and top_bookings_list:
        selected_entry = next(b for b in top_bookings_list if b["label"] == selected_label)

        cache_key = f"explain_{selected_entry['rank']}"
        if cache_key not in st.session_state:
            with st.spinner("Running prediction and SHAP explanation …"):
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
        st.markdown(
            """
            <div style="text-align: center; padding: 3rem 2rem;">
                <div style="font-size: 2.5rem; margin-bottom: 0.8rem; opacity: 0.4;">🔍</div>
                <p style="color: #6B7280; font-size: 0.92rem;">
                    Select a booking above to load its prediction.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        booking = st.session_state["single_booking"]
        actual = st.session_state["single_actual"]
        explain = st.session_state["single_explain"]

        prob = explain["cancellation_probability"]
        prediction = explain.get("prediction", int(prob >= 0.5))

        # Top metrics row
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", "Will Cancel" if prediction == 1 else "Won't Cancel")
        col2.metric("Cancellation Probability", f"{prob * 100:.1f}%")
        col3.metric("Actual Outcome", "Canceled" if actual == 1 else "Not Canceled")

        st.divider()

        # Two-column layout
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.subheader("Booking Details")
            details = pd.DataFrame(
                {"Field": list(booking.keys()), "Value": [str(v) for v in booking.values()]}
            ).set_index("Field")
            st.dataframe(details, use_container_width=True)

        with right_col:
            st.subheader("SHAP — Top 5 Risk Factors")
            st.caption("Features pushing this booking toward cancellation")

            shap_df = pd.DataFrame(explain["grouped_local_shap"]).copy()
            shap_df["shap_value"] = pd.to_numeric(shap_df["shap_value"], errors="coerce")
            shap_df["feature"] = (
                shap_df["feature_group"]
                .astype(str)
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
                st.info("No cancellation risk factors found for this booking.")
            else:
                plot_df = top_shap.iloc[::-1].reset_index(drop=True)

                fig = go.Figure(
                    go.Bar(
                        x=plot_df["shap_value"].tolist(),
                        y=plot_df["feature"].tolist(),
                        orientation="h",
                        marker_color=PLOTLY_PRIMARY_LIGHT,
                        marker_line=dict(width=0),
                    )
                )

                fig.update_layout(
                    **PLOTLY_LAYOUT_DEFAULTS,
                    height=520,
                    xaxis_title="SHAP Value",
                    yaxis_title="",
                )
                fig.update_yaxes(
                    type="category",
                    categoryorder="array",
                    categoryarray=plot_df["feature"].tolist(),
                    automargin=True,
                )
                fig.update_xaxes(automargin=True)
                st.plotly_chart(fig, use_container_width=True)
