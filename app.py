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
TOP_BOOKINGS_URL = BASE_URI + 'top-bookings'
EXPLAIN_LOCAL_URL = BASE_URI + 'explain/local'


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NoShowShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------
# Custom CSS
# ------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --shield-blue: #1B4F72;
        --shield-blue-light: #2E86C1;
        --shield-accent: #F39C12;
        --shield-green: #27AE60;
        --shield-red: #E74C3C;
        --shield-bg: #F8F9FB;
        --shield-card: #FFFFFF;
        --shield-border: #E8ECF1;
        --shield-text: #2C3E50;
        --shield-text-muted: #7F8C8D;
        --shield-shadow: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
        --shield-shadow-md: 0 4px 12px rgba(0,0,0,0.07);
        --shield-radius: 12px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── Header ── */
    .dashboard-header {
        background: linear-gradient(135deg, #1B4F72 0%, #2E86C1 60%, #3498DB 100%);
        border-radius: var(--shield-radius);
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
        position: relative;
        overflow: hidden;
    }
    .dashboard-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -20%;
        width: 400px;
        height: 400px;
        background: radial-gradient(circle, rgba(255,255,255,0.08) 0%, transparent 70%);
        border-radius: 50%;
    }
    .dashboard-header h1 {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        margin: 0 0 0.3rem 0 !important;
        color: white !important;
        letter-spacing: -0.02em;
    }
    .dashboard-header .tagline {
        font-size: 0.95rem;
        color: rgba(255,255,255,0.85);
        font-weight: 400;
        line-height: 1.5;
        max-width: 720px;
    }

    /* ── Sidebar ── */
    section[data-testid="stSidebar"] {
        background: #FAFBFD;
        border-right: 1px solid var(--shield-border);
    }
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        padding-top: 1.5rem;
    }
    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.6rem;
        padding: 0 0 1.2rem 0;
        border-bottom: 1px solid var(--shield-border);
        margin-bottom: 1.2rem;
    }
    .sidebar-brand .logo { font-size: 1.6rem; line-height: 1; }
    .sidebar-brand .name {
        font-family: 'DM Sans', sans-serif;
        font-weight: 700;
        font-size: 1.1rem;
        color: var(--shield-blue);
    }
    .sidebar-section-title {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: var(--shield-text-muted);
        margin: 1.4rem 0 0.6rem 0;
    }

    /* ── Metric cards ── */
    div[data-testid="stMetric"] {
        background: var(--shield-card);
        border: 1px solid var(--shield-border);
        border-radius: var(--shield-radius);
        padding: 1.1rem 1.3rem;
        box-shadow: var(--shield-shadow);
        transition: box-shadow 0.2s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: var(--shield-shadow-md);
    }
    div[data-testid="stMetric"] label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.72rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: var(--shield-text-muted) !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.65rem !important;
        font-weight: 700 !important;
        color: var(--shield-text) !important;
    }

    /* ── Tabs ── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0;
        background: var(--shield-card);
        border: 1px solid var(--shield-border);
        border-radius: var(--shield-radius);
        padding: 0.3rem;
        margin-bottom: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.88rem !important;
        font-weight: 500;
        color: var(--shield-text-muted);
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        border: none !important;
        background: transparent;
        transition: all 0.15s ease;
    }
    .stTabs [aria-selected="true"] {
        background: var(--shield-blue) !important;
        color: white !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-highlight"],
    .stTabs [data-baseweb="tab-border"] { display: none; }

    /* ── Section headers ── */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.55rem;
        margin: 1.8rem 0 1rem 0;
    }
    .section-header .icon {
        width: 32px; height: 32px;
        border-radius: 8px;
        display: flex; align-items: center; justify-content: center;
        font-size: 1rem; flex-shrink: 0;
    }
    .section-header .icon.blue  { background: rgba(46,134,193,0.12); }
    .section-header .icon.amber { background: rgba(243,156,18,0.12); }
    .section-header .icon.green { background: rgba(39,174,96,0.12); }
    .section-header .icon.red   { background: rgba(231,76,60,0.12); }
    .section-header h3 {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 1.1rem !important;
        font-weight: 600 !important;
        color: var(--shield-text) !important;
        margin: 0 !important;
    }

    /* ── Context bar ── */
    .context-bar {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-bottom: 1.2rem;
    }
    .ctx-pill {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 0.35rem 0.85rem;
        border-radius: 20px;
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
    }
    .ctx-pill.hotel { background: rgba(46,134,193,0.1); color: #2E86C1; }
    .ctx-pill.date  { background: rgba(39,174,96,0.1);  color: #27AE60; }
    .ctx-pill.room  { background: rgba(243,156,18,0.1); color: #E67E22; }
    .ctx-pill.auc   { background: #F0F2F5; color: var(--shield-text-muted); }

    /* ── Insight callout ── */
    .insight-box {
        background: linear-gradient(135deg, rgba(46,134,193,0.06), rgba(39,174,96,0.06));
        border: 1px solid rgba(46,134,193,0.15);
        border-left: 4px solid #2E86C1;
        border-radius: 0 var(--shield-radius) var(--shield-radius) 0;
        padding: 1rem 1.3rem;
        margin: 1rem 0 0.5rem 0;
        font-size: 0.88rem;
        line-height: 1.6;
        color: var(--shield-text);
    }
    .insight-box strong { color: var(--shield-blue); }
    .insight-box.warning {
        background: linear-gradient(135deg, rgba(243,156,18,0.06), rgba(231,76,60,0.06));
        border-left-color: var(--shield-accent);
    }
    .insight-box.warning strong { color: #E67E22; }

    /* ── Prediction result badge ── */
    .pred-result {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.82rem;
        font-weight: 600;
        padding: 0.4rem 1rem;
        border-radius: 8px;
        margin-top: 0.3rem;
    }
    .pred-result.correct {
        background: rgba(39,174,96,0.1);
        color: #27AE60;
    }
    .pred-result.incorrect {
        background: rgba(231,76,60,0.1);
        color: #E74C3C;
    }

    /* ── Probability badges ── */
    .prob-badge {
        display: inline-block;
        margin: 3px;
        padding: 5px 10px;
        border-radius: 8px;
        color: white;
        font-size: 0.75rem;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
    }
    .prob-badge.current { background: var(--shield-blue-light); }
    .prob-badge.extra   { background: var(--shield-green); }

    .badge-legend {
        font-family: 'DM Sans', sans-serif;
        font-size: 0.78rem;
        color: var(--shield-text-muted);
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }
    .legend-dot {
        width: 10px; height: 10px;
        border-radius: 3px;
        display: inline-block;
    }

    /* ── Chart legend strip ── */
    .chart-legend {
        display: flex;
        gap: 1.2rem;
        margin: 0.5rem 0 0.2rem 0;
        font-family: 'DM Sans', sans-serif;
        font-size: 0.76rem;
        color: var(--shield-text-muted);
    }
    .chart-legend span {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
    }
    .chart-legend .dot {
        width: 10px; height: 10px;
        border-radius: 3px;
        display: inline-block;
    }

    /* ── Tables ── */
    div[data-testid="stTable"] table {
        font-family: 'DM Sans', sans-serif !important;
        border-radius: var(--shield-radius);
        overflow: hidden;
    }

    hr {
        border: none;
        border-top: 1px solid var(--shield-border);
        margin: 1.2rem 0;
    }

    .stPlotlyChart {
        background: var(--shield-card);
        border: 1px solid var(--shield-border);
        border-radius: var(--shield-radius);
        padding: 0.8rem;
        box-shadow: var(--shield-shadow);
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1B4F72, #2E86C1) !important;
        border: none !important;
        border-radius: 8px !important;
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        padding: 0.6rem 1.2rem !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button[kind="primary"]:hover {
        box-shadow: 0 4px 14px rgba(27,79,114,0.35) !important;
        transform: translateY(-1px);
    }

    .stSelectbox label, .stSlider label, .stNumberInput label {
        font-family: 'DM Sans', sans-serif !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
        color: var(--shield-text) !important;
    }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Shared constants
# ------------------------------------------------------------------
COLORS = {
    "blue": "#2E86C1", "green": "#27AE60", "amber": "#F39C12",
    "red": "#E74C3C", "slate": "#7F8C8D", "dark": "#2C3E50",
}

PLOTLY_LAYOUT = dict(
    font=dict(family="DM Sans, sans-serif", color=COLORS["dark"]),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=10, r=10, t=10, b=10),
    showlegend=False,
)


# ------------------------------------------------------------------
# Poisson-Binomial PMF (local computation)
# ------------------------------------------------------------------
def poisson_binomial_pmf(probs):
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
# API helpers
# ------------------------------------------------------------------
def api_get(url, params, timeout=180, max_retries=3):
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.get(url, params=params, timeout=timeout)
            if r.status_code != 200:
                return {"error": f"API returned status {r.status_code}: {r.text}"}
            return r.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(2); continue
            return {"error": "Request timed out — the API may still be starting. Try again in a minute."}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep(2); continue
            return {"error": "Could not connect to the API. Verify the Cloud Run service is running."}
    return {"error": "Unexpected error during API call."}


def api_post(url, payload, timeout=60, max_retries=2):
    for attempt in range(1, max_retries + 1):
        try:
            r = requests.post(url, json=payload, timeout=timeout)
            if r.status_code != 200:
                return {"error": f"API returned status {r.status_code}: {r.text}"}
            return r.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(2); continue
            return {"error": "Request timed out."}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep(2); continue
            return {"error": "Could not connect to the API."}
    return {"error": "Unexpected error during API call."}


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
        <div class="sidebar-brand">
            <span class="logo">🛡️</span>
            <span class="name">NoShowShield</span>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Optimization Parameters</div>', unsafe_allow_html=True)

    relocation_cost = st.number_input(
        "Relocation cost (€)",
        min_value=0.0, max_value=1000.0, value=300.0, step=50.0,
        help="Average cost of relocating one guest to a partner hotel.",
    )

    max_risk = st.slider(
        "Max relocation risk (%)",
        min_value=0.0, max_value=10.0, value=2.0, step=1.0,
        help="Upper limit on the probability of needing to relocate any guest.",
    )
    max_risk_frac = max_risk / 100.0

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    if st.button("🚀  Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Fetching predictions … (first call may take ~2 min for cold start)"):
            results = api_get(OPTIMISE_URL, {
                "relocation_cost": relocation_cost,
                "max_risk": max_risk_frac,
            })
        if "error" in results:
            st.error(results["error"])
        else:
            st.session_state["results"] = results
            st.session_state["relocation_cost"] = relocation_cost
            st.session_state["max_risk"] = max_risk_frac


# ------------------------------------------------------------------
# Header
# ------------------------------------------------------------------
st.markdown("""
    <div class="dashboard-header">
        <h1>🛡️ NoShowShield</h1>
        <p class="tagline">
            Predict cancellations. Optimise overbooking. Protect revenue
            — with full SHAP explainability.
        </p>
    </div>
""", unsafe_allow_html=True)


# ==================================================================
# TABS
# ==================================================================
tab1, tab2 = st.tabs(["📊  Overbooking Recommendations", "🔍  Single Booking Prediction"])


# ==================================================================
# TAB 1 — Overbooking Recommendations
# ==================================================================
with tab1:
    if "results" not in st.session_state:
        st.markdown("""
            <div style="text-align:center; padding:4rem 2rem; color:var(--shield-text-muted);">
                <div style="font-size:3rem; margin-bottom:1rem;">📊</div>
                <div style="font-size:1.1rem; font-weight:500; color:var(--shield-text); margin-bottom:0.5rem;">
                    Ready to optimise
                </div>
                <div style="font-size:0.9rem; max-width:420px; margin:0 auto; line-height:1.6;">
                    Set your relocation cost and risk tolerance in the sidebar,
                    then click <strong>Get Recommendations</strong>.
                </div>
            </div>
        """, unsafe_allow_html=True)
    else:
        results = st.session_state["results"]
        recs = pd.DataFrame(results["recommendations"])
        recs["arrival_date"] = pd.to_datetime(recs["arrival_date"])
        metrics = results["metrics"]
        model_info = results["model_info"]

        # ── Sidebar filters ──
        with st.sidebar:
            st.markdown('<div class="sidebar-section-title">Filters</div>', unsafe_allow_html=True)

            available_hotels = sorted(recs["hotel"].unique())
            selected_hotel = st.selectbox("Hotel", available_hotels)

            available_dates = sorted(
                recs[recs["hotel"] == selected_hotel]["arrival_date"].dt.date.unique()
            )
            selected_date = st.selectbox("Arrival date", available_dates)

            available_rooms = sorted(
                recs[
                    (recs["hotel"] == selected_hotel)
                    & (recs["arrival_date"].dt.date == selected_date)
                ]["assigned_room_type"].unique()
            )
            selected_room = st.selectbox("Room type", available_rooms)

            st.markdown('<div class="sidebar-section-title">Model Performance</div>', unsafe_allow_html=True)
            st.caption(model_info.get("model_type", "XGBoost"))
            metrics_df = pd.DataFrame(
                {"Metric": list(metrics.keys()), "Score": list(metrics.values())}
            ).set_index("Metric")
            st.table(metrics_df)

        # ── Context bar — what the user is looking at ──
        auc_val = metrics.get('auc', '—')
        st.markdown(f"""
            <div class="context-bar">
                <span class="ctx-pill hotel">🏨 {selected_hotel}</span>
                <span class="ctx-pill date">📅 {selected_date}</span>
                <span class="ctx-pill room">🚪 Room {selected_room}</span>
                <span class="ctx-pill auc">Model AUC: {auc_val}</span>
            </div>
        """, unsafe_allow_html=True)

        # ── Filter data ──
        filtered = recs[
            (recs["hotel"] == selected_hotel)
            & (recs["arrival_date"].dt.date == selected_date)
            & (recs["assigned_room_type"] == selected_room)
        ]

        # ── CAPACITY & DEMAND ──
        st.markdown("""
            <div class="section-header">
                <div class="icon blue">📋</div>
                <h3>Capacity & Demand Overview</h3>
            </div>
        """, unsafe_allow_html=True)

        if filtered.empty:
            st.warning("No data available for this combination of hotel, date, and room type.")
        else:
            row = filtered.iloc[0]

            capacity = int(row["capacity"])
            total_bookings = int(row["total_bookings"])
            expected_show = round(row["expected_show_ups"], 1)
            expected_cancel = round(total_bookings - expected_show, 1)
            occupancy_pct = round((expected_show / capacity) * 100, 1) if capacity > 0 else 0
            rec_extra = int(row["recommended_extra"])
            net_benefit = row["net_benefit"]
            reloc_risk = row["relocation_probability"]

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Room Capacity", capacity)
            c2.metric("Current Bookings", total_bookings)
            c3.metric("Expected Cancellations", expected_cancel)
            c4.metric("Expected Occupancy", f"{occupancy_pct}%")

            st.divider()

            # ── OVERBOOKING RECOMMENDATION ──
            st.markdown("""
                <div class="section-header">
                    <div class="icon green">✅</div>
                    <h3>Overbooking Recommendation</h3>
                </div>
            """, unsafe_allow_html=True)

            r1, r2, r3 = st.columns(3)
            r1.metric("Extra Bookings to Accept", rec_extra)
            r2.metric("Estimated Revenue Gain", f"€{net_benefit:.2f}")
            r3.metric("Relocation Risk", f"{reloc_risk * 100:.2f}%")

            # ── Actionable insight ──
            if rec_extra > 0:
                new_total = total_bookings + rec_extra
                risk_threshold = st.session_state.get("max_risk", 0.02) * 100
                st.markdown(f"""
                    <div class="insight-box">
                        <strong>Recommendation:</strong> Accept <strong>{rec_extra} extra booking{"s" if rec_extra != 1 else ""}</strong>
                        (total {new_total}) for an estimated <strong>€{net_benefit:.2f}</strong> revenue gain.
                        The probability of needing to relocate a guest stays at
                        <strong>{reloc_risk * 100:.2f}%</strong> — within your {risk_threshold:.0f}% threshold.
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                    <div class="insight-box warning">
                        <strong>No overbooking recommended.</strong>
                        Current bookings ({total_bookings}) already approach capacity ({capacity})
                        given the low cancellation rate for this segment. Accepting more bookings
                        would exceed your risk threshold.
                    </div>
                """, unsafe_allow_html=True)

            st.divider()

            # ──────────────────────────────────────────────────
            # Show-up Distribution
            # ──────────────────────────────────────────────────
            st.markdown("""
                <div class="section-header">
                    <div class="icon green">📈</div>
                    <h3>Show-up Distribution</h3>
                </div>
            """, unsafe_allow_html=True)
            st.caption(
                "How many guests will actually arrive? Drag the slider "
                "to see how adding bookings shifts the probability curve."
            )

            recommended_total = int(row["recommended_total"])
            n_current = total_bookings

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
                help="Current bookings included first; extras use group mean cancellation rate.",
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
                reloc_prob = float(show_pmf[capacity + 1:].sum()) if capacity + 1 <= n_simulate else 0.0

            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            dcol1.metric("Simulated Bookings", n_simulate)
            dcol2.metric("Expected Show-ups", f"{mean_su:.1f}")
            dcol3.metric("Std Deviation", f"{std_su:.2f}")
            dcol4.metric("P(Overcapacity)", f"{reloc_prob * 100:.2f}%")

            # Chart legend
            st.markdown(f"""
                <div class="chart-legend">
                    <span><span class="dot" style="background:{COLORS['blue']}"></span> Current bookings (≤{n_current})</span>
                    <span><span class="dot" style="background:{COLORS['green']}"></span> Within capacity (≤{capacity})</span>
                    <span><span class="dot" style="background:{COLORS['red']}"></span> Over capacity → relocation risk</span>
                </div>
            """, unsafe_allow_html=True)

            x_vals = list(range(len(show_pmf)))
            bar_colors = [
                COLORS["blue"] if k <= n_current
                else COLORS["green"] if k <= capacity
                else COLORS["red"]
                for k in x_vals
            ]

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(
                x=x_vals, y=show_pmf.tolist(),
                marker_color=bar_colors,
                hovertemplate="Show-ups: %{x}<br>Probability: %{y:.4f}<extra></extra>",
            ))
            fig_dist.add_vline(
                x=capacity + 0.5, line_dash="dash",
                line_color=COLORS["red"], line_width=2,
                annotation_text=f"Capacity = {capacity}",
                annotation_position="top left",
                annotation_font_color=COLORS["red"],
                annotation_font_size=12,
            )
            fig_dist.update_layout(
                **PLOTLY_LAYOUT,
                xaxis_title="Number of Show-ups",
                yaxis_title="Probability",
                height=380,
                margin=dict(l=10, r=20, t=30, b=10),
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.04)"),
                bargap=0.05,
            )
            st.plotly_chart(fig_dist, use_container_width=True)

            # Per-booking badges
            if n_simulate > 0:
                show_pcts = (indiv_show * 100).astype(int).tolist()
                display_limit = 15
                st.markdown("""
                    <div class="badge-legend">
                        <span>Per-booking show-up probability (%)</span>
                        <span><span class="legend-dot" style="background:#2E86C1"></span> Current</span>
                        <span><span class="legend-dot" style="background:#27AE60"></span> Extra</span>
                    </div>
                """, unsafe_allow_html=True)
                badges = ""
                for i, pct in enumerate(show_pcts[:display_limit]):
                    cls = "current" if i < n_current else "extra"
                    badges += f'<span class="prob-badge {cls}">{pct}%</span>'
                remaining = len(show_pcts) - display_limit
                if remaining > 0:
                    badges += f'<span style="font-size:0.8rem;color:var(--shield-text-muted);margin-left:4px;">…+{remaining} more</span>'
                st.markdown(badges, unsafe_allow_html=True)

            st.divider()

            # ──────────────────────────────────────────────────
            # Top Cancellations + SHAP
            # ──────────────────────────────────────────────────
            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.markdown("""
                    <div class="section-header">
                        <div class="icon red">⚠️</div>
                        <h3>Highest-Risk Bookings</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.caption("The 3 individual bookings most likely to cancel.")

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
                    st.warning(f"Could not load data: {top_3_result['error']}")
                else:
                    top_3_data = top_3_result.get("top_3", [])
                    if not top_3_data:
                        st.info("No booking-level data available.")
                    else:
                        top_3_df = pd.DataFrame(top_3_data)
                        top_3_df["cancel_prob"] = (top_3_df["cancel_prob"] * 100).map("{:.1f}%".format)
                        top_3_df["adr"] = top_3_df["adr"].map("€{:.2f}".format)
                        col_mapping = {
                            "lead_time": "Lead Time",
                            "adr": "Avg. Daily Rate",
                            "market_segment": "Segment",
                            "deposit_type": "Deposit",
                            "customer_type": "Customer Type",
                            "cancel_prob": "Cancel Risk",
                        }
                        top_3_df = top_3_df.rename(columns=col_mapping)
                        display_cols = [v for k, v in col_mapping.items() if v in top_3_df.columns]
                        st.table(top_3_df[display_cols])

            with right_col:
                st.markdown("""
                    <div class="section-header">
                        <div class="icon amber">🧠</div>
                        <h3>Why Are They Cancelling?</h3>
                    </div>
                """, unsafe_allow_html=True)
                st.caption("Top features driving cancellation predictions (SHAP).")

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
                            timeout=60, max_retries=2,
                        )
                    st.session_state[cache_key] = shap_result
                else:
                    shap_result = st.session_state[cache_key]

                if "error" in shap_result:
                    st.warning(f"Could not load SHAP data: {shap_result['error']}")
                elif shap_result.get("message"):
                    st.info(shap_result["message"])
                elif shap_result.get("grouped_global_shap"):
                    shap_df = pd.DataFrame(shap_result["grouped_global_shap"])
                    shap_df["feature"] = (
                        shap_df["feature_group"]
                        .str.replace("cat_ordinal__", "", regex=False)
                        .str.replace("_", " ", regex=False)
                        .str.title()
                    )
                    top_shap = shap_df.nlargest(5, "mean_abs_shap").sort_values("mean_abs_shap")

                    fig = px.bar(
                        top_shap, x="mean_abs_shap", y="feature",
                        orientation="h",
                        labels={"mean_abs_shap": "Mean |SHAP|", "feature": ""},
                        color="mean_abs_shap",
                        color_continuous_scale=[COLORS["amber"], COLORS["red"]],
                    )
                    fig.update_layout(
                        **PLOTLY_LAYOUT,
                        height=350,
                        margin=dict(l=0, r=0, t=10, b=0),
                        coloraxis_showscale=False,
                        yaxis=dict(tickfont=dict(size=12)),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No SHAP data available for this selection.")


# ==================================================================
# TAB 2 — Single Booking Prediction
# ==================================================================
with tab2:
    st.caption(
        "Inspect individual high-risk bookings: see the model's prediction, "
        "the actual outcome, and which features drove the cancellation score."
    )

    if "top_bookings_list" not in st.session_state and "results" in st.session_state:
        with st.spinner("Loading high-risk bookings …"):
            top_result = api_get(TOP_BOOKINGS_URL, {}, timeout=60, max_retries=2)
        if "error" in top_result:
            st.error(top_result["error"])
            st.session_state["top_bookings_list"] = []
        else:
            st.session_state["top_bookings_list"] = top_result["top_bookings"]

    top_bookings_list = st.session_state.get("top_bookings_list", [])

    placeholder = "Choose a booking …"
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
            with st.spinner("Running prediction …"):
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
        if "results" not in st.session_state:
            st.markdown("""
                <div style="text-align:center; padding:3rem 2rem; color:var(--shield-text-muted);">
                    <div style="font-size:2.5rem; margin-bottom:0.8rem;">🔍</div>
                    <div style="font-size:0.9rem; max-width:380px; margin:0 auto; line-height:1.6;">
                        Load recommendations first (Tab 1), then return here to inspect individual bookings.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div style="text-align:center; padding:3rem 2rem; color:var(--shield-text-muted);">
                    <div style="font-size:2.5rem; margin-bottom:0.8rem;">👆</div>
                    <div style="font-size:0.9rem; max-width:380px; margin:0 auto; line-height:1.6;">
                        Select a booking from the dropdown above.
                    </div>
                </div>
            """, unsafe_allow_html=True)
    else:
        booking = st.session_state["single_booking"]
        actual = st.session_state["single_actual"]
        explain = st.session_state["single_explain"]

        prob = explain["cancellation_probability"]
        prediction = explain.get("prediction", int(prob >= 0.5))
        is_correct = (prediction == actual)

        # ── Prediction metrics ──
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", "Cancel" if prediction == 1 else "No Cancel")
        col2.metric("Cancel Probability", f"{prob * 100:.1f}%")
        col3.metric("Actual Outcome", "Canceled" if actual == 1 else "Kept")

        # ── Accuracy badge ──
        if is_correct:
            st.markdown(
                '<div class="pred-result correct">✅ Prediction was correct</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="pred-result incorrect">❌ Prediction was incorrect</div>',
                unsafe_allow_html=True,
            )

        st.divider()

        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.markdown("""
                <div class="section-header">
                    <div class="icon blue">📝</div>
                    <h3>Booking Details</h3>
                </div>
            """, unsafe_allow_html=True)

            label_map = {
                "lead_time": "Lead Time (days)",
                "adr": "Avg. Daily Rate (€)",
                "arrival_date_year": "Arrival Year",
                "arrival_date_month": "Arrival Month",
                "arrival_date_week_number": "Week Number",
                "arrival_date_day_of_month": "Day of Month",
                "stays_in_weekend_nights": "Weekend Nights",
                "stays_in_week_nights": "Weekday Nights",
                "adults": "Adults",
                "children": "Children",
                "babies": "Babies",
                "is_repeated_guest": "Repeat Guest",
                "previous_cancellations": "Previous Cancellations",
                "previous_bookings_not_canceled": "Previous Completed Bookings",
                "booking_changes": "Booking Changes",
                "days_in_waiting_list": "Days on Waitlist",
                "required_car_parking_spaces": "Parking Spaces",
                "total_of_special_requests": "Special Requests",
                "hotel": "Hotel",
                "meal": "Meal Plan",
                "market_segment": "Market Segment",
                "distribution_channel": "Distribution Channel",
                "reserved_room_type": "Reserved Room",
                "assigned_room_type": "Assigned Room",
                "deposit_type": "Deposit Type",
                "customer_type": "Customer Type",
                "country": "Country",
            }
            detail_data = []
            for key, val in booking.items():
                label = label_map.get(key, key.replace("_", " ").title())
                detail_data.append({"Field": label, "Value": str(val)})

            details_df = pd.DataFrame(detail_data).set_index("Field")
            st.dataframe(details_df, use_container_width=True)

        with right_col:
            st.markdown("""
                <div class="section-header">
                    <div class="icon amber">🧠</div>
                    <h3>Why This Prediction?</h3>
                </div>
            """, unsafe_allow_html=True)
            st.caption("Features that increased the cancellation score for this booking.")

            shap_df = pd.DataFrame(explain["grouped_local_shap"])
            top_shap = (
                shap_df[shap_df["shap_value"] > 0]
                .nlargest(5, "shap_value")
                .sort_values("shap_value")
            )
            top_shap["feature"] = (
                top_shap["feature_group"]
                .str.replace("_", " ", regex=False)
                .str.title()
            )

            if top_shap.empty:
                st.info("No strong risk factors found — this booking has a low cancellation score.")
            else:
                fig = px.bar(
                    top_shap, x="shap_value", y="feature",
                    orientation="h",
                    labels={"shap_value": "SHAP Value (→ more likely to cancel)", "feature": ""},
                    color="shap_value",
                    color_continuous_scale=[COLORS["amber"], COLORS["red"]],
                )
                fig.update_layout(
                    **PLOTLY_LAYOUT,
                    height=380,
                    margin=dict(l=0, r=0, t=10, b=0),
                    coloraxis_showscale=False,
                    yaxis=dict(tickfont=dict(size=12)),
                )
                st.plotly_chart(fig, use_container_width=True)
