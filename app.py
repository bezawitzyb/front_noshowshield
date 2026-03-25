"""
NoShowShield — Streamlit dashboard (Apple HIG redesign)
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


# ──────────────────────────────────────────────────────────────────────
# API configuration
# ──────────────────────────────────────────────────────────────────────
if 'API_URI' in os.environ:
    BASE_URI = st.secrets[os.environ.get('API_URI')]
else:
    BASE_URI = st.secrets['cloud_api_uri']
BASE_URI = BASE_URI if BASE_URI.endswith('/') else BASE_URI + '/'

OPTIMISE_URL         = BASE_URI + 'optimise'
EXPLAIN_GLOBAL_URL   = BASE_URI + 'explain/global-by-date'
TOP_CANCELLATIONS_URL = BASE_URI + 'top-cancellations'
GROUP_PROBS_URL      = BASE_URI + 'group-probs'
TOP_BOOKINGS_URL     = BASE_URI + 'top-bookings'
EXPLAIN_LOCAL_URL    = BASE_URI + 'explain/local'


# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NoShowShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────────────────────────────
# Apple HIG–inspired theme: custom CSS
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Import SF Pro–like system font stack ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --hig-bg:           #F5F5F7;
    --hig-surface:      #FFFFFF;
    --hig-border:       #D2D2D7;
    --hig-border-light: #E8E8ED;
    --hig-text-primary: #1D1D1F;
    --hig-text-secondary: #6E6E73;
    --hig-text-tertiary: #86868B;
    --hig-accent:       #0071E3;
    --hig-accent-hover: #0077ED;
    --hig-green:        #34C759;
    --hig-green-bg:     #F0FDF4;
    --hig-red:          #FF3B30;
    --hig-red-bg:       #FEF2F2;
    --hig-orange:       #FF9500;
    --hig-orange-bg:    #FFFBEB;
    --hig-blue-bg:      #EFF6FF;
    --hig-radius:       12px;
    --hig-radius-sm:    8px;
    --hig-shadow:       0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --hig-shadow-md:    0 4px 12px rgba(0,0,0,0.08);
}

/* ── Global overrides ── */
html, body, .stApp {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'SF Pro Text',
                 'Helvetica Neue', sans-serif !important;
    background-color: var(--hig-bg) !important;
    color: var(--hig-text-primary) !important;
}

/* ── Main content area ── */
.block-container {
    padding: 2rem 2.5rem 3rem 2.5rem !important;
    max-width: 1200px !important;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header {visibility: hidden;}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background-color: var(--hig-surface) !important;
    border-right: 1px solid var(--hig-border-light) !important;
}

section[data-testid="stSidebar"] .block-container {
    padding: 1.5rem 1.25rem !important;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: var(--hig-text-secondary);
    font-size: 0.85rem;
    line-height: 1.5;
}

/* ── Sidebar headers ── */
section[data-testid="stSidebar"] [data-testid="stHeadingWithActionElements"] {
    margin-top: 0.75rem;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0px;
    border-bottom: 1px solid var(--hig-border-light);
    background: transparent;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
    color: var(--hig-text-secondary) !important;
    border-bottom: 2px solid transparent;
    padding: 0.75rem 1.25rem !important;
    background: transparent !important;
}

.stTabs [aria-selected="true"] {
    color: var(--hig-text-primary) !important;
    border-bottom: 2px solid var(--hig-accent) !important;
    font-weight: 600 !important;
}

.stTabs [data-baseweb="tab-highlight"] {
    display: none;
}

.stTabs [data-baseweb="tab-border"] {
    display: none;
}

/* ── Metric cards ── */
[data-testid="stMetric"] {
    background: var(--hig-surface);
    border: 1px solid var(--hig-border-light);
    border-radius: var(--hig-radius);
    padding: 1rem 1.25rem;
    box-shadow: var(--hig-shadow);
}

[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    font-weight: 500 !important;
    color: var(--hig-text-tertiary) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
}

[data-testid="stMetricValue"] {
    font-size: 1.6rem !important;
    font-weight: 700 !important;
    color: var(--hig-text-primary) !important;
}

/* ── Buttons ── */
.stButton > button {
    background-color: var(--hig-accent) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--hig-radius) !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 0.6rem 1.25rem !important;
    transition: background-color 0.15s ease !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

.stButton > button:hover {
    background-color: var(--hig-accent-hover) !important;
}

/* ── Selectboxes & inputs ── */
[data-baseweb="select"] > div {
    border-radius: var(--hig-radius-sm) !important;
    border-color: var(--hig-border) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
}

.stNumberInput > div > div > input,
.stTextInput > div > div > input {
    border-radius: var(--hig-radius-sm) !important;
    border-color: var(--hig-border) !important;
}

/* ── Sliders ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--hig-accent) !important;
}

/* ── Tables ── */
.stTable table,
.stDataFrame table {
    border-collapse: separate !important;
    border-spacing: 0 !important;
    border-radius: var(--hig-radius-sm) !important;
    overflow: hidden !important;
    font-size: 0.85rem !important;
}

.stTable thead th {
    background: var(--hig-bg) !important;
    font-weight: 600 !important;
    color: var(--hig-text-secondary) !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.03em !important;
    border-bottom: 1px solid var(--hig-border-light) !important;
}

.stTable tbody td {
    border-bottom: 1px solid var(--hig-border-light) !important;
    color: var(--hig-text-primary) !important;
}

/* ── Dividers ── */
hr {
    border: none !important;
    border-top: 1px solid var(--hig-border-light) !important;
    margin: 1.5rem 0 !important;
}

/* ── Info/warning/error boxes ── */
.stAlert {
    border-radius: var(--hig-radius-sm) !important;
    font-size: 0.88rem !important;
}

/* ── Expander ── */
.streamlit-expanderHeader {
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    color: var(--hig-text-primary) !important;
}

/* ── Captions ── */
.stCaption, [data-testid="stCaptionContainer"] {
    color: var(--hig-text-tertiary) !important;
    font-size: 0.8rem !important;
    line-height: 1.5 !important;
}

/* ── Subheaders ── */
h2 {
    font-weight: 600 !important;
    font-size: 1.2rem !important;
    color: var(--hig-text-primary) !important;
    letter-spacing: -0.01em !important;
    margin-top: 0.5rem !important;
}

h3 {
    font-weight: 600 !important;
    font-size: 1rem !important;
    color: var(--hig-text-primary) !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: var(--hig-accent) transparent transparent transparent !important;
}

/* ── Card helper class ── */
div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
    gap: 0.75rem;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Plotly theme (Apple HIG colors & typography)
# ──────────────────────────────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    font=dict(
        family="Inter, -apple-system, BlinkMacSystemFont, SF Pro Text, Helvetica Neue, sans-serif",
        size=13,
        color="#1D1D1F",
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=0, r=16, t=24, b=48),
    showlegend=False,
    xaxis=dict(
        showgrid=False,
        zeroline=False,
        title_font=dict(size=12, color="#86868B"),
        tickfont=dict(size=11, color="#6E6E73"),
    ),
    yaxis=dict(
        showgrid=True,
        gridcolor="rgba(0,0,0,0.04)",
        zeroline=False,
        title_font=dict(size=12, color="#86868B"),
        tickfont=dict(size=11, color="#6E6E73"),
    ),
)

HIG_BLUE   = "#0071E3"
HIG_GREEN  = "#34C759"
HIG_RED    = "#FF3B30"
HIG_ORANGE = "#FF9500"
HIG_GRAY   = "#86868B"
HIG_TEAL   = "#30B0C7"


# ──────────────────────────────────────────────────────────────────────
# Poisson-Binomial PMF (local computation)
# ──────────────────────────────────────────────────────────────────────
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


# ──────────────────────────────────────────────────────────────────────
# API helpers
# ──────────────────────────────────────────────────────────────────────
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
            return {"error": "Request timed out. The API may still be waking up — try again shortly."}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"error": "Could not connect to the API."}
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
            return {"error": "Request timed out."}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"error": "Could not connect to the API."}
    return {"error": "Unexpected error during API call."}


# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-bottom: 1.5rem;">
    <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 0.25rem;">
        <span style="font-size: 1.75rem;">🛡️</span>
        <span style="font-size: 1.75rem; font-weight: 700; color: #1D1D1F;
                     letter-spacing: -0.03em;">NoShowShield</span>
    </div>
    <p style="color: #86868B; font-size: 0.95rem; margin: 0; line-height: 1.5;">
        AI-powered overbooking recommendations that maximise revenue
        while keeping guest relocation risk under control.
    </p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 1.25rem;">
        <span style="font-size: 1.25rem;">🛡️</span>
        <span style="font-weight: 700; font-size: 1.05rem; color: #1D1D1F;">
            NoShowShield
        </span>
    </div>
    """, unsafe_allow_html=True)

    # ── Optimisation parameters ──
    st.markdown("##### Settings")

    relocation_cost = st.number_input(
        "Relocation cost (€)",
        min_value=0.0,
        max_value=1000.0,
        value=300.0,
        step=50.0,
        help="Cost of relocating a guest when overbooked.",
    )

    max_risk = st.slider(
        "Max relocation risk",
        min_value=0.0,
        max_value=0.10,
        value=0.02,
        step=0.01,
        format="%.0f%%",
        help="Maximum acceptable probability of relocating a guest.",
    )

    st.markdown("<div style='height: 0.5rem'></div>", unsafe_allow_html=True)

    if st.button("Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Connecting to prediction engine…"):
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

    # ── Filters (appear after data loads) ──
    if "results" in st.session_state:
        recs_sidebar = pd.DataFrame(st.session_state["results"]["recommendations"])
        recs_sidebar["arrival_date"] = pd.to_datetime(recs_sidebar["arrival_date"])

        st.markdown("---")
        st.markdown("##### Filters")

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

        # ── Model performance (collapsed) ──
        st.markdown("---")
        metrics = st.session_state["results"]["metrics"]
        model_info = st.session_state["results"]["model_info"]

        with st.expander("Model performance", expanded=False):
            st.caption(model_info.get("model_type", "XGBoost"))
            for k, v in metrics.items():
                display_val = f"{v:.3f}" if isinstance(v, float) else str(v)
                st.markdown(
                    f"<div style='display:flex; justify-content:space-between; "
                    f"padding:0.25rem 0; font-size:0.82rem;'>"
                    f"<span style='color:#6E6E73; text-transform:uppercase; "
                    f"letter-spacing:0.03em; font-size:0.72rem; font-weight:500;'>{k}</span>"
                    f"<span style='font-weight:600; color:#1D1D1F;'>{display_val}</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )


# ══════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════
tab1, tab2 = st.tabs(["Overbooking Recommendations", "Single Booking Prediction"])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — Overbooking Recommendations
# ══════════════════════════════════════════════════════════════════════
with tab1:
    if "results" not in st.session_state:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🛡️</div>
            <p style="font-size: 1.1rem; font-weight: 600; color: #1D1D1F; margin-bottom: 0.5rem;">
                Ready when you are
            </p>
            <p style="color: #86868B; font-size: 0.9rem; max-width: 360px; margin: 0 auto;">
                Configure your relocation cost and risk threshold in the sidebar,
                then tap <strong>Get Recommendations</strong>.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        results = st.session_state["results"]
        recs = pd.DataFrame(results["recommendations"])
        recs["arrival_date"] = pd.to_datetime(recs["arrival_date"])
        metrics = results["metrics"]

        # ── Context bar ──
        st.caption(
            f"€{st.session_state.get('relocation_cost', relocation_cost):.0f} relocation cost  ·  "
            f"{st.session_state.get('max_risk', max_risk) * 100:.0f}% max risk  ·  "
            f"AUC {metrics.get('auc', '—')}"
        )

        # ── Filter ──
        filtered = recs[
            (recs["hotel"] == selected_hotel)
            & (recs["arrival_date"].dt.date == selected_date)
            & (recs["assigned_room_type"] == selected_room)
        ]

        if filtered.empty:
            st.warning("No data available for this selection.")
        else:
            row = filtered.iloc[0]

            # ── Recommendation hero card ──
            rec_extra = int(row["recommended_extra"])
            net_benefit = row["net_benefit"]
            reloc_risk = row["relocation_probability"]

            # Determine sentiment color
            if rec_extra > 0 and net_benefit > 0:
                sentiment = "green"
                icon = "✅"
                verdict = f"Accept {rec_extra} additional booking{'s' if rec_extra != 1 else ''}"
            elif rec_extra == 0:
                sentiment = "orange"
                icon = "⚖️"
                verdict = "No additional bookings recommended"
            else:
                sentiment = "red"
                icon = "⚠️"
                verdict = "Reduce exposure — risk too high"

            bg_map = {"green": "#F0FDF4", "orange": "#FFFBEB", "red": "#FEF2F2"}
            border_map = {"green": "#BBF7D0", "orange": "#FDE68A", "red": "#FECACA"}
            text_map = {"green": "#166534", "orange": "#92400E", "red": "#991B1B"}

            st.markdown(f"""
            <div style="background: {bg_map[sentiment]}; border: 1px solid {border_map[sentiment]};
                        border-radius: 12px; padding: 1.25rem 1.5rem; margin-bottom: 1.25rem;">
                <div style="display: flex; align-items: center; gap: 0.5rem; margin-bottom: 0.25rem;">
                    <span style="font-size: 1.2rem;">{icon}</span>
                    <span style="font-weight: 700; font-size: 1.1rem; color: {text_map[sentiment]};">
                        {verdict}
                    </span>
                </div>
                <span style="font-size: 0.85rem; color: {text_map[sentiment]}99;">
                    Net benefit of <strong>€{net_benefit:.2f}</strong> with
                    <strong>{reloc_risk * 100:.2f}%</strong> relocation risk
                </span>
            </div>
            """, unsafe_allow_html=True)

            # ── Key metrics row ──
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Capacity", int(row["capacity"]))
            m2.metric("Bookings", int(row["total_bookings"]))
            m3.metric("Expected Show-ups", round(row["expected_show_ups"], 1))
            m4.metric("Extra Bookings", f"+{int(row['recommended_extra'])}")

            st.markdown("<div style='height: 0.75rem'></div>", unsafe_allow_html=True)

            # ──────────────────────────────────────────────────────────
            # Show-up Distribution
            # ──────────────────────────────────────────────────────────
            st.markdown("##### Show-up Distribution")
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
                mean_cp = row.get(
                    "cancel_prob_mean",
                    row["expected_cancellations"] / row["total_bookings"],
                )
                cancel_probs_arr = np.full(n_current, float(mean_cp))

            n_simulate = st.slider(
                "Total bookings to simulate",
                min_value=0,
                max_value=recommended_total,
                value=recommended_total,
                help="Drag to see how the show-up distribution shifts as bookings are added.",
            )

            # Compute PMF
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

            # Stats row
            d1, d2, d3, d4 = st.columns(4)
            d1.metric("Simulated", n_simulate)
            d2.metric("Expected", f"{mean_su:.1f}")
            d3.metric("Std Dev", f"{std_su:.2f}")
            d4.metric("Relocation Risk", f"{reloc_prob * 100:.2f}%")

            # ── Plotly chart ──
            x_vals = list(range(len(show_pmf)))

            bar_colors = [
                HIG_BLUE if k <= n_current
                else HIG_GREEN if k <= capacity
                else HIG_RED
                for k in x_vals
            ]

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(
                x=x_vals,
                y=show_pmf.tolist(),
                marker_color=bar_colors,
                marker_line_width=0,
                hovertemplate="Show-ups: %{x}<br>Probability: %{y:.4f}<extra></extra>",
            ))

            fig_dist.add_vline(
                x=capacity + 0.5,
                line_dash="dash",
                line_color=HIG_RED,
                line_width=1.5,
                annotation_text=f"Capacity ({capacity})",
                annotation_position="top left",
                annotation_font=dict(color=HIG_RED, size=11, family="Inter"),
            )

            fig_dist.update_layout(
                **PLOTLY_LAYOUT,
                height=360,
                xaxis_title="Number of Show-ups",
                yaxis_title="Probability",
                bargap=0.08,
            )

            st.plotly_chart(fig_dist, use_container_width=True)

            # ── Individual show-up badges ──
            if n_simulate > 0:
                show_pcts = (indiv_show * 100).astype(int).tolist()
                display_limit = 15
                badges = ""
                for i, pct in enumerate(show_pcts[:display_limit]):
                    bg = HIG_BLUE if i < n_current else HIG_GREEN
                    badges += (
                        f'<span style="display:inline-block; margin:2px; padding:3px 8px;'
                        f'border-radius:6px; background:{bg}18; color:{bg};'
                        f'font-size:0.75rem; font-weight:600;">{pct}%</span>'
                    )
                remaining = len(show_pcts) - display_limit
                if remaining > 0:
                    badges += (
                        f'<span style="font-size:0.78rem; color:#86868B;'
                        f'margin-left:4px;">+{remaining} more</span>'
                    )

                st.markdown(
                    f"<div style='margin-top: -0.5rem;'>"
                    f"<span style='font-size:0.78rem; font-weight:500; color:#86868B;'>"
                    f"Individual show-up probabilities "
                    f"<span style='color:{HIG_BLUE};'>●</span> Current  "
                    f"<span style='color:{HIG_GREEN};'>●</span> Extra</span><br>"
                    f"{badges}</div>",
                    unsafe_allow_html=True,
                )

            st.divider()

            # ──────────────────────────────────────────────────────────
            # Two-column: Top Cancellations + SHAP
            # ──────────────────────────────────────────────────────────
            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.markdown("##### Highest-Risk Bookings")
                st.caption(
                    f"Top 3 bookings for {selected_room} on {selected_date} "
                    f"most likely to cancel"
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
                            "cancel_prob": "Risk",
                        }
                        top_3_df = top_3_df.rename(columns=col_mapping)
                        display_cols = [v for k, v in col_mapping.items() if v in top_3_df.columns]
                        st.table(top_3_df[display_cols])

            with right_col:
                st.markdown("##### Key Cancellation Drivers")
                st.caption(f"SHAP feature importance for this date and room type")

                selected_date_str = str(selected_date)
                cache_key = f"shap_{selected_hotel}_{selected_date_str}_{selected_room}"

                if cache_key not in st.session_state:
                    with st.spinner("Loading explanations…"):
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
                    shap_df["mean_abs_shap"] = pd.to_numeric(
                        shap_df["mean_abs_shap"], errors="coerce"
                    )
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
                        st.info("No SHAP data available.")
                    else:
                        plot_df = top_shap.iloc[::-1].reset_index(drop=True)

                        fig_shap = go.Figure(go.Bar(
                            x=plot_df["mean_abs_shap"].tolist(),
                            y=plot_df["feature"].tolist(),
                            orientation="h",
                            marker_color=HIG_BLUE,
                            marker_line_width=0,
                        ))

                        fig_shap.update_layout(
                            **PLOTLY_LAYOUT,
                            height=340,
                            xaxis_title="Mean |SHAP|",
                            yaxis_title="",
                        )
                        fig_shap.update_yaxes(
                            type="category",
                            categoryorder="array",
                            categoryarray=plot_df["feature"].tolist(),
                            automargin=True,
                        )
                        fig_shap.update_xaxes(automargin=True)

                        st.plotly_chart(fig_shap, use_container_width=True)
                else:
                    st.info("No SHAP data available for this selection.")


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — Single Booking Prediction
# ══════════════════════════════════════════════════════════════════════
with tab2:
    if "results" not in st.session_state:
        st.markdown("""
        <div style="text-align:center; padding: 4rem 2rem;">
            <div style="font-size: 2.5rem; margin-bottom: 0.75rem;">🔍</div>
            <p style="font-size: 1.1rem; font-weight: 600; color: #1D1D1F; margin-bottom: 0.5rem;">
                Inspect individual bookings
            </p>
            <p style="color: #86868B; font-size: 0.9rem; max-width: 380px; margin: 0 auto;">
                Load recommendations first, then explore the highest-risk bookings
                with full SHAP explanations.
            </p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("##### Booking Risk Explorer")
        st.caption(
            "Select a high-risk booking to see the model's prediction "
            "and which features are driving the cancellation risk."
        )

        # Load top bookings once
        if "top_bookings_list" not in st.session_state:
            with st.spinner("Loading high-risk bookings…"):
                top_result = api_get(TOP_BOOKINGS_URL, {}, timeout=60, max_retries=2)
            if "error" in top_result:
                st.error(top_result["error"])
                st.session_state["top_bookings_list"] = []
            else:
                st.session_state["top_bookings_list"] = top_result["top_bookings"]

        top_bookings_list = st.session_state.get("top_bookings_list", [])

        placeholder = "Select a booking…"
        dropdown_labels = [placeholder] + [b["label"] for b in top_bookings_list]
        selected_label = st.selectbox(
            "Booking",
            options=dropdown_labels,
            index=0,
            disabled=not top_bookings_list,
            label_visibility="collapsed",
        )

        if selected_label != placeholder and top_bookings_list:
            selected_entry = next(
                b for b in top_bookings_list if b["label"] == selected_label
            )

            cache_key = f"explain_{selected_entry['rank']}"
            if cache_key not in st.session_state:
                with st.spinner("Running prediction…"):
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
                st.info("Choose a booking above to see its prediction details.")
        else:
            booking = st.session_state["single_booking"]
            actual = st.session_state["single_actual"]
            explain = st.session_state["single_explain"]

            prob = explain["cancellation_probability"]
            prediction = explain.get("prediction", int(prob >= 0.5))

            # ── Prediction result card ──
            if prediction == 1:
                pred_bg = "#FEF2F2"
                pred_border = "#FECACA"
                pred_color = "#991B1B"
                pred_icon = "⚠️"
                pred_label = "Will Cancel"
            else:
                pred_bg = "#F0FDF4"
                pred_border = "#BBF7D0"
                pred_color = "#166534"
                pred_icon = "✅"
                pred_label = "Won't Cancel"

            actual_text = "Canceled" if actual == 1 else "Not Canceled"
            match = prediction == actual
            match_icon = "✓ Correct" if match else "✗ Incorrect"
            match_color = "#166534" if match else "#991B1B"

            st.markdown(f"""
            <div style="background: {pred_bg}; border: 1px solid {pred_border};
                        border-radius: 12px; padding: 1.25rem 1.5rem; margin: 0.75rem 0 1.25rem 0;">
                <div style="display: flex; align-items: baseline; gap: 2rem; flex-wrap: wrap;">
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 500; color: {pred_color}99;
                                    text-transform: uppercase; letter-spacing: 0.04em;">Prediction</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: {pred_color};
                                    margin-top: 0.15rem;">
                            {pred_icon} {pred_label}
                        </div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 500; color: {pred_color}99;
                                    text-transform: uppercase; letter-spacing: 0.04em;">Probability</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: {pred_color};
                                    margin-top: 0.15rem;">{prob * 100:.1f}%</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 500; color: {pred_color}99;
                                    text-transform: uppercase; letter-spacing: 0.04em;">Actual Outcome</div>
                        <div style="font-size: 1.3rem; font-weight: 700; color: {pred_color};
                                    margin-top: 0.15rem;">{actual_text}</div>
                    </div>
                    <div>
                        <div style="font-size: 0.72rem; font-weight: 500; color: {match_color}99;
                                    text-transform: uppercase; letter-spacing: 0.04em;">Accuracy</div>
                        <div style="font-size: 1.1rem; font-weight: 700; color: {match_color};
                                    margin-top: 0.25rem;">{match_icon}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Two columns: details + SHAP ──
            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.markdown("##### Booking Details")

                # Render as a clean key-value list instead of raw dataframe
                detail_html = ""
                for field, value in booking.items():
                    clean_field = str(field).replace("_", " ").title()
                    detail_html += (
                        f"<div style='display:flex; justify-content:space-between; "
                        f"padding: 0.5rem 0; border-bottom: 1px solid #E8E8ED;'>"
                        f"<span style='color:#6E6E73; font-size:0.85rem;'>{clean_field}</span>"
                        f"<span style='font-weight:500; font-size:0.85rem; color:#1D1D1F;'>"
                        f"{value}</span></div>"
                    )

                st.markdown(
                    f"<div style='background:white; border:1px solid #E8E8ED; "
                    f"border-radius:12px; padding: 0.25rem 1.25rem; "
                    f"max-height: 420px; overflow-y:auto;'>{detail_html}</div>",
                    unsafe_allow_html=True,
                )

            with right_col:
                st.markdown("##### Top Risk Factors")
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
                    st.info("No cancellation risk factors found.")
                else:
                    plot_df = top_shap.iloc[::-1].reset_index(drop=True)

                    fig_shap2 = go.Figure(go.Bar(
                        x=plot_df["shap_value"].tolist(),
                        y=plot_df["feature"].tolist(),
                        orientation="h",
                        marker_color=HIG_RED,
                        marker_line_width=0,
                    ))

                    fig_shap2.update_layout(
                        **PLOTLY_LAYOUT,
                        height=340,
                        xaxis_title="SHAP Value",
                        yaxis_title="",
                    )
                    fig_shap2.update_yaxes(
                        type="category",
                        categoryorder="array",
                        categoryarray=plot_df["feature"].tolist(),
                        automargin=True,
                    )
                    fig_shap2.update_xaxes(automargin=True)

                    st.plotly_chart(fig_shap2, use_container_width=True)
