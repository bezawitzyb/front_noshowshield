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
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Theme / color constants
# ------------------------------------------------------------------
# Page background
CLR_PAGE_BG       = "#F0F4F8"
# Card fills — row 1 (top metrics)
CLR_CARD_BLUE     = "#DBEAFE"   # Capacity, Confirmed Bookings
CLR_CARD_GREEN    = "#D1FAE5"   # Expected Show-ups
CLR_CARD_RED      = "#FEE2E2"   # Relocation Risk / Cost
CLR_CARD_AMBER    = "#FEF3C7"   # Rec. Extra Bookings
# Card fills — row 2 (financial / risk metrics)
CLR_CARD_YELLOW   = "#FEF9C3"   # Current Revenue
CLR_CARD_RED2     = "#FEE2E2"   # Relocation Risk (shared)
CLR_CARD_PINK     = "#FCE7F3"   # Relocation Cost
CLR_CARD_SOFTGRN  = "#F0FDF4"   # Net Benefit
# Text / accent
CLR_TEXT_PRI      = "#1E293B"
CLR_TEXT_MUT      = "#64748B"
CLR_BLUE          = "#3B82F6"
CLR_GREEN_POS     = "#059669"
CLR_RED_RISK      = "#DC2626"
# Chart bars — 3 colours using primary blue family
CLR_BAR_BELOW     = "#93C5FD"   # below current bookings (light blue)
CLR_BAR_SAFE      = "#3B82F6"   # current → capacity (primary blue)
CLR_BAR_RISK      = "#EF4444"   # over capacity (red)
# Sidebar model info
CLR_MODEL_BG      = "#EFF6FF"


# ------------------------------------------------------------------
# API configuration
# ------------------------------------------------------------------
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


# ------------------------------------------------------------------
# Page config & global CSS injection
# ------------------------------------------------------------------
st.set_page_config(
    page_title="NoShowShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  /* ---- Page background ---- */
  .stApp {{ background-color: {CLR_PAGE_BG}; }}

  /* ---- Metric card base ---- */
  .nss-card {{
    border-radius: 12px;
    padding: 18px 20px 16px;
    margin-bottom: 4px;
  }}
  .nss-card .card-icon {{
    font-size: 18px;
    margin-bottom: 6px;
    display: block;
  }}
  .nss-card .card-label {{
    font-size: 12px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: {CLR_TEXT_MUT};
    margin-bottom: 4px;
  }}
  .nss-card .card-value {{
    font-size: 28px;
    font-weight: 700;
    color: {CLR_TEXT_PRI};
    line-height: 1.1;
  }}
  .nss-card .card-sub {{
    font-size: 12px;
    color: {CLR_TEXT_MUT};
    margin-top: 5px;
    line-height: 1.4;
  }}
  .nss-card .card-badge {{
    display: inline-block;
    font-size: 11px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 999px;
    margin-top: 5px;
  }}

  /* ---- Section header ---- */
  .nss-section-title {{
    font-size: 18px;
    font-weight: 700;
    color: {CLR_TEXT_PRI};
    margin: 28px 0 4px;
  }}
  .nss-section-sub {{
    font-size: 13px;
    color: {CLR_TEXT_MUT};
    margin-bottom: 14px;
  }}

  /* ---- Distribution stats row ---- */
  .dist-stat {{
    text-align: center;
    padding: 10px 0;
  }}
  .dist-stat .ds-label {{
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: {CLR_TEXT_MUT};
  }}
  .dist-stat .ds-value {{
    font-size: 22px;
    font-weight: 700;
    color: {CLR_TEXT_PRI};
    margin-top: 2px;
  }}
  .dist-stat .ds-value.risk {{ color: {CLR_RED_RISK}; }}
  .dist-stat .ds-value.conf {{ color: {CLR_GREEN_POS}; }}

  /* ---- Individual show-up pills ---- */
  .showup-pills {{ display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }}
  .sup-pill {{
    display: inline-flex; align-items: center; gap: 6px;
    padding: 4px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 600;
    background: {CLR_CARD_BLUE}; color: {CLR_BLUE};
    border: 1px solid #BFDBFE;
  }}
  .sup-pill .pill-id {{ color: {CLR_TEXT_MUT}; font-weight: 500; }}
  .sup-pill .pill-bar {{
    display: inline-block; height: 4px; border-radius: 2px;
    background: {CLR_BLUE}; vertical-align: middle;
  }}

  /* ---- Cancel risk badge ---- */
  .risk-pill {{
    display: inline-block; padding: 3px 9px; border-radius: 999px;
    font-size: 12px; font-weight: 700;
  }}
  .risk-high  {{ background: #FEE2E2; color: {CLR_RED_RISK}; }}
  .risk-med   {{ background: {CLR_CARD_AMBER}; color: #92400E; }}
  .risk-low   {{ background: {CLR_CARD_GREEN}; color: #065F46; }}

  /* ---- Model info box in sidebar ---- */
  .model-box {{
    background: {CLR_MODEL_BG};
    border-radius: 10px;
    padding: 12px 14px;
    margin-top: 8px;
  }}
  .model-box .mb-version {{
    font-size: 13px; font-weight: 700; color: {CLR_BLUE};
    margin-bottom: 10px; display: flex; align-items: center; gap: 6px;
  }}
  .model-box .mb-row {{
    display: flex; justify-content: space-between;
    font-size: 12px; padding: 3px 0;
    border-bottom: 0.5px solid #BFDBFE;
  }}
  .model-box .mb-row:last-child {{ border-bottom: none; }}
  .model-box .mb-key {{ color: {CLR_TEXT_MUT}; text-transform: uppercase; font-size: 11px; letter-spacing: 0.04em; }}
  .model-box .mb-val {{ color: {CLR_TEXT_PRI}; font-weight: 600; }}

  /* ---- Page 1 header ---- */
  .nss-page-header {{
    background: white;
    border-radius: 14px;
    padding: 20px 24px 16px;
    margin-bottom: 20px;
    border: 0.5px solid #E2E8F0;
  }}
  .nss-page-header h1 {{
    font-size: 24px; font-weight: 800; color: {CLR_TEXT_PRI}; margin: 0 0 4px;
  }}
  .nss-page-header p {{
    font-size: 13px; color: {CLR_TEXT_MUT}; margin: 0;
  }}

  /* ---- Prediction result box (tab 2) ---- */
  .pred-box {{
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    background: white;
    padding: 20px 24px;
    margin-bottom: 16px;
  }}
  .pred-box .pred-label {{ font-size: 12px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: {CLR_TEXT_MUT}; }}
  .pred-box .pred-verdict {{ font-size: 32px; font-weight: 800; color: {CLR_RED_RISK}; margin: 4px 0; }}
  .pred-box .pred-verdict.no-cancel {{ color: {CLR_GREEN_POS}; }}
  .pred-box .pred-prob {{ font-size: 28px; font-weight: 800; color: {CLR_BLUE}; }}
  .pred-box .pred-sub {{ font-size: 12px; color: {CLR_TEXT_MUT}; margin-top: 4px; }}

  /* ---- Action bar (tab 2) ---- */
  .action-bar {{
    background: {CLR_CARD_BLUE};
    border-radius: 12px;
    padding: 16px 20px;
    display: flex; align-items: center; justify-content: space-between;
    margin-top: 20px;
    border: 1px solid #BFDBFE;
  }}
  .action-bar .ab-text {{ font-size: 14px; font-weight: 600; color: {CLR_TEXT_PRI}; }}
  .action-bar .ab-sub {{ font-size: 12px; color: {CLR_TEXT_MUT}; margin-top: 2px; }}

  /* ---- Booking characteristics table ---- */
  .bc-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
  .bc-table tr {{ border-bottom: 0.5px solid #F1F5F9; }}
  .bc-table tr:last-child {{ border-bottom: none; }}
  .bc-table td {{ padding: 8px 4px; }}
  .bc-table td:first-child {{ color: {CLR_TEXT_MUT}; width: 45%; }}
  .bc-table td:last-child {{ font-weight: 600; color: {CLR_TEXT_PRI}; }}

  /* ---- Hide default streamlit metric styling ---- */
  [data-testid="stMetric"] {{ display: none; }}

  /* ---- Divider ---- */
  hr {{ border: none; border-top: 1px solid #E2E8F0; margin: 20px 0; }}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def card_html(bg_color, icon, label, value, sub="", badge_text="", badge_color=""):
    badge = ""
    if badge_text:
        badge = f'<span class="card-badge" style="background:{badge_color};color:{CLR_TEXT_MUT};">{badge_text}</span>'
    return f"""
    <div class="nss-card" style="background:{bg_color};">
      <span class="card-icon">{icon}</span>
      <div class="card-label">{label}</div>
      <div class="card-value">{value}</div>
      {"<div class='card-sub'>" + sub + "</div>" if sub else ""}
      {badge}
    </div>
    """


def dist_stat_html(label, value, extra_class=""):
    return f"""
    <div class="dist-stat">
      <div class="ds-label">{label}</div>
      <div class="ds-value {extra_class}">{value}</div>
    </div>
    """


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


def api_get(url, params, timeout=180, max_retries=3):
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
            return {"error": "API request timed out. Try again in a moment."}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep(2)
                continue
            return {"error": "Could not connect to the API."}
    return {"error": "Unexpected error during API call."}


def api_post(url, payload, timeout=60, max_retries=2):
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
# Sidebar — settings (always visible)
# ------------------------------------------------------------------
with st.sidebar:
    selected_tab = st.selectbox("Navigation", ["📊 Booking Recommendations", "🔍 Single Booking Analysis"])

    st.markdown("### ⚙️ Optimization Settings")

    relocation_cost = st.number_input(
        "Relocation cost (€)",
        min_value=0.0, max_value=1000.0, value=150.0, step=10.0,
        help="Cost of relocating a guest when overbooked.",
    )
    max_risk = st.slider(
        "Max relocation risk",
        min_value=0.0, max_value=0.10, value=0.05, step=0.01,
        format="%.0f%%",
        help="Maximum acceptable probability of having to relocate a guest.",
    )
    get_recs = st.button("Get Recommendations", type="primary", use_container_width=True)

    if get_recs:
        with st.spinner("Fetching predictions … (first load may take ~2 min)"):
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


# ------------------------------------------------------------------
# Sidebar — filters + model info (only after data loaded)
# ------------------------------------------------------------------
if "results" in st.session_state:
    results  = st.session_state["results"]
    recs     = pd.DataFrame(results["recommendations"])
    recs["arrival_date"] = pd.to_datetime(recs["arrival_date"])
    metrics    = results["metrics"]
    model_info = results["model_info"]

    with st.sidebar:
        st.markdown("### 🔍 Filters")
        available_hotels = sorted(recs["hotel"].unique())
        selected_hotel = st.selectbox("Hotel", available_hotels)

        available_dates = sorted(
            recs[recs["hotel"] == selected_hotel]["arrival_date"].dt.date.unique()
        )
        selected_date = st.selectbox("Date", available_dates)

        available_rooms = sorted(
            recs[
                (recs["hotel"] == selected_hotel)
                & (recs["arrival_date"].dt.date == selected_date)
            ]["assigned_room_type"].unique()
        )
        selected_room = st.selectbox("Room type", available_rooms)

        # Model info box
        model_version = model_info.get("model_version", model_info.get("model_type", "v2.4.1"))
        acc   = metrics.get("accuracy", metrics.get("acc", "—"))
        f1    = metrics.get("f1",  metrics.get("f1_score", "—"))
        auc   = metrics.get("auc", "—")
        prec  = metrics.get("precision", "—")
        rec   = metrics.get("recall", "—")

        def fmt_metric(v):
            if isinstance(v, float):
                return f"{v:.1%}" if v <= 1.0 else f"{v:.1f}"
            return str(v)

        st.markdown(f"""
        <div class="model-box">
          <div class="mb-version">🤖 Model {model_version}</div>
          <div class="mb-row"><span class="mb-key">Accuracy</span><span class="mb-val">{fmt_metric(acc)}</span></div>
          <div class="mb-row"><span class="mb-key">F1 Score</span><span class="mb-val">{fmt_metric(f1)}</span></div>
          <div class="mb-row"><span class="mb-key">AUC</span><span class="mb-val">{fmt_metric(auc)}</span></div>
          <div class="mb-row"><span class="mb-key">Precision</span><span class="mb-val">{fmt_metric(prec)}</span></div>
          <div class="mb-row"><span class="mb-key">Recall</span><span class="mb-val">{fmt_metric(rec)}</span></div>
        </div>
        """, unsafe_allow_html=True)


# ------------------------------------------------------------------
# MAIN CONTENT
# ------------------------------------------------------------------
# ==================================================================
# BOOKING RECOMMENDATIONS
# ==================================================================
if selected_tab == "📊 Booking Recommendations":
    if "results" not in st.session_state:
        st.markdown("""
        <div class="nss-page-header">
          <h1>🛡️ NoShowShield</h1>
          <p>Revenue Protection Intelligence — adjust settings in the sidebar and click <b>Get Recommendations</b> to start.</p>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # Filter to selection
    filtered = recs[
        (recs["hotel"] == selected_hotel)
        & (recs["arrival_date"].dt.date == selected_date)
        & (recs["assigned_room_type"] == selected_room)
    ]

    # Page header
    st.markdown(f"""
    <div class="nss-page-header">
      <h1>Booking Recommendations</h1>
      <p>Optimize your inventory for <b>{selected_date}</b> · Room Type: <b>{selected_room}</b></p>
    </div>
    """, unsafe_allow_html=True)

    if filtered.empty:
        st.warning("No data available for this selection.")
    else:
        row = filtered.iloc[0]

        capacity        = int(row["capacity"])
        total_bookings  = int(row["total_bookings"])
        expected_showup = float(row["expected_show_ups"])
        rec_extra       = int(row["recommended_extra"])
        net_benefit     = float(row["net_benefit"])
        reloc_prob      = float(row["relocation_probability"])
        reloc_cost_val  = float(row["expected_relocation_cost"])
        mean_adr        = float(row["mean_adr"])

        # Derived values
        occupancy_pct   = f"{total_bookings / capacity * 100:.1f}%"
        current_revenue = total_bookings * mean_adr

        # ---- Row 1: 4 metric cards ----
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(card_html(
                CLR_CARD_BLUE, "🏨",
                "Physical Capacity",
                str(capacity),
                "Maximum rooms available."
            ), unsafe_allow_html=True)
        with c2:
            st.markdown(card_html(
                CLR_CARD_BLUE, "📋",
                "Confirmed Bookings",
                str(total_bookings),
                "Current total bookings in the PMS.",
                badge_text=f"{occupancy_pct} Occ",
                badge_color="#BFDBFE",
            ), unsafe_allow_html=True)
        with c3:
            std_cancel = float(row.get("std_cancellations", 0))
            st.markdown(card_html(
                CLR_CARD_GREEN, "✅",
                "Expected Use-Shows",
                f"{expected_showup:.1f}",
                f"All predicted arrivals based on optimised cancel corrections.",
                badge_text=f"±{std_cancel:.1f} std dev",
                badge_color="#A7F3D0",
            ), unsafe_allow_html=True)
        with c4:
            st.markdown(card_html(
                CLR_CARD_AMBER, "➕",
                "Rec. Extra Bookings",
                f"+{rec_extra}",
                "Desired overbooking to maximise revenue safety."
            ), unsafe_allow_html=True)

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

        # ---- Row 2: 4 metric cards ----
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            st.markdown(card_html(
                CLR_CARD_YELLOW, "€",
                "Current Revenue",
                f"€{current_revenue:,.0f}",
                "Approximate baseline revenue ex tax."
            ), unsafe_allow_html=True)
        with c6:
            risk_pct = reloc_prob * 100
            st.markdown(card_html(
                CLR_CARD_RED2, "⚠️",
                "Relocation Risk",
                f"{risk_pct:.2f}%",
                "Probability of exceeding physical capacity."
            ), unsafe_allow_html=True)
        with c7:
            st.markdown(card_html(
                CLR_CARD_PINK, "€",
                "Relocation Cost",
                f"€{reloc_cost_val:.0f}",
                "Risk-adjusted relocation cost."
            ), unsafe_allow_html=True)
        with c8:
            st.markdown(card_html(
                CLR_CARD_SOFTGRN, "📈",
                "Predicted Net Benefit",
                f"€{net_benefit:,.0f}",
                "Additional revenue minus relocation cost."
            ), unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ---- Show-up Distribution ----
        st.markdown('<div class="nss-section-title">Show-up Distribution</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="nss-section-sub">Poisson-Binomial probability density. '
            'Slide to simulate how additional bookings affect arrival risk.</div>',
            unsafe_allow_html=True
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
                else:
                    gp_result = {}
            except Exception:
                gp_result = {}
        else:
            gp_result = st.session_state[probs_cache_key]

        if "cancel_probs" in gp_result:
            cancel_probs_arr = np.array(gp_result["cancel_probs"], dtype=np.float64)
        else:
            mean_cp = row.get("cancel_prob_mean",
                              row["expected_cancellations"] / row["total_bookings"])
            cancel_probs_arr = np.full(n_current, float(mean_cp))

        n_simulate = st.slider(
            "Total bookings to simulate",
            min_value=0,
            max_value=recommended_total,
            value=recommended_total,
        )

        # Compute PMF
        if n_simulate == 0:
            show_pmf = np.array([1.0])
            mean_su = 0.0
            std_su = 0.0
            reloc_prob_sim = 0.0
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
            reloc_prob_sim = float(show_pmf[capacity + 1:].sum()) if capacity + 1 <= n_simulate else 0.0

        target_confidence = (1 - reloc_prob_sim) * 100

        # Distribution stats row (5 columns)
        ds1, ds2, ds3, ds4, ds5 = st.columns(5)
        with ds1:
            st.markdown(dist_stat_html("Bookings simulated", str(n_simulate)), unsafe_allow_html=True)
        with ds2:
            st.markdown(dist_stat_html("Expected show-ups", f"{mean_su:.1f}"), unsafe_allow_html=True)
        with ds3:
            st.markdown(dist_stat_html("Std deviation", f"{std_su:.2f}"), unsafe_allow_html=True)
        with ds4:
            st.markdown(dist_stat_html("Relocation risk", f"{reloc_prob_sim * 100:.2f}%", "risk"), unsafe_allow_html=True)
        with ds5:
            st.markdown(dist_stat_html("Target confidence", f"{target_confidence:.1f}%", "conf"), unsafe_allow_html=True)

        # PMF chart — 3 colours: light-blue (≤ n_current), primary blue (n_current → capacity), red (> capacity)
        x_vals = list(range(len(show_pmf)))
        bar_colors = [
            CLR_BAR_BELOW if k <= n_current
            else CLR_BAR_SAFE if k <= capacity
            else CLR_BAR_RISK
            for k in x_vals
        ]

        fig_dist = go.Figure()
        fig_dist.add_trace(go.Bar(
            x=x_vals,
            y=show_pmf.tolist(),
            marker_color=bar_colors,
            hovertemplate="Show-ups: %{x}<br>Probability: %{y:.4f}<extra></extra>",
        ))
        fig_dist.add_vline(
            x=capacity + 0.5,
            line_dash="dash", line_color=CLR_RED_RISK, line_width=1.5,
            annotation_text=f"Capacity: {capacity}",
            annotation_position="top left",
            annotation_font_color=CLR_RED_RISK,
            annotation_font_size=11,
        )
        fig_dist.update_layout(
            xaxis_title="Number of Guests Showing Up",
            yaxis_title="Probability",
            height=320,
            margin=dict(l=0, r=10, t=20, b=0),
            showlegend=False,
            plot_bgcolor="white",
            paper_bgcolor="white",
            xaxis=dict(showgrid=False, tickfont=dict(size=11)),
            yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)", tickfont=dict(size=11)),
            bargap=0.03,
        )
        st.plotly_chart(fig_dist, use_container_width=True)

        # ---- Individual show-up probability pills ----
        if n_simulate > 0 and len(indiv_show) > 0:
            st.markdown(
                '<div class="nss-section-sub" style="margin-top:0;">⚡ Individual show-up probability</div>',
                unsafe_allow_html=True,
            )
            show_pcts = (indiv_show * 100).astype(int).tolist()
            display_limit = 15
            pills_html = '<div class="showup-pills">'
            for i, pct in enumerate(show_pcts[:display_limit]):
                bar_w = max(20, int(pct * 0.5))
                pills_html += (
                    f'<span class="sup-pill">'
                    f'<span class="pill-id">#{i+1}</span>'
                    f'<span class="pill-bar" style="width:{bar_w}px;"></span>'
                    f'{pct}%'
                    f'</span>'
                )
            remaining = len(show_pcts) - display_limit
            if remaining > 0:
                pills_html += f'<span style="font-size:12px;color:{CLR_TEXT_MUT};align-self:center;">+{remaining} more</span>'
            pills_html += '</div>'
            st.markdown(pills_html, unsafe_allow_html=True)

        st.markdown("<hr>", unsafe_allow_html=True)

        # ---- Bottom two-column layout: Top Cancellations | SHAP ----
        left_col, right_col = st.columns([3, 2])

        with left_col:
            st.markdown('<div class="nss-section-title">Top High-Risk Cancellations</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="nss-section-sub">Most likely no-shows based on lead time and market segment.</div>',
                unsafe_allow_html=True,
            )

            top_result = api_get(
                TOP_CANCELLATIONS_URL,
                {"hotel": selected_hotel, "arrival_date": str(selected_date), "room_type": selected_room},
                timeout=30, max_retries=2,
            )

            if "error" in top_result:
                st.warning(f"Could not load top cancellations: {top_result['error']}")
            else:
                top_data = top_result.get("top_3", [])
                if not top_data:
                    st.info("No booking data available.")
                else:
                    top_df = pd.DataFrame(top_data)

                    # Build styled HTML table
                    col_map = {
                        "lead_time": "Lead Time (days)",
                        "adr": "ADR",
                        "market_segment": "Segment",
                        "deposit_type": "Deposit",
                        "customer_type": "Customer",
                        "cancel_prob": "Cancel Risk",
                    }
                    top_df = top_df.rename(columns=col_map)
                    display_cols = [v for v in col_map.values() if v in top_df.columns]
                    top_df = top_df[display_cols].head(5)

                    # Format
                    if "ADR" in top_df.columns:
                        top_df["ADR"] = top_df["ADR"].apply(lambda x: f"€{float(x):.0f}")
                    if "Lead Time (days)" in top_df.columns:
                        top_df["Lead Time (days)"] = top_df["Lead Time (days)"].apply(lambda x: f"{int(x)} days")

                    # Build HTML table with risk pills
                    table_html = '<table class="bc-table" style="width:100%;">'
                    table_html += '<thead><tr style="border-bottom:1.5px solid #E2E8F0;">'
                    for col in display_cols:
                        table_html += f'<th style="font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;color:{CLR_TEXT_MUT};padding:6px 4px;text-align:left;">{col}</th>'
                    table_html += '</tr></thead><tbody>'

                    for _, r in top_df.iterrows():
                        table_html += '<tr style="border-bottom:0.5px solid #F1F5F9;">'
                        for col in display_cols:
                            val = r[col]
                            if col == "Cancel Risk":
                                raw = float(str(val).replace("%", ""))
                                pill_cls = "risk-high" if raw >= 50 else "risk-med" if raw >= 25 else "risk-low"
                                cell = f'<span class="risk-pill {pill_cls}">{raw:.1f}% ↑</span>'
                            else:
                                cell = str(val)
                            table_html += f'<td style="padding:8px 4px;font-size:13px;">{cell}</td>'
                        table_html += '</tr>'
                    table_html += '</tbody></table>'
                    st.markdown(table_html, unsafe_allow_html=True)

        with right_col:
            st.markdown('<div class="nss-section-title">SHAP — Top Risk Factors</div>', unsafe_allow_html=True)
            st.markdown(
                f'<div class="nss-section-sub">Main drivers for cancellations for current segment.</div>',
                unsafe_allow_html=True,
            )

            shap_cache = f"shap_{selected_hotel}_{selected_date}_{selected_room}"
            if shap_cache not in st.session_state:
                with st.spinner("Loading SHAP …"):
                    shap_result = api_get(
                        EXPLAIN_GLOBAL_URL,
                        {"selected_date": str(selected_date), "room_type": selected_room, "hotel": selected_hotel},
                        timeout=60, max_retries=2,
                    )
                st.session_state[shap_cache] = shap_result
            else:
                shap_result = st.session_state[shap_cache]

            if "error" in shap_result:
                st.warning(f"Could not load SHAP data: {shap_result['error']}")
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
                    shap_df[["feature", "mean_abs_shap"]].dropna()
                    .sort_values("mean_abs_shap", ascending=False)
                    .head(5).reset_index(drop=True)
                )
                if top_shap.empty:
                    st.info("No SHAP data available.")
                else:
                    plot_df = top_shap.iloc[::-1].reset_index(drop=True)
                    fig_shap = go.Figure(go.Bar(
                        x=plot_df["mean_abs_shap"].tolist(),
                        y=plot_df["feature"].tolist(),
                        orientation="h",
                        marker_color=CLR_BLUE,
                    ))
                    fig_shap.update_layout(
                        height=320,
                        margin=dict(l=10, r=10, t=10, b=40),
                        showlegend=False,
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        xaxis_title="Mean |SHAP Value|",
                        yaxis_title="",
                        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
                        yaxis=dict(automargin=True),
                    )
                    st.plotly_chart(fig_shap, use_container_width=True)
            else:
                st.info("No SHAP data available for this selection.")


# ==================================================================
# SINGLE BOOKING ANALYSIS
# ==================================================================
else:
    st.markdown("""
    <div class="nss-page-header">
      <h1>Single Booking Prediction</h1>
      <p>Select a high-risk booking to analyse the underlying patterns detected by our AI model.
         Use SHAP values to understand specific cancellation drivers.</p>
    </div>
    """, unsafe_allow_html=True)

    # Load top bookings once
    if "top_bookings_list" not in st.session_state and "results" in st.session_state:
        with st.spinner("Loading top high-risk bookings …"):
            top_result = api_get(TOP_BOOKINGS_URL, {}, timeout=60, max_retries=2)
        if "error" in top_result:
            st.error(top_result["error"])
            st.session_state["top_bookings_list"] = []
        else:
            st.session_state["top_bookings_list"] = top_result["top_bookings"]

    if "results" not in st.session_state:
        st.info("Load recommendations first using the sidebar.")
    else:
        top_bookings_list = st.session_state.get("top_bookings_list", [])

        # Booking selector + export button in same row
        sel_col, btn_col = st.columns([5, 1])
        with sel_col:
            placeholder = "Select booking for prediction"
            dropdown_labels = [placeholder] + [b["label"] for b in top_bookings_list]
            selected_label = st.selectbox(
                "Analyse high-risk booking",
                options=dropdown_labels,
                index=0,
                disabled=not top_bookings_list,
                label_visibility="collapsed",
            )
        with btn_col:
            export_clicked = st.button("⬇ Export", use_container_width=True, disabled=True)

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
                st.session_state["single_actual"]  = selected_entry["actual_outcome"]
                st.session_state["single_explain"] = st.session_state[cache_key]

        if "single_booking" not in st.session_state:
            st.info("Select a booking above to load its prediction.")
        else:
            booking    = st.session_state["single_booking"]
            actual     = st.session_state["single_actual"]
            explain    = st.session_state["single_explain"]
            prob       = explain["cancellation_probability"]
            prediction = explain.get("prediction", int(prob >= 0.5))

            # ---- Prediction result box ----
            verdict_cls   = "" if prediction == 1 else "no-cancel"
            verdict_label = "Will Cancel ⚠️" if prediction == 1 else "Will Show ✓"
            pred_sub      = "High confidence cancellation expected." if prediction == 1 else "Low cancellation risk detected."

            pred_l, pred_r = st.columns(2)
            with pred_l:
                st.markdown(f"""
                <div class="pred-box">
                  <div class="pred-label">Model Prediction</div>
                  <div class="pred-verdict {verdict_cls}">{verdict_label}</div>
                  <div class="pred-sub">{pred_sub}</div>
                </div>
                """, unsafe_allow_html=True)
            with pred_r:
                st.markdown(f"""
                <div class="pred-box">
                  <div class="pred-label">Cancellation Probability</div>
                  <div class="pred-prob">{prob * 100:.1f}%</div>
                  <div class="pred-sub">Probability calculated by KGBoost Classifier.</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<hr>", unsafe_allow_html=True)

            # ---- Booking characteristics | SHAP ----
            left_col, right_col = st.columns([1, 1])

            with left_col:
                st.markdown('<div class="nss-section-title">Booking Characteristics</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="nss-section-sub">Comprehensive feature set used for this prediction instance.</div>',
                    unsafe_allow_html=True,
                )
                DISPLAY_LABELS = {
                    "hotel": "Hotel",
                    "lead_time": "Lead Time",
                    "arrival_date_month": "Arrival Month",
                    "stays_in_weekend_nights": "# Weekend Nights",
                    "stays_in_week_nights": "# Week Nights",
                    "meal": "Meal",
                    "country": "Country",
                    "market_segment": "Market Segment",
                    "adr": "Avg Daily Rate",
                    "customer_type": "Customer Type",
                    "deposit_type": "Deposit Type",
                    "total_special_requests": "Total Special Requests",
                    "reserved_room_type": "Reserved Room",
                    "assigned_room_type": "Assigned Room",
                    "booking_changes": "Booking Changes",
                    "days_in_waiting_list": "Days on Waitlist",
                    "required_car_parking_spaces": "Parking Spaces",
                    "previous_cancellations": "Prev. Cancellations",
                    "previous_bookings_not_canceled": "Prev. Non-Cancels",
                    "is_repeated_guest": "Repeated Guest",
                    "adults": "Adults",
                    "children": "Children",
                    "babies": "Babies",
                    "agent": "Agent",
                    "company": "Company",
                    "distribution_channel": "Distribution Channel",
                }

                rows_html = ""
                for key, val in booking.items():
                    label = DISPLAY_LABELS.get(key, key.replace("_", " ").title())
                    display_val = str(val) if val is not None else "—"
                    if key == "adr":
                        try:
                            display_val = f"€ {float(val):.2f}"
                        except Exception:
                            pass
                    rows_html += f'<tr><td>{label}</td><td>{display_val}</td></tr>'

                st.markdown(
                    f'<table class="bc-table">{rows_html}</table>',
                    unsafe_allow_html=True,
                )

            with right_col:
                st.markdown('<div class="nss-section-title">SHAP — Top 5 Risk Factors</div>', unsafe_allow_html=True)
                st.markdown(
                    '<div class="nss-section-sub">Main drivers contributing to cancellation for this booking segment.</div>',
                    unsafe_allow_html=True,
                )

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
                    .head(5).reset_index(drop=True)
                )

                if top_shap.empty:
                    st.info("No cancellation risk factors found for this booking.")
                else:
                    plot_df = top_shap.iloc[::-1].reset_index(drop=True)
                    fig_local = go.Figure(go.Bar(
                        x=plot_df["shap_value"].tolist(),
                        y=plot_df["feature"].tolist(),
                        orientation="h",
                        marker_color=CLR_BLUE,
                    ))
                    fig_local.update_layout(
                        height=400,
                        margin=dict(l=10, r=10, t=10, b=40),
                        showlegend=False,
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                        xaxis_title="SHAP Value",
                        yaxis_title="",
                        xaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.06)"),
                        yaxis=dict(automargin=True),
                    )
                    st.plotly_chart(fig_local, use_container_width=True)

            # ---- Action bar (decorative) ----
            st.markdown(f"""
            <div class="action-bar">
              <div>
                <div class="ab-text">🛡️ Ready to take action?</div>
                <div class="ab-sub">Use the options below to optimise owner or contact the guest.</div>
              </div>
              <div style="display:flex;gap:10px;">
                <button style="padding:8px 18px;border-radius:8px;border:1px solid #CBD5E1;
                  background:white;color:{CLR_TEXT_PRI};font-size:13px;font-weight:600;cursor:pointer;">
                  Ignore Prediction
                </button>
                <button style="padding:8px 18px;border-radius:8px;border:none;
                  background:{CLR_RED_RISK};color:white;font-size:13px;font-weight:600;cursor:pointer;">
                  Apply Relocation Strategy
                </button>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ------------------------------------------------------------------
# Footer
# ------------------------------------------------------------------
st.markdown(f"""
<div style="text-align:center;padding:24px 0 12px;font-size:12px;color:{CLR_TEXT_MUT};border-top:0.5px solid #E2E8F0;margin-top:32px;">
  © 2024 NoShowShield Intelligence. All rights reserved. &nbsp;·&nbsp;
  <a href="#" style="color:{CLR_BLUE};text-decoration:none;">Support</a> &nbsp;·&nbsp;
  <a href="#" style="color:{CLR_BLUE};text-decoration:none;">Privacy Policy</a> &nbsp;·&nbsp;
  <a href="#" style="color:{CLR_BLUE};text-decoration:none;">Documentation</a>
</div>
""", unsafe_allow_html=True)
