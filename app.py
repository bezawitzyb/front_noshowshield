"""
NoShowShield — Streamlit Dashboard
Revenue protection intelligence for hotel managers.
"""

import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, date

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="NoShowShield",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# THEME & CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080f1e;
    color: #e2e8f0;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem 2.5rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1a2f;
    border-right: 1px solid #1e3050;
}
[data-testid="stSidebar"] .stSlider label,
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] label {
    color: #94a3b8 !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    font-family: 'DM Mono', monospace !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e3050;
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.75rem 1.5rem;
    border: none;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #2dd4bf !important;
    border-bottom: 2px solid #2dd4bf !important;
    background: transparent !important;
}

/* ── Metric cards ── */
.metric-card {
    background: #0d1a2f;
    border: 1px solid #1e3050;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, #2dd4bf, #0ea5e9);
}
.metric-card .label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #64748b;
    margin-bottom: 0.4rem;
}
.metric-card .value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #f1f5f9;
    line-height: 1;
}
.metric-card .sub {
    font-size: 0.75rem;
    color: #475569;
    margin-top: 0.3rem;
    font-family: 'DM Mono', monospace;
}

/* ── Recommendation card ── */
.rec-card {
    background: linear-gradient(135deg, #0d2a2a 0%, #0d1a2f 100%);
    border: 1px solid #2dd4bf40;
    border-radius: 16px;
    padding: 1.75rem 2rem;
    position: relative;
    overflow: hidden;
}
.rec-card::after {
    content: '';
    position: absolute;
    bottom: -40px; right: -40px;
    width: 140px; height: 140px;
    background: radial-gradient(circle, #2dd4bf15, transparent 70%);
    border-radius: 50%;
}
.rec-title {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2dd4bf;
    margin-bottom: 1rem;
}
.rec-main {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    color: #f1f5f9;
    line-height: 1;
}
.rec-sub {
    font-size: 0.875rem;
    color: #94a3b8;
    margin-top: 0.5rem;
}
.rec-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #1e3050;
    font-size: 0.875rem;
}
.rec-row:last-child { border-bottom: none; }
.rec-row .key { color: #64748b; font-family: 'DM Mono', monospace; font-size: 0.78rem; }
.rec-row .val { color: #e2e8f0; font-weight: 500; }

/* ── Financial card ── */
.fin-card {
    background: #0d1a2f;
    border: 1px solid #1e3050;
    border-radius: 16px;
    padding: 1.75rem 2rem;
}
.fin-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.7rem 0;
    border-bottom: 1px solid #1a2744;
    font-size: 0.875rem;
}
.fin-row:last-child {
    border-bottom: none;
    padding-top: 1rem;
    margin-top: 0.5rem;
    border-top: 1px solid #2dd4bf30;
}
.fin-row .key { color: #64748b; font-family: 'DM Mono', monospace; font-size: 0.78rem; }
.fin-positive { color: #34d399; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.1rem; }
.fin-negative { color: #f87171; font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1.1rem; }
.fin-net { color: #2dd4bf; font-family: 'Syne', sans-serif; font-weight: 800; font-size: 1.4rem; }

/* ── SHAP bars ── */
.shap-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    font-size: 0.82rem;
}
.shap-label { color: #94a3b8; font-family: 'DM Mono', monospace; width: 160px; flex-shrink: 0; font-size: 0.75rem; }
.shap-bar-wrap { flex: 1; height: 8px; background: #1e3050; border-radius: 4px; overflow: hidden; }
.shap-bar-pos { height: 100%; background: linear-gradient(90deg, #f59e0b, #ef4444); border-radius: 4px; }
.shap-bar-neg { height: 100%; background: linear-gradient(90deg, #0ea5e9, #2dd4bf); border-radius: 4px; }
.shap-val-pos { color: #fbbf24; font-family: 'DM Mono', monospace; font-size: 0.72rem; width: 48px; text-align: right; }
.shap-val-neg { color: #38bdf8; font-family: 'DM Mono', monospace; font-size: 0.72rem; width: 48px; text-align: right; }

/* ── Section headers ── */
.section-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #2dd4bf;
    margin-bottom: 1rem;
    padding-bottom: 0.5rem;
    border-bottom: 1px solid #1e3050;
}

/* ── Pill badge ── */
.pill {
    display: inline-block;
    padding: 0.2rem 0.7rem;
    border-radius: 999px;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    letter-spacing: 0.05em;
    font-weight: 500;
}
.pill-green { background: #052e16; color: #4ade80; border: 1px solid #166534; }
.pill-red   { background: #2d0a0a; color: #f87171; border: 1px solid #7f1d1d; }
.pill-teal  { background: #022c22; color: #2dd4bf; border: 1px solid #134e4a; }
.pill-amber { background: #2d1a00; color: #fbbf24; border: 1px solid #78350f; }

/* ── Page header ── */
.page-header {
    display: flex;
    align-items: center;
    gap: 1rem;
    margin-bottom: 2rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid #1e3050;
}
.page-header .logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #f1f5f9;
    letter-spacing: -0.02em;
}
.page-header .logo span { color: #2dd4bf; }
.page-header .tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #475569;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-top: 0.2rem;
}

/* ── Warning / info boxes ── */
.info-box {
    background: #0d2a3a;
    border: 1px solid #0ea5e940;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-size: 0.82rem;
    color: #7dd3fc;
    font-family: 'DM Mono', monospace;
    margin: 0.5rem 0;
}
.warn-box {
    background: #2d1a00;
    border: 1px solid #f59e0b40;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    font-size: 0.82rem;
    color: #fbbf24;
    font-family: 'DM Mono', monospace;
    margin: 0.5rem 0;
}

/* ── Booking inspector ── */
.booking-field {
    display: flex;
    justify-content: space-between;
    padding: 0.4rem 0;
    border-bottom: 1px solid #1a2744;
    font-size: 0.8rem;
}
.booking-field .fk { color: #64748b; font-family: 'DM Mono', monospace; font-size: 0.72rem; }
.booking-field .fv { color: #cbd5e1; }

/* ── Plotly override ── */
.js-plotly-plot .plotly .modebar { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="DM Mono, monospace", color="#94a3b8", size=11),
    margin=dict(l=12, r=12, t=36, b=12),
    xaxis=dict(gridcolor="#1e3050", zerolinecolor="#1e3050"),
    yaxis=dict(gridcolor="#1e3050", zerolinecolor="#1e3050"),
)

TEAL  = "#2dd4bf"
AMBER = "#fbbf24"
RED   = "#f87171"
BLUE  = "#38bdf8"
GREEN = "#4ade80"


def api(path: str, method="get", **kwargs):
    base = st.session_state.get("api_url", "http://localhost:8000")
    try:
        fn = getattr(requests, method)
        r = fn(f"{base}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def fmt_eur(val):
    if val is None:
        return "—"
    sign = "+" if val >= 0 else ""
    return f"{sign}€{val:,.0f}"


def render_metric(label, value, sub="", accent=False):
    color = TEAL if accent else "#f1f5f9"
    return f"""
    <div class="metric-card">
        <div class="label">{label}</div>
        <div class="value" style="color:{color}">{value}</div>
        {"<div class='sub'>" + sub + "</div>" if sub else ""}
    </div>"""


def render_shap_bars(shap_list, max_abs, positive=True):
    html = ""
    for item in shap_list:
        feat = item.get("feature_group", "")
        val  = item.get("shap_value", 0)
        pct  = abs(val) / max_abs * 100 if max_abs > 0 else 0
        bar_class = "shap-bar-pos" if positive else "shap-bar-neg"
        val_class = "shap-val-pos" if positive else "shap-val-neg"
        sign = "+" if val > 0 else ""
        html += f"""
        <div class="shap-row">
            <div class="shap-label">{feat}</div>
            <div class="shap-bar-wrap"><div class="{bar_class}" style="width:{pct:.1f}%"></div></div>
            <div class="{val_class}">{sign}{val:.3f}</div>
        </div>"""
    return html


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 1rem 0 1.5rem 0;">
        <div style="font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:800;
                    color:#f1f5f9; letter-spacing:-0.02em;">
            🛡️ <span style="color:#2dd4bf">NoShow</span>Shield
        </div>
        <div style="font-family:'DM Mono',monospace; font-size:0.65rem;
                    color:#475569; letter-spacing:0.1em; text-transform:uppercase;
                    margin-top:0.3rem;">Revenue Intelligence</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">API Connection</div>', unsafe_allow_html=True)
    api_url = st.text_input("API URL", value="http://localhost:8000", label_visibility="collapsed")
    st.session_state["api_url"] = api_url

    # Test connection
    health = api("/")
    if "error" not in health:
        st.markdown('<div class="info-box">● API connected</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="warn-box">✕ API unreachable</div>', unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-header">Risk Parameters</div>', unsafe_allow_html=True)

    relocation_cost = st.slider(
        "Relocation Cost (€ / guest)",
        min_value=100, max_value=600, value=300, step=50,
    )
    max_risk_pct = st.slider(
        "Max Relocation Risk (%)",
        min_value=1, max_value=10, value=2, step=1,
    )
    max_risk = max_risk_pct / 100.0

    st.divider()
    st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)
    hotel_options = ["All Hotels", "City Hotel", "Resort Hotel"]
    hotel_filter = st.selectbox("Hotel", hotel_options, index=0)
    selected_hotel = None if hotel_filter == "All Hotels" else hotel_filter

    st.divider()
    st.markdown("""
    <div style="font-family:'DM Mono',monospace; font-size:0.65rem;
                color:#334155; padding: 0.5rem 0; text-align:center;">
        Bezawit Zerayacob · March 2026
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE HEADER
# ─────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <div>
        <div class="logo">🛡️ <span>NoShow</span>Shield</div>
        <div class="tagline">AI-Powered Hotel Revenue Protection · Cancellation Intelligence Dashboard</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab_revenue, tab_metrics, tab_inspector = st.tabs([
    "📊  Revenue Dashboard",
    "🎯  Model Performance",
    "🔍  Booking Inspector",
])


# ══════════════════════════════════════════════
# TAB 1 — REVENUE DASHBOARD
# ══════════════════════════════════════════════
with tab_revenue:

    # ── Load available dates ──────────────────
    @st.cache_data(ttl=300)
    def load_available_dates():
        return api("/explain/available-dates")

    dates_resp = load_available_dates()
    if "error" in dates_resp:
        st.markdown(f'<div class="warn-box">Could not load dates: {dates_resp["error"]}</div>',
                    unsafe_allow_html=True)
        st.stop()

    all_dates = [d["arrival_date"] for d in dates_resp["dates"]]
    date_counts = {d["arrival_date"]: d["n_bookings"] for d in dates_resp["dates"]}

    # ── Load recommendations (cached by params) ──
    @st.cache_data(ttl=120)
    def load_recommendations(relocation_cost, max_risk, hotel):
        params = {"relocation_cost": relocation_cost, "max_risk": max_risk}
        if hotel:
            params["hotel"] = hotel
        return api("/optimise", params=params)

    with st.spinner("Loading recommendations…"):
        rec_resp = load_recommendations(relocation_cost, max_risk, selected_hotel)

    if "error" in rec_resp:
        st.markdown(f'<div class="warn-box">API error: {rec_resp["error"]}</div>',
                    unsafe_allow_html=True)
        st.stop()

    recs_df = pd.DataFrame(rec_resp.get("recommendations", []))
    if recs_df.empty:
        st.markdown('<div class="warn-box">No recommendations returned from API.</div>',
                    unsafe_allow_html=True)
        st.stop()

    # ── Date & Room Selectors ──────────────────
    st.markdown('<div class="section-header">Select Date & Room Type</div>',
                unsafe_allow_html=True)

    available_dates_in_recs = sorted(recs_df["arrival_date"].astype(str).unique())
    col_date, col_room, col_spacer = st.columns([2, 2, 4])

    with col_date:
        chosen_date_str = st.selectbox(
            "Arrival Date",
            options=available_dates_in_recs,
            index=min(30, len(available_dates_in_recs) - 1),
        )

    # Filter room types for chosen date
    date_recs = recs_df[recs_df["arrival_date"].astype(str) == chosen_date_str]
    available_rooms = sorted(date_recs["assigned_room_type"].unique()) if not date_recs.empty else ["A"]

    with col_room:
        chosen_room = st.selectbox("Room Type", options=available_rooms)

    # ── Single row for chosen date + room ─────
    row_mask = (
        (recs_df["arrival_date"].astype(str) == chosen_date_str) &
        (recs_df["assigned_room_type"] == chosen_room)
    )
    if selected_hotel:
        row_mask &= (recs_df["hotel"] == selected_hotel)

    filtered = recs_df[row_mask]

    if filtered.empty:
        st.markdown('<div class="warn-box">No data for this date / room combination.</div>',
                    unsafe_allow_html=True)
        st.stop()

    row = filtered.iloc[0]

    capacity        = int(row["capacity"])
    total_bookings  = int(row["total_bookings"])
    exp_cancel      = float(row["expected_cancellations"])
    std_cancel      = float(row.get("std_cancellations", 0))
    exp_showup      = float(row["expected_show_ups"])
    rec_extra       = int(row["recommended_extra"])
    rec_total       = int(row["recommended_total"])
    add_rev         = float(row["additional_revenue"])
    exp_reloc       = float(row["expected_relocation_cost"])
    net_benefit     = float(row["net_benefit"])
    reloc_prob      = float(row["relocation_probability"])
    mean_adr        = float(row["mean_adr"])

    st.divider()

    # ── KPI row ───────────────────────────────
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(render_metric("Physical Capacity", f"{capacity}", "rooms"), unsafe_allow_html=True)
    with k2:
        over = total_bookings - capacity
        sub = f"+{over} over capacity" if over > 0 else "at capacity"
        st.markdown(render_metric("Confirmed Bookings", f"{total_bookings}", sub), unsafe_allow_html=True)
    with k3:
        st.markdown(render_metric("Expected Cancellations", f"{exp_cancel:.1f}",
                                  f"σ = {std_cancel:.1f}"), unsafe_allow_html=True)
    with k4:
        occ_pct = min(exp_showup / capacity * 100, 100)
        st.markdown(render_metric("Expected Show-ups", f"{exp_showup:.0f}",
                                  f"{occ_pct:.1f}% occupancy", accent=True),
                    unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Recommendation + Financial ─────────────
    col_rec, col_fin = st.columns([1.1, 1])

    with col_rec:
        risk_badge = (
            f'<span class="pill pill-green">▲ {reloc_prob*100:.1f}% relocation risk</span>'
            if reloc_prob <= max_risk
            else f'<span class="pill pill-red">▲ {reloc_prob*100:.1f}% relocation risk — HIGH</span>'
        )
        rec_badge = (
            f'<span class="pill pill-teal">OVERBOOK +{rec_extra}</span>'
            if rec_extra > 0
            else '<span class="pill pill-amber">HOLD — no extra bookings recommended</span>'
        )

        st.markdown(f"""
        <div class="rec-card">
            <div class="rec-title">Overbooking Recommendation</div>
            <div class="rec-main">+{rec_extra}</div>
            <div class="rec-sub">additional bookings → {rec_total} total</div>
            <br>
            <div style="display:flex; gap:0.5rem; flex-wrap:wrap; margin-bottom:1rem;">
                {rec_badge}
                {risk_badge}
            </div>
            <div class="rec-row">
                <span class="key">CAPACITY</span>
                <span class="val">{capacity} rooms</span>
            </div>
            <div class="rec-row">
                <span class="key">CONFIRMED BOOKINGS</span>
                <span class="val">{total_bookings}</span>
            </div>
            <div class="rec-row">
                <span class="key">EXPECTED CANCELLATIONS</span>
                <span class="val">{exp_cancel:.1f} <span style="color:#475569; font-size:0.75rem;">(± {std_cancel:.1f})</span></span>
            </div>
            <div class="rec-row">
                <span class="key">EXPECTED SHOW-UPS</span>
                <span class="val">{exp_showup:.0f} guests</span>
            </div>
            <div class="rec-row">
                <span class="key">MEAN ADR</span>
                <span class="val">€{mean_adr:.0f} / night</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_fin:
        st.markdown(f"""
        <div class="fin-card">
            <div class="rec-title">Financial Impact</div>
            <div class="fin-row">
                <span class="key">ADDITIONAL REVENUE</span>
                <span class="fin-positive">{fmt_eur(add_rev)}/day</span>
            </div>
            <div class="fin-row">
                <span class="key">EXPECTED RELOCATION COST</span>
                <span class="fin-negative">−€{exp_reloc:.0f}/day</span>
            </div>
            <div class="fin-row">
                <span class="key">NET DAILY BENEFIT</span>
                <span class="fin-net">{fmt_eur(net_benefit)}/day</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Annualised projection
        ann = net_benefit * 365
        st.markdown(f"""
        <div style="margin-top:0.75rem; padding:1rem 1.25rem;
                    background:#0a1f1a; border:1px solid #2dd4bf20;
                    border-radius:10px;">
            <div style="font-family:'DM Mono',monospace; font-size:0.68rem;
                        color:#475569; letter-spacing:0.1em; text-transform:uppercase;
                        margin-bottom:0.4rem;">Annualised Projection</div>
            <div style="font-family:'Syne',sans-serif; font-size:1.6rem;
                        font-weight:800; color:#2dd4bf;">
                {fmt_eur(ann)}/year
            </div>
            <div style="font-family:'DM Mono',monospace; font-size:0.7rem;
                        color:#334155; margin-top:0.2rem;">per 100-room property</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # ── Cancellation Distribution Chart ───────
    col_dist, col_shap = st.columns([1.2, 1])

    with col_dist:
        st.markdown('<div class="section-header">Cancellation Distribution — Current vs Recommended</div>',
                    unsafe_allow_html=True)

        @st.cache_data(ttl=120)
        def load_distribution(date_str, room, hotel, rel_cost, m_risk):
            params = {
                "selected_date": date_str,
                "room_type": room,
                "relocation_cost": rel_cost,
                "max_risk": m_risk,
            }
            if hotel:
                params["hotel"] = hotel
            return api("/optimise/cancellation-distribution", params=params)

        dist_resp = load_distribution(chosen_date_str, chosen_room, selected_hotel,
                                      relocation_cost, max_risk)

        if "error" not in dist_resp and "distribution" in dist_resp:
            d = dist_resp["distribution"]

            x_cur = d["current"]["x"]
            pmf_cur = d["current"]["pmf"]
            x_rec = d["recommended"]["x"]
            pmf_rec = d["recommended"]["pmf"]
            min_needed = d.get("min_cancellations_needed", 0)

            fig_dist = go.Figure()
            fig_dist.add_trace(go.Bar(
                x=x_cur, y=pmf_cur,
                name=f"Current ({d['current']['n_bookings']} bookings)",
                marker_color="rgba(56, 189, 248, 0.65)",
                marker_line_width=0,
                hovertemplate="Cancellations: %{x}<br>P = %{y:.4f}<extra></extra>",
            ))
            fig_dist.add_trace(go.Bar(
                x=x_rec, y=pmf_rec,
                name=f"Recommended ({d['recommended']['n_bookings']} bookings)",
                marker_color="rgba(251, 191, 36, 0.55)",
                marker_line_width=0,
                hovertemplate="Cancellations: %{x}<br>P = %{y:.4f}<extra></extra>",
            ))

            if min_needed > 0:
                fig_dist.add_vline(
                    x=min_needed - 0.5,
                    line_dash="dot",
                    line_color=RED,
                    line_width=1.5,
                    annotation_text=f" ≥{min_needed} to avoid walks",
                    annotation_font_size=10,
                    annotation_font_color=RED,
                    annotation_position="top right",
                )

            fig_dist.update_layout(
                **PLOTLY_LAYOUT,
                barmode="overlay",
                bargap=0.05,
                height=320,
                xaxis_title="Number of Cancellations",
                yaxis_title="Probability",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1,
                    bgcolor="rgba(0,0,0,0)",
                    font=dict(size=10),
                ),
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        else:
            st.markdown('<div class="warn-box">Distribution endpoint not available yet. '
                        'Add /optimise/cancellation-distribution to fast.py.</div>',
                        unsafe_allow_html=True)

    # ── SHAP Explainability ───────────────────
    with col_shap:
        st.markdown('<div class="section-header">Risk Drivers — SHAP Explainability</div>',
                    unsafe_allow_html=True)

        @st.cache_data(ttl=120)
        def load_shap(date_str, room, hotel):
            params = {"selected_date": date_str, "room_type": room, "min_rows": 3}
            if hotel:
                params["hotel"] = hotel
            return api("/explain/global-by-date", params=params)

        shap_resp = load_shap(chosen_date_str, chosen_room, selected_hotel)

        if "error" not in shap_resp and shap_resp.get("grouped_global_shap"):
            shap_data = shap_resp["grouped_global_shap"][:8]
            max_shap = max(abs(s["mean_abs_shap"]) for s in shap_data) if shap_data else 1

            st.markdown(f"""
            <div style="font-family:'DM Mono',monospace; font-size:0.7rem; color:#475569;
                        margin-bottom:1rem;">
                Based on {shap_resp.get('n_bookings', '—')} bookings on {chosen_date_str}
            </div>
            """, unsafe_allow_html=True)

            bars_html = ""
            for item in shap_data:
                feat = item.get("feature_group", "")
                val  = float(item.get("mean_abs_shap", 0))
                pct  = val / max_shap * 100
                bars_html += f"""
                <div class="shap-row">
                    <div class="shap-label">{feat}</div>
                    <div class="shap-bar-wrap">
                        <div class="shap-bar-pos" style="width:{pct:.1f}%"></div>
                    </div>
                    <div class="shap-val-pos">{val:.3f}</div>
                </div>"""

            st.markdown(bars_html, unsafe_allow_html=True)

        elif shap_resp.get("message"):
            st.markdown(f'<div class="info-box">{shap_resp["message"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">SHAP data unavailable for this date.</div>',
                        unsafe_allow_html=True)

    st.divider()

    # ── Portfolio View: Net Benefit Over Time ──
    st.markdown('<div class="section-header">Portfolio View — Net Benefit by Date</div>',
                unsafe_allow_html=True)

    if not recs_df.empty:
        portfolio = (
            recs_df.groupby("arrival_date")
            .agg(
                net_benefit=("net_benefit", "sum"),
                total_bookings=("total_bookings", "sum"),
                expected_cancellations=("expected_cancellations", "sum"),
            )
            .reset_index()
            .sort_values("arrival_date")
        )
        portfolio["arrival_date"] = pd.to_datetime(portfolio["arrival_date"])
        portfolio["cancel_rate"] = (
            portfolio["expected_cancellations"] / portfolio["total_bookings"] * 100
        )

        fig_port = go.Figure()
        fig_port.add_trace(go.Bar(
            x=portfolio["arrival_date"],
            y=portfolio["net_benefit"],
            name="Net Benefit (€)",
            marker_color=[
                "rgba(45, 212, 191, 0.8)" if v >= 0 else "rgba(248, 113, 113, 0.8)"
                for v in portfolio["net_benefit"]
            ],
            marker_line_width=0,
            hovertemplate="%{x|%b %d}<br>Net: €%{y:,.0f}<extra></extra>",
        ))
        fig_port.update_layout(
            **PLOTLY_LAYOUT,
            height=220,
            xaxis_title=None,
            yaxis_title="€",
            showlegend=False,
            bargap=0.2,
        )
        st.plotly_chart(fig_port, use_container_width=True)

        # Summary stats row
        s1, s2, s3, s4 = st.columns(4)
        total_net = portfolio["net_benefit"].sum()
        profitable_days = (portfolio["net_benefit"] > 0).sum()
        avg_cancel_rate = portfolio["cancel_rate"].mean()
        avg_daily = portfolio["net_benefit"].mean()

        with s1:
            st.markdown(render_metric("Total Net Benefit", fmt_eur(total_net), "across all dates"),
                        unsafe_allow_html=True)
        with s2:
            st.markdown(render_metric("Profitable Dates",
                                      f"{profitable_days}/{len(portfolio)}",
                                      "dates with positive net benefit"),
                        unsafe_allow_html=True)
        with s3:
            st.markdown(render_metric("Avg Cancel Rate",
                                      f"{avg_cancel_rate:.1f}%",
                                      "portfolio-wide"),
                        unsafe_allow_html=True)
        with s4:
            st.markdown(render_metric("Avg Daily Benefit", fmt_eur(avg_daily),
                                      "per date", accent=True),
                        unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 2 — MODEL PERFORMANCE
# ══════════════════════════════════════════════
with tab_metrics:

    @st.cache_data(ttl=300)
    def load_metrics(rel_cost, m_risk, hotel):
        params = {"relocation_cost": rel_cost, "max_risk": m_risk}
        if hotel:
            params["hotel"] = hotel
        return api("/optimise", params=params)

    m_resp = load_metrics(relocation_cost, max_risk, selected_hotel)

    if "error" in m_resp:
        st.markdown(f'<div class="warn-box">{m_resp["error"]}</div>', unsafe_allow_html=True)
    else:
        metrics = m_resp.get("metrics", {})
        model_info = m_resp.get("model_info", {})

        st.markdown('<div class="section-header">Classifier Performance — Holdout Test Set</div>',
                    unsafe_allow_html=True)

        m1, m2, m3, m4, m5 = st.columns(5)
        metric_items = [
            ("AUC-ROC", "auc", "≥ 0.85 target"),
            ("Precision", "precision", "of predicted cancellations"),
            ("Recall", "recall", "of actual cancellations"),
            ("F1 Score", "f1", "harmonic mean"),
            ("Accuracy", "accuracy", "overall"),
        ]
        for col, (label, key, sub) in zip([m1, m2, m3, m4, m5], metric_items):
            val = metrics.get(key, "—")
            formatted = f"{val:.2f}" if isinstance(val, float) else str(val)
            accent = key == "auc"
            with col:
                st.markdown(render_metric(label, formatted, sub, accent=accent),
                            unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gauge for AUC
        col_gauge, col_info = st.columns([1, 1.5])

        with col_gauge:
            auc_val = metrics.get("auc", 0)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=auc_val,
                number=dict(font=dict(family="Syne", color="#2dd4bf", size=36)),
                gauge=dict(
                    axis=dict(range=[0.5, 1.0], tickfont=dict(color="#64748b", size=10)),
                    bar=dict(color="#2dd4bf", thickness=0.25),
                    bgcolor="rgba(0,0,0,0)",
                    borderwidth=0,
                    steps=[
                        dict(range=[0.5, 0.7], color="#1e2a3a"),
                        dict(range=[0.7, 0.85], color="#1a2f2a"),
                        dict(range=[0.85, 1.0], color="#0d2a22"),
                    ],
                    threshold=dict(
                        line=dict(color=AMBER, width=2),
                        thickness=0.75, value=0.85,
                    ),
                ),
                title=dict(text="AUC-ROC", font=dict(
                    family="DM Mono", color="#64748b", size=11)),
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(family="DM Mono"),
                height=240,
                margin=dict(l=20, r=20, t=20, b=0),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

        with col_info:
            st.markdown('<div class="section-header" style="margin-top:1rem">Model Configuration</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="label">Model Type</div>
                <div style="font-family:'DM Mono',monospace; color:#2dd4bf; font-size:1rem;
                            font-weight:500; margin-top:0.4rem;">
                    {model_info.get('model_type', 'XGBClassifier')}
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)

            eval_table = {
                "Target AUC": "≥ 0.85",
                "Target Precision": "≥ 0.80",
                "Target Recall": "≥ 0.75",
                "Brier Score target": "≤ 0.15",
                "Dataset": "119,390 bookings",
                "Train/Test split": "Temporal (2017 = test)",
                "Class balance": "37% cancel / 63% complete",
            }
            rows = "".join(
                f'<div class="rec-row"><span class="key">{k}</span>'
                f'<span class="val">{v}</span></div>'
                for k, v in eval_table.items()
            )
            st.markdown(f'<div class="rec-card">{rows}</div>', unsafe_allow_html=True)

        st.divider()

        # Room type breakdown
        st.markdown('<div class="section-header">Recommendations Breakdown by Room Type</div>',
                    unsafe_allow_html=True)

        recs_df2 = pd.DataFrame(m_resp.get("recommendations", []))
        if not recs_df2.empty:
            breakdown = (
                recs_df2.groupby("assigned_room_type")
                .agg(
                    total_bookings=("total_bookings", "sum"),
                    expected_cancellations=("expected_cancellations", "sum"),
                    net_benefit=("net_benefit", "sum"),
                    recommended_extra=("recommended_extra", "sum"),
                )
                .reset_index()
                .sort_values("net_benefit", ascending=False)
            )
            breakdown["cancel_rate"] = (
                breakdown["expected_cancellations"] / breakdown["total_bookings"] * 100
            )

            fig_room = go.Figure()
            fig_room.add_trace(go.Bar(
                x=breakdown["assigned_room_type"],
                y=breakdown["net_benefit"],
                name="Net Benefit",
                marker_color="rgba(45, 212, 191, 0.75)",
                marker_line_width=0,
                hovertemplate="Room %{x}<br>Net: €%{y:,.0f}<extra></extra>",
            ))
            fig_room.add_trace(go.Scatter(
                x=breakdown["assigned_room_type"],
                y=breakdown["cancel_rate"],
                name="Cancel Rate %",
                yaxis="y2",
                line=dict(color=AMBER, width=2),
                mode="lines+markers",
                marker=dict(size=6),
                hovertemplate="Room %{x}<br>Cancel Rate: %{y:.1f}%<extra></extra>",
            ))
            fig_room.update_layout(
                **PLOTLY_LAYOUT,
                height=300,
                yaxis=dict(title="Net Benefit (€)", gridcolor="#1e3050", zerolinecolor="#1e3050"),
                yaxis2=dict(
                    title="Cancel Rate (%)",
                    overlaying="y", side="right",
                    gridcolor="rgba(0,0,0,0)",
                    tickfont=dict(color=AMBER),
                ),
                legend=dict(orientation="h", yanchor="bottom", y=1.02,
                            bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
                bargap=0.3,
            )
            st.plotly_chart(fig_room, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3 — BOOKING INSPECTOR
# ══════════════════════════════════════════════
with tab_inspector:

    st.markdown('<div class="section-header">Single Booking Risk Assessment</div>',
                unsafe_allow_html=True)

    col_btn, col_spacer = st.columns([1, 4])
    with col_btn:
        if st.button("🎲 Load Random Booking", use_container_width=True):
            st.session_state["random_booking"] = api("/random-booking")

    if "random_booking" in st.session_state:
        rb = st.session_state["random_booking"]
        if "error" in rb:
            st.markdown(f'<div class="warn-box">{rb["error"]}</div>', unsafe_allow_html=True)
        else:
            booking = rb.get("booking", {})
            actual = rb.get("actual_outcome", None)

            col_booking, col_predict = st.columns([1, 1.2])

            with col_booking:
                st.markdown('<div class="section-header">Booking Details</div>',
                            unsafe_allow_html=True)
                actual_label = (
                    '<span class="pill pill-red">CANCELLED</span>'
                    if actual == 1
                    else '<span class="pill pill-green">COMPLETED</span>'
                )
                st.markdown(f"**Actual outcome:** {actual_label}", unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)

                display_fields = [
                    "hotel", "lead_time", "market_segment", "deposit_type",
                    "previous_cancellations", "adr", "arrival_date_year",
                    "arrival_date_month", "adults", "is_repeated_guest",
                    "total_of_special_requests", "customer_type",
                    "distribution_channel", "reserved_room_type",
                ]
                rows = ""
                for f in display_fields:
                    if f in booking:
                        rows += f"""
                        <div class="booking-field">
                            <span class="fk">{f}</span>
                            <span class="fv">{booking[f]}</span>
                        </div>"""
                st.markdown(f'<div class="fin-card">{rows}</div>', unsafe_allow_html=True)

            with col_predict:
                st.markdown('<div class="section-header">Model Prediction & SHAP</div>',
                            unsafe_allow_html=True)

                with st.spinner("Running prediction…"):
                    predict_resp = api("/explain/local", method="post", json=booking)

                if "error" in predict_resp:
                    st.markdown(f'<div class="warn-box">{predict_resp["error"]}</div>',
                                unsafe_allow_html=True)
                else:
                    prob = predict_resp.get("cancellation_probability", 0)
                    prob_pct = prob * 100

                    # Probability gauge
                    if prob_pct >= 60:
                        prob_color = RED
                        prob_label = "HIGH RISK"
                        pill_class = "pill-red"
                    elif prob_pct >= 35:
                        prob_color = AMBER
                        prob_label = "MEDIUM RISK"
                        pill_class = "pill-amber"
                    else:
                        prob_color = GREEN
                        prob_label = "LOW RISK"
                        pill_class = "pill-green"

                    st.markdown(f"""
                    <div class="rec-card" style="text-align:center; padding: 1.5rem;">
                        <div class="rec-title">Cancellation Probability</div>
                        <div style="font-family:'Syne',sans-serif; font-size:3.5rem;
                                    font-weight:800; color:{prob_color}; line-height:1;">
                            {prob_pct:.1f}%
                        </div>
                        <div style="margin-top:0.75rem;">
                            <span class="pill {pill_class}">{prob_label}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown("<br>", unsafe_allow_html=True)

                    # SHAP drivers
                    higher = predict_resp.get("higher_cancellation_risk", [])
                    lower  = predict_resp.get("lower_cancellation_risk", [])

                    all_vals = [abs(s.get("shap_value", 0))
                                for s in (higher + lower)]
                    max_abs = max(all_vals) if all_vals else 1

                    if higher:
                        st.markdown("""
                        <div style="font-family:'DM Mono',monospace; font-size:0.68rem;
                                    color:#f87171; letter-spacing:0.1em; text-transform:uppercase;
                                    margin-bottom:0.5rem;">↑ Increasing Risk</div>
                        """, unsafe_allow_html=True)
                        st.markdown(render_shap_bars(higher, max_abs, positive=True),
                                    unsafe_allow_html=True)

                    if lower:
                        st.markdown("""
                        <div style="font-family:'DM Mono',monospace; font-size:0.68rem;
                                    color:#38bdf8; letter-spacing:0.1em; text-transform:uppercase;
                                    margin: 1rem 0 0.5rem 0;">↓ Reducing Risk</div>
                        """, unsafe_allow_html=True)
                        st.markdown(render_shap_bars(lower, max_abs, positive=False),
                                    unsafe_allow_html=True)

    else:
        st.markdown("""
        <div class="info-box" style="text-align:center; padding: 2rem;">
            Press "Load Random Booking" to sample a booking from the test set
            and inspect the model's prediction with SHAP explainability.
        </div>
        """, unsafe_allow_html=True)
