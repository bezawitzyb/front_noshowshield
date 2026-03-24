"""
NoShowShield — Modern Hotel Revenue Protection Dashboard
Redesigned with focus on visual hierarchy, progressive disclosure, and UX best practices.

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
from plotly.subplots import make_subplots

# ==================================================================
# PAGE CONFIGURATION & DESIGN SYSTEM
# ==================================================================
st.set_page_config(
    page_title="NoShowShield | Revenue Protection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design system
st.markdown("""
<style>
    /* Typography & Base Styles */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header Styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 10px 40px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0.5rem 0 0 0;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
        transition: transform 0.2s, box-shadow 0.2s;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.12);
    }

    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-bottom: 0.5rem;
    }

    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #111827;
        line-height: 1.2;
    }

    .metric-delta {
        font-size: 0.875rem;
        margin-top: 0.25rem;
        font-weight: 500;
    }

    .positive { color: #10b981; }
    .negative { color: #ef4444; }
    .warning { color: #f59e0b; }

    /* Status Badges */
    .status-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.375rem 0.875rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 600;
        gap: 0.375rem;
    }

    .status-optimal {
        background: #d1fae5;
        color: #065f46;
    }

    .status-warning {
        background: #fef3c7;
        color: #92400e;
    }

    .status-danger {
        background: #fee2e2;
        color: #991b1b;
    }

    /* Section Headers */
    .section-header {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    /* Sidebar Styling */
    .sidebar-section {
        background: #f9fafb;
        padding: 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        border: 1px solid #e5e7eb;
    }

    /* Chart Containers */
    .chart-container {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        border: 1px solid #e5e7eb;
    }

    /* Data Tables */
    .styled-table {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    /* Info Cards */
    .info-card {
        background: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }

    /* Risk Indicators */
    .risk-low { color: #10b981; }
    .risk-medium { color: #f59e0b; }
    .risk-high { color: #ef4444; }

    /* Button Styling Override */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.2s;
    }

    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }

    /* Slider Customization */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }

    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 12px 24px;
        font-weight: 500;
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    /* Tooltip Styling */
    .tooltip {
        position: relative;
        cursor: help;
    }

    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Loading Animation */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }

    .loading-pulse {
        animation: pulse 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
    }
</style>
""", unsafe_allow_html=True)

# ==================================================================
# API CONFIGURATION
# ==================================================================
if 'API_URI' in os.environ:
    BASE_URI = st.secrets[os.environ.get('API_URI')]
else:
    BASE_URI = st.secrets.get('cloud_api_uri', 'http://localhost:8000')

BASE_URI = BASE_URI if BASE_URI.endswith('/') else BASE_URI + '/'

OPTIMISE_URL = BASE_URI + 'optimise'
EXPLAIN_GLOBAL_URL = BASE_URI + 'explain/global-by-date'
TOP_CANCELLATIONS_URL = BASE_URI + 'top-cancellations'
GROUP_PROBS_URL = BASE_URI + 'group-probs'
TOP_BOOKINGS_URL = BASE_URI + 'top-bookings'
EXPLAIN_LOCAL_URL = BASE_URI + 'explain/local'

# ==================================================================
# UTILITY FUNCTIONS
# ==================================================================
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
    """GET request with retries for Cloud Run cold starts."""
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
    """POST request with retries."""
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

def get_risk_color(prob):
    """Return color based on risk probability."""
    if prob < 0.02:
        return "#10b981"  # Green
    elif prob < 0.05:
        return "#f59e0b"  # Yellow
    else:
        return "#ef4444"  # Red

def get_risk_status(prob):
    """Return status text based on risk probability."""
    if prob < 0.02:
        return "Low Risk", "status-optimal"
    elif prob < 0.05:
        return "Moderate Risk", "status-warning"
    else:
        return "High Risk", "status-danger"

# ==================================================================
# SIDEBAR - SETTINGS PANEL
# ==================================================================
with st.sidebar:
    # Logo/Brand Area
    st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 3rem; margin-bottom: 0.5rem;">🛡️</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #1f2937;">NoShowShield</div>
            <div style="font-size: 0.875rem; color: #6b7280; margin-top: 0.25rem;">Revenue Protection System</div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Optimization Settings Section
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown("### ⚙️ Optimization Settings")

    relocation_cost = st.number_input(
        "Relocation Cost (€)",
        min_value=0.0,
        max_value=1000.0,
        value=300.0,
        step=50.0,
        help="Cost incurred when relocating an overbooked guest to another hotel.",
        format="%.0f"
    )

    max_risk = st.slider(
        "Risk Tolerance",
        min_value=0.0,
        max_value=0.10,
        value=0.02,
        step=0.005,
        format="%.1%%",
        help="Maximum acceptable probability of guest relocation."
    )

    # Risk indicator
    risk_color = get_risk_color(max_risk)
    st.markdown(f"""
        <div style="margin-top: 0.5rem; padding: 0.75rem; background: {risk_color}15; border-radius: 8px; border-left: 4px solid {risk_color};">
            <div style="font-size: 0.875rem; color: {risk_color}; font-weight: 600;">
                Current Threshold: {max_risk:.1%}
            </div>
        </div>
    """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    # Action Button
    if st.button("🚀 Get Recommendations", type="primary", use_container_width=True):
        with st.spinner("Analyzing booking patterns..."):
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
                st.success("Analysis complete!")
                time.sleep(0.5)
                st.rerun()

    st.markdown("---")

    # Filters Section (only show if results exist)
    if "results" in st.session_state:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🔍 Filters")

        recs = pd.DataFrame(st.session_state["results"]["recommendations"])
        recs["arrival_date"] = pd.to_datetime(recs["arrival_date"])

        available_hotels = sorted(recs["hotel"].unique())
        selected_hotel = st.selectbox("Hotel", available_hotels, key="hotel_filter")

        available_dates = sorted(
            recs[recs["hotel"] == selected_hotel]["arrival_date"].dt.date.unique()
        )
        selected_date = st.selectbox("Arrival Date", available_dates, key="date_filter")

        available_rooms = sorted(
            recs[
                (recs["hotel"] == selected_hotel) &
                (recs["arrival_date"].dt.date == selected_date)
            ]["assigned_room_type"].unique()
        )
        selected_room = st.selectbox("Room Type", available_rooms, key="room_filter")

        st.session_state["selected_hotel"] = selected_hotel
        st.session_state["selected_date"] = selected_date
        st.session_state["selected_room"] = selected_room

        st.markdown('</div>', unsafe_allow_html=True)

        # Model Performance Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 📊 Model Performance")

        metrics = st.session_state["results"]["metrics"]
        model_info = st.session_state["results"]["model_info"]

        col1, col2 = st.columns(2)
        with col1:
            st.metric("AUC", f"{metrics.get('auc', 0):.3f}")
        with col2:
            st.metric("Accuracy", f"{metrics.get('accuracy', 0):.1%}")

        st.caption(f"Model: {model_info.get('model_type', 'XGBoost')}")
        st.markdown('</div>', unsafe_allow_html=True)

# ==================================================================
# MAIN HEADER
# ==================================================================
st.markdown("""
    <div class="main-header">
        <h1>🛡️ NoShowShield</h1>
        <p>AI-Powered Revenue Protection — Optimize overbooking while minimizing relocation risk</p>
    </div>
""", unsafe_allow_html=True)

# ==================================================================
# TABS
# ==================================================================
tab1, tab2 = st.tabs(["📈 Revenue Optimization", "🔍 Single Booking Analysis"])

# ==================================================================
# TAB 1: REVENUE OPTIMIZATION
# ==================================================================
with tab1:
    if "results" not in st.session_state:
        # Empty State
        st.markdown("""
            <div style="text-align: center; padding: 4rem 2rem; background: #f9fafb; border-radius: 16px; border: 2px dashed #e5e7eb;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">📊</div>
                <h3 style="color: #374151; margin-bottom: 0.5rem;">Ready to Analyze</h3>
                <p style="color: #6b7280; max-width: 400px; margin: 0 auto;">
                    Configure your settings in the sidebar and click <strong>Get Recommendations</strong> to start optimizing your revenue.
                </p>
            </div>
        """, unsafe_allow_html=True)

    else:
        results = st.session_state["results"]
        recs = pd.DataFrame(results["recommendations"])
        recs["arrival_date"] = pd.to_datetime(recs["arrival_date"])

        # Get current selection
        selected_hotel = st.session_state.get("selected_hotel")
        selected_date = st.session_state.get("selected_date")
        selected_room = st.session_state.get("selected_room")

        if not all([selected_hotel, selected_date, selected_room]):
            st.warning("Please select filters from the sidebar to view recommendations.")
        else:
            # Filter data
            filtered = recs[
                (recs["hotel"] == selected_hotel) &
                (recs["arrival_date"].dt.date == selected_date) &
                (recs["assigned_room_type"] == selected_room)
            ]

            if filtered.empty:
                st.warning("No data available for the selected criteria.")
            else:
                row = filtered.iloc[0]

                # Status Badge
                risk_status, risk_class = get_risk_status(row["relocation_probability"])

                # KEY METRICS ROW
                st.markdown('<div class="section-header">📊 Key Metrics</div>', unsafe_allow_html=True)

                cols = st.columns(4)

                metrics_data = [
                    ("Capacity", f"{int(row['capacity'])}", "rooms", "neutral"),
                    ("Current Bookings", f"{int(row['total_bookings'])}", "booked", "neutral"),
                    ("Expected Show-ups", f"{row['expected_show_ups']:.1f}", "guests", "neutral"),
                    ("Recommended Overbook", f"+{int(row['recommended_extra'])}", "additional", "positive" if row['recommended_extra'] > 0 else "neutral")
                ]

                for col, (label, value, subtext, sentiment) in zip(cols, metrics_data):
                    with col:
                        st.markdown(f"""
                            <div class="metric-card">
                                <div class="metric-label">{label}</div>
                                <div class="metric-value">{value}</div>
                                <div class="metric-delta {'positive' if sentiment == 'positive' else ''}">{subtext}</div>
                            </div>
                        """, unsafe_allow_html=True)

                # FINANCIAL & RISK ROW
                st.markdown('<div class="section-header">💰 Financial Impact & Risk Assessment</div>', unsafe_allow_html=True)

                fin_cols = st.columns(3)

                with fin_cols[0]:
                    net_benefit = row['net_benefit']
                    benefit_color = "positive" if net_benefit > 0 else "negative"
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Net Revenue Impact</div>
                            <div class="metric-value {benefit_color}">€{net_benefit:,.2f}</div>
                            <div class="metric-delta">per day</div>
                        </div>
                    """, unsafe_allow_html=True)

                with fin_cols[1]:
                    reloc_prob = row['relocation_probability']
                    risk_color_class = "positive" if reloc_prob < 0.02 else "warning" if reloc_prob < 0.05 else "negative"
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Relocation Risk</div>
                            <div class="metric-value {risk_color_class}">{reloc_prob:.1%}</div>
                            <div class="metric-delta">probability</div>
                        </div>
                    """, unsafe_allow_html=True)

                with fin_cols[2]:
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Risk Status</div>
                            <div style="margin-top: 0.5rem;">
                                <span class="status-badge {risk_class}">{risk_status}</span>
                            </div>
                            <div class="metric-delta">threshold: {max_risk:.1%}</div>
                        </div>
                    """, unsafe_allow_html=True)

                # SHOW-UP DISTRIBUTION CHART
                st.markdown('<div class="section-header">📉 Show-up Probability Distribution</div>', unsafe_allow_html=True)

                # Fetch probabilities
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

                # Resolve probabilities
                n_current = int(row["total_bookings"])
                capacity = int(row["capacity"])
                recommended_total = int(row["recommended_total"])

                if "cancel_probs" in gp_result:
                    cancel_probs_arr = np.array(gp_result["cancel_probs"], dtype=np.float64)
                else:
                    mean_cp = row.get("cancel_prob_mean", row["expected_cancellations"] / row["total_bookings"])
                    cancel_probs_arr = np.full(n_current, float(mean_cp))

                # Interactive Slider
                st.markdown("""
                    <div class="info-card">
                        <strong>💡 Tip:</strong> Adjust the slider to simulate different booking levels and see how
                        the relocation risk changes. The green zone represents safe capacity levels.
                    </div>
                """, unsafe_allow_html=True)

                n_simulate = st.slider(
                    "Simulate Total Bookings",
                    min_value=0,
                    max_value=recommended_total + 5,
                    value=recommended_total,
                    help="Drag to see risk at different booking levels"
                )

                # Calculate distribution
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

                # Distribution Chart
                x_vals = list(range(len(show_pmf)))

                # Color coding: blue (current), green (safe), red (over capacity)
                colors = []
                for k in x_vals:
                    if k <= n_current:
                        colors.append("#3b82f6")  # Blue - existing
                    elif k <= capacity:
                        colors.append("#10b981")  # Green - safe overbooking
                    else:
                        colors.append("#ef4444")  # Red - risk zone

                fig_dist = go.Figure()

                fig_dist.add_trace(go.Bar(
                    x=x_vals,
                    y=show_pmf.tolist(),
                    marker_color=colors,
                    name="Probability",
                    hovertemplate="Show-ups: %{x}<br>Probability: %{y:.4f}<extra></extra>",
                ))

                # Capacity line
                fig_dist.add_vline(
                    x=capacity + 0.5,
                    line_dash="dash",
                    line_color="#ef4444",
                    line_width=3,
                    annotation_text=f"Capacity: {capacity}",
                    annotation_position="top",
                    annotation_font_size=12,
                    annotation_font_color="#ef4444"
                )

                # Current bookings line
                fig_dist.add_vline(
                    x=n_current + 0.5,
                    line_dash="dot",
                    line_color="#3b82f6",
                    line_width=2,
                    annotation_text=f"Current: {n_current}",
                    annotation_position="bottom",
                    annotation_font_size=11,
                    annotation_font_color="#3b82f6"
                )

                fig_dist.update_layout(
                    title={
                        'text': f"Distribution of Expected Show-ups (μ={mean_su:.1f}, σ={std_su:.2f})",
                        'x': 0.5,
                        'xanchor': 'center',
                        'font': {'size': 16, 'color': '#374151'}
                    },
                    xaxis_title="Number of Guests Showing Up",
                    yaxis_title="Probability Density",
                    height=450,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    xaxis=dict(
                        showgrid=True,
                        gridcolor='#f3f4f6',
                        zeroline=False
                    ),
                    yaxis=dict(
                        showgrid=True,
                        gridcolor='#f3f4f6',
                        zeroline=False
                    ),
                    bargap=0.1,
                    margin=dict(l=60, r=40, t=80, b=60)
                )

                st.plotly_chart(fig_dist, use_container_width=True)

                # Stats row below chart
                stat_cols = st.columns(4)
                stat_cols[0].metric("Simulated Bookings", n_simulate)
                stat_cols[1].metric("Expected Show-ups", f"{mean_su:.1f}")
                stat_cols[2].metric("Std Deviation", f"{std_su:.2f}")
                stat_cols[3].metric("Relocation Risk", f"{reloc_prob:.1%}",
                                   delta=f"{(reloc_prob - max_risk):.1%} vs threshold",
                                   delta_color="inverse")

                # INSIGHTS SECTION
                st.markdown('<div class="section-header">🔍 Insights & Risk Factors</div>', unsafe_allow_html=True)

                insight_cols = st.columns([2, 3])

                # Left: Top Cancellations Table
                with insight_cols[0]:
                    st.markdown("#### Top Risk Bookings")
                    st.caption(f"Highest cancellation probability for {selected_room} on {selected_date}")

                    top_3_result = api_get(
                        TOP_CANCELLATIONS_URL,
                        {
                            "hotel": selected_hotel,
                            "arrival_date": str(selected_date),
                            "room_type": selected_room,
                        },
                        timeout=30,
                        max_retries=2
                    )

                    if "error" not in top_3_result and top_3_result.get("top_3"):
                        top_3_df = pd.DataFrame(top_3_result["top_3"])

                        # Format for display
                        display_df = pd.DataFrame({
                            'Risk': (top_3_df["cancel_prob"] * 100).map("{:.0f}%".format),
                            'Lead Time': top_3_df["lead_time"].map("{} days".format),
                            'ADR': top_3_df["adr"].map("€{:.0f}".format),
                            'Segment': top_3_df["market_segment"]
                        })

                        st.dataframe(
                            display_df,
                            use_container_width=True,
                            hide_index=True,
                            column_config={
                                "Risk": st.column_config.TextColumn("Risk", help="Cancellation probability"),
                                "Lead Time": st.column_config.TextColumn("Lead Time"),
                                "ADR": st.column_config.TextColumn("ADR"),
                                "Segment": st.column_config.TextColumn("Segment")
                            }
                        )
                    else:
                        st.info("No high-risk bookings identified for this selection.")

                # Right: SHAP Explanation
                with insight_cols[1]:
                    st.markdown("#### Key Risk Drivers")
                    st.caption("Features most predictive of cancellation for this segment")

                    cache_key = f"shap_{selected_hotel}_{str(selected_date)}_{selected_room}"

                    if cache_key not in st.session_state:
                        with st.spinner("Loading feature importance..."):
                            shap_result = api_get(
                                EXPLAIN_GLOBAL_URL,
                                {
                                    "selected_date": str(selected_date),
                                    "room_type": selected_room,
                                    "hotel": selected_hotel,
                                },
                                timeout=60,
                                max_retries=2,
                            )
                        st.session_state[cache_key] = shap_result
                    else:
                        shap_result = st.session_state[cache_key]

                    if "error" not in shap_result and shap_result.get("grouped_global_shap"):
                        shap_df = pd.DataFrame(shap_result["grouped_global_shap"])
                        shap_df["mean_abs_shap"] = pd.to_numeric(shap_df["mean_abs_shap"], errors="coerce")

                        # Clean feature names
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
                            .sort_values("mean_abs_shap", ascending=True)
                            .tail(5)
                        )

                        fig_shap = go.Figure(go.Bar(
                            x=top_shap["mean_abs_shap"].tolist(),
                            y=top_shap["feature"].tolist(),
                            orientation="h",
                            marker_color="#667eea"
                        ))

                        fig_shap.update_layout(
                            height=300,
                            showlegend=False,
                            xaxis_title="Impact on Prediction",
                            yaxis_title="",
                            plot_bgcolor='white',
                            paper_bgcolor='white',
                            margin=dict(l=20, r=20, t=20, b=40)
                        )

                        st.plotly_chart(fig_shap, use_container_width=True)
                    else:
                        st.info("Feature importance data not available.")

# ==================================================================
# TAB 2: SINGLE BOOKING ANALYSIS
# ==================================================================
with tab2:
    st.markdown('<div class="section-header">🔍 Individual Booking Risk Assessment</div>', unsafe_allow_html=True)

    st.markdown("""
        <div class="info-card">
            <strong>About this tool:</strong> Select a high-risk booking to analyze individual cancellation
            probability and understand the specific factors driving the risk prediction.
        </div>
    """, unsafe_allow_html=True)

    # Load top bookings
    if "top_bookings_list" not in st.session_state and "results" in st.session_state:
        with st.spinner("Loading high-risk bookings..."):
            top_result = api_get(TOP_BOOKINGS_URL, {}, timeout=60, max_retries=2)
        if "error" not in top_result:
            st.session_state["top_bookings_list"] = top_result.get("top_bookings", [])

    top_bookings_list = st.session_state.get("top_bookings_list", [])

    if not top_bookings_list:
        st.info("Load recommendations in Tab 1 to enable single booking analysis.")
    else:
        # Booking selector
        booking_options = {b["label"]: b for b in top_bookings_list}
        selected_label = st.selectbox(
            "Select Booking to Analyze",
            options=["Choose a booking..."] + list(booking_options.keys()),
            index=0
        )

        if selected_label != "Choose a booking...":
            selected_entry = booking_options[selected_label]

            # Fetch explanation
            cache_key = f"explain_{selected_entry['rank']}"
            if cache_key not in st.session_state:
                with st.spinner("Analyzing booking..."):
                    explain_result = api_post(EXPLAIN_LOCAL_URL, selected_entry["booking"])
                    if "error" not in explain_result:
                        st.session_state[cache_key] = explain_result

            if cache_key in st.session_state:
                explain = st.session_state[cache_key]
                booking = selected_entry["booking"]
                actual = selected_entry["actual_outcome"]

                prob = explain["cancellation_probability"]
                prediction = explain.get("prediction", int(prob >= 0.5))

                # Prediction Cards
                pred_cols = st.columns(3)

                with pred_cols[0]:
                    pred_text = "Will Cancel" if prediction == 1 else "Will Not Cancel"
                    pred_color = "negative" if prediction == 1 else "positive"
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">AI Prediction</div>
                            <div class="metric-value {pred_color}">{pred_text}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with pred_cols[1]:
                    prob_color = "negative" if prob > 0.7 else "warning" if prob > 0.3 else "positive"
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Confidence</div>
                            <div class="metric-value {prob_color}">{prob:.0%}</div>
                        </div>
                    """, unsafe_allow_html=True)

                with pred_cols[2]:
                    actual_text = "Canceled" if actual == 1 else "No Show"
                    actual_color = "negative" if actual == 1 else "positive"
                    st.markdown(f"""
                        <div class="metric-card">
                            <div class="metric-label">Actual Outcome</div>
                            <div class="metric-value {actual_color}">{actual_text}</div>
                        </div>
                    """, unsafe_allow_html=True)

                # Two column layout
                detail_cols = st.columns([1, 1])

                with detail_cols[0]:
                    st.markdown("#### Booking Details")

                    details_df = pd.DataFrame([
                        {"Attribute": k.replace("_", " ").title(), "Value": str(v)}
                        for k, v in booking.items()
                    ])

                    st.dataframe(
                        details_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Attribute": st.column_config.TextColumn("Field", width=150),
                            "Value": st.column_config.TextColumn("Value")
                        }
                    )

                with detail_cols[1]:
                    st.markdown("#### Risk Factor Breakdown")
                    st.caption("Top factors increasing cancellation probability")

                    if "grouped_local_shap" in explain:
                        shap_df = pd.DataFrame(explain["grouped_local_shap"])
                        shap_df["shap_value"] = pd.to_numeric(shap_df["shap_value"], errors="coerce")

                        # Only positive contributions (increasing risk)
                        risk_factors = shap_df[shap_df["shap_value"] > 0].sort_values("shap_value", ascending=False).head(5)

                        if not risk_factors.empty:
                            risk_factors["feature"] = (
                                risk_factors["feature_group"]
                                .astype(str)
                                .str.replace("_", " ", regex=False)
                                .str.title()
                            )

                            fig_local = go.Figure(go.Bar(
                                x=risk_factors["shap_value"].tolist(),
                                y=risk_factors["feature"].tolist(),
                                orientation="h",
                                marker_color="#ef4444"
                            ))

                            fig_local.update_layout(
                                height=350,
                                showlegend=False,
                                xaxis_title="Risk Contribution",
                                yaxis_title="",
                                plot_bgcolor='white',
                                paper_bgcolor='white',
                                margin=dict(l=20, r=20, t=20, b=40)
                            )

                            st.plotly_chart(fig_local, use_container_width=True)
                        else:
                            st.success("No significant risk factors identified — this booking appears stable.")
                    else:
                        st.info("Detailed explanation not available.")

# ==================================================================
# FOOTER
# ==================================================================
st.markdown("---")
st.markdown("""
    <div style="text-align: center; padding: 1rem; color: #9ca3af; font-size: 0.875rem;">
        NoShowShield v2.0 • Powered by Machine Learning •
        <span style="color: #6b7280;">Maximize Revenue, Minimize Risk</span>
    </div>
""", unsafe_allow_html=True)
