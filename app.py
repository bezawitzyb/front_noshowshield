import os
import streamlit as st
import pandas as pd
import requests
import numpy as np
import plotly.graph_objects as go

# ==========================================================
# DESIGN SYSTEM
# ==========================================================
COLORS = {
    "primary": "#5078AA",
    "primary_hover": "#3F6291",
    "secondary": "#5EB0AD",
    "accent": "#6C9BBD",
    "background": "#D7D5D0",
    "surface": "#A2B5C8",
    "surface_light": "#EEF1F4",
    "text": "#2F3A44",
    "text_secondary": "#5A6B7C",
    "text_muted": "#A0A0B2",
    "border": "#99A09A",
    "border_light": "#C5CCD3",
    "success": "#5EB0AD",
    "warning": "#DCC88C",
    "danger": "#E07A7A",
}

st.set_page_config(page_title="NoShowShield", layout="wide")

# ==========================================================
# GLOBAL STYLE
# ==========================================================
st.markdown(f"""
<style>
.stApp {{
    background-color: {COLORS['background']};
    color: {COLORS['text']};
}}
.stButton>button {{
    background-color: {COLORS['primary']};
    color: white;
    border-radius: 10px;
}}
</style>
""", unsafe_allow_html=True)

# ==========================================================
# COMPONENTS
# ==========================================================
def metric_card(title, value, subtitle="", bg="#fff", color="#000"):
    st.markdown(f"""
    <div style="
        background:{bg};
        padding:16px;
        border-radius:16px;
        box-shadow:0 2px 6px rgba(0,0,0,0.05);
    ">
        <div style="font-size:12px;color:{COLORS['text_secondary']};">{title}</div>
        <div style="font-size:28px;font-weight:700;color:{color};">{value}</div>
        <div style="font-size:12px;color:{COLORS['text_muted']};">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def section(title):
    st.markdown(f"### {title}")

# ==========================================================
# API CONFIG
# ==========================================================
BASE_URI = st.secrets['cloud_api_uri']
OPTIMISE_URL = BASE_URI + "optimise"
TOP_BOOKINGS_URL = BASE_URI + "top-bookings"
EXPLAIN_LOCAL_URL = BASE_URI + "explain/local"
GROUP_PROBS_URL = BASE_URI + "group-probs"
TOP_CANCELLATIONS_URL = BASE_URI + "top-cancellations"

def api_get(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=120)
        return r.json()
    except:
        return {"error": "API error"}

def api_post(url, payload=None):
    try:
        r = requests.post(url, json=payload, timeout=120)
        return r.json()
    except:
        return {"error": "API error"}

# ==========================================================
# SIDEBAR NAVIGATION
# ==========================================================
st.sidebar.title("🛡️ NoShowShield")
page = st.sidebar.radio("Navigation", ["📊 Recommendations", "🔍 Single Booking"])
st.sidebar.divider()

relocation_cost = st.sidebar.number_input("Relocation cost (€)", value=150.0)
max_risk = st.sidebar.slider("Max relocation risk", 0.0, 0.1, 0.05)

if st.sidebar.button("Run Optimization"):
    res = api_get(OPTIMISE_URL, {
        "relocation_cost": relocation_cost,
        "max_risk": max_risk
    })
    st.session_state["data"] = res

# ==========================================================
# LOAD DATA
# ==========================================================
if "data" not in st.session_state:
    st.info("Run optimization to begin")
    st.stop()

data = st.session_state["data"]
df = pd.DataFrame(data["recommendations"])
row = df.iloc[0]

# ==========================================================
# DASHBOARD 1 — RECOMMENDATIONS
# ==========================================================
if page == "📊 Recommendations":

    st.title("Revenue Optimization Dashboard")

    occupancy = row["total_bookings"] / row["capacity"]
    revenue = row["total_bookings"] * row["mean_adr"]

    # KPI Row 1
    c1, c2, c3, c4 = st.columns(4)
    with c1: metric_card("Capacity", row["capacity"], bg=COLORS["surface"])
    with c2: metric_card("Bookings", row["total_bookings"], f"{occupancy*100:.1f}% occ", bg=COLORS["surface"])
    with c3: metric_card("Expected Show-ups", f"{row['expected_show_ups']:.1f}", bg="#CFE5DD")
    with c4: metric_card("Extra Bookings", f"+{row['recommended_extra']}", bg="#D6E2F0")

    # KPI Row 2
    c5, c6, c7, c8 = st.columns(4)
    with c5: metric_card("Revenue", f"€{revenue:,.0f}", bg="#E8D8C3")
    with c6: metric_card("Risk", f"{row['relocation_probability']*100:.2f}%", bg=COLORS["surface"])
    with c7: metric_card("Relocation Cost", f"€{row['expected_relocation_cost']:.0f}", bg=COLORS["danger"], color="white")
    with c8: metric_card("Net Benefit", f"€{row['net_benefit']:,.0f}", bg="#CFE5DD")

    # Show-up Distribution Chart
    section("Show-up Distribution")
    n = int(row["recommended_total"])
    probs = np.full(n, 0.8)
    pmf = np.random.dirichlet(np.ones(n))
    x = list(range(n))
    colors = [
        COLORS["accent"] if k < row["total_bookings"]
        else COLORS["secondary"] if k < row["capacity"]
        else COLORS["danger"]
        for k in x
    ]
    fig = go.Figure()
    fig.add_bar(x=x, y=pmf, marker_color=colors)
    fig.add_vline(x=row["capacity"], line_color=COLORS["danger"], line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# DASHBOARD 2 — SINGLE BOOKING
# ==========================================================
elif page == "🔍 Single Booking":

    st.title("Single Booking Prediction")

    if "top_bookings_list" not in st.session_state:
        with st.spinner("Loading top high-risk bookings …"):
            top_result = api_get(TOP_BOOKINGS_URL)
        if "error" in top_result:
            st.error(top_result["error"])
            st.session_state["top_bookings_list"] = []
        else:
            st.session_state["top_bookings_list"] = top_result.get("top_bookings", [])

    top_bookings = st.session_state.get("top_bookings_list", [])
    placeholder = "Select a high-risk booking"
    dropdown_options = [placeholder] + [b["label"] for b in top_bookings]

    selected_label = st.selectbox("Select a booking", dropdown_options, index=0, disabled=not top_bookings)

    if selected_label != placeholder and top_bookings:
        selected_entry = next(b for b in top_bookings if b["label"] == selected_label)

        # Fetch prediction & SHAP
        cache_key = f"explain_{selected_entry['rank']}"
        if cache_key not in st.session_state:
            with st.spinner("Fetching prediction and SHAP …"):
                explain_result = api_post(EXPLAIN_LOCAL_URL, selected_entry["booking"])
            st.session_state[cache_key] = explain_result
        else:
            explain_result = st.session_state[cache_key]

        prob = explain_result.get("cancellation_probability", 0.0)
        prediction = explain_result.get("prediction", int(prob >= 0.5))

        # Metric cards
        c1, c2 = st.columns(2)
        with c1:
            metric_card("Prediction", "Will Cancel" if prediction else "Will Show",
                        bg=COLORS["danger"] if prediction else COLORS["secondary"], color="white")
        with c2:
            metric_card("Cancellation Probability", f"{prob*100:.1f}%", bg=COLORS["surface"])

        # Booking details
        section("Booking Details")
        details = pd.DataFrame({
            "Field": list(selected_entry["booking"].keys()),
            "Value": [str(v) for v in selected_entry["booking"].values()]
        }).set_index("Field")
        st.dataframe(details, use_container_width=True)

        # SHAP chart
        section("Top Risk Factors (SHAP)")
        shap_df = pd.DataFrame(explain_result.get("grouped_local_shap", []))
        if not shap_df.empty:
            shap_df["shap_value"] = pd.to_numeric(shap_df["shap_value"], errors="coerce")
            shap_df["feature"] = shap_df["feature_group"].astype(str).str.replace("_"," ").str.title()
            top_shap = shap_df[shap_df["shap_value"] > 0].nlargest(5, "shap_value").iloc[::-1]
            fig = go.Figure(go.Bar(
                x=top_shap["shap_value"].tolist(),
                y=top_shap["feature"].tolist(),
                orientation="h",
                marker_color=COLORS["primary"]
            ))
            fig.update_layout(height=400, margin=dict(l=20,r=20,t=20,b=20),
                              xaxis_title="SHAP Value", yaxis_title="", showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No SHAP risk factors found for this booking.")
