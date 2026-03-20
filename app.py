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


# ------------------------------------------------------------------
# API configuration
# ------------------------------------------------------------------
# Define the base URI of the API
#   - Potential sources are in `.streamlit/secrets.toml` or in the Secrets section
#     on Streamlit Cloud
#   - The source selected is based on the shell variable passed when launching streamlit
#     (shortcuts are included in Makefile). By default it takes the cloud API url
if 'API_URI' in os.environ:
    BASE_URI = st.secrets[os.environ.get('API_URI')]
else:
    BASE_URI = st.secrets['cloud_api_uri']
# Add a '/' at the end if it's not there
BASE_URI = BASE_URI if BASE_URI.endswith('/') else BASE_URI + '/'
# Define the url to be used by requests.get to get a prediction
url = BASE_URI + 'optimise'


# ------------------------------------------------------------------
# page config
# ------------------------------------------------------------------
st.set_page_config(page_title="NoShowShield", page_icon="🛡️", layout="wide")
st.title("🛡️ NoShowShield")
st.markdown("**AI-Powered Hotel Revenue Protection Against Cancellations**")
st.markdown(
    "This dashboard predicts which bookings are likely to cancel and "
    "generates smart overbooking recommendations that protect hotel "
    "revenue without putting guests at risk."
)


# ------------------------------------------------------------------
# Request user input (sidebar)
# ------------------------------------------------------------------
st.sidebar.header("⚙️ Optimization Settings")

relocation_cost = st.sidebar.number_input(
    "Relocation cost (€)",
    min_value=0.0,
    max_value=1000.0,
    value=300.0,
    step=50.0,
    help="Cost of relocating a guest to another hotel when overbooked.",
)

max_risk = st.sidebar.slider(
    "Max relocation risk",
    min_value=0.0,
    max_value=0.10,
    value=0.02,
    step=0.01,
    help="Maximum acceptable probability of having to relocate a guest.",
)


# ------------------------------------------------------------------
# Call the API using the user's input
# ------------------------------------------------------------------
@st.cache_data(show_spinner="Fetching predictions from API …")
def load_results(relocation_cost: float, max_risk: float):
    """Call the live FastAPI GET /optimise endpoint on Cloud Run."""
    params = {
        "relocation_cost": relocation_cost,
        "max_risk": max_risk,
    }
    try:
        response = requests.get(url, params=params, timeout=130)

        # Check if we got an ok response (status code 200)
        if response.status_code != 200:
            st.error(
                f"API returned status {response.status_code}: "
                f"{response.text}"
            )
            st.stop()

        # Retrieve the prediction from the JSON
        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "Could not connect to the API. "
            "Check that the Cloud Run service is running."
        )
        st.stop()
    except requests.exceptions.Timeout:
        st.error("API request timed out. Try again in a moment.")
        st.stop()


results = load_results(relocation_cost, max_risk)


# ------------------------------------------------------------------
# Retrieve the results
# ------------------------------------------------------------------
recs = pd.DataFrame(results["recommendations"])
recs["arrival_date"] = pd.to_datetime(recs["arrival_date"])

metrics = results["metrics"]
model_info = results["model_info"]


# ------------------------------------------------------------------
# Sidebar — filters
# ------------------------------------------------------------------
st.sidebar.header("🔍 Filters")

available_dates = sorted(recs["arrival_date"].dt.date.unique())
selected_date = st.sidebar.selectbox("Select date", available_dates)

available_rooms = sorted(recs["assigned_room_type"].unique())
selected_room = st.sidebar.selectbox("Select room type", available_rooms)


# ------------------------------------------------------------------
# Sidebar — model & evaluation metrics
# ------------------------------------------------------------------
st.sidebar.header("📊 Model Info")
st.sidebar.caption(model_info.get("model_type", "XGBoost"))

metrics_df = pd.DataFrame(
    {"Metric": list(metrics.keys()), "Score": list(metrics.values())}
).set_index("Metric")
st.sidebar.table(metrics_df)

st.caption(
    f"Relocation cost = €{relocation_cost:.0f}  ·  "
    f"Max risk = {max_risk * 100:.1f}%  ·  "
    f"Model AUC = {metrics.get('auc', '—')}"
)


# ------------------------------------------------------------------
# Filter recommendations for selected date + room type
# ------------------------------------------------------------------
filtered = recs[
    (recs["arrival_date"].dt.date == selected_date)
    & (recs["assigned_room_type"] == selected_room)
]


# ------------------------------------------------------------------
# Display the prediction
# ------------------------------------------------------------------
st.subheader("Recommendation")

if filtered.empty:
    st.warning("No data available for this selection.")
else:
    row = filtered.iloc[0]

    col1, col2, col3 = st.columns(3)
    col1.metric("🏨 Capacity", int(row["capacity"]))
    col2.metric("📋 Current Bookings", int(row["total_bookings"]))
    col3.metric("👥 Expected Show-ups", round(row["expected_show_ups"], 1))

    st.divider()

    col4, col5, col6 = st.columns(3)
    col4.metric(
        "✅ Recommended Extra Bookings",
        int(row["recommended_extra"]),
    )
    col5.metric(
        "💰 Net Benefit (€)",
        f"€{row['net_benefit']:.2f}",
    )
    col6.metric(
        "⚠️ Relocation Risk",
        f"{row['relocation_probability'] * 100:.2f}%",
    )

    st.divider()

    # ------------------------------------------------------------------
    # Detailed table
    # ------------------------------------------------------------------
    st.subheader("Detailed View")

    display_cols = [
        "arrival_date",
        "assigned_room_type",
        "capacity",
        "total_bookings",
        "expected_show_ups",
        "expected_cancellations",
        "recommended_extra",
        "net_benefit",
        "relocation_probability",
    ]

    nice_df = filtered[display_cols].rename(columns={
        "arrival_date": "Date",
        "assigned_room_type": "Room",
        "capacity": "Capacity",
        "total_bookings": "Bookings",
        "expected_show_ups": "Expected Show-ups",
        "expected_cancellations": "Expected Cancels",
        "recommended_extra": "Recommended Extra",
        "net_benefit": "Net €",
        "relocation_probability": "Relocation Risk",
    })

    st.dataframe(
        nice_df.style.format({
            "Expected Show-ups": "{:.1f}",
            "Expected Cancels": "{:.1f}",
            "Net €": "€{:.2f}",
            "Relocation Risk": "{:.2%}",
        }),
        use_container_width=True,
    )
