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
import plotly.express as px


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


# ------------------------------------------------------------------
# page config
# ------------------------------------------------------------------
st.set_page_config(page_title="NoShowShield", page_icon="🛡️", layout="wide")
st.title("🛡️ NoShowShield")
st.markdown("**AI-Powered Hotel Revenue Protection Against Cancellations**")
st.markdown(
    "NoShowShield uses machine learning to predict booking cancellations "
    "and recommend optimal overbooking levels: maximising hotel revenue "
    "while keeping guest relocation risk below a configurable threshold. "
    "Select a date and room type to see actionable recommendations backed "
    "by SHAP explainability."
)


# ------------------------------------------------------------------
# Request user input (sidebar)
# ------------------------------------------------------------------
st.sidebar.header("Optimization Settings")

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
# API helpers
# ------------------------------------------------------------------
def api_get(url: str, params: dict, timeout: int = 180, max_retries: int = 3):
    """
    GET request with retries for Cloud Run cold starts.
    Returns the JSON response or a dict with an 'error' key.
    """
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


# ------------------------------------------------------------------
# Load optimisation data — triggered by button
# ------------------------------------------------------------------
if st.sidebar.button("Get Recommendations", type="primary", use_container_width=True):
    with st.spinner("Fetching predictions from API … (first load may take up to 2 min while the API wakes up)"):
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
tab1, tab2 = st.tabs(["Overbooking Recommendations", "Single Booking Prediction"])


# ==================================================================
# TAB 1 — Bezas CODE
# Note: st.stop() replaced with if/else so tab 2 can render.
# Only other change: nlargest(10) → nlargest(5) per product request.
# ==================================================================
with tab1:
    # ------------------------------------------------------------------
    # Display results (only if loaded)
    # ------------------------------------------------------------------
    if "results" not in st.session_state:
        st.info("Adjust settings in the sidebar and click **Get Recommendations** to start.")
    else:
        results = st.session_state["results"]

        # ------------------------------------------------------------------
        # Parse API response
        # ------------------------------------------------------------------
        recs = pd.DataFrame(results["recommendations"])
        recs["arrival_date"] = pd.to_datetime(recs["arrival_date"])

        metrics = results["metrics"]
        model_info = results["model_info"]


        # ------------------------------------------------------------------
        # Sidebar — filters
        # ------------------------------------------------------------------
        st.sidebar.header("Filters")

        available_hotels = sorted(recs["hotel"].unique())
        selected_hotel = st.sidebar.selectbox("Select hotel", available_hotels)

        available_dates = sorted(
            recs[recs["hotel"] == selected_hotel]["arrival_date"].dt.date.unique()
        )
        selected_date = st.sidebar.selectbox("Select date", available_dates)

        available_rooms = sorted(
            recs[
                (recs["hotel"] == selected_hotel)
                & (recs["arrival_date"].dt.date == selected_date)
            ]["assigned_room_type"].unique()
        )
        selected_room = st.sidebar.selectbox("Select room type", available_rooms)


        # ------------------------------------------------------------------
        # Sidebar — model & evaluation metrics
        # ------------------------------------------------------------------
        st.sidebar.header("Model Info")
        st.sidebar.caption(model_info.get("model_type", "XGBoost"))

        metrics_df = pd.DataFrame(
            {"Metric": list(metrics.keys()), "Score": list(metrics.values())}
        ).set_index("Metric")
        st.sidebar.table(metrics_df)

        st.caption(
            f"Relocation cost = €{st.session_state.get('relocation_cost', relocation_cost):.0f}  ·  "
            f"Max risk = {st.session_state.get('max_risk', max_risk) * 100:.1f}%  ·  "
            f"Model AUC = {metrics.get('auc', '—')}"
        )


        # ------------------------------------------------------------------
        # Filter recommendations for selected hotel + date + room type
        # ------------------------------------------------------------------
        filtered = recs[
            (recs["hotel"] == selected_hotel)
            & (recs["arrival_date"].dt.date == selected_date)
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

            # --- Top metrics row ---
            col1, col2, col3 = st.columns(3)
            col1.metric("Capacity", int(row["capacity"]))
            col2.metric("Current Bookings", int(row["total_bookings"]))
            col3.metric("Expected Show-ups", round(row["expected_show_ups"], 1))

            st.divider()

            col4, col5, col6 = st.columns(3)
            col4.metric(
                "Recommended Extra Bookings",
                int(row["recommended_extra"]),
            )
            col5.metric(
                "Net Benefit (€)",
                f"€{row['net_benefit']:.2f}",
            )
            col6.metric(
                "Relocation Risk",
                f"{row['relocation_probability'] * 100:.2f}%",
            )

            st.divider()

            # ------------------------------------------------------------------
            # SHAP Explainability + Detailed table — two-column layout
            # ------------------------------------------------------------------
            left_col, right_col = st.columns([3, 2])

            # --- Left: Top 3 likely cancellations ---
            with left_col:
                st.subheader("Top 3 Likely Cancellations")
                st.caption(
                    f"Bookings for **{selected_room}** on **{selected_date}** with highest risk"
                )

                # Fetch top 3 from API
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

                if "error" in top_3_result:
                    st.warning(f"Could not load top cancellations: {top_3_result['error']}")
                else:
                    top_3_data = top_3_result.get("top_3", [])
                    if not top_3_data:
                        st.info("No booking data available.")
                    else:
                        top_3_df = pd.DataFrame(top_3_data)

                        # Formatting for display
                        top_3_df["cancel_prob"] = (top_3_df["cancel_prob"] * 100).map("{:.1f}%".format)
                        top_3_df["adr"] = top_3_df["adr"].map("€{:.2f}".format)

                        # Rename columns for presentation
                        col_mapping = {
                            "lead_time": "Lead Time (days)",
                            "adr": "ADR",
                            "market_segment": "Market Segment",
                            "deposit_type": "Deposit",
                            "customer_type": "Customer",
                            "cancel_prob": "Cancel Risk"
                        }
                        top_3_df = top_3_df.rename(columns=col_mapping)

                        # Reorder to match mapping order
                        display_cols = [v for k, v in col_mapping.items() if v in top_3_df.columns]
                        st.table(top_3_df[display_cols])

            # --- Right: SHAP chart ---
            with right_col:
                st.subheader("SHAP — Top Risk Factors")
                st.caption(
                    f"Why bookings on **{selected_date}** for room type **{selected_room}** "
                    f"are likely to cancel"
                )

                # Cache key includes hotel, date and room type
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
                    shap_df = pd.DataFrame(shap_result["grouped_global_shap"])

                    # Clean up feature names for display
                    shap_df["feature"] = (
                        shap_df["feature_group"]
                        .str.replace("cat_ordinal__", "", regex=False)
                        .str.replace("_", " ", regex=False)
                        .str.title()
                    )

                    # Top 5 features, sorted ascending for horizontal bar
                    top_shap = (
                        shap_df
                        .nlargest(5, "mean_abs_shap")
                        .sort_values("mean_abs_shap")
                    )

                    fig = px.bar(
                        top_shap,
                        x="mean_abs_shap",
                        y="feature",
                        orientation="h",
                        labels={
                            "mean_abs_shap": "Mean |SHAP Value|",
                            "feature": "",
                        },
                        color="mean_abs_shap",
                        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                    )
                    fig.update_layout(
                        height=400,
                        margin=dict(l=0, r=0, t=10, b=40),
                        showlegend=False,
                        coloraxis_showscale=False,
                        yaxis=dict(tickfont=dict(size=12)),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No SHAP data available for this date and room type.")


# ==================================================================
# TAB 2 — Alex CODE (single booking prediction)
# ==================================================================

# Additional URL constants used only by tab 2
TOP_BOOKINGS_URL  = BASE_URI + 'top-bookings'
EXPLAIN_LOCAL_URL = BASE_URI + 'explain/local'


def api_post(url: str, payload: dict, timeout: int = 60, max_retries: int = 2):
    """POST request with retries. Returns JSON or a dict with an 'error' key."""
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


with tab2:
    st.subheader("Single Booking Prediction")
    st.markdown(
        "Select one of the top 3 bookings with the highest predicted cancellation "
        "risk to see the model's prediction and the SHAP values explaining it."
    )

    # Load top bookings once per session, but only after the main results are ready
    # (avoids a double API call when the user first clicks "Get Recommendations")
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

        # Fetch explanation when selection changes
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
        booking = st.session_state["single_booking"]
        actual  = st.session_state["single_actual"]
        explain = st.session_state["single_explain"]

        prob       = explain["cancellation_probability"]
        prediction = explain.get("prediction", int(prob >= 0.5))

        # --- Top metrics row ---
        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction", "Will Cancel" if prediction == 1 else "Won't Cancel")
        col2.metric("Cancellation Probability", f"{prob * 100:.1f}%")
        col3.metric("Actual Outcome", "Canceled" if actual == 1 else "Not Canceled")

        st.divider()

        # --- Two-column layout: booking details | SHAP chart ---
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

            # Keep only positive SHAP values (reasons to cancel), top 5
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
                st.info("No cancellation risk factors found for this booking.")
            else:
                fig = px.bar(
                    top_shap,
                    x="shap_value",
                    y="feature",
                    orientation="h",
                    labels={"shap_value": "SHAP Value", "feature": ""},
                    color="shap_value",
                    color_continuous_scale=["#f39c12", "#e74c3c"],
                )
                fig.update_layout(
                    height=400,
                    margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False,
                    coloraxis_showscale=False,
                    yaxis=dict(tickfont=dict(size=12)),
                )
                st.plotly_chart(fig, use_container_width=True)
