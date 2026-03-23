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
CANCELLATION_DIST_URL = BASE_URI + 'optimise/cancellation-distribution'


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
def api_get(url: str, params: dict, timeout: int = 120, max_retries: int = 3):
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

            # --- Left: Revenue comparison chart ---
            with left_col:
                st.subheader("Revenue Comparison")
                st.caption(
                    f"Expected revenue for **{selected_room}** on **{selected_date}**"
                )

                cancel_rate  = row["expected_cancellations"] / row["total_bookings"] if row["total_bookings"] > 0 else 0
                extra_shows  = row["recommended_extra"] * (1 - cancel_rate)
                mean_adr     = row["mean_adr"]

                rev_without = row["expected_show_ups"] * mean_adr
                rev_with    = rev_without + row["net_benefit"]

                fig_rev = go.Figure()

                fig_rev.add_trace(go.Bar(
                    name="Without Overbooking",
                    x=["Without Overbooking", "With Overbooking"],
                    y=[rev_without, rev_with],
                    marker_color=["#2ecc71", "#1abc9c"],
                    text=[f"€{rev_without:,.0f}", f"€{rev_with:,.0f}"],
                    textposition="outside",
                    cliponaxis=False,
                    showlegend=False,
                ))

                # Arrow annotation: bottom of right bar top → net benefit label
                fig_rev.add_annotation(
                    x="With Overbooking",
                    y=rev_with,
                    ax="Without Overbooking",
                    ay=rev_without,
                    axref="x",
                    ayref="y",
                    text=f"<b>+€{row['net_benefit']:,.0f}</b>",
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=1.2,
                    arrowwidth=2,
                    arrowcolor="#f39c12",
                    font=dict(size=14, color="#f39c12"),
                    xanchor="left",
                    yanchor="middle",
                )

                fig_rev.update_layout(
                    height=380,
                    margin=dict(l=0, r=20, t=40, b=0),
                    showlegend=False,
                    yaxis=dict(title="Revenue (€)", showgrid=False, tickprefix="€", tickformat=","),
                    xaxis=dict(showgrid=False),
                    bargap=0.4,
                )

                st.plotly_chart(fig_rev, use_container_width=True)

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
                        margin=dict(l=0, r=0, t=10, b=0),
                        showlegend=False,
                        coloraxis_showscale=False,
                        yaxis=dict(tickfont=dict(size=12)),
                    )

                    st.plotly_chart(fig, use_container_width=True)

                else:
                    st.info("No SHAP data available for this date and room type.")

            # ------------------------------------------------------------------
            # Cancellation Distribution Chart — full width below
            # ------------------------------------------------------------------
            st.divider()
            st.subheader("Cancellation Risk Distribution")
            st.caption(
                f"Probability distribution of cancellation counts for **{selected_room}** on **{selected_date}**"
            )

            # Cache key for distribution data
            dist_cache_key = f"dist_{selected_hotel}_{selected_date_str}_{selected_room}"

            if dist_cache_key not in st.session_state:
                with st.spinner("Loading cancellation distribution …"):
                    dist_result = api_get(
                        CANCELLATION_DIST_URL,
                        {
                            "selected_date": selected_date_str,
                            "room_type": selected_room,
                            "hotel": selected_hotel,
                            "relocation_cost": st.session_state.get("relocation_cost", relocation_cost),
                            "max_risk": st.session_state.get("max_risk", max_risk),
                        },
                        timeout=60,
                        max_retries=2,
                    )
                st.session_state[dist_cache_key] = dist_result
            else:
                dist_result = st.session_state[dist_cache_key]

            if "error" in dist_result:
                st.warning(f"Could not load distribution data: {dist_result['error']}")
            else:
                dist_data = dist_result["distribution"]

                fig_dist = go.Figure()

                # Current state PMF
                fig_dist.add_trace(go.Bar(
                    x=dist_data["current"]["x"],
                    y=dist_data["current"]["pmf"],
                    name=f"Current ({dist_data['current']['n_bookings']} bookings)",
                    marker_color="rgba(56, 189, 248, 0.7)",
                    hovertemplate="Cancellations: %{x}<br>Probability: %{y:.3f}<extra></extra>"
                ))

                # Recommended state PMF
                fig_dist.add_trace(go.Bar(
                    x=dist_data["recommended"]["x"],
                    y=dist_data["recommended"]["pmf"],
                    name=f"Recommended ({dist_data['recommended']['n_bookings']} bookings)",
                    marker_color="rgba(251, 146, 60, 0.7)",
                    hovertemplate="Cancellations: %{x}<br>Probability: %{y:.3f}<extra></extra>"
                ))

                # Vertical line: minimum cancellations needed to avoid relocation
                min_needed = dist_data["min_cancellations_needed"]
                if min_needed > 0:
                    fig_dist.add_vline(
                        x=min_needed - 0.5,
                        line_dash="dash",
                        line_color="red",
                        annotation_text=f"Need ≥{min_needed} cancellations to avoid walks",
                        annotation_position="top right"
                    )

                fig_dist.update_layout(
                    title="Current vs Recommended Booking State",
                    xaxis_title="Number of Cancellations",
                    yaxis_title="Probability",
                    barmode="overlay",
                    bargap=0.1,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                    template="plotly_dark",
                    height=400,
                )

                st.plotly_chart(fig_dist, use_container_width=True)

                # Explanation text
                if min_needed > 0:
                    st.markdown(
                        f"**How to read this:** The blue bars show the current cancellation probability distribution. "
                        f"The orange bars show the distribution if you accept **{int(row['recommended_extra'])}** extra bookings. "
                        f"The red line indicates the minimum cancellations needed ({min_needed}) to avoid overbooking incidents. "
                        f"The area under the orange curve to the right of this line represents your safety buffer."
                    )


# ==================================================================
# TAB 2 — Alex CODE (single booking prediction)
# ==================================================================

# Additional URL constants used only by tab 2
RANDOM_BOOKING_URL = BASE_URI + 'random-booking'
EXPLAIN_LOCAL_URL  = BASE_URI + 'explain/local'


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
        "Pick a random booking from the test set to see the model's "
        "cancellation prediction and the SHAP values explaining it."
    )

    if st.button("🎲 Pick Random Booking", type="primary"):
        with st.spinner("Fetching a random booking …"):
            booking_result = api_get(RANDOM_BOOKING_URL, {}, timeout=30, max_retries=2)
        if "error" in booking_result:
            st.error(booking_result["error"])
        else:
            with st.spinner("Running prediction and SHAP explanation …"):
                explain_result = api_post(EXPLAIN_LOCAL_URL, booking_result["booking"])
            if "error" in explain_result:
                st.error(explain_result["error"])
            else:
                st.session_state["single_booking"] = booking_result["booking"]
                st.session_state["single_actual"]  = booking_result["actual_outcome"]
                st.session_state["single_explain"] = explain_result

    if "single_booking" not in st.session_state:
        st.info("Click **Pick Random Booking** to load a booking from the test set.")
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
