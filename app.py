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
GROUP_PROBS_URL = BASE_URI + 'group-probs'


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
# Inline Poisson-Binomial PMF (avoids API round-trip per slider tick)
# ------------------------------------------------------------------
import numpy as np


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
            # Show-up Distribution (Poisson-Binomial)
            # ------------------------------------------------------------------
            st.subheader("Show-up Distribution")
            st.caption(
                f"Poisson-Binomial distribution of expected show-ups for "
                f"**{selected_room}** on **{selected_date}**. "
                f"Slide to see how adding bookings shifts the distribution."
            )

            recommended_total = int(row["recommended_total"])
            n_current = int(row["total_bookings"])
            capacity = int(row["capacity"])

            # Fetch individual cancel probs for this group (cached per selection)
            probs_cache_key = f"gprobs_{selected_hotel}_{selected_date}_{selected_room}"

            if probs_cache_key not in st.session_state:
                # Try the /group-probs endpoint (only works after API redeploy)
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

            # --- Resolve cancel probs: API first, fallback to mean approx ---
            if "error" not in gp_result and "cancel_probs" in gp_result:
                cancel_probs_arr = np.array(gp_result["cancel_probs"], dtype=np.float64)
            else:
                # Fallback: approximate with n identical probs = group mean
                mean_cp = row.get("cancel_prob_mean", row["expected_cancellations"] / row["total_bookings"])
                cancel_probs_arr = np.full(n_current, float(mean_cp))
                st.caption("⚠️ Using mean cancel probability (redeploy API with `/group-probs` for exact per-booking probs)")

            n_simulate = st.slider(
                "Total bookings to simulate",
                min_value=0,
                max_value=recommended_total,
                value=recommended_total,
                help=(
                    "Drag to see how the show-up distribution changes "
                    "as bookings are added. Current bookings are included "
                    "first; extra bookings use the group mean cancel rate."
                ),
            )

            # --- Compute show-up PMF locally (instant) ---
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

            # --- Stats row ---
            dcol1, dcol2, dcol3, dcol4 = st.columns(4)
            dcol1.metric("Bookings Simulated", n_simulate)
            dcol2.metric("Expected Show-ups", f"{mean_su:.1f}")
            dcol3.metric("Std Deviation", f"{std_su:.2f}")
            dcol4.metric("Relocation Risk", f"{reloc_prob * 100:.2f}%")

            # --- Plotly PMF chart ---
            import plotly.graph_objects as go

            x_vals = list(range(len(show_pmf)))

            fig_dist = go.Figure()

            # colour bars: blue ≤ current, green current..capacity, red > capacity
            bar_colors = [
                "#3498db" if k <= n_current
                else "#2ecc71" if k <= capacity
                else "#e74c3c"
                for k in x_vals
            ]

            fig_dist.add_trace(go.Bar(
                x=x_vals,
                y=show_pmf.tolist(),
                marker_color=bar_colors,
                name="P(show-ups = k)",
                hovertemplate="Show-ups: %{x}<br>Probability: %{y:.4f}<extra></extra>",
            ))

            # capacity vertical line
            fig_dist.add_vline(
                x=capacity + 0.5,
                line_dash="dash",
                line_color="#e74c3c",
                line_width=2,
                annotation_text=f"Capacity = {capacity}",
                annotation_position="top left",
                annotation_font_color="#e74c3c",
                annotation_font_size=12,
            )

            fig_dist.update_layout(
                xaxis_title="Number of Show-ups",
                yaxis_title="Probability",
                height=400,
                margin=dict(l=0, r=20, t=30, b=0),
                showlegend=False,
                xaxis=dict(showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="rgba(0,0,0,0.05)"),
                bargap=0.05,
            )

            st.plotly_chart(fig_dist, use_container_width=True)

            # --- Individual show-up probabilities (like the reference image) ---
            if n_simulate > 0:
                show_pcts = (indiv_show * 100).astype(int).tolist()
                current_pcts = show_pcts[:min(n_simulate, n_current)]
                extra_pcts = show_pcts[n_current:] if n_simulate > n_current else []

                display_limit = 15
                badges = ""
                for i, pct in enumerate(show_pcts[:display_limit]):
                    color = "#3498db" if i < n_current else "#2ecc71"
                    badges += (
                        f'<span style="display:inline-block;margin:2px;padding:4px 8px;'
                        f'border-radius:12px;background:{color};color:white;'
                        f'font-size:12px;font-weight:600;">{pct}</span>'
                    )
                remaining = len(show_pcts) - display_limit
                if remaining > 0:
                    badges += f' <span style="font-size:13px;color:#888;">…+{remaining} more</span>'

                st.markdown(
                    f"Individual show-up probabilities % "
                    f"(<span style='color:#3498db'>■</span> Current "
                    f"<span style='color:#2ecc71'>■</span> Extra)",
                    unsafe_allow_html=True,
                )
                st.markdown(badges, unsafe_allow_html=True)

            st.divider()

            # ------------------------------------------------------------------
            # SHAP Explainability + Detailed table — two-column layout
            # ------------------------------------------------------------------
            left_col, right_col = st.columns([3, 2])

            # --- Left: Revenue comparison chart ---
            with left_col:
                import plotly.graph_objects as go

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
                {"Field": list(booking.keys()), "Value": list(booking.values())}
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
