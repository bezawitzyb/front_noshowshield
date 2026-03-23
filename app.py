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
import numpy as np


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
# PMF computation helper (client-side) - FIXED VERSION
# ------------------------------------------------------------------
def compute_pmf_exact(probs):
    """
    Compute exact Poisson-Binomial PMF using dynamic programming.
    Returns: x (list of ints), pmf (list of floats)
    """
    probs = np.asarray(probs, dtype=np.float64)
    n = len(probs)

    if n == 0:
        return [0], [1.0]

    # DP array: pmf[k] = probability of k cancellations
    pmf = np.zeros(n + 1)
    pmf[0] = 1.0

    for p in probs:
        new_pmf = np.zeros(n + 1)
        for k in range(n + 1):
            if k == 0:
                new_pmf[k] = pmf[k] * (1 - p)
            else:
                new_pmf[k] = pmf[k-1] * p + pmf[k] * (1 - p)
        pmf = new_pmf

    x = list(range(n + 1))
    pmf = pmf.tolist()

    return x, pmf


def compute_pmf_normal(probs):
    """
    Normal approximation for large n.
    """
    probs = np.asarray(probs, dtype=np.float64)
    n = len(probs)
    mean = np.sum(probs)
    variance = np.sum(probs * (1 - probs))

    if variance < 1e-6:
        x = [int(round(mean))]
        pmf = [1.0]
        return x, pmf

    std = np.sqrt(variance)

    # Create range covering mean ± 4 std devs
    x_min = max(0, int(mean - 4*std))
    x_max = min(n, int(mean + 4*std))

    # Ensure reasonable range
    if x_max <= x_min:
        x_max = x_min + 1

    x = list(range(x_min, x_max + 1))

    # Normal PDF
    pmf = [(1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((xi - mean) / std) ** 2) for xi in x]

    # Normalize
    total = sum(pmf)
    pmf = [p / total for p in pmf]

    return x, pmf


def compute_pmf_fast(cancel_probs):
    """
    Compute PMF using exact method for small n, Normal approximation for large n.
    Returns Python lists (not numpy arrays) for Plotly compatibility.
    """
    try:
        # Ensure numpy array and clean data
        probs = np.asarray(cancel_probs, dtype=np.float64)

        # Handle NaN/inf and clip to valid range
        probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
        probs = np.clip(probs, 0.0, 1.0)

        n = len(probs)

        if n == 0:
            return [0], [1.0]
        elif n <= 80:
            return compute_pmf_exact(probs)
        else:
            return compute_pmf_normal(probs)

    except Exception as e:
        st.error(f"PMF computation error: {e}")
        # Fallback to uniform
        n = len(cancel_probs) if cancel_probs is not None else 1
        return list(range(n + 1)), [1.0/(n+1)] * (n + 1)


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
# TAB 1 — Overbooking Recommendations
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
            # Revenue Comparison + SHAP — two-column layout
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
            # Cancellation Distribution Charts — Two interactive charts with sliders
            # ------------------------------------------------------------------
            st.divider()
            st.subheader("Cancellation Risk Analysis")

            # Validate that we have individual_probs
            if "individual_probs" not in row or row["individual_probs"] is None:
                st.error("No individual probabilities found in the data.")
            else:
                try:
                    # Convert to numpy array and ensure valid values
                    cancel_probs_raw = np.array(row["individual_probs"])

                    if len(cancel_probs_raw) == 0:
                        st.warning("No cancellation probabilities available.")
                    else:
                        # Clean data - ensure they're proper probabilities between 0 and 1
                        cancel_probs_raw = np.clip(cancel_probs_raw, 0.0, 1.0)

                        mean_cancel = float(np.mean(cancel_probs_raw))
                        recommended_extra = int(row["recommended_extra"])
                        recommended_total = int(row["total_bookings"] + recommended_extra)
                        capacity = int(row["capacity"])
                        min_needed = max(0, recommended_total - capacity)

                        # Prepare extended probabilities for recommended scenario
                        if recommended_extra > 0:
                            extended_probs = np.concatenate([
                                cancel_probs_raw,
                                np.full(recommended_extra, mean_cancel)
                            ])
                        else:
                            extended_probs = cancel_probs_raw

                        dist_col1, dist_col2 = st.columns(2)

                        # --- LEFT CHART: Current Bookings ---
                        with dist_col1:
                            st.markdown("**Current Bookings**")

                            n_current_max = len(cancel_probs_raw)

                            n_current = st.slider(
                                "Bookings processed",
                                min_value=1,
                                max_value=int(n_current_max),
                                value=int(n_current_max),
                                key="current_slider"
                            )

                            # Subset probabilities based on slider
                            current_probs = cancel_probs_raw[:n_current]

                            # Metrics
                            outcomes = int(n_current + 1)
                            expected = float(np.sum(current_probs))
                            std_dev = float(np.sqrt(np.sum(current_probs * (1 - current_probs))))

                            m1, m2, m3 = st.columns(3)
                            m1.metric("Possible outcomes", outcomes)
                            m2.metric("Expected cancellations", f"{expected:.1f}")
                            m3.metric("Std deviation", f"{std_dev:.2f}")

                            # Compute distribution - returns Python lists
                            x_cur, pmf_cur = compute_pmf_fast(current_probs)

                            # DEBUG: Show data being passed to Plotly
                            # st.write(f"Debug: x_cur={x_cur[:5]}..., pmf_cur={pmf_cur[:5]}..., sum={sum(pmf_cur):.4f}")

                            if len(x_cur) > 0 and len(pmf_cur) > 0 and sum(pmf_cur) > 0:
                                # Create bar chart with explicit data
                                fig_cur = go.Figure(data=[
                                    go.Bar(
                                        x=x_cur,
                                        y=pmf_cur,
                                        marker_color="#6366f1",
                                        hovertemplate="Cancellations: %{x}<br>Probability: %{y:.4f}<extra></extra>"
                                    )
                                ])

                                # Set layout with proper ranges
                                y_max = max(pmf_cur) * 1.2 if pmf_cur else 0.1

                                fig_cur.update_layout(
                                    height=300,
                                    margin=dict(l=0, r=0, t=20, b=0),
                                    xaxis=dict(
                                        title="Number of Cancellations",
                                        tickmode='linear',
                                        tick0=0,
                                        dtick=max(1, n_current // 5),
                                        range=[-0.5, n_current + 0.5]
                                    ),
                                    yaxis=dict(
                                        title="Probability",
                                        range=[0, y_max]
                                    ),
                                    showlegend=False,
                                    template="plotly_white",
                                    bargap=0.2
                                )

                                st.plotly_chart(fig_cur, use_container_width=True)
                            else:
                                st.error("Failed to compute distribution")

                            # Individual probability pills
                            st.caption("Individual booking cancel probabilities:")
                            if len(current_probs) > 0:
                                prob_vals = [int(float(p)*100) for p in current_probs[:15]]
                                pills_html = "".join([
                                    f'<span style="background-color: #e0e7ff; color: #4338ca; padding: 4px 10px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 13px; font-weight: 600; border: 1px solid #c7d2fe;">{p}</span>'
                                    for p in prob_vals
                                ])
                                if len(current_probs) > 15:
                                    pills_html += f'<span style="padding: 4px;">...+{len(current_probs)-15} more</span>'
                                st.markdown(pills_html, unsafe_allow_html=True)

                        # --- RIGHT CHART: Recommended Bookings ---
                        with dist_col2:
                            st.markdown("**Recommended Bookings** (Current + Extras)")

                            n_rec_max = len(extended_probs)

                            n_rec = st.slider(
                                "Bookings processed",
                                min_value=1,
                                max_value=int(n_rec_max),
                                value=int(n_rec_max),
                                key="rec_slider"
                            )

                            # Subset probabilities based on slider
                            rec_probs = extended_probs[:n_rec]

                            # Metrics
                            outcomes_rec = int(n_rec + 1)
                            expected_rec = float(np.sum(rec_probs))
                            std_dev_rec = float(np.sqrt(np.sum(rec_probs * (1 - rec_probs))))

                            m1, m2, m3 = st.columns(3)
                            m1.metric("Possible outcomes", outcomes_rec)
                            m2.metric("Expected cancellations", f"{expected_rec:.1f}")
                            m3.metric("Std deviation", f"{std_dev_rec:.2f}")

                            # Compute distribution
                            x_rec, pmf_rec = compute_pmf_fast(rec_probs)

                            if len(x_rec) > 0 and len(pmf_rec) > 0 and sum(pmf_rec) > 0:
                                # Determine color based on whether showing extras
                                is_all_current = n_rec <= len(cancel_probs_raw)
                                bar_color = "#6366f1" if is_all_current else "#f97316"

                                fig_rec = go.Figure(data=[
                                    go.Bar(
                                        x=x_rec,
                                        y=pmf_rec,
                                        marker_color=bar_color,
                                        hovertemplate="Cancellations: %{x}<br>Probability: %{y:.4f}<extra></extra>"
                                    )
                                ])

                                # Add threshold line if needed
                                shapes = []
                                annotations = []
                                if min_needed > 0:
                                    shapes.append(dict(
                                        type='line',
                                        x0=min_needed - 0.5,
                                        x1=min_needed - 0.5,
                                        y0=0,
                                        y1=max(pmf_rec) * 1.1,
                                        line=dict(color='red', width=2, dash='dash')
                                    ))
                                    annotations.append(dict(
                                        x=min_needed,
                                        y=max(pmf_rec) * 1.05,
                                        text=f"Need ≥{min_needed}",
                                        showarrow=False,
                                        font=dict(color='red', size=12)
                                    ))

                                y_max_rec = max(pmf_rec) * 1.2 if pmf_rec else 0.1

                                fig_rec.update_layout(
                                    height=300,
                                    margin=dict(l=0, r=0, t=20, b=0),
                                    xaxis=dict(
                                        title="Number of Cancellations",
                                        tickmode='linear',
                                        tick0=0,
                                        dtick=max(1, n_rec // 5),
                                        range=[-0.5, max(n_rec, min_needed + 5) + 0.5]
                                    ),
                                    yaxis=dict(
                                        title="Probability",
                                        range=[0, y_max_rec]
                                    ),
                                    showlegend=False,
                                    template="plotly_white",
                                    bargap=0.2,
                                    shapes=shapes,
                                    annotations=annotations
                                )

                                st.plotly_chart(fig_rec, use_container_width=True)

                                # Safety buffer calculation
                                if min_needed > 0 and len(x_rec) > 0:
                                    prob_safe = sum([p for x, p in zip(x_rec, pmf_rec) if x >= min_needed])
                                    st.markdown(f"**Safety buffer:** {prob_safe*100:.1f}% chance of ≥{min_needed} cancellations")
                            else:
                                st.error("Failed to compute distribution")

                            # Probability pills
                            st.caption("Individual cancel probabilities (Blue=Current, Orange=Extra):")
                            if len(rec_probs) > 0:
                                pills_rec = []
                                for i, p in enumerate(rec_probs[:15]):
                                    p_int = int(float(p)*100)
                                    if i < len(cancel_probs_raw):
                                        style = 'background-color: #e0e7ff; color: #4338ca; border: 1px solid #c7d2fe;'
                                    else:
                                        style = 'background-color: #ffedd5; color: #9a3412; border: 1px solid #fed7aa;'

                                    pills_rec.append(
                                        f'<span style="{style} padding: 4px 10px; border-radius: 12px; margin: 2px; display: inline-block; font-size: 13px; font-weight: 600;">{p_int}</span>'
                                    )

                                if len(rec_probs) > 15:
                                    pills_rec.append(f'<span style="padding: 4px;">...+{len(rec_probs)-15} more</span>')

                                st.markdown("".join(pills_rec), unsafe_allow_html=True)

                except Exception as e:
                    st.error(f"Error rendering charts: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())


# ==================================================================
# TAB 2 — Single Booking Prediction
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
