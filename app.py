"""
NoShowShield — Streamlit dashboard for overbooking recommendations
Connects to the live FastAPI on Google Cloud Run.

Run with:
    streamlit run app.py
"""

import os
import math
import numpy as np
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

OPTIMISE_URL       = BASE_URI + 'optimise'
EXPLAIN_GLOBAL_URL = BASE_URI + 'explain/global-by-date'
RANDOM_BOOKING_URL = BASE_URI + 'random-booking'
EXPLAIN_LOCAL_URL  = BASE_URI + 'explain/local'


# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="NoShowShield", page_icon="🛡️", layout="wide")

st.markdown("""
<style>
  .booking-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin: 10px 0 14px 0;
  }
  .bk-sq {
    width: 36px; height: 36px;
    border-radius: 6px;
    display: flex; align-items: center; justify-content: center;
    font-size: 10px; font-weight: 700; color: #fff;
    border: 2px solid rgba(255,255,255,0.10);
    cursor: default;
  }
  .bk-low    { background: #1a7a5c; }
  .bk-medium { background: #c07a00; }
  .bk-high   { background: #a93226; }
  .legend-row {
    display: flex; gap: 18px;
    font-size: 12px; color: #8fa8c8; margin-bottom: 6px;
  }
  .legend-dot {
    width: 11px; height: 11px; border-radius: 3px;
    display: inline-block; margin-right: 4px; vertical-align: middle;
  }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ NoShowShield")
st.markdown("**AI-Powered Hotel Revenue Protection Against Cancellations**")
st.markdown(
    "NoShowShield uses machine learning to predict booking cancellations "
    "and recommend optimal overbooking levels — maximising revenue while "
    "keeping guest relocation risk below a configurable threshold. "
    "Select a date and room type to see actionable recommendations backed "
    "by SHAP explainability."
)


# ------------------------------------------------------------------
# Sidebar — optimization settings
# ------------------------------------------------------------------
st.sidebar.header("Optimization Settings")

relocation_cost = st.sidebar.number_input(
    "Relocation cost (€)",
    min_value=0.0, max_value=1000.0, value=300.0, step=50.0,
    help="Cost of relocating a guest to another hotel when overbooked.",
)

max_risk = st.sidebar.slider(
    "Max relocation risk",
    min_value=0.0, max_value=0.10, value=0.02, step=0.01,
    help="Maximum acceptable probability of having to relocate a guest.",
)


# ------------------------------------------------------------------
# API helpers
# ------------------------------------------------------------------
def api_get(url: str, params: dict, timeout: int = 120, max_retries: int = 3):
    for attempt in range(1, max_retries + 1):
        try:
            response = requests.get(url, params=params, timeout=timeout)
            if response.status_code != 200:
                return {"error": f"API returned status {response.status_code}: {response.text}"}
            return response.json()
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                time.sleep(2); continue
            return {"error": "API request timed out. The API may still be waking up — try again in a minute."}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep(2); continue
            return {"error": "Could not connect to the API. Check that the Cloud Run service is running."}
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
                time.sleep(2); continue
            return {"error": "API request timed out."}
        except requests.exceptions.ConnectionError:
            if attempt < max_retries:
                time.sleep(2); continue
            return {"error": "Could not connect to the API."}
    return {"error": "Unexpected error during API call."}


# ------------------------------------------------------------------
# Poisson-Binomial helpers
# ------------------------------------------------------------------
def pb_pmf(probs: list) -> np.ndarray:
    """Exact Poisson-Binomial PMF via dynamic programming — same algorithm as optimiser.py."""
    probs = np.asarray(probs, dtype=np.float64)
    n = len(probs)
    pmf = np.zeros(n + 1)
    pmf[0] = 1.0
    for p in probs:
        new = np.empty_like(pmf)
        new[0]  = pmf[0] * (1 - p)
        new[1:] = pmf[1:] * (1 - p) + pmf[:-1] * p
        pmf = new
    return pmf


def sq_color_class(p: float) -> str:
    if p < 0.30: return "bk-low"
    if p < 0.65: return "bk-medium"
    return "bk-high"


# ------------------------------------------------------------------
# Load optimisation data — triggered by button
# ------------------------------------------------------------------
if st.sidebar.button("Get Recommendations", type="primary", use_container_width=True):
    with st.spinner("Fetching predictions from API … (first load may take up to 2 min while the API wakes up)"):
        results = api_get(OPTIMISE_URL, {"relocation_cost": relocation_cost, "max_risk": max_risk})
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
# TAB 1 — Overbooking Recommendations + Poisson-Binomial Explainer
# ==================================================================
with tab1:
    if "results" not in st.session_state:
        st.info("Adjust settings in the sidebar and click **Get Recommendations** to start.")
    else:
        results = st.session_state["results"]

        recs = pd.DataFrame(results["recommendations"])
        recs["arrival_date"] = pd.to_datetime(recs["arrival_date"])

        metrics    = results["metrics"]
        model_info = results["model_info"]

        # ── Sidebar filters ──────────────────────────────────────────
        st.sidebar.header("Filters")
        available_hotels = sorted(recs["hotel"].unique())
        selected_hotel   = st.sidebar.selectbox("Select hotel", available_hotels)

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

        # ── Sidebar model info ───────────────────────────────────────
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

        # ── Filter for selection ─────────────────────────────────────
        filtered = recs[
            (recs["hotel"] == selected_hotel)
            & (recs["arrival_date"].dt.date == selected_date)
            & (recs["assigned_room_type"] == selected_room)
        ]

        st.subheader("Recommendation")

        if filtered.empty:
            st.warning("No data available for this selection.")
        else:
            row = filtered.iloc[0]

            # ── Top metrics ──────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            col1.metric("Capacity",           int(row["capacity"]))
            col2.metric("Current Bookings",   int(row["total_bookings"]))
            col3.metric("Expected Show-ups",  round(row["expected_show_ups"], 1))

            st.divider()

            col4, col5, col6 = st.columns(3)
            col4.metric("Recommended Extra Bookings", int(row["recommended_extra"]))
            col5.metric("Net Benefit (€)",            f"€{row['net_benefit']:.2f}")
            col6.metric("Relocation Risk",            f"{row['relocation_probability'] * 100:.2f}%")

            st.divider()

            # ── Revenue chart + SHAP ─────────────────────────────────
            left_col, right_col = st.columns([3, 2])

            with left_col:
                st.subheader("Revenue Comparison")
                st.caption(f"Expected revenue for **{selected_room}** on **{selected_date}**")

                mean_adr    = row["mean_adr"]
                rev_without = row["expected_show_ups"] * mean_adr
                rev_with    = rev_without + row["net_benefit"]

                fig_rev = go.Figure()
                fig_rev.add_trace(go.Bar(
                    x=["Without Overbooking", "With Overbooking"],
                    y=[rev_without, rev_with],
                    marker_color=["#2ecc71", "#1abc9c"],
                    text=[f"€{rev_without:,.0f}", f"€{rev_with:,.0f}"],
                    textposition="outside",
                    cliponaxis=False,
                    showlegend=False,
                ))
                fig_rev.add_annotation(
                    x="With Overbooking", y=rev_with,
                    ax="Without Overbooking", ay=rev_without,
                    axref="x", ayref="y",
                    text=f"<b>+€{row['net_benefit']:,.0f}</b>",
                    showarrow=True, arrowhead=2, arrowsize=1.2,
                    arrowwidth=2, arrowcolor="#f39c12",
                    font=dict(size=14, color="#f39c12"),
                    xanchor="left", yanchor="middle",
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

            with right_col:
                st.subheader("SHAP — Top Risk Factors")
                st.caption(
                    f"Why bookings on **{selected_date}** for room type **{selected_room}** "
                    f"are likely to cancel"
                )

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
                            timeout=60, max_retries=2,
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
                    shap_df["feature"] = (
                        shap_df["feature_group"]
                        .str.replace("cat_ordinal__", "", regex=False)
                        .str.replace("_", " ", regex=False)
                        .str.title()
                    )
                    top_shap = shap_df.nlargest(5, "mean_abs_shap").sort_values("mean_abs_shap")

                    fig = px.bar(
                        top_shap, x="mean_abs_shap", y="feature", orientation="h",
                        labels={"mean_abs_shap": "Mean |SHAP Value|", "feature": ""},
                        color="mean_abs_shap",
                        color_continuous_scale=["#2ecc71", "#f39c12", "#e74c3c"],
                    )
                    fig.update_layout(
                        height=400, margin=dict(l=0, r=0, t=10, b=0),
                        showlegend=False, coloraxis_showscale=False,
                        yaxis=dict(tickfont=dict(size=12)),
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No SHAP data available for this date and room type.")

            # ==========================================================
            # POISSON-BINOMIAL EXPLAINER — for the currently selected
            # hotel / date / room type
            # ==========================================================
            st.divider()
            st.subheader("🧮 How the Engine Decided — Poisson-Binomial Distribution")
            st.markdown(
                f"The recommendation above is driven by this distribution. "
                f"Each of the **{int(row['total_bookings'])} bookings** for "
                f"**{selected_room}** on **{selected_date}** has its own "
                f"XGBoost-predicted cancellation probability. The Poisson-Binomial "
                f"engine builds the full distribution of possible cancellation "
                f"counts — then finds the overbooking level that keeps relocation "
                f"risk below **{st.session_state.get('max_risk', max_risk) * 100:.0f}%**."
            )

            # Use the real per-booking cancellation probabilities from the optimiser
            ind_probs = sorted(row["individual_probs"])
            n_books   = int(row["total_bookings"])
            p_mean    = float(np.mean(ind_probs)) if ind_probs else 0.0
            cap       = int(row["capacity"])
            cur_risk  = st.session_state.get("max_risk", max_risk)

            # ── Booking grid ──────────────────────────────────────────
            st.markdown(
                """<div class="legend-row">
                  <span><span class="legend-dot" style="background:#1a7a5c"></span>Low risk (&lt;30%)</span>
                  <span><span class="legend-dot" style="background:#c07a00"></span>Medium risk (30–65%)</span>
                  <span><span class="legend-dot" style="background:#a93226"></span>High risk (&gt;65%)</span>
                </div>""",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"**{n_books} confirmed bookings** — each square shows one booking's "
                f"estimated cancellation probability:"
            )

            squares_html = '<div class="booking-grid">'
            for p in ind_probs:
                css   = sq_color_class(p)
                label = f"{int(round(p * 100))}%"
                squares_html += (
                    f'<div class="bk-sq {css}" title="Cancel prob: {p:.1%}">{label}</div>'
                )
            squares_html += "</div>"
            st.markdown(squares_html, unsafe_allow_html=True)

            # ── PMF bar chart ─────────────────────────────────────────
            pmf    = pb_pmf(ind_probs)
            k_vals = list(range(len(pmf)))

            mean_c = float(np.dot(k_vals, pmf))
            var_c  = float(np.dot([(k - mean_c) ** 2 for k in k_vals], pmf))
            std_c  = math.sqrt(var_c)

            # Relocation: show-ups > cap  ↔  cancellations < n_books - cap
            boundary   = n_books - cap
            reloc_prob = float(sum(pmf[k] for k in range(max(0, boundary))))

            # Colour each bar
            bar_colors = []
            for k in k_vals:
                shows = n_books - k
                if shows > cap:
                    bar_colors.append("#e74c3c")   # relocation needed
                elif shows == cap:
                    bar_colors.append("#f39c12")   # exactly full
                else:
                    bar_colors.append("#00c9a7")   # safe

            # 95% CI band
            cum = np.cumsum(pmf)
            lo  = int(np.searchsorted(cum, 0.025))
            hi  = int(np.searchsorted(cum, 0.975))

            fig_pb = go.Figure()

            fig_pb.add_trace(go.Bar(
                x=k_vals,
                y=pmf,
                marker_color=bar_colors,
                hovertemplate=(
                    "<b>%{x} cancellations</b><br>"
                    "Probability: %{y:.2%}<br>"
                    "Show-ups: %{customdata}<extra></extra>"
                ),
                customdata=[n_books - k for k in k_vals],
            ))

            # Expected cancellations line
            fig_pb.add_vline(
                x=mean_c,
                line_dash="dash", line_color="#f39c12", line_width=2,
                annotation_text=f"  E[cancel] = {mean_c:.1f}  (σ = {std_c:.1f})",
                annotation_font_color="#f39c12",
                annotation_position="top right",
            )

            # Capacity boundary
            if n_books > cap:
                fig_pb.add_vline(
                    x=boundary - 0.5,
                    line_color="#e74c3c", line_width=2.5,
                    annotation_text="  ← relocation zone",
                    annotation_font_color="#e74c3c",
                    annotation_position="top left",
                )

            # 95% CI shading
            fig_pb.add_vrect(
                x0=lo - 0.5, x1=hi + 0.5,
                fillcolor="rgba(0,201,167,0.07)",
                line_width=0,
                annotation_text="95% CI",
                annotation_font_color="#00c9a7",
                annotation_position="bottom right",
            )

            fig_pb.update_layout(
                title=(
                    f"Distribution of cancellation counts — {n_books} bookings, "
                    f"capacity {cap}"
                ),
                xaxis_title="Number of cancellations (k)",
                yaxis_title="P(exactly k cancel)",
                yaxis_tickformat=".1%",
                height=400,
                margin=dict(l=0, r=0, t=50, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                bargap=0.1,
                showlegend=False,
            )
            fig_pb.update_xaxes(showgrid=False, dtick=1 if n_books <= 30 else 2)
            fig_pb.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

            st.plotly_chart(fig_pb, use_container_width=True)

            # ── Stats row ─────────────────────────────────────────────
            s1, s2, s3, s4 = st.columns(4)
            s1.metric(
                "Expected cancellations",
                f"{mean_c:.1f}",
                delta=f"σ = {std_c:.1f}",
                delta_color="off",
            )
            s2.metric("Expected show-ups", f"{n_books - mean_c:.1f}")
            s3.metric(
                "Relocation probability",
                f"{reloc_prob:.1%}",
                delta=(
                    f"{'⚠️ above' if reloc_prob > cur_risk else '✅ below'} "
                    f"{cur_risk:.0%} threshold"
                ),
                delta_color="inverse",
            )
            s4.metric(
                "Extra bookings recommended",
                int(row["recommended_extra"]),
                delta=f"→ {int(row['recommended_total'])} total",
                delta_color="off",
            )

            # ── Key insight callout ───────────────────────────────────
            st.markdown(
                f"> **Why not just use the average?**  \n"
                f"These {n_books} bookings have a mean cancellation probability of "
                f"**{p_mean:.0%}**, implying {mean_c:.1f} expected cancellations — but "
                f"that tells you nothing about the *variance*. The Poisson-Binomial "
                f"computes the full distribution, so the engine knows there is a "
                f"**{reloc_prob:.1%}** chance of needing to relocate a guest. "
                f"That tail probability is what determines whether accepting "
                f"**{int(row['recommended_extra'])} extra booking"
                f"{'s' if row['recommended_extra'] != 1 else ''}** is safe."
            )


# ==================================================================
# TAB 2 — Single Booking Prediction
# ==================================================================
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

        col1, col2, col3 = st.columns(3)
        col1.metric("Prediction",               "Will Cancel" if prediction == 1 else "Won't Cancel")
        col2.metric("Cancellation Probability", f"{prob * 100:.1f}%")
        col3.metric("Actual Outcome",           "Canceled" if actual == 1 else "Not Canceled")

        st.divider()

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

            shap_df  = pd.DataFrame(explain["grouped_local_shap"])
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
                    top_shap, x="shap_value", y="feature", orientation="h",
                    labels={"shap_value": "SHAP Value", "feature": ""},
                    color="shap_value",
                    color_continuous_scale=["#f39c12", "#e74c3c"],
                )
                fig.update_layout(
                    height=400, margin=dict(l=0, r=0, t=10, b=0),
                    showlegend=False, coloraxis_showscale=False,
                    yaxis=dict(tickfont=dict(size=12)),
                )
                st.plotly_chart(fig, use_container_width=True)
