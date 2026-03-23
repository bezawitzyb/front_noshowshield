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

OPTIMISE_URL         = BASE_URI + 'optimise'
EXPLAIN_GLOBAL_URL   = BASE_URI + 'explain/global-by-date'
RANDOM_BOOKING_URL   = BASE_URI + 'random-booking'
EXPLAIN_LOCAL_URL    = BASE_URI + 'explain/local'


# ------------------------------------------------------------------
# page config
# ------------------------------------------------------------------
st.set_page_config(page_title="NoShowShield", page_icon="🛡️", layout="wide")

# ---- Custom CSS for the brand palette and grid cards ----
st.markdown("""
<style>
  :root {
    --teal:   #00c9a7;
    --navy:   #0d1b2a;
    --slate:  #1a2d42;
    --muted:  #8fa8c8;
    --danger: #e74c3c;
    --warn:   #f39c12;
  }
  /* booking squares grid */
  .booking-grid {
    display: flex;
    flex-wrap: wrap;
    gap: 4px;
    margin: 12px 0;
  }
  .bk-sq {
    width: 38px;
    height: 38px;
    border-radius: 6px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 11px;
    font-weight: 700;
    color: #fff;
    border: 2px solid rgba(255,255,255,0.15);
    transition: transform .15s;
  }
  .bk-sq:hover { transform: scale(1.15); }
  .bk-low    { background: #1a7a5c; }
  .bk-medium { background: #c07a00; }
  .bk-high   { background: #a93226; }
  .bk-future { background: #1a2d42; border: 2px dashed #3a5068; color: #3a5068; }
  .legend-row { display:flex; gap:18px; margin-bottom:8px; font-size:13px; color:#8fa8c8; }
  .legend-dot { width:12px; height:12px; border-radius:3px; display:inline-block; margin-right:4px; vertical-align:middle; }
</style>
""", unsafe_allow_html=True)

st.title("🛡️ NoShowShield")
st.markdown("**AI-Powered Hotel Revenue Protection Against Cancellations**")
st.markdown(
    "NoShowShield uses machine learning to predict booking cancellations "
    "and recommend optimal overbooking levels — maximising revenue while "
    "keeping guest relocation risk below a configurable threshold."
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
# Load optimisation data
# ------------------------------------------------------------------
if st.sidebar.button("Get Recommendations", type="primary", use_container_width=True):
    with st.spinner("Fetching predictions from API … (first load may take up to 2 min)"):
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
tab1, tab2, tab3 = st.tabs([
    "📊 Overbooking Recommendations",
    "🔍 Single Booking Prediction",
    "🧮 How It Works — Poisson-Binomial",
])


# ==================================================================
# TAB 1 — Overbooking Recommendations
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

        # Sidebar filters
        st.sidebar.header("Filters")
        available_hotels = sorted(recs["hotel"].unique())
        selected_hotel   = st.sidebar.selectbox("Select hotel", available_hotels)

        available_dates  = sorted(recs[recs["hotel"] == selected_hotel]["arrival_date"].dt.date.unique())
        selected_date    = st.sidebar.selectbox("Select date", available_dates)

        available_rooms  = sorted(
            recs[(recs["hotel"] == selected_hotel) & (recs["arrival_date"].dt.date == selected_date)]
            ["assigned_room_type"].unique()
        )
        selected_room = st.sidebar.selectbox("Select room type", available_rooms)

        # Sidebar model info
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

        # Filter for selection
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

            col1, col2, col3 = st.columns(3)
            col1.metric("Capacity",            int(row["capacity"]))
            col2.metric("Current Bookings",    int(row["total_bookings"]))
            col3.metric("Expected Show-ups",   round(row["expected_show_ups"], 1))

            st.divider()

            col4, col5, col6 = st.columns(3)
            col4.metric("Recommended Extra Bookings", int(row["recommended_extra"]))
            col5.metric("Net Benefit (€)",            f"€{row['net_benefit']:.2f}")
            col6.metric("Relocation Risk",            f"{row['relocation_probability'] * 100:.2f}%")

            st.divider()

            left_col, right_col = st.columns([3, 2])

            # --- Revenue comparison chart ---
            with left_col:
                st.subheader("Revenue Comparison")
                st.caption(f"Expected revenue for **{selected_room}** on **{selected_date}**")

                cancel_rate = row["expected_cancellations"] / row["total_bookings"] if row["total_bookings"] > 0 else 0
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

            # --- SHAP chart ---
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
                            {"selected_date": selected_date_str, "room_type": selected_room, "hotel": selected_hotel},
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
        col1.metric("Prediction",              "Will Cancel" if prediction == 1 else "Won't Cancel")
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


# ==================================================================
# TAB 3 — Poisson-Binomial Interactive Explainer
# ==================================================================
with tab3:
    st.subheader("🧮 The Poisson-Binomial Distribution — Live")
    st.markdown(
        "This visualiser shows **exactly** how the overbooking engine works. "
        "Each booking has its own cancellation probability from the XGBoost model. "
        "Drag the slider to add bookings one at a time and watch the probability "
        "distribution over *total cancellations* evolve from trivial to bell-shaped."
    )

    # ----------------------------------------------------------------
    # Helper functions (pure Python, no API needed)
    # ----------------------------------------------------------------
    def pb_pmf(probs: list[float]) -> np.ndarray:
        """Exact Poisson-Binomial PMF via dynamic programming."""
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
        if p < 0.30:  return "bk-low"
        if p < 0.65:  return "bk-medium"
        return "bk-high"

    def sq_label(p: float) -> str:
        return f"{int(round(p * 100))}%"

    # ----------------------------------------------------------------
    # Scenario selector
    # ----------------------------------------------------------------
    scenario = st.selectbox(
        "Choose a scenario",
        [
            "Mixed risk (realistic hotel mix)",
            "Mostly low-risk (corporate bookings)",
            "Mostly high-risk (OTA, long lead time)",
            "Bimodal (two very different guest groups)",
        ],
        index=0,
    )

    np.random.seed(42)

    if scenario == "Mixed risk (realistic hotel mix)":
        ALL_PROBS = sorted(
            list(np.random.beta(1.5, 4, 8))       # low risk cluster
            + list(np.random.beta(4, 4, 5))        # medium risk
            + list(np.random.beta(6, 2, 5)),       # high risk cluster
        )
    elif scenario == "Mostly low-risk (corporate bookings)":
        ALL_PROBS = sorted(np.random.beta(1, 6, 18).tolist())
    elif scenario == "Mostly high-risk (OTA, long lead time)":
        ALL_PROBS = sorted(np.random.beta(5, 2, 18).tolist())
    else:  # bimodal
        ALL_PROBS = sorted(
            list(np.random.beta(1, 8, 9))
            + list(np.random.beta(8, 1, 9))
        )

    ALL_PROBS = [round(p, 3) for p in ALL_PROBS]
    N_TOTAL   = len(ALL_PROBS)

    # ----------------------------------------------------------------
    # Capacity & risk settings
    # ----------------------------------------------------------------
    c1, c2, c3 = st.columns([2, 2, 2])
    with c1:
        capacity = st.number_input(
            "Room capacity", min_value=1, max_value=N_TOTAL, value=min(12, N_TOTAL),
            help="Physical number of rooms. Used to draw the relocation-risk threshold.",
        )
    with c2:
        risk_threshold = st.slider(
            "Max relocation risk", 0.01, 0.10, 0.02, 0.01,
            help="The tail probability above capacity that we want to keep below this level.",
        )
    with c3:
        adr = st.number_input("ADR (€)", min_value=50, max_value=500, value=150, step=10)

    # ----------------------------------------------------------------
    # Booking slider
    # ----------------------------------------------------------------
    n_bookings = st.slider(
        "Number of bookings added",
        min_value=0,
        max_value=N_TOTAL,
        value=0,
        step=1,
        help="Drag right to add bookings one at a time. Watch the distribution evolve.",
    )

    active_probs = ALL_PROBS[:n_bookings]

    # ----------------------------------------------------------------
    # Booking grid
    # ----------------------------------------------------------------
    st.markdown("**Booking pool** — each square shows one booking's cancellation probability:")

    legend_html = """
    <div class="legend-row">
      <span><span class="legend-dot" style="background:#1a7a5c"></span>Low risk (&lt;30%)</span>
      <span><span class="legend-dot" style="background:#c07a00"></span>Medium risk (30–65%)</span>
      <span><span class="legend-dot" style="background:#a93226"></span>High risk (&gt;65%)</span>
      <span><span class="legend-dot" style="background:#1a2d42;border:2px dashed #3a5068"></span>Not yet added</span>
    </div>
    """
    st.markdown(legend_html, unsafe_allow_html=True)

    squares_html = '<div class="booking-grid">'
    for i, p in enumerate(ALL_PROBS):
        if i < n_bookings:
            css = sq_color_class(p)
            label = sq_label(p)
        else:
            css   = "bk-future"
            label = "—"
        squares_html += f'<div class="bk-sq {css}" title="Cancel prob: {p:.1%}">{label}</div>'
    squares_html += "</div>"
    st.markdown(squares_html, unsafe_allow_html=True)

    # ----------------------------------------------------------------
    # Distribution chart + insights
    # ----------------------------------------------------------------
    if n_bookings == 0:
        st.info("☝️ Drag the slider above to start adding bookings.")
    else:
        pmf        = pb_pmf(active_probs)
        k_vals     = list(range(len(pmf)))
        mean_c     = float(np.dot(k_vals, pmf))
        var_c      = float(np.dot([(k - mean_c) ** 2 for k in k_vals], pmf))
        std_c      = math.sqrt(var_c)
        mean_shows = n_bookings - mean_c

        # Tail probability: P(show-ups > capacity) = P(cancellations < n - capacity)
        # show-ups = n_bookings - cancellations
        # show-ups > capacity  ↔  cancellations < n_bookings - capacity
        min_cancel_for_reloc = n_bookings - capacity  # at this many cancellations, show-ups == capacity
        # relocation when show-ups > capacity ↔ cancellations < (n_bookings - capacity)
        reloc_prob = float(sum(pmf[k] for k in range(max(0, min_cancel_for_reloc))))

        # Color each bar: grey if safe show-up, red if overbooked (cancellations too few)
        bar_colors = []
        for k in k_vals:
            shows = n_bookings - k
            if shows > capacity:
                bar_colors.append("#e74c3c")   # red: relocation needed
            elif shows == capacity:
                bar_colors.append("#f39c12")   # amber: exactly at capacity
            else:
                bar_colors.append("#00c9a7")   # teal: under capacity

        # Mean annotation
        mean_idx = int(round(mean_c))

        fig = go.Figure()

        fig.add_trace(go.Bar(
            x=k_vals,
            y=pmf,
            marker_color=bar_colors,
            name="P(k cancellations)",
            hovertemplate=(
                "<b>%{x} cancellations</b><br>"
                "Probability: %{y:.2%}<br>"
                "Show-ups: %{customdata}<extra></extra>"
            ),
            customdata=[n_bookings - k for k in k_vals],
        ))

        # Mean vertical line
        fig.add_vline(
            x=mean_c,
            line_dash="dash",
            line_color="#f39c12",
            line_width=2,
            annotation_text=f"  E[cancel] = {mean_c:.1f}",
            annotation_font_color="#f39c12",
            annotation_position="top right",
        )

        # Capacity boundary (vertical line at n_bookings - capacity cancellations)
        if n_bookings > capacity:
            boundary = n_bookings - capacity
            fig.add_vline(
                x=boundary - 0.5,
                line_color="#e74c3c",
                line_width=2.5,
                annotation_text="  ← relocation zone",
                annotation_font_color="#e74c3c",
                annotation_position="top left",
            )

        # 95% confidence band shading
        cum = np.cumsum(pmf)
        lo  = int(np.searchsorted(cum, 0.025))
        hi  = int(np.searchsorted(cum, 0.975))
        fig.add_vrect(
            x0=lo - 0.5, x1=hi + 0.5,
            fillcolor="rgba(0,201,167,0.08)",
            line_width=0,
            annotation_text="95% CI",
            annotation_font_color="#00c9a7",
            annotation_position="bottom right",
        )

        fig.update_layout(
            title=f"Distribution of Total Cancellations — {n_bookings} booking{'s' if n_bookings != 1 else ''} added",
            xaxis_title="Number of cancellations (k)",
            yaxis_title="Probability P(exactly k cancel)",
            yaxis_tickformat=".1%",
            height=420,
            margin=dict(l=0, r=0, t=50, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            bargap=0.1,
            showlegend=False,
        )
        fig.update_xaxes(showgrid=False, dtick=1)
        fig.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")

        st.plotly_chart(fig, use_container_width=True)

        # ----------------------------------------------------------------
        # Key stats row
        # ----------------------------------------------------------------
        s1, s2, s3, s4 = st.columns(4)
        s1.metric("Expected cancellations",  f"{mean_c:.1f}  (σ = {std_c:.1f})")
        s2.metric("Expected show-ups",        f"{mean_shows:.1f}")
        s3.metric("Relocation probability",   f"{reloc_prob:.1%}",
                  delta=f"{'⚠️ above' if reloc_prob > risk_threshold else '✅ below'} {risk_threshold:.0%} threshold",
                  delta_color="inverse")
        s4.metric("Avg cancel prob",          f"{np.mean(active_probs):.1%}" if active_probs else "—")

        # ----------------------------------------------------------------
        # Why NOT just use the mean? explainer
        # ----------------------------------------------------------------
        st.divider()
        st.markdown("#### 💡 Why Not Just Use the Mean?")

        eg_c1, eg_c2 = st.columns(2)

        with eg_c1:
            st.markdown("**Same mean, different shape**")
            # Example A: one 90% + one 10%   vs   B: two 50%
            pmf_A = pb_pmf([0.90, 0.10])
            pmf_B = pb_pmf([0.50, 0.50])

            figAB = go.Figure()
            figAB.add_trace(go.Bar(
                x=[0, 1, 2], y=pmf_A,
                name="[90%, 10%]",
                marker_color="#e74c3c",
                opacity=0.8,
            ))
            figAB.add_trace(go.Bar(
                x=[0, 1, 2], y=pmf_B,
                name="[50%, 50%]",
                marker_color="#00c9a7",
                opacity=0.8,
            ))
            figAB.update_layout(
                barmode="group",
                height=260,
                xaxis=dict(title="# cancellations", dtick=1),
                yaxis=dict(title="Probability", tickformat=".0%"),
                legend=dict(orientation="h", y=1.12),
                margin=dict(l=0, r=0, t=30, b=0),
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            figAB.update_xaxes(showgrid=False)
            figAB.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
            st.plotly_chart(figAB, use_container_width=True)
            st.caption(
                "Both groups have a mean of 1.0 cancellation. But [90%, 10%] "
                "is heavily bimodal — it almost always cancels 0 or 2, rarely 1. "
                "[50%, 50%] peaks at exactly 1. The Poisson-Binomial captures this difference; "
                "a simple average cannot."
            )

        with eg_c2:
            st.markdown("**Overbooking decision depends on the tail, not the mean**")
            st.markdown("""
The optimiser asks:

> *Given these exact cancellation probabilities, what is the chance that more guests
show up than I have rooms for?*

That question requires the **full distribution** — specifically the right tail
above the capacity line (shown in red on the chart above).

If you only knew the mean expected cancellations, you could compute an expected
number of show-ups — but you could not compute the *probability* of exceeding
capacity. You would have no idea whether that 2% relocation risk target is met.

**The Poisson-Binomial gives you the tail. The mean does not.**
""")
            # Show tail shrinking as bookings increase
            if n_bookings >= 3:
                tail_vals = []
                for nb in range(1, n_bookings + 1):
                    p_sub  = ALL_PROBS[:nb]
                    pmf_nb = pb_pmf(p_sub)
                    shows  = nb - capacity
                    tail   = float(sum(pmf_nb[k] for k in range(max(0, nb - capacity))))
                    tail_vals.append({"bookings": nb, "reloc_prob": tail})
                tail_df = pd.DataFrame(tail_vals)

                fig_tail = go.Figure()
                fig_tail.add_trace(go.Scatter(
                    x=tail_df["bookings"],
                    y=tail_df["reloc_prob"],
                    mode="lines+markers",
                    line=dict(color="#00c9a7", width=2),
                    marker=dict(size=6),
                    name="Relocation probability",
                ))
                fig_tail.add_hline(
                    y=risk_threshold,
                    line_dash="dash",
                    line_color="#e74c3c",
                    annotation_text=f"  max risk = {risk_threshold:.0%}",
                    annotation_font_color="#e74c3c",
                )
                fig_tail.update_layout(
                    height=260,
                    xaxis=dict(title="Bookings added", dtick=1),
                    yaxis=dict(title="P(relocation needed)", tickformat=".1%"),
                    margin=dict(l=0, r=0, t=10, b=0),
                    plot_bgcolor="rgba(0,0,0,0)",
                    paper_bgcolor="rgba(0,0,0,0)",
                )
                fig_tail.update_xaxes(showgrid=False)
                fig_tail.update_yaxes(showgrid=True, gridcolor="rgba(255,255,255,0.06)")
                st.plotly_chart(fig_tail, use_container_width=True)
                st.caption(
                    "Relocation risk as each booking is added. "
                    "The engine stops accepting extra bookings the moment this line "
                    f"would cross the {risk_threshold:.0%} threshold."
                )

        # ----------------------------------------------------------------
        # Simple vs PB comparison table
        # ----------------------------------------------------------------
        if n_bookings > 0:
            st.divider()
            st.markdown("#### Simple Mean vs Poisson-Binomial — side by side")

            naive_mean   = np.mean(active_probs) * n_bookings
            pb_mean      = mean_c
            naive_std    = "unknown (needs full dist.)"
            pb_std       = f"{std_c:.2f}"
            naive_tail   = "cannot compute"
            pb_tail      = f"{reloc_prob:.2%}"

            comparison = pd.DataFrame({
                "Approach": ["Simple mean", "Poisson-Binomial"],
                "Expected cancellations": [f"{naive_mean:.1f}", f"{pb_mean:.1f}"],
                "Std deviation": [naive_std, pb_std],
                "P(relocation needed)": [naive_tail, pb_tail],
                "Can optimise at 2% risk?": ["❌ No", "✅ Yes"],
            }).set_index("Approach")

            st.dataframe(comparison, use_container_width=True)
