"""Next-Day Violation Forecast — 7-day ahead predictions per junction and city-wide.
   Adapted from ParkSight AI's NextDayForecast page."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import ui, core

ui.page("Next-Day Forecast", "N")
ui.brand_sidebar()

st.markdown("## Next-Day Violation Forecast")
st.caption("7-day ahead predictions per junction — RandomForest / GradientBoosting trained on BTP historical data.")

with st.spinner("Training models and generating forecasts…"):
    forecast = ui.get_nextday_forecast()

if isinstance(forecast, dict) and "error" in forecast:
    st.error(f"Forecast unavailable: {forecast['error']}")
    st.info("Ensure the dataset contains `junction_name` and sufficient temporal data (20+ days per junction).")
    st.stop()

cw = forecast.get("city_wide", {})
junctions = forecast.get("junctions", {})
jr = forecast.get("junction_results", {})

# ---------------- header KPIs ----------------
h1, h2, h3 = st.columns(3)
ui.kpi(h1, "City-wide Accuracy", f"{cw.get('accuracy', 0)}%",
       help=f"R² = {cw.get('r2', '—')}")
ui.kpi(h2, "Avg daily violations", f"{cw.get('avg_daily', 0):,.0f}")
ui.kpi(h3, "Data range", forecast.get("data_date_range", "—"))

# ---------------- city-wide 7-day bar chart ----------------
st.markdown("---")
st.subheader("City-Wide Predicted Violations — Next 7 Days")
city_fc = cw.get("forecast", [])

if city_fc:
    city_df = pd.DataFrame(city_fc)
    city_df["day"] = city_df["dayName"].str[:3]

    color_map = {"HIGH": "#EF4444", "MEDIUM": "#F59E0B", "LOW": "#10B981"}
    city_df["color"] = city_df["risk"].map(color_map)

    fig = go.Figure()
    for _, row in city_df.iterrows():
        fig.add_trace(go.Bar(
            x=[row["day"]], y=[row["predicted"]],
            marker_color=color_map.get(row["risk"], "#10B981"),
            showlegend=False,
            hovertemplate=f"<b>{row['dayName']} {row['date']}</b><br>"
                         f"Predicted: {row['predicted']:,}<br>"
                         f"Risk: {row['risk']}<extra></extra>"
        ))

    # Threshold lines
    high_thresh = cw.get("high_threshold", 0)
    avg_daily = cw.get("avg_daily", 0)
    if high_thresh > 0:
        fig.add_hline(y=high_thresh, line_dash="dash", line_color="#EF4444",
                     annotation_text=f"HIGH threshold ({high_thresh:.0f})",
                     annotation_position="top right")
    if avg_daily > 0:
        fig.add_hline(y=avg_daily, line_dash="dash", line_color="#8DA4BE",
                     annotation_text=f"Average ({avg_daily:.0f})",
                     annotation_position="top right")

    fig.update_layout(
        height=280, margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="", yaxis_title="Predicted violations",
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        bargap=0.3,
    )
    st.plotly_chart(fig, width="stretch")

    # Risk legend
    legend_cols = st.columns(3)
    legend_cols[0].markdown("🔴 **HIGH** — Pre-position teams early")
    legend_cols[1].markdown("🟡 **MEDIUM** — Standard monitoring")
    legend_cols[2].markdown("🟢 **LOW** — Normal operations")

# ---------------- junction forecast grid ----------------
if junctions:
    st.markdown("---")
    st.subheader("Per-Junction 7-Day Forecast")
    st.caption("Risk level per day for each monitored junction. Click to expand.")

    # Display in rows of 3
    junc_items = list(junctions.items())
    for row_start in range(0, len(junc_items), 3):
        cols = st.columns(min(3, len(junc_items) - row_start))
        for j, (junc_name, junc_data) in enumerate(junc_items[row_start:row_start + 3]):
            with cols[j]:
                short = junc_data["shortName"]
                has_high = any(d["risk"] == "HIGH" for d in junc_data["days"])

                # Header
                st.markdown(f"**{short}**")
                st.caption(f"{junc_data['modelUsed']} · {junc_data['accuracy']}% acc · MAE ±{junc_data['testMAE']}")

                # 7-day risk blocks
                day_cols = st.columns(7)
                for i, d in enumerate(junc_data["days"]):
                    icon = "🔴" if d["risk"] == "HIGH" else "🟡" if d["risk"] == "MEDIUM" else "🟢"
                    with day_cols[i]:
                        st.markdown(f"<div style='text-align:center;font-size:0.7rem;color:#8A93A6'>"
                                   f"{d['dayName'][:3]}</div>", unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align:center;font-size:1.1rem'>{icon}</div>",
                                   unsafe_allow_html=True)
                        st.markdown(f"<div style='text-align:center;font-size:0.7rem;color:#8A93A6'>"
                                   f"{d['predicted']}</div>", unsafe_allow_html=True)

                # Peak day info
                peak_day = max(junc_data["days"], key=lambda d: d["predicted"])
                if has_high:
                    st.warning(f"⚠️ Peak: {peak_day['dayName'][:3]} ({peak_day['predicted']}) — pre-position recommended")
                else:
                    st.success(f"✅ Peak: {peak_day['dayName'][:3]} ({peak_day['predicted']}) — normal ops")

                st.markdown("")

# ---------------- detailed forecast table ----------------
if junctions:
    st.markdown("---")
    with st.expander("Detailed forecast table (all junctions × all days)"):
        rows = []
        for junc_name, junc_data in junctions.items():
            for d in junc_data["days"]:
                rows.append({
                    "Junction": junc_data["shortName"],
                    "Date": d["date"],
                    "Day": d["dayName"],
                    "Predicted": d["predicted"],
                    "CI Low": d["ciLow"],
                    "CI High": d["ciHigh"],
                    "Risk": d["risk"],
                    "Recommendation": d["recommendation"],
                })
        detail_df = pd.DataFrame(rows)
        st.dataframe(detail_df, hide_index=True, width="stretch", height=500)

        st.download_button("Download forecast (CSV)",
                          detail_df.to_csv(index=False).encode(),
                          file_name="parksensei_nextday_forecast.csv",
                          mime="text/csv")

# ---------------- model accuracy table ----------------
if jr:
    st.markdown("---")
    st.subheader("Model Performance Summary")

    perf_cols = st.columns(2)
    with perf_cols[0]:
        st.markdown("**Per-Junction Accuracy**")
        rows = []
        for junc, r in jr.items():
            accuracy = r["accuracy"]
            badge = "🟢" if accuracy >= 75 else "🟡" if accuracy >= 60 else "🔴"
            rows.append({
                "Junction": r["short"],
                "Model": r["best_model"],
                "MAE": r["test_mae"],
                "R²": r["test_r2"],
                "Accuracy": f"{badge} {accuracy}%",
            })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    with perf_cols[1]:
        # Feature importance from first junction
        first_junc = next(iter(junctions.values()))
        fi = first_junc.get("featureImportance", {})
        if fi:
            st.markdown(f"**Top Features — {first_junc['shortName']}**")
            fi_items = list(fi.items())
            for fname, fimp in fi_items:
                pct = min(100, fimp * 500)
                st.markdown(
                    f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px'>"
                    f"<span style='width:120px;font-size:0.75rem;color:#8DA4BE'>{fname}</span>"
                    f"<div style='flex:1;background:#1E2638;border-radius:4px;height:8px'>"
                    f"<div style='width:{pct}%;background:linear-gradient(to right,#4C8BF5,#06B6D4);height:8px;border-radius:4px'></div>"
                    f"</div>"
                    f"<span style='width:40px;text-align:right;font-size:0.75rem;color:#8DA4BE'>{fimp:.3f}</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

with st.expander("How next-day prediction works"):
    st.markdown(
        "- **Algorithm:** RandomForest / GradientBoosting (best selected per junction via TimeSeriesSplit CV)\n"
        "- **Features (20):** lag_1d/2d/3d/7d, rolling avg 3/7/14d, rolling std 7d, "
        "DOW/month historical averages, 7-day trend slope, calendar features, severity ratios\n"
        "- **Validation:** TimeSeriesSplit (5 folds) — respects temporal order, no future leakage\n"
        "- **Risk levels:** HIGH = predicted ≥ mean + 0.75σ, MEDIUM = ≥ mean − 0.25σ, LOW = below\n"
        "- **Confidence interval:** ±1.25 × test RMSE\n"
        "- **City-wide:** Separate RF model with 12 aggregate features"
    )
