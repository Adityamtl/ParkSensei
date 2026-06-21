"""Operations & Dispatch — Forecast & Patrol, Officer Allocation, Next Day Forecast."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import ui, core

ui.page("Operations & Dispatch", "O")
ui.brand_sidebar()


# =====================================================================
# TAB 1: FORECAST & PATROL
# =====================================================================
def _render_forecast_patrol():
    zones = ui.get_zones()
    fc    = ui.get_forecaster()
    bt    = ui.get_backtest()

    st.markdown("## Forecast & Patrol Planner")
    st.caption("Predict where violations will cluster next, then generate a deployment plan with recommended actions.")

    # ---------------- controls ----------------
    c1, c2, c3, c4 = st.columns([1.2, 1.6, 1, 1])
    day  = c1.selectbox("Weekday", ui.DOW_NAMES, index=5)        # default Saturday
    dow  = ui.DOW_NAMES.index(day)
    win  = c2.slider("Shift window (IST)", 0, 23, (9, 13))
    hours = list(range(win[0], win[1] + 1))
    teams = c3.slider("Patrol teams", 3, 20, 8)
    sep   = c4.slider("Min spacing (m)", 200, 1500, 600, step=100)

    pred = core.predict_load(fc, dow, hours)
    plan = core.allocate_patrols(zones, pred, k=teams, min_sep_m=sep)
    pm   = pred.merge(zones, on="gh6")

    total_pred = pred["pred_load"].sum()
    covered    = plan["pred_load"].sum() if len(plan) else 0
    k = st.columns(4)
    ui.kpi(k[0], "Predicted violations", f"{total_pred:,.0f}", help=f"{day} {win[0]:02d}:00–{win[1]:02d}:59")
    ui.kpi(k[1], f"Captured by {teams} teams", f"{covered/total_pred*100:,.0f}%" if total_pred else "—")
    ui.kpi(k[2], "Forecast accuracy", f"r = {bt['pearson_r']}",
           help=f"held-out time-split (after {bt['cutoff']}), {bt['cells']:,} zone·day·hour cells")
    ui.kpi(k[3], "Mean abs. error", f"{bt['mae']:.2f}", help="violations per zone·hour cell")

    # ---------------- map + plan ----------------
    left, right = st.columns([2, 1], gap="large")
    with left:
        pm2 = pm.copy()
        pm2["radius"] = (np.sqrt(pm2["pred_load"]) * 30).clip(40, 700)
        pm2["color"] = pm2["impact_score"].map(ui.impact_color)
        base = pdk.Layer("ScatterplotLayer", data=pm2, get_position=["lon", "lat"],
                         get_radius="radius", get_fill_color="color", opacity=0.45, pickable=False)
        layers = [base] + ui.plan_layers(plan)
        centre = plan.iloc[0] if len(plan) else {"lat": ui.BLR["lat"], "lon": ui.BLR["lon"]}
        st.pydeck_chart(ui.deck(layers, ui.view(centre["lat"], centre["lon"], zoom=11, pitch=40),
                                ui.TIP_PLAN), width="stretch")
        st.caption("Faint bubbles = predicted load · blue pins = recommended team deployment")
    with right:
        st.markdown(f"**Deployment plan — {day} {win[0]:02d}:00–{win[1]:02d}:59**")
        tbl_cols = ["team", "label", "pred_load", "impact_score"]
        if "recommended_action" in plan.columns:
            tbl_cols.append("recommended_action")
        if "place_type" in plan.columns:
            tbl_cols.append("place_type")
        tbl = plan[tbl_cols].copy()
        rename = {"team": "Team", "label": "Deploy to", "pred_load": "Exp. catches",
                  "impact_score": "Impact", "recommended_action": "Action", "place_type": "Type"}
        tbl.columns = [rename.get(c, c) for c in tbl.columns]
        st.dataframe(tbl, hide_index=True, width="stretch", height=380)

        # Download buttons
        dl1, dl2 = st.columns(2)
        with dl1:
            st.download_button("Download plan (CSV)",
                               plan.to_csv(index=False).encode(),
                               file_name=f"patrol_plan_{day}_{win[0]:02d}{win[1]:02d}.csv",
                               width="stretch")
        with dl2:
            pdf_bytes = ui.generate_pdf_brief(zones, plan=plan, backtest_result=bt)
            if pdf_bytes:
                st.download_button("Download PDF Brief",
                                   pdf_bytes,
                                   file_name=f"ParkSensei_brief_{day}_{win[0]:02d}{win[1]:02d}.pdf",
                                   mime="application/pdf",
                                   width="stretch")

    # ---------------- recommended actions per zone ----------------
    if "recommended_action" in plan.columns and len(plan):
        st.markdown("---")
        st.subheader("Recommended actions for each deployment zone")
        for _, p in plan.iterrows():
            with st.expander(f"{p['team']} → {p['label']} (Impact {p.get('impact_score', 0):.0f})"):
                recs = core.generate_recommendations(p)
                for r in recs:
                    ui.render_recommendation_card(r)

    with st.expander("How the forecast works"):
        st.markdown(
            "- **Target:** expected violations per *zone × weekday × hour*, learned from 150 days of history.\n"
            "- **Method:** Bayesian-shrunk historical rates — sparse weekday cells borrow strength from the "
            "zone's overall hour profile, so estimates stay stable.\n"
            f"- **Validation:** time-split backtest (train → predict the held-out tail after {bt['cutoff']}): "
            f"Pearson r = **{bt['pearson_r']}**, MAE = **{bt['mae']}** across {bt['cells']:,} cells.\n"
            "- **Allocation:** greedy maximisation of predicted load with a minimum-spacing constraint so "
            "teams don't stack on one street.\n"
            "- **Scoring:** 7-factor Congestion Impact Score: "
            "0.30×obstruction + 0.18×density + 0.15×junction + 0.13×arterial + 0.10×peak + 0.08×recurrence + 0.06×severity.\n"
            "- **Recommendations:** rule-based action engine assigns tow-away zones, peak-hour enforcement, "
            "camera monitoring, repeat-offender escalation, or signage audits based on zone characteristics.")


# =====================================================================
# TAB 2: OFFICER ALLOCATION
# =====================================================================
def _render_officer_allocation():
    zones = ui.get_zones()

    st.markdown("## Officer Allocation & Route Optimizer")
    st.caption("Proportional officer deployment based on zone impact scores, with optimized patrol routing.")

    # Controls
    c1, c2, c3, _ = st.columns([1, 1, 1, 1])
    total_officers = c1.slider("Total officers", 20, 300, 100, 10)
    top_n = c2.slider("Zones to cover", 5, 40, 20, 5)
    route_stops = c3.slider("Route stops", 5, 20, 10, 1)

    # Officer allocation
    alloc = core.officer_allocation(zones, total_officers=total_officers, top_n=top_n)

    # KPIs
    k = st.columns(5)
    ui.kpi(k[0], "Total officers", f"{total_officers}")
    ui.kpi(k[1], "Zones covered", f"{len(alloc)}")
    ui.kpi(k[2], "Max per zone", f"{alloc['allocated_officers'].max()}")
    ui.kpi(k[3], "Min per zone", f"{alloc['allocated_officers'].min()}")
    ui.kpi(k[4], "Avg per zone", f"{alloc['allocated_officers'].mean():.1f}")

    # Allocation chart + table
    st.markdown("---")
    al, ar = st.columns([2, 1], gap="large")

    with al:
        st.subheader("Officer Allocation by Zone")
        st.caption("Bar width proportional to allocated officers. Colour = impact score.")

        show_alloc = alloc.head(20).copy()
        show_alloc["label_short"] = show_alloc["label"].str.split(" - ").str[-1].str[:25]

        fig = go.Figure(go.Bar(
            x=show_alloc["allocated_officers"],
            y=show_alloc["label_short"],
            orientation="h",
            marker_color=[f"rgb({','.join(map(str, ui.impact_color(s)))})"
                          for s in show_alloc["impact_score"]],
            text=[f"{v} officers" for v in show_alloc["allocated_officers"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Officers: %{x}<br>"
                          "Impact: %{customdata[0]:.0f}<br>"
                          "Violations: %{customdata[1]:,}<extra></extra>",
            customdata=show_alloc[["impact_score", "violations"]].values,
        ))
        fig.update_layout(
            height=max(350, len(show_alloc) * 28),
            margin=dict(l=0, r=60, t=10, b=0),
            xaxis_title="Allocated Officers",
            yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, width="stretch")

    with ar:
        st.subheader("Allocation Summary")
        st.markdown(
            f"**{total_officers}** officers distributed across **{len(alloc)} zones** "
            f"proportional to their Congestion Impact Score.\n\n"
            f"The top zone receives **{alloc.iloc[0]['allocated_officers']}** officers "
            f"({alloc.iloc[0]['label'].split(' - ')[-1][:20]}, "
            f"impact {alloc.iloc[0]['impact_score']:.0f})."
        )

        # Officer share pie
        top5 = alloc.head(5).copy()
        top5["label_short"] = top5["label"].str.split(" - ").str[-1].str[:18]
        others = pd.DataFrame([{
            "label_short": "Others",
            "allocated_officers": alloc.iloc[5:]["allocated_officers"].sum()
        }])
        pie_data = pd.concat([top5[["label_short", "allocated_officers"]], others], ignore_index=True)
        fig = px.pie(pie_data, values="allocated_officers", names="label_short",
                     hole=0.45, color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    # Allocation table
    st.markdown("---")
    with st.expander("Full allocation table", expanded=False):
        show_tbl = alloc.copy()
        show_tbl["officer_share"] = (show_tbl["officer_share"] * 100).round(1)
        show_tbl.columns = ["Rank", "Zone", "Violations", "Impact",
                             "Share %", "Officers"]
        st.dataframe(show_tbl, hide_index=True, width="stretch", height=450)

        st.download_button("Download allocation (CSV)",
                           alloc.to_csv(index=False).encode(),
                           file_name="parksensei_officer_allocation.csv",
                           mime="text/csv")

    # ===================== Patrol Route Optimizer =====================
    st.markdown("---")
    st.subheader("Optimized Patrol Route")
    st.caption(f"Nearest-neighbor TSP route through top {route_stops} hotspots — minimizes total patrol distance.")

    with st.spinner("Computing optimal route…"):
        route_result = core.optimal_patrol_route(zones, top_n=route_stops)

    route_df = route_result["route"]
    total_dist = route_result["total_distance_km"]

    # Route KPIs
    rk = st.columns(4)
    ui.kpi(rk[0], "Route stops", f"{route_result['stops']}")
    ui.kpi(rk[1], "Total distance", f"{total_dist:.1f} km")
    ui.kpi(rk[2], "Avg leg", f"{total_dist / max(route_result['stops'] - 1, 1):.1f} km")
    ui.kpi(rk[3], "Longest leg", f"{route_df['leg_distance_km'].max():.1f} km")

    # Route map
    rl, rr = st.columns([2, 1], gap="large")

    with rl:
        st.markdown("**Route Map**")
        st.caption("Blue line = optimized patrol path connecting hotspots in order.")

        # Build path data for PathLayer
        path_coords = [[float(row["lon"]), float(row["lat"])] for _, row in route_df.iterrows()]
        path_data = pd.DataFrame([{"path": path_coords, "name": "Patrol Route"}])

        path_layer = pdk.Layer(
            "PathLayer",
            data=path_data,
            get_path="path",
            get_color=[76, 139, 245, 200],
            width_min_pixels=3,
            get_width=5,
            pickable=False,
        )

        # Stop markers
        route_dots = route_df.copy()
        route_dots["color"] = route_dots["impact_score"].map(ui.impact_color) if "impact_score" in route_dots.columns else [[76, 139, 245]] * len(route_dots)

        stop_layer = pdk.Layer(
            "ScatterplotLayer", data=route_dots, get_position=["lon", "lat"],
            get_radius=200, get_fill_color="color",
            opacity=0.9, stroked=True, get_line_color=[255, 255, 255],
            line_width_min_pixels=2, pickable=True, auto_highlight=True,
        )

        stop_text = pdk.Layer(
            "TextLayer", data=route_dots, get_position=["lon", "lat"],
            get_text="stop", get_size=12, get_color=[255, 255, 255],
            get_pixel_offset=[0, -22], get_alignment_baseline="'bottom'",
        )

        tip_route = {"html": "<b>{stop}</b> → {label}<br/>Impact: {impact_score}<br/>"
                             "Leg distance: {leg_distance_km} km",
                     "style": {"backgroundColor": "#161B26", "color": "#E6E9EF", "fontSize": "12px"}}

        center_lat = route_df["lat"].mean()
        center_lon = route_df["lon"].mean()
        st.pydeck_chart(
            ui.deck([path_layer, stop_layer, stop_text],
                    ui.view(center_lat, center_lon, zoom=11.2, pitch=35),
                    tip_route),
            width="stretch"
        )

    with rr:
        st.markdown("**Route Sequence**")
        for _, row in route_df.iterrows():
            zone_label = row.get("label", "Zone").split(" - ")[-1][:22]
            leg = row["leg_distance_km"]
            if leg > 0:
                st.markdown(f"↓ *{leg:.1f} km*")
            impact = row.get("impact_score", 0)
            badge = "🔴" if impact >= 70 else "🟠" if impact >= 50 else "🟡" if impact >= 30 else "🔵"
            st.markdown(f"**{row['stop']}** {badge} {zone_label}")

        st.markdown("---")
        st.metric("Total patrol distance", f"{total_dist:.1f} km")

    # Route table
    with st.expander("Full route details"):
        show_route = route_df.copy()
        cols_show = ["stop", "label", "impact_score", "violations", "leg_distance_km"]
        cols_available = [c for c in cols_show if c in show_route.columns]
        st.dataframe(show_route[cols_available], hide_index=True, width="stretch")


# =====================================================================
# TAB 3: NEXT DAY FORECAST
# =====================================================================
def _render_nextday_forecast():
    st.markdown("## Next-Day Violation Forecast")
    st.caption("7-day ahead predictions per junction — RandomForest / GradientBoosting trained on BTP historical data.")

    with st.spinner("Training models and generating forecasts…"):
        forecast = ui.get_nextday_forecast()

    if isinstance(forecast, dict) and "error" in forecast:
        st.error(f"Forecast unavailable: {forecast['error']}")
        st.info("Ensure the dataset contains `junction_name` and sufficient temporal data (20+ days per junction).")
        return

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


# =====================================================================
# RENDER TABS
# =====================================================================
tab1, tab2, tab3 = st.tabs(["📡 Forecast & Patrol", "👮 Officer Allocation", "📅 Next Day Forecast"])

with tab1:
    _render_forecast_patrol()
with tab2:
    _render_officer_allocation()
with tab3:
    _render_nextday_forecast()
