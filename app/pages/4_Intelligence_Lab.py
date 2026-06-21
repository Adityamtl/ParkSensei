"""Intelligence Lab - ML Impact Analysis, What-If Simulator, Alerts."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import ui, core

ui.page("Intelligence Lab", "I")
ui.brand_sidebar()


# =====================================================================
# TAB 1: ML IMPACT ANALYSIS
# =====================================================================
def _render_ml_impact():
    df = ui.load_data()

    st.markdown("## ML Impact Analysis")
    st.caption("Model performance metrics, DBSCAN cluster quality, and predictive feature importance — from ParkSight AI.")

    # ---------------- DBSCAN clustering stats ----------------
    st.subheader("DBSCAN Spatial Clustering")
    st.caption("300-metre radius, haversine metric — identifies tight violation hotspot clusters from GPS coordinates.")

    with st.spinner("Running DBSCAN clustering and quality evaluation…"):
        clusters = ui.get_dbscan_clusters()
        quality = ui.get_cluster_quality()

    if clusters is not None and not clusters.empty:
        k = st.columns(6)
        ui.kpi(k[0], "DBSCAN Clusters", f"{quality.get('n_clusters', 0)}")
        ui.kpi(k[1], "Noise Points", f"{quality.get('n_noise', 0):,}",
               help=f"{quality.get('noise_ratio', 0)*100:.1f}% of points are noise")
        ui.kpi(k[2], "Silhouette Score",
               f"{quality.get('silhouette_score', 'N/A')}",
               help=f"Interpretation: {quality.get('silhouette_interp', 'N/A')} — higher = better separated clusters")
        ui.kpi(k[3], "Davies-Bouldin",
               f"{quality.get('davies_bouldin_score', 'N/A')}",
               help=f"Interpretation: {quality.get('davies_bouldin_interp', 'N/A')} — lower = better separation")
        ui.kpi(k[4], "Calinski-Harabasz",
               f"{quality.get('calinski_harabasz_score', 'N/A')}",
               help="Higher = more distinct, compact clusters")
        ui.kpi(k[5], "Clustered Points", f"{quality.get('n_clustered', 0):,}")

        # Cluster map + impact score distribution
        st.markdown("---")
        cl, cr = st.columns([2, 1], gap="large")

        with cl:
            st.markdown("**DBSCAN Cluster Spatial Distribution**")
            st.caption("Bubble size = violation count, colour = impact score. Top 6 clusters annotated.")

            cdf = clusters.copy()
            cdf["radius"] = (np.sqrt(cdf["violation_count"]) * 5).clip(60, 900)
            cdf["color"] = cdf["impact_score"].apply(
                lambda s: ui.impact_color(s * 10))  # scale 0-10 to 0-100 for color
            cdf.rename(columns={"centroid_lat": "lat", "centroid_lon": "lon"}, inplace=True)

            scatter = pdk.Layer(
                "ScatterplotLayer", data=cdf, get_position=["lon", "lat"],
                get_radius="radius", get_fill_color="color",
                opacity=0.7, stroked=True, get_line_color=[255, 255, 255, 60],
                line_width_min_pixels=0.5, pickable=True, auto_highlight=True)

            tip = {"html": "<b>{dominant_junction}</b><br/>Violations: {violation_count}<br/>"
                           "Impact: {impact_score}<br/>Station: {dominant_station}",
                   "style": {"backgroundColor": "#161B26", "color": "#E6E9EF", "fontSize": "12px"}}

            st.pydeck_chart(ui.deck([scatter], ui.view(zoom=10.6, pitch=35), tip), width="stretch")

        with cr:
            st.markdown("**Impact Score Distribution**")
            fig = px.histogram(clusters, x="impact_score", nbins=20,
                              color_discrete_sequence=[ui.ACCENT])
            mean_score = clusters["impact_score"].mean()
            fig.add_vline(x=mean_score, line_dash="dash", line_color="#F5CD5A",
                         annotation_text=f"Mean: {mean_score:.1f}")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                              xaxis_title="Impact Score (0-10)",
                              yaxis_title="Number of clusters",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

            # Score components scatter
            st.markdown("**Peak vs Severity (coloured by impact)**")
            fig2 = px.scatter(clusters, x="peak_hour_ratio", y="high_severity_ratio",
                             color="impact_score", size="violation_count",
                             color_continuous_scale="RdYlGn_r",
                             hover_data=["dominant_junction"])
            fig2.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                               xaxis_title="Peak Hour Ratio",
                               yaxis_title="High Severity Ratio",
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig2, width="stretch")

        # Top clusters table
        st.markdown("---")
        st.markdown("**Top DBSCAN Clusters by Impact Score**")
        show = clusters.head(15)[[
            "dominant_junction", "violation_count", "impact_score",
            "peak_hour_ratio", "high_severity_ratio", "dominant_station",
            "dominant_vehicle", "top_violation"
        ]].copy()
        show.columns = ["Zone", "Violations", "Impact", "Peak%", "High Sev%",
                         "Station", "Vehicle", "Top Offence"]
        show["Peak%"] = (show["Peak%"] * 100).round(1)
        show["High Sev%"] = (show["High Sev%"] * 100).round(1)
        st.dataframe(show, hide_index=True, width="stretch", height=440)
    else:
        st.warning("DBSCAN clustering returned no clusters. Check that scikit-learn is installed: `pip install scikit-learn`")

    # ---------------- Congestion Probability Model ----------------
    st.markdown("---")
    st.subheader("Congestion Probability Model")
    st.caption("Random Forest classifier — predicts binary congestion risk from violation patterns.")

    with st.spinner("Training congestion model…"):
        cong = ui.get_congestion_model()

    if "error" not in cong:
        m1, m2, m3, m4 = st.columns(4)
        ui.kpi(m1, "ROC-AUC Score", f"{cong['roc_auc']:.3f}",
               help="Area under receiver operating characteristic curve")
        ui.kpi(m2, "F1 Score", f"{cong['f1_score']:.3f}",
               help="Harmonic mean of precision and recall")
        ui.kpi(m3, "Accuracy", f"{cong['accuracy']*100:.1f}%")
        ui.kpi(m4, "High-congestion threshold", f"\u2265{cong['threshold']} viol/day")

        # Feature importance
        if cong.get("feature_importance"):
            st.markdown("**Congestion Model Feature Importance**")
            fi = cong["feature_importance"]
            fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
            fi_df = fi_df.sort_values("importance", ascending=True)
            fig = go.Figure(go.Bar(
                x=fi_df["importance"], y=fi_df["feature"], orientation="h",
                marker_color=[ui.ACCENT if v > 0.15 else "#3a4254" for v in fi_df["importance"]],
                text=[f"{v:.3f}" for v in fi_df["importance"]],
                textposition="outside"
            ))
            fig.update_layout(height=250, margin=dict(l=0, r=50, t=10, b=0),
                              xaxis_title="Feature Importance",
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")
    else:
        st.info(f"Congestion model: {cong.get('error', 'Unknown error')}")

    # ---------------- Next-Day Model Performance ----------------
    st.markdown("---")
    st.subheader("Next-Day Prediction Model Performance")
    st.caption("Per-junction RandomForest / GradientBoosting models trained with TimeSeriesSplit cross-validation.")

    with st.spinner("Loading next-day model results…"):
        forecast = ui.get_nextday_forecast()

    if isinstance(forecast, dict) and "error" not in forecast:
        jr = forecast.get("junction_results", {})
        cw = forecast.get("city_wide", {})

        if jr:
            rows = []
            for junc, r in jr.items():
                rows.append({
                    "Junction": r["short"],
                    "Model": r["best_model"],
                    "MAE": r["test_mae"],
                    "RMSE": r["test_rmse"],
                    "R\u00b2": r["test_r2"],
                    "Accuracy": f"{r['accuracy']}%",
                    "Days": r["n_days"],
                })
            # Add city-wide
            rows.append({
                "Junction": "City-wide",
                "Model": "RandomForest",
                "MAE": cw.get("mae", None),
                "RMSE": None,
                "R\u00b2": cw.get("r2", None),
                "Accuracy": f"{cw.get('accuracy', 0)}%",
                "Days": None,
            })
            results_df = pd.DataFrame(rows)
            # Ensure numeric columns are proper floats (not mixed str/float)
            for col in ["MAE", "RMSE", "R\u00b2"]:
                results_df[col] = pd.to_numeric(results_df[col], errors="coerce")
            results_df["Days"] = pd.to_numeric(results_df["Days"], errors="coerce").astype("Int64")
            st.dataframe(results_df, hide_index=True, width="stretch")

        # Feature importance from top junction
        junctions = forecast.get("junctions", {})
        if junctions:
            first_junc = next(iter(junctions.values()))
            fi = first_junc.get("featureImportance", {})
            if fi:
                st.markdown(f"**Top Predictive Features \u2014 {first_junc['shortName']}**")
                fi_df = pd.DataFrame({"feature": list(fi.keys()), "importance": list(fi.values())})
                fi_df = fi_df.sort_values("importance", ascending=True)

                fig = go.Figure(go.Bar(
                    x=fi_df["importance"], y=fi_df["feature"], orientation="h",
                    marker_color=ui.ACCENT,
                    text=[f"{v:.3f}" for v in fi_df["importance"]],
                    textposition="outside"
                ))
                fig.update_layout(height=300, margin=dict(l=0, r=50, t=10, b=0),
                                  xaxis_title="Feature Importance",
                                  paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig, width="stretch")
    else:
        error_msg = forecast.get("error", "Unknown error") if isinstance(forecast, dict) else "Model training failed"
        st.info(f"Next-day model: {error_msg}")

    with st.expander("About the ML pipeline"):
        st.markdown(
            "**DBSCAN Clustering:** eps=300m, min_samples=30, haversine metric. "
            "Groups violations into tight spatial hotspots.\n\n"
            "**Impact Score:** `(log_norm_count \u00d7 0.40 + high_severity_ratio \u00d7 0.35 + peak_hour_ratio \u00d7 0.25) \u00d7 10`\n\n"
            "**Congestion Model:** Random Forest classifier predicting binary high-congestion risk "
            "from severity, peak ratios, and temporal features.\n\n"
            "**Next-Day Prediction:** Per-junction RF/GBT with 20 features including "
            "lags (1,2,3,7d), rolling averages (3,7,14d), rolling std 7d, DOW/month historical averages, "
            "and 7-day trend slope. Validated with TimeSeriesSplit (5 folds)."
        )


# =====================================================================
# TAB 2: WHAT-IF SIMULATOR
# =====================================================================
def _render_whatif_simulator():
    zones = ui.get_zones()

    st.markdown("## What-If Enforcement Simulator")
    st.caption("Estimate the impact of adding officers to specific zones \u2014 risk, congestion, and propagation reduction.")

    # ===================== SINGLE ZONE SIMULATION =====================
    st.subheader("Single Zone Simulation")

    zone_labels = zones.head(30)["label"].tolist()
    c1, c2, _ = st.columns([2, 1, 1])
    selected_zone = c1.selectbox("Target zone", zone_labels, index=0)
    additional = c2.slider("Additional officers", 1, 25, 5)

    target_idx = zones[zones["label"] == selected_zone].index[0]
    sim = core.what_if_simulation(zones, target_idx=target_idx, additional_officers=additional)

    if "error" in sim:
        st.error(sim["error"])
        return

    # Before / After gauges
    st.markdown("---")
    g1, g2, g3, g4 = st.columns(4)

    with g1:
        st.markdown("""
        <div class="rec-card" style="text-align:center">
            <span style="color:#606878;font-size:0.75rem;text-transform:uppercase">Current Impact</span>
            <div style="font-size:2rem;font-weight:700;color:#EF4444">{:.0f}</div>
            <span style="color:#606878;font-size:0.75rem">{}</span>
        </div>
        """.format(sim["current_impact"], sim["zone_label"].split(" - ")[-1][:20]),
        unsafe_allow_html=True)

    with g2:
        new_color = "#10B981" if sim["new_impact"] < 50 else "#F59E0B" if sim["new_impact"] < 70 else "#EF4444"
        st.markdown("""
        <div class="rec-card" style="text-align:center">
            <span style="color:#606878;font-size:0.75rem;text-transform:uppercase">New Impact</span>
            <div style="font-size:2rem;font-weight:700;color:{color}">{score:.0f}</div>
            <span style="color:#10B981;font-size:0.85rem;font-weight:600">\u25bc {reduction:.0f} pts ({pct:.0f}%)</span>
        </div>
        """.format(color=new_color, score=sim["new_impact"],
                   reduction=sim["impact_reduction"], pct=sim["impact_reduction_pct"]),
        unsafe_allow_html=True)

    with g3:
        st.markdown("""
        <div class="rec-card" style="text-align:center">
            <span style="color:#606878;font-size:0.75rem;text-transform:uppercase">Congestion Reduction</span>
            <div style="font-size:2rem;font-weight:700;color:#06B6D4">{:.1f}%</div>
            <span style="color:#606878;font-size:0.75rem">Estimated flow improvement</span>
        </div>
        """.format(sim["congestion_reduction"]),
        unsafe_allow_html=True)

    with g4:
        st.markdown("""
        <div class="rec-card" style="text-align:center">
            <span style="color:#606878;font-size:0.75rem;text-transform:uppercase">Propagation Reduction</span>
            <div style="font-size:2rem;font-weight:700;color:#8B5CF6">{:.1f}%</div>
            <span style="color:#606878;font-size:0.75rem">Spill-over risk decrease</span>
        </div>
        """.format(sim["propagation_reduction"]),
        unsafe_allow_html=True)

    # Visual before/after comparison
    st.markdown("---")
    bl, br = st.columns([2, 1], gap="large")

    with bl:
        st.markdown("**Impact Score Gauge \u2014 Before vs After**")

        fig = go.Figure()
        fig.add_trace(go.Indicator(
            mode="gauge+number+delta",
            value=sim["new_impact"],
            delta={"reference": sim["current_impact"], "decreasing": {"color": "#10B981"}},
            gauge={
                "axis": {"range": [0, 100], "tickwidth": 1},
                "bar": {"color": "#4C8BF5"},
                "steps": [
                    {"range": [0, 35], "color": "#1A3A1A"},
                    {"range": [35, 70], "color": "#3A2A1A"},
                    {"range": [70, 100], "color": "#3A1A1A"},
                ],
                "threshold": {
                    "line": {"color": "#EF4444", "width": 3},
                    "thickness": 0.8,
                    "value": sim["current_impact"],
                },
            },
            title={"text": f"Impact Score \u2014 {sim['zone_label'].split(' - ')[-1][:22]}"},
            domain={"x": [0, 1], "y": [0, 1]},
        ))
        fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=10),
                          paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    with br:
        st.markdown("**Zone Details**")
        st.metric("Violations", f"{sim['violations']:,}")
        st.metric("Avg severity", f"{sim['avg_severity']:.2f}")
        st.metric("Junction exposure", f"{sim['junction_frac']:.0f}%")
        st.metric("Officers added", f"+{sim['additional_officers']}")

        recs = core.generate_recommendations(zones.iloc[target_idx])
        if recs:
            st.markdown("**Top recommendation:**")
            ui.render_recommendation_card(recs[0])

    # ===================== MULTI-ZONE COMPARISON =====================
    st.markdown("---")
    st.subheader("Multi-Zone Strategy Comparison")
    st.caption("Compare the effect of adding officers across multiple zones to find the best investment.")

    mc1, mc2, _ = st.columns([2, 1, 1])
    num_zones = mc1.slider("Compare top N zones", 5, 20, 10)
    officers_each = mc2.slider("Officers per zone", 1, 20, 5, key="multi_officers")

    target_indices = list(range(num_zones))
    batch = core.what_if_batch(zones, target_indices, officers_each)
    batch_df = pd.DataFrame(batch)

    batch_df = batch_df.sort_values("impact_reduction", ascending=False).reset_index(drop=True)
    batch_df["label_short"] = batch_df["zone_label"].str.split(" - ").str[-1].str[:22]

    # Comparison chart
    comp_l, comp_r = st.columns([2, 1], gap="large")

    with comp_l:
        st.markdown("**Impact Reduction per Zone**")
        st.caption(f"Adding {officers_each} officers to each zone. Sorted by effectiveness.")

        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=batch_df["label_short"],
            x=batch_df["current_impact"],
            name="Current Impact",
            orientation="h",
            marker_color="rgba(239, 68, 68, 0.3)",
        ))
        fig.add_trace(go.Bar(
            y=batch_df["label_short"],
            x=batch_df["new_impact"],
            name="New Impact",
            orientation="h",
            marker_color="#4C8BF5",
        ))
        fig.update_layout(
            height=max(350, len(batch_df) * 30),
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_title="Impact Score",
            barmode="overlay",
            yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            legend=dict(orientation="h", y=-0.1),
        )
        st.plotly_chart(fig, width="stretch")

    with comp_r:
        st.markdown("**Best Investment Zones**")
        st.caption("Zones where adding officers has the highest effect.")

        top3 = batch_df.head(3)
        for _, row in top3.iterrows():
            pct = row["impact_reduction_pct"]
            badge = "\U0001f7e2" if pct >= 15 else "\U0001f7e1" if pct >= 8 else "\U0001f535"
            st.markdown(
                f"{badge} **{row['label_short']}** \u2014 "
                f"{row['impact_reduction']:.0f} pt reduction "
                f"({pct:.0f}%)"
            )

        st.markdown("---")
        st.markdown("**Lowest ROI Zones**")
        bottom3 = batch_df.tail(3)
        for _, row in bottom3.iterrows():
            st.markdown(
                f"\u26aa **{row['label_short']}** \u2014 "
                f"{row['impact_reduction']:.0f} pt "
                f"({row['impact_reduction_pct']:.0f}%)"
            )

    # Comparison table
    with st.expander("Full simulation results"):
        show_batch = batch_df[["zone_label", "current_impact", "new_impact",
                                "impact_reduction", "impact_reduction_pct",
                                "congestion_reduction", "propagation_reduction",
                                "violations"]].copy()
        show_batch.columns = ["Zone", "Current", "After", "\u0394 Impact",
                               "\u0394 %", "Congest. \u0394%", "Propag. \u0394%", "Violations"]
        st.dataframe(show_batch, hide_index=True, width="stretch", height=400)

        st.download_button("Download simulation results (CSV)",
                           batch_df.to_csv(index=False).encode(),
                           file_name="parksensei_whatif_simulation.csv",
                           mime="text/csv")

    with st.expander("How the simulator works"):
        st.markdown(
            "**Formula (from GRIDLOCK2.0 Prototype):**\n\n"
            "- Each additional officer reduces the zone's impact score by **2 points**\n"
            "- Congestion reduction = impact change \u00d7 **0.8** (80% of risk reduction flows to traffic)\n"
            "- Propagation reduction = impact change \u00d7 **0.6** (60% reduces spill-over to neighbours)\n\n"
            "**Limitations:**\n"
            "- Linear estimation \u2014 real-world effects are non-linear with diminishing returns\n"
            "- Does not account for officer fatigue, shift patterns, or inter-zone coordination\n"
            "- Best used for comparative analysis (which zones benefit most) rather than absolute numbers"
        )


# =====================================================================
# TAB 3: ALERTS
# =====================================================================
def _render_alerts():
    st.markdown("## Active Alerts")
    st.caption("Real-time violation zone notifications derived from ML predictions and DBSCAN cluster analysis.")

    # ---------------- generate alerts from ML models ----------------
    def _generate_alerts():
        alerts = []
        alert_id = 0

        # 1. Alerts from next-day forecast (HIGH risk days)
        try:
            forecast = ui.get_nextday_forecast()
            if isinstance(forecast, dict) and "error" not in forecast:
                junctions = forecast.get("junctions", {})
                for junc_name, junc_data in junctions.items():
                    for d in junc_data["days"]:
                        if d["risk"] == "HIGH":
                            alert_id += 1
                            alerts.append({
                                "id": alert_id,
                                "type": "critical",
                                "zone": f"{junc_data['shortName']} \u2014 {d['dayName']} {d['date']}",
                                "message": f"Predicted {d['predicted']} violations "
                                          f"(CI: {d['ciLow']}\u2013{d['ciHigh']}). "
                                          f"{d['recommendation']}",
                                "time": f"Forecast for {d['date']}",
                                "source": "Next-Day RF Model",
                            })
                        elif d["risk"] == "MEDIUM":
                            alert_id += 1
                            alerts.append({
                                "id": alert_id,
                                "type": "medium",
                                "zone": f"{junc_data['shortName']} \u2014 {d['dayName']} {d['date']}",
                                "message": f"Predicted {d['predicted']} violations. "
                                          f"{d['recommendation']}",
                                "time": f"Forecast for {d['date']}",
                                "source": "Next-Day RF Model",
                            })

                # City-wide HIGH alerts
                cw = forecast.get("city_wide", {})
                for d in cw.get("forecast", []):
                    if d["risk"] == "HIGH":
                        alert_id += 1
                        alerts.append({
                            "id": alert_id,
                            "type": "high",
                            "zone": f"City-wide \u2014 {d['dayName']} {d['date']}",
                            "message": f"City-wide predicted {d['predicted']:,} violations \u2014 "
                                      f"above high-congestion threshold.",
                            "time": f"Forecast for {d['date']}",
                            "source": "City-wide RF Model",
                        })
        except Exception:
            pass

        # 2. Alerts from DBSCAN mega-clusters
        try:
            clusters = ui.get_dbscan_clusters()
            if clusters is not None and not clusters.empty:
                mega = clusters[clusters["violation_count"] > 10000]
                for _, c in mega.iterrows():
                    alert_id += 1
                    alerts.append({
                        "id": alert_id,
                        "type": "high",
                        "zone": c["dominant_junction"],
                        "message": f"Mega-cluster: {c['violation_count']:,} violations in 300m radius. "
                                  f"Impact score {c['impact_score']:.1f}. "
                                  f"Consider permanent tow-away zone.",
                        "time": "Historical analysis",
                        "source": "DBSCAN Clustering",
                    })

                # High-impact clusters
                high_impact = clusters[
                    (clusters["impact_score"] >= 7) & (clusters["violation_count"] <= 10000)
                ].head(5)
                for _, c in high_impact.iterrows():
                    alert_id += 1
                    alerts.append({
                        "id": alert_id,
                        "type": "medium",
                        "zone": c["dominant_junction"],
                        "message": f"High-impact cluster: {c['violation_count']:,} violations, "
                                  f"impact {c['impact_score']:.1f}. "
                                  f"Peak-hour ratio: {c['peak_hour_ratio']*100:.0f}%.",
                        "time": "Historical analysis",
                        "source": "DBSCAN Clustering",
                    })
        except Exception:
            pass

        # 3. Alerts from congestion model
        try:
            cong = ui.get_congestion_model()
            if isinstance(cong, dict) and "error" not in cong:
                if cong.get("f1_score", 0) < 0.6:
                    alert_id += 1
                    alerts.append({
                        "id": alert_id,
                        "type": "low",
                        "zone": "Congestion Model",
                        "message": f"Model F1 score is {cong['f1_score']:.2f} \u2014 consider retraining "
                                  f"with more recent data or additional features.",
                        "time": "Model diagnostic",
                        "source": "Congestion RF Classifier",
                    })
        except Exception:
            pass

        return alerts

    # Priority ordering
    PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    PRIORITY_CONFIG = {
        "critical": {"icon": "\U0001f534", "label": "CRITICAL", "color": "#E2352B"},
        "high":     {"icon": "\U0001f7e0", "label": "HIGH",     "color": "#E8923E"},
        "medium":   {"icon": "\U0001f7e1", "label": "MEDIUM",   "color": "#F5CD5A"},
        "low":      {"icon": "\U0001f535", "label": "LOW",      "color": "#4C8BF5"},
    }

    # Session state for dismissed alerts
    if "dismissed_alerts" not in st.session_state:
        st.session_state.dismissed_alerts = set()

    with st.spinner("Analysing predictions for alerts\u2026"):
        all_alerts = _generate_alerts()

    # Filter out dismissed
    active_alerts = [a for a in all_alerts if a["id"] not in st.session_state.dismissed_alerts]
    active_alerts.sort(key=lambda a: PRIORITY_ORDER.get(a["type"], 4))

    # KPIs
    k = st.columns(5)
    ui.kpi(k[0], "Active alerts", f"{len(active_alerts)}")
    ui.kpi(k[1], "Critical", f"{sum(1 for a in active_alerts if a['type'] == 'critical')}")
    ui.kpi(k[2], "High", f"{sum(1 for a in active_alerts if a['type'] == 'high')}")
    ui.kpi(k[3], "Medium", f"{sum(1 for a in active_alerts if a['type'] == 'medium')}")
    ui.kpi(k[4], "Dismissed", f"{len(st.session_state.dismissed_alerts)}")

    st.markdown("---")

    # Clear all button
    if active_alerts:
        c1, c2, _ = st.columns([1, 1, 4])
        with c1:
            if st.button("Clear all alerts", type="secondary"):
                st.session_state.dismissed_alerts.update(a["id"] for a in all_alerts)
                st.rerun()
        with c2:
            if st.session_state.dismissed_alerts:
                if st.button("Reset dismissed"):
                    st.session_state.dismissed_alerts.clear()
                    st.rerun()

    # Alert cards
    if not active_alerts:
        st.markdown(
            "<div style='text-align:center;padding:60px 0'>"
            "<div style='font-size:2.5rem;margin-bottom:12px'>\u2705</div>"
            "<div style='font-size:1rem;font-weight:600;color:#8A93A6'>No active alerts</div>"
            "<div style='font-size:0.8rem;color:#606878;margin-top:4px'>All zones operating normally</div>"
            "</div>",
            unsafe_allow_html=True
        )
    else:
        for alert in active_alerts:
            config = PRIORITY_CONFIG.get(alert["type"], PRIORITY_CONFIG["low"])

            col_alert, col_dismiss = st.columns([20, 1])
            with col_alert:
                st.markdown(f"""
                <div class="rec-card">
                    <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                        <span class="rec-badge rec-badge-{alert['type'].upper()}"
                              style="background:{config['color']}">{config['label']}</span>
                        <strong>{alert['zone']}</strong>
                    </div>
                    <div style="color:#8A93A6;font-size:0.85rem">{alert['message']}</div>
                    <div style="display:flex;justify-content:space-between;margin-top:8px">
                        <span style="color:#606878;font-size:0.75rem">{alert['time']}</span>
                        <span style="color:#606878;font-size:0.75rem">Source: {alert['source']}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            with col_dismiss:
                if st.button("\u2715", key=f"dismiss_{alert['id']}", help="Dismiss this alert"):
                    st.session_state.dismissed_alerts.add(alert["id"])
                    st.rerun()


# =====================================================================
# RENDER TABS
# =====================================================================
tab1, tab2, tab3 = st.tabs(["\U0001f916 ML Impact Analysis", "\U0001f3ae What-If Simulator", "\U0001f514 Alerts"])

with tab1:
    _render_ml_impact()
with tab2:
    _render_whatif_simulator()
with tab3:
    _render_alerts()
