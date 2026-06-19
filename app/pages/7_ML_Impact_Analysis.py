"""ML Impact Analysis — Model performance metrics, cluster quality, and feature importance.
   Adapted from ParkSight AI's Impact Analysis page."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import ui, core

ui.page("ML Impact Analysis", "M")
ui.brand_sidebar()

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

        import pydeck as pdk
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
    ui.kpi(m4, "High-congestion threshold", f"≥{cong['threshold']} viol/day")

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
                "R²": r["test_r2"],
                "Accuracy": f"{r['accuracy']}%",
                "Days": r["n_days"],
            })
        # Add city-wide
        rows.append({
            "Junction": "City-wide",
            "Model": "RandomForest",
            "MAE": cw.get("mae", "—"),
            "RMSE": "—",
            "R²": cw.get("r2", "—"),
            "Accuracy": f"{cw.get('accuracy', 0)}%",
            "Days": "—",
        })
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

    # Feature importance from top junction
    junctions = forecast.get("junctions", {})
    if junctions:
        first_junc = next(iter(junctions.values()))
        fi = first_junc.get("featureImportance", {})
        if fi:
            st.markdown(f"**Top Predictive Features — {first_junc['shortName']}**")
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
        "**Impact Score:** `(log_norm_count × 0.40 + high_severity_ratio × 0.35 + peak_hour_ratio × 0.25) × 10`\n\n"
        "**Congestion Model:** Random Forest classifier predicting binary high-congestion risk "
        "from severity, peak ratios, and temporal features.\n\n"
        "**Next-Day Prediction:** Per-junction RF/GBT with 20 features including "
        "lags (1,2,3,7d), rolling averages (3,7,14d), DOW/month historical averages, "
        "and 7-day trend slope. Validated with TimeSeriesSplit (5 folds)."
    )
