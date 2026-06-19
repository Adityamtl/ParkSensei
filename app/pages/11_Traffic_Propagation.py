"""Traffic Propagation & Blast Radius — Haversine-based pairwise hotspot proximity analysis.
   Adapted from GRIDLOCK2.0 Prototype script 09_traffic_propagation.py."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import ui, core

ui.page("Traffic Propagation", "T")
ui.brand_sidebar()

st.markdown("## Traffic Propagation & Blast Radius")
st.caption("Identify which violation hotspots affect each other — haversine proximity within 2 km radius.")

zones = ui.get_zones()

# Controls
c1, c2, _ = st.columns([1, 1, 2])
radius = c1.slider("Propagation radius (km)", 0.5, 4.0, 2.0, 0.5)
top_n = c2.slider("Analyse top N zones", 10, 50, 30, 5)

with st.spinner("Computing propagation network…"):
    prop = core.traffic_propagation(zones.head(top_n), radius_km=radius)

if prop.empty:
    st.info("No propagation links found at this radius. Try increasing the radius or number of zones.")
    st.stop()

# KPIs
risk_counts = prop["propagation_risk"].value_counts()
k = st.columns(5)
ui.kpi(k[0], "Propagation links", f"{len(prop):,}")
ui.kpi(k[1], "Very High risk", f"{risk_counts.get('Very High', 0)}", help="Zones within 500m")
ui.kpi(k[2], "High risk", f"{risk_counts.get('High', 0)}", help="Zones within 0.5–1 km")
ui.kpi(k[3], "Avg distance", f"{prop['distance_km'].mean():.2f} km")
ui.kpi(k[4], "Closest pair", f"{prop['distance_km'].min():.3f} km")

# Propagation network map
st.markdown("---")
ml, mr = st.columns([2, 1], gap="large")

with ml:
    st.subheader("Propagation Network")
    st.caption("Arcs connect hotspots within propagation radius. Colour = risk level (red = Very High).")

    risk_colors = {
        "Very High": [226, 53, 43, 200],
        "High":      [245, 158, 65, 180],
        "Medium":    [245, 205, 90, 140],
        "Low":       [76, 139, 245, 100],
    }
    prop_map = prop.copy()
    prop_map["color"] = prop_map["propagation_risk"].map(risk_colors)

    arc_layer = pdk.Layer(
        "ArcLayer",
        data=prop_map,
        get_source_position=["source_lon", "source_lat"],
        get_target_position=["affected_lon", "affected_lat"],
        get_source_color="color",
        get_target_color="color",
        get_width=2,
        pickable=True,
        auto_highlight=True,
    )

    # Zone dots
    zone_dots = zones.head(top_n).copy()
    zone_dots["color"] = zone_dots["impact_score"].map(ui.impact_color)
    zone_dots["radius"] = (np.sqrt(zone_dots["violations"]) * 5).clip(60, 600)

    scatter = pdk.Layer(
        "ScatterplotLayer", data=zone_dots, get_position=["lon", "lat"],
        get_radius="radius", get_fill_color="color",
        opacity=0.8, stroked=True, get_line_color=[255, 255, 255, 80],
        line_width_min_pixels=1, pickable=True,
    )

    tip = {"html": "<b>{source_zone}</b> → <b>{affected_zone}</b><br/>"
                   "Distance: {distance_km} km<br/>Risk: {propagation_risk}",
           "style": {"backgroundColor": "#161B26", "color": "#E6E9EF", "fontSize": "12px"}}

    st.pydeck_chart(ui.deck([arc_layer, scatter], ui.view(zoom=10.8, pitch=30), tip),
                    width="stretch")

with mr:
    st.subheader("Risk Breakdown")
    risk_df = risk_counts.reset_index()
    risk_df.columns = ["risk", "count"]
    risk_order = ["Very High", "High", "Medium", "Low"]
    risk_df["risk"] = pd.Categorical(risk_df["risk"], categories=risk_order, ordered=True)
    risk_df = risk_df.sort_values("risk")

    colors = {"Very High": "#E2352B", "High": "#E8923E", "Medium": "#F5CD5A", "Low": "#4C8BF5"}
    fig = px.bar(risk_df, x="risk", y="count", color="risk",
                 color_discrete_map=colors)
    fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                      showlegend=False, xaxis_title="", yaxis_title="Links",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

    # Most connected zones
    st.markdown("**Most connected zones**")
    src_counts = prop["source_zone"].value_counts().head(5)
    for zone_name, cnt in src_counts.items():
        st.markdown(f"- **{zone_name}**: {cnt} outbound links")

# Blast Radius Analysis
st.markdown("---")
st.subheader("Blast Radius Analysis")
st.caption("Select a hotspot to see all zones it affects at different distance rings.")

zone_names = zones.head(top_n)["label"].tolist()
selected = st.selectbox("Select zone", zone_names, index=0)
sel_row = zones[zones["label"] == selected].iloc[0]

# Find all links from this zone
blast = prop[prop["source_zone"] == selected].copy()

if blast.empty:
    st.info(f"No propagation links from **{selected}** at {radius} km radius.")
else:
    blast["ring"] = blast["distance_km"].apply(
        lambda d: "0–0.5 km" if d <= 0.5 else "0.5–1 km" if d <= 1.0 else "1–1.5 km" if d <= 1.5 else "1.5–2 km"
    )

    b1, b2 = st.columns([1, 1], gap="large")
    with b1:
        ring_counts = blast["ring"].value_counts().reindex(
            ["0–0.5 km", "0.5–1 km", "1–1.5 km", "1.5–2 km"]).fillna(0)
        ring_df = ring_counts.reset_index()
        ring_df.columns = ["ring", "zones_affected"]
        ring_colors = ["#E2352B", "#E8923E", "#F5CD5A", "#4C8BF5"]
        fig = px.bar(ring_df, x="ring", y="zones_affected",
                     color="ring", color_discrete_sequence=ring_colors)
        fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0),
                          showlegend=False, xaxis_title="Distance ring",
                          yaxis_title="Affected zones",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")

    with b2:
        st.metric("Total affected zones", len(blast))
        st.metric("Closest zone", f"{blast['distance_km'].min():.3f} km",
                   help=blast.iloc[0]["affected_zone"])
        st.metric("Farthest zone", f"{blast['distance_km'].max():.3f} km")
        st.metric("Zone impact score", f"{sel_row['impact_score']:.0f}")

    # Blast radius table
    show_blast = blast[["affected_zone", "distance_km", "propagation_risk",
                        "affected_impact"]].copy()
    show_blast.columns = ["Affected Zone", "Distance (km)", "Risk", "Impact Score"]
    st.dataframe(show_blast, hide_index=True, width="stretch", height=300)

# Top propagation risk table
st.markdown("---")
st.subheader("Top Propagation Links")
show = prop.head(20)[["source_zone", "affected_zone", "distance_km",
                       "propagation_risk", "source_impact", "affected_impact"]].copy()
show.columns = ["Source", "Affected", "Distance (km)", "Risk", "Src Impact", "Aff Impact"]
st.dataframe(show, hide_index=True, width="stretch", height=400)

st.download_button("Download propagation data (CSV)",
                   prop.to_csv(index=False).encode(),
                   file_name="parksensei_propagation.csv",
                   mime="text/csv")
