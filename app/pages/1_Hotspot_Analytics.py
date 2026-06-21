"""Hotspot Analytics - Hotspot Explorer, Parking DNA, Traffic Propagation."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import ui, core

ui.page("Hotspot Analytics", "H")
ui.brand_sidebar()

# =====================================================================
# TAB 1: HOTSPOT EXPLORER
# =====================================================================
def _render_hotspot_explorer():
    df = ui.load_data()

    st.markdown("## Hotspot Explorer")
    st.caption("Filter the 298K records and watch hotspots re-rank in real time. Scored with 7-factor impact breakdown.")

    # ---------------- filters ----------------
    f1, f2, f3, f4 = st.columns([1.1, 1.3, 1.3, 1.3])
    days = f1.multiselect("Weekday", ui.DOW_NAMES, default=ui.DOW_NAMES)
    hr   = f2.slider("Hour of day (IST)", 0, 23, (6, 13))
    vt_all = sorted(df["primary_type"].unique())
    vts  = f3.multiselect("Violation type", vt_all, default=vt_all)
    st_all = sorted(s for s in df["police_station"].unique() if s.lower() != "nan")
    sts  = f4.multiselect("Police station", st_all, default=[])

    m = (df["dow"].map(dict(enumerate(ui.DOW_NAMES))).isin(days)
         & df["hour"].between(hr[0], hr[1])
         & df["primary_type"].isin(vts))
    if sts:
        m &= df["police_station"].isin(sts)
    sub = df[m]

    if sub.empty:
        st.warning("No violations match these filters.")
        return

    zones = core.add_impact(core.build_zones(sub))

    # KPIs
    k = st.columns(5)
    ui.kpi(k[0], "Matching violations", f"{len(sub):,}", help=f"{len(sub)/len(df)*100:.1f}% of all records")
    ui.kpi(k[1], "Active zones", f"{zones.shape[0]:,}")
    ui.kpi(k[2], "Avg severity", f"{sub['severity'].mean():.2f}")
    avg_pcu_filtered = sub["pcu"].mean() if "pcu" in sub.columns else 1.0
    ui.kpi(k[3], "Avg PCU weight", f"{avg_pcu_filtered:.2f}")
    ui.kpi(k[4], "Top zone", zones.iloc[0]["label"].split(" - ")[-1][:18])

    # ---------------- map + table ----------------
    left, right = st.columns([2, 1], gap="large")
    with left:
        st.pydeck_chart(ui.deck([ui.zone_layer(zones)], ui.view(zoom=10.6, pitch=40), ui.TIP_ZONE),
                        width="stretch")
        st.caption("Bubble size = violations, colour = impact (blue to red)")
    with right:
        show_cols = ["label", "violations", "impact_score", "top_violation"]
        if "place_type" in zones.columns:
            show_cols.append("place_type")
        if "avg_pcu" in zones.columns:
            show_cols.append("avg_pcu")
        show = zones.head(15)[show_cols].copy()
        rename = {"label": "Zone", "violations": "Viol.", "impact_score": "Impact",
                  "top_violation": "Top type", "place_type": "Place", "avg_pcu": "PCU"}
        show.columns = [rename.get(c, c) for c in show.columns]
        st.dataframe(show, hide_index=True, width="stretch", height=430,
                     column_config={"Impact": st.column_config.ProgressColumn(
                         "Impact", min_value=0, max_value=100, format="%.0f")})

    # ---------------- impact breakdown for top zone ----------------
    st.markdown("---")
    st.subheader("Impact Score Breakdown — Top Zone")
    st.caption("Explainable 7-factor scoring: see exactly which factors drive the congestion impact.")

    top_zone = zones.iloc[0]
    bd = core.impact_breakdown(top_zone)

    bc1, bc2 = st.columns([2, 1], gap="large")
    with bc1:
        factors = list(bd.keys())
        weighted_vals = [bd[f]["weighted"] for f in factors]
        labels = [bd[f]["label"] for f in factors]

        fig = go.Figure(go.Bar(
            x=weighted_vals, y=labels, orientation="h",
            marker_color=[ui.ACCENT if v > 8 else "#3a4254" for v in weighted_vals],
            text=[f"{v:.1f}" for v in weighted_vals],
            textposition="outside"
        ))
        fig.update_layout(
            height=280, margin=dict(l=0, r=40, t=10, b=0),
            xaxis_title="Contribution to impact score (0–100)",
            yaxis={"categoryorder": "total ascending"},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig, width="stretch")

    with bc2:
        st.markdown(f"**{top_zone['label']}**")
        st.markdown(f"Impact score: **{top_zone['impact_score']:.0f}** / 100")
        if "place_type" in top_zone.index:
            st.markdown(f"Place type: **{top_zone['place_type']}**")
        st.markdown(f"Violations: **{int(top_zone['violations']):,}**")
        if "avg_pcu" in top_zone.index:
            st.markdown(f"Avg PCU: **{top_zone['avg_pcu']:.2f}**")
        if "peak_share" in top_zone.index:
            st.markdown(f"Peak-hour share: **{top_zone['peak_share']*100:.0f}%**")

        # Recommendations for this zone
        st.markdown("**Recommended actions:**")
        recs = core.generate_recommendations(top_zone)
        for r in recs[:3]:
            ui.render_recommendation_card(r)

    # ---------------- breakdowns ----------------
    st.markdown("---")
    b1, b2 = st.columns(2, gap="large")
    with b1:
        vc = sub["primary_type"].value_counts().head(8).reset_index()
        vc.columns = ["type", "n"]
        fig = px.bar(vc, x="n", y="type", orientation="h", color="n",
                     color_continuous_scale="Blues")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0), yaxis_title="",
                          xaxis_title="violations", coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          yaxis={"categoryorder": "total ascending"})
        st.markdown("**Violation mix**"); st.plotly_chart(fig, width="stretch")
    with b2:
        hv = sub.groupby("hour").size().reset_index(name="n")
        fig = px.area(hv, x="hour", y="n")
        fig.update_traces(line_color=ui.ACCENT, fillcolor="rgba(76,139,245,0.25)")
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="hour of day (IST)", yaxis_title="violations",
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.markdown("**Daily rhythm**"); st.plotly_chart(fig, width="stretch")

    # vehicle type distribution (new)
    if "pcu" in sub.columns:
        st.markdown("---")
        vt1, vt2 = st.columns(2, gap="large")
        with vt1:
            vt_dist = sub["vehicle_type"].value_counts().head(10).reset_index()
            vt_dist.columns = ["vehicle", "count"]
            fig = px.bar(vt_dist, x="count", y="vehicle", orientation="h",
                         color="count", color_continuous_scale="Oranges")
            fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title="", xaxis_title="violations",
                              coloraxis_showscale=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              yaxis={"categoryorder": "total ascending"})
            st.markdown("**Vehicle type mix**"); st.plotly_chart(fig, width="stretch")
        with vt2:
            if "place_type" in sub.columns:
                pt_dist = sub["place_type"].value_counts().head(8).reset_index()
                pt_dist.columns = ["place", "count"]
                fig = px.pie(pt_dist, values="count", names="place", hole=0.4,
                             color_discrete_sequence=px.colors.qualitative.Set2)
                fig.update_layout(height=300, margin=dict(l=0, r=0, t=10, b=0),
                                  paper_bgcolor="rgba(0,0,0,0)")
                st.markdown("**Place-type distribution**"); st.plotly_chart(fig, width="stretch")


# =====================================================================
# TAB 2: PARKING DNA
# =====================================================================
def _render_parking_dna():
    df = ui.load_data()

    st.markdown("## Parking DNA & Emerging Hotspots")
    st.caption("Behavioural fingerprinting per police station and violation growth trend detection.")

    # ===================== PARKING DNA PROFILES =====================
    st.subheader("Station DNA Profiles")
    st.caption("Each police station's unique violation fingerprint — dominant vehicle, violation type, peak behaviour.")

    with st.spinner("Building DNA profiles…"):
        dna = ui.get_parking_dna()

    if dna.empty:
        st.warning("Not enough data to build DNA profiles (minimum 50 violations per station).")
    else:
        # KPIs
        k = st.columns(5)
        ui.kpi(k[0], "Stations profiled", f"{len(dna)}")
        ui.kpi(k[1], "Total violations", f"{dna['total_violations'].sum():,}")
        ui.kpi(k[2], "Top station", dna.iloc[0]["police_station"][:20])
        ui.kpi(k[3], "Most common vehicle", dna["dominant_vehicle"].mode().iat[0] if len(dna) else "—")
        avg_weekend = dna["weekend_ratio"].mean()
        ui.kpi(k[4], "Avg weekend ratio", f"{avg_weekend:.1f}%")

        # DNA cards — top 6 stations
        st.markdown("---")
        st.markdown("**Top Station Fingerprints**")
        top_stations = dna.head(6)

        for row_start in range(0, len(top_stations), 3):
            cols = st.columns(min(3, len(top_stations) - row_start))
            for j, (_, station) in enumerate(top_stations.iloc[row_start:row_start + 3].iterrows()):
                with cols[j]:
                    st.markdown(f"""
                    <div class="rec-card">
                        <strong>{station['police_station']}</strong>
                        <br/><span style="color:#8A93A6;font-size:0.85rem">{station['total_violations']:,} violations</span>
                        <hr style="margin:8px 0;border-color:#1E2638"/>
                        <div style="display:grid;grid-template-columns:1fr 1fr;gap:6px;font-size:0.8rem">
                            <div><span style="color:#606878">Vehicle</span><br/><strong>{station['dominant_vehicle']}</strong></div>
                            <div><span style="color:#606878">Violation</span><br/><strong>{str(station['dominant_violation'])[:20]}</strong></div>
                            <div><span style="color:#606878">Peak Hour</span><br/><strong>{station['peak_hour']:02d}:00</strong></div>
                            <div><span style="color:#606878">Weekend %</span><br/><strong>{station['weekend_ratio']:.1f}%</strong></div>
                            <div><span style="color:#606878">Severity</span><br/><strong>{station['avg_severity']:.3f}</strong></div>
                            <div><span style="color:#606878">Junction %</span><br/><strong>{station['junction_frac']:.1f}%</strong></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        # DNA comparison radar chart — top 5
        st.markdown("---")
        dl, dr = st.columns([2, 1], gap="large")

        with dl:
            st.markdown("**DNA Comparison Radar — Top 5 Stations**")
            st.caption("Normalized traits across stations. Each axis = 0–100% of max value.")

            top5 = dna.head(5).copy()
            traits = ["total_violations", "avg_severity", "weekend_ratio", "junction_frac", "peak_hour_share"]
            trait_labels = ["Violations", "Severity", "Weekend %", "Junction %", "Peak Hour %"]

            fig = go.Figure()
            colors = ["#4C8BF5", "#EF4444", "#F59E0B", "#10B981", "#8B5CF6"]

            for i, (_, row) in enumerate(top5.iterrows()):
                values = []
                for t in traits:
                    col_max = dna[t].max()
                    values.append(round(row[t] / col_max * 100, 1) if col_max > 0 else 0)
                values.append(values[0])  # close the polygon
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=trait_labels + [trait_labels[0]],
                    fill="toself",
                    fillcolor=f"rgba({int(colors[i][1:3], 16)},{int(colors[i][3:5], 16)},{int(colors[i][5:7], 16)},0.1)",
                    line_color=colors[i],
                    name=row["police_station"][:20],
                ))

            fig.update_layout(
                height=400, margin=dict(l=60, r=60, t=30, b=30),
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                paper_bgcolor="rgba(0,0,0,0)",
                legend=dict(orientation="h", y=-0.15),
            )
            st.plotly_chart(fig, width="stretch")

        with dr:
            st.markdown("**Vehicle Type Distribution**")
            veh_counts = dna["dominant_vehicle"].value_counts().head(8).reset_index()
            veh_counts.columns = ["vehicle", "stations"]
            fig = px.bar(veh_counts, x="stations", y="vehicle", orientation="h",
                         color="stations", color_continuous_scale="Blues")
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                              yaxis_title="", xaxis_title="stations",
                              coloraxis_showscale=False,
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                              yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, width="stretch")

            st.markdown("**Violation Type Distribution**")
            viol_counts = dna["dominant_violation"].value_counts().head(6).reset_index()
            viol_counts.columns = ["violation", "stations"]
            fig = px.pie(viol_counts, values="stations", names="violation",
                         hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel)
            fig.update_layout(height=260, margin=dict(l=0, r=0, t=10, b=0),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

        # Full DNA table
        with st.expander("Full DNA profile table"):
            show_dna = dna.copy()
            show_dna.columns = ["Station", "Vehicle", "Violation", "Peak Hr",
                                "Weekend %", "Violations", "Severity", "Junction %", "Peak %"]
            st.dataframe(show_dna, hide_index=True, width="stretch", height=400)

            st.download_button("Download DNA profiles (CSV)",
                               dna.to_csv(index=False).encode(),
                               file_name="parksensei_parking_dna.csv",
                               mime="text/csv")

    # ===================== EMERGING HOTSPOTS =====================
    st.markdown("---")
    st.subheader("Emerging Hotspot Detection")
    st.caption("Growth analysis — which stations are seeing accelerating or declining violations?")

    with st.spinner("Analysing growth trends…"):
        growth = ui.get_emerging_hotspots()

    if growth.empty:
        st.warning("Not enough temporal data for growth analysis.")
    else:
        early_period = growth.attrs.get("early_period", "first half")
        late_period = growth.attrs.get("late_period", "second half")
        st.caption(f"Comparing **{early_period}** (early) vs **{late_period}** (late)")

        # KPIs
        trend_counts = growth["trend"].value_counts()
        gk = st.columns(5)
        ui.kpi(gk[0], "Stations analysed", f"{len(growth)}")
        ui.kpi(gk[1], "Rapidly emerging", f"{trend_counts.get('Rapidly Emerging', 0)}",
               help="50%+ growth")
        ui.kpi(gk[2], "Emerging", f"{trend_counts.get('Emerging', 0)}", help="20-50% growth")
        ui.kpi(gk[3], "Stable", f"{trend_counts.get('Stable', 0)}", help="-10% to +20%")
        declining = trend_counts.get("Declining", 0) + trend_counts.get("Rapidly Declining", 0)
        ui.kpi(gk[4], "Declining", f"{declining}", help=">10% decline")

        # Growth chart
        st.markdown("---")
        gl, gr = st.columns([2, 1], gap="large")

        with gl:
            st.markdown("**Growth Rate by Station**")
            show_growth = growth.head(20).copy()
            show_growth["station_short"] = show_growth["police_station"].str[:22]

            trend_colors = {
                "Rapidly Emerging": "#EF4444",
                "Emerging": "#F59E0B",
                "Stable": "#8DA4BE",
                "Declining": "#06B6D4",
                "Rapidly Declining": "#4C8BF5",
            }
            show_growth["color"] = show_growth["trend"].map(trend_colors)

            fig = go.Figure(go.Bar(
                x=show_growth["growth_percent"],
                y=show_growth["station_short"],
                orientation="h",
                marker_color=show_growth["color"],
                text=[f"{v:+.1f}%" for v in show_growth["growth_percent"]],
                textposition="outside",
            ))
            fig.add_vline(x=0, line_color="#4A5568", line_width=1)
            fig.update_layout(
                height=max(350, len(show_growth) * 25),
                margin=dict(l=0, r=50, t=10, b=0),
                xaxis_title="Growth %",
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, width="stretch")

        with gr:
            st.markdown("**Trend Classification**")
            trend_df = trend_counts.reset_index()
            trend_df.columns = ["trend", "count"]
            fig = px.pie(trend_df, values="count", names="trend",
                         color="trend", color_discrete_map=trend_colors,
                         hole=0.4)
            fig.update_layout(height=280, margin=dict(l=0, r=0, t=10, b=0),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")

            # Top emerging alerts
            emerging = growth[growth["trend"].isin(["Rapidly Emerging", "Emerging"])].head(3)
            if not emerging.empty:
                st.markdown("**⚠️ Top Emerging Hotspots**")
                for _, e in emerging.iterrows():
                    icon = "🔴" if e["trend"] == "Rapidly Emerging" else "🟡"
                    st.markdown(
                        f"{icon} **{e['police_station'][:22]}** — "
                        f"{e['growth_percent']:+.1f}% growth "
                        f"({e['early_count']} → {e['late_count']})"
                    )

        # Full growth table
        with st.expander("Full growth analysis table"):
            show_g = growth.copy()
            show_g.columns = ["Station", "Early", "Late", "Change",
                              "Growth %", "Total", "Trend"]
            st.dataframe(show_g, hide_index=True, width="stretch", height=400)

            st.download_button("Download growth analysis (CSV)",
                               growth.to_csv(index=False).encode(),
                               file_name="parksensei_emerging_hotspots.csv",
                               mime="text/csv")


# =====================================================================
# TAB 3: TRAFFIC PROPAGATION
# =====================================================================
def _render_traffic_propagation():
    zones = ui.get_zones()

    st.markdown("## Traffic Propagation & Blast Radius")
    st.caption("Identify which violation hotspots affect each other — haversine proximity within 2 km radius.")

    # Controls
    c1, c2, _ = st.columns([1, 1, 2])
    radius = c1.slider("Propagation radius (km)", 0.5, 4.0, 2.0, 0.5)
    top_n = c2.slider("Analyse top N zones", 10, 50, 30, 5)

    with st.spinner("Computing propagation network…"):
        prop = core.traffic_propagation(zones.head(top_n), radius_km=radius)

    if prop.empty:
        st.info("No propagation links found at this radius. Try increasing the radius or number of zones.")
        return

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

        hex_colors = {"Very High": "#E2352B", "High": "#E8923E", "Medium": "#F5CD5A", "Low": "#4C8BF5"}
        fig = px.bar(risk_df, x="risk", y="count", color="risk",
                     color_discrete_map=hex_colors)
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


# =====================================================================
# RENDER TABS
# =====================================================================
tab1, tab2, tab3 = st.tabs(["🔍 Hotspot Explorer", "🧬 Parking DNA", "🔗 Traffic Propagation"])

with tab1:
    _render_hotspot_explorer()
with tab2:
    _render_parking_dna()
with tab3:
    _render_traffic_propagation()
