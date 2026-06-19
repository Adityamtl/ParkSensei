"""Parking DNA & Emerging Hotspots — Per-station behavioural fingerprints and growth analysis.
   Adapted from GRIDLOCK2.0 Prototype scripts 12 (parking_dna) and 06 (emerging_hotspots)."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import ui, core

ui.page("Parking DNA", "D")
ui.brand_sidebar()

st.markdown("## Parking DNA & Emerging Hotspots")
st.caption("Behavioural fingerprinting per police station and violation growth trend detection.")

df = ui.load_data()

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
        # Normalize traits to 0-1
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
