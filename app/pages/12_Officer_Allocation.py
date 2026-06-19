"""Officer Allocation & Route Optimizer — Priority-proportional headcount and TSP patrol route.
   Adapted from GRIDLOCK2.0 Prototype scripts 10 (officer_allocation) and 11 (patrol_route_optimizer)."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import ui, core

ui.page("Officer Allocation", "O")
ui.brand_sidebar()

st.markdown("## Officer Allocation & Route Optimizer")
st.caption("Proportional officer deployment based on zone impact scores, with optimized patrol routing.")

zones = ui.get_zones()

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
