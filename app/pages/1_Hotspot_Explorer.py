"""Hotspot Explorer — slice the city by time, day, violation type and station.
   Enhanced with impact breakdown visualization and place-type badges."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import ui, core

ui.page("Hotspot Explorer", "H")
ui.brand_sidebar()
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
    st.stop()

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
    # Horizontal bar chart of factor contributions
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
