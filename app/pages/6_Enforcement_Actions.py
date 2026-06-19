"""Enforcement Action Planner — AI-generated actionable recommendations per zone.
   Combines ParkWatch AI's recommendation engine with ParkSensei's intelligence layer."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import ui, core

ui.page("Enforcement Actions", "E")
ui.brand_sidebar()

df    = ui.load_data()
zones = ui.get_zones()

st.markdown("## Enforcement Action Planner")
st.caption("Explainable enforcement recommendations for every high-impact zone.")

# ---------------- generate recommendations ----------------
zone_recs = ui.get_zone_recommendations(top_n=20)

# Aggregate action stats
all_actions = []
all_priorities = []
for zr in zone_recs:
    for r in zr["recommendations"]:
        all_actions.append(r["action"])
        all_priorities.append(r["priority"])

action_counts = pd.Series(all_actions).value_counts()
priority_counts = pd.Series(all_priorities).value_counts()

# KPIs
k = st.columns(5)
ui.kpi(k[0], "Zones analysed", f"{len(zone_recs)}")
ui.kpi(k[1], "Total recommendations", f"{len(all_actions)}")
critical_count = priority_counts.get("CRITICAL", 0)
ui.kpi(k[2], "Critical actions", f"{critical_count}")
high_count = priority_counts.get("HIGH", 0)
ui.kpi(k[3], "High-priority actions", f"{high_count}")
tow_count = action_counts.get("Tow-Away Zone", 0)
ui.kpi(k[4], "Tow-away zones", f"{tow_count}")

# ---------------- action summary charts ----------------
st.markdown("---")
ch1, ch2 = st.columns(2, gap="large")

with ch1:
    st.markdown("**Actions by type**")
    ac_df = action_counts.reset_index()
    ac_df.columns = ["action", "count"]
    fig = px.bar(ac_df, x="count", y="action", orientation="h",
                 color="count", color_continuous_scale="Reds")
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                      yaxis_title="", xaxis_title="zones needing this action",
                      coloraxis_showscale=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      yaxis={"categoryorder": "total ascending"})
    st.plotly_chart(fig, width="stretch")

with ch2:
    st.markdown("**Actions by priority**")
    pr_df = priority_counts.reindex(["CRITICAL", "HIGH", "MEDIUM", "LOW"]).fillna(0).reset_index()
    pr_df.columns = ["priority", "count"]
    colors = [ui.REC_COLORS.get(p, "#4C8BF5") for p in pr_df["priority"]]
    fig = px.bar(pr_df, x="priority", y="count", color="priority",
                 color_discrete_map=ui.REC_COLORS)
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="", yaxis_title="recommendations",
                      showlegend=False,
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

# ---------------- zone-by-zone recommendations ----------------
st.markdown("---")
st.subheader("Zone-by-zone enforcement recommendations")
st.caption("Each zone's 7-factor impact breakdown and actionable enforcement steps.")

for i, zr in enumerate(zone_recs):
    z = zr["zone"]
    recs = zr["recommendations"]
    top_priority = zr["top_priority"]

    # Zone header
    priority_label = {"CRITICAL": "CRIT", "HIGH": "HIGH", "MEDIUM": "MED", "LOW": "LOW"}.get(top_priority, "--")
    with st.expander(
        f"[{priority_label}] #{i+1}  {z.get('label', 'Zone')} — "
        f"Impact {z.get('impact_score', 0):.0f} | "
        f"{int(z.get('violations', 0)):,} violations | "
        f"{zr['top_action']}",
        expanded=(i < 3)
    ):
        # Zone stats row
        zc = st.columns(6)
        zc[0].metric("Impact Score", f"{z.get('impact_score', 0):.0f}")
        zc[1].metric("Violations", f"{int(z.get('violations', 0)):,}")
        zc[2].metric("Avg PCU", f"{z.get('avg_pcu', 1):.2f}")
        zc[3].metric("Junction %", f"{z.get('junction_frac', 0)*100:.0f}%")
        zc[4].metric("Peak %", f"{z.get('peak_share', 0)*100:.0f}%")
        zc[5].metric("Repeat %", f"{z.get('repeat_vehicle_share', 0)*100:.0f}%")

        # Impact breakdown + recommendations side by side
        zl, zr_col = st.columns([1.2, 1], gap="large")
        with zl:
            st.markdown("**Impact score breakdown:**")
            bd = core.impact_breakdown(z)
            factors = list(bd.keys())
            weighted_vals = [bd[f]["weighted"] for f in factors]
            labels = [bd[f]["label"] for f in factors]
            fig = go.Figure(go.Bar(
                x=weighted_vals, y=labels, orientation="h",
                marker_color=[ui.ACCENT if v > 5 else "#3a4254" for v in weighted_vals],
                text=[f"{v:.1f}" for v in weighted_vals],
                textposition="outside"
            ))
            fig.update_layout(
                height=220, margin=dict(l=0, r=40, t=5, b=5),
                xaxis_title="contribution (0-100)",
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, width="stretch")

        with zr_col:
            st.markdown("**Recommended actions:**")
            for r in recs:
                ui.render_recommendation_card(r)

        # Additional zone context
        extra_cols = st.columns(3)
        if "place_type" in z:
            extra_cols[0].markdown(f"**Place type:** {z['place_type']}")
        if "top_violation" in z:
            extra_cols[1].markdown(f"**Top offence:** {z['top_violation']}")
        if "avg_delay_mins" in z and pd.notna(z.get("avg_delay_mins")):
            extra_cols[2].markdown(f"**Avg delay:** {z['avg_delay_mins']:.0f} mins")

# ---------------- enforcement delay by station ----------------
st.markdown("---")
st.subheader("Enforcement delay by police station")
st.caption("Which stations are slowest to respond? Higher delay indicates more time violations go unchallenged.")

delay_df = core.delay_by_station(df)
if len(delay_df) and delay_df["avg_delay"].notna().any():
    dl, dr = st.columns([2, 1], gap="large")
    with dl:
        top_delay = delay_df.head(15)
        fig = px.bar(top_delay, x="avg_delay", y="police_station", orientation="h",
                     color="avg_delay", color_continuous_scale="YlOrRd")
        fig.update_layout(height=400, margin=dict(l=0, r=0, t=10, b=0),
                          yaxis_title="", xaxis_title="average delay (minutes)",
                          coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
    with dr:
        st.markdown("**Key insight:** Stations with high enforcement delay are prime candidates for "
                    "**automated camera / ANPR monitoring** — manual patrols can't reach violations fast enough.")
        avg_overall = delay_df["avg_delay"].mean()
        st.metric("City-wide avg delay", f"{avg_overall:.0f} mins")
        slowest = delay_df.iloc[0]
        st.metric("Slowest station", slowest["police_station"],
                  help=f"Avg delay: {slowest['avg_delay']:.0f} mins")
else:
    st.info("Enforcement delay data not available — re-run `prep.py` with the raw CSV to generate it.")

def _build_recs_csv(zone_recs_data):
    """Build a CSV string from zone recommendations."""
    rows = []
    for zr in zone_recs_data:
        z = zr["zone"]
        for r in zr["recommendations"]:
            rows.append({
                "zone": z.get("label", ""),
                "impact_score": z.get("impact_score", 0),
                "violations": z.get("violations", 0),
                "avg_pcu": z.get("avg_pcu", 1),
                "place_type": z.get("place_type", ""),
                "action": r["action"],
                "priority": r["priority"],
                "reason": r["reason"],
                "deployment_window": r["window"],
            })
    return pd.DataFrame(rows).to_csv(index=False)

# CSV download
st.markdown("---")
st.download_button(
    "Download full recommendations (CSV)",
    _build_recs_csv(zone_recs).encode(),
    file_name="parksensei_enforcement_actions.csv",
    mime="text/csv"
)
