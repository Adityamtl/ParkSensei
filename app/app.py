"""
ParkSensei — Command Center (entry page).
Run from round2/:  streamlit run app/app.py
"""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import ui, core

ui.page("Command Center")
ui.brand_sidebar()

df    = ui.load_data()
zones = ui.get_zones()
grid  = ui.get_grid()

# ---------------- hero ----------------
st.markdown("# \U0001F6A6 ParkSensei")
st.markdown("#### Turning 298K parking-violation records into targeted enforcement for Bengaluru")
st.caption("Theme 1 · Poor Visibility on Parking-Induced Congestion · Gridlock Hackathon 2.0")

# ---------------- KPIs ----------------
total      = len(df)
vc         = core.vehicle_counts(df)
repeat_sh  = vc[vc >= 2].sum() / total
per_day    = total / df["ymd"].nunique()
before1pm  = (df["hour"] < 13).mean()
top_zone   = zones.iloc[0]
avg_pcu    = df["pcu"].mean() if "pcu" in df.columns else 1.0

c = st.columns(6)
ui.kpi(c[0], "Violations logged", f"{total:,}")
ui.kpi(c[1], "Hotspot zones", f"{zones.shape[0]:,}", help="~1.2 km neighbourhoods (geohash-6)")
ui.kpi(c[2], "Avg / day", f"{per_day:,.0f}")
ui.kpi(c[3], "Repeat-offender load", f"{repeat_sh*100:,.0f}%",
       help="share of violations from vehicles caught more than once")
ui.kpi(c[4], "Avg PCU weight", f"{avg_pcu:.2f}",
       help="Passenger Car Unit — higher = bigger vehicles blocking more road")
ui.kpi(c[5], "#1 hotspot", top_zone["label"].split(" - ")[-1][:18])

st.markdown("")

# ---------------- Judge Brief (from ParkSight AI) ----------------
junction_pct = int(zones["junction_frac"].mean() * 100) if "junction_frac" in zones.columns else 0
peak_pct = int(zones["peak_share"].mean() * 100) if "peak_share" in zones.columns else 0
critical_zones = len(zones[zones["impact_score"] >= 70])
top_violation = df["primary_type"].mode().iat[0] if len(df) else "unknown"

st.info(
    f"**\U0001F4CA Judge Brief** — ParkSensei's enhanced 7-factor Congestion Impact Score identifies "
    f"**{critical_zones} critical zones** (impact ≥ 70). "
    f"On average, **{junction_pct}%** of violations are junction-linked, "
    f"**{peak_pct}%** recur during peak hours, "
    f"and the dominant offence is **{top_violation.lower()}**. "
    f"Average vehicle obstruction weight is **{avg_pcu:.2f} PCU** — enforcement should prioritize "
    f"heavy-vehicle zones and junction-clearing over simple ticket counting."
)

# ---------------- PDF download ----------------
bt = ui.get_backtest()
pdf_bytes = ui.generate_pdf_brief(zones, backtest_result=bt)
if pdf_bytes:
    col_dl1, col_dl2, _ = st.columns([1, 1, 4])
    with col_dl1:
        st.download_button("📄 Download PDF Brief", pdf_bytes,
                           file_name="parksensei_enforcement_brief.pdf",
                           mime="application/pdf")
    with col_dl2:
        csv_data = zones.head(30).to_csv(index=False).encode()
        st.download_button("📊 Download Top Zones (CSV)", csv_data,
                           file_name="parksensei_top_zones.csv")

# ---------------- city map ----------------
left, right = st.columns([2, 1], gap="large")
with left:
    st.subheader("City-wide violation density")
    st.caption("3-D hex bins · height & colour = violation volume. Drag to rotate.")
    st.pydeck_chart(ui.deck([ui.hex_layer(grid)], ui.view(zoom=10.4, pitch=50)),
                    width="stretch")

with right:
    st.subheader("Top impact zones")
    st.caption("Ranked by 7-factor Congestion Impact Score")
    show_cols = ["label", "violations", "impact_score"]
    if "avg_pcu" in zones.columns:
        show_cols.append("avg_pcu")
    if "place_type" in zones.columns:
        show_cols.append("place_type")
    show = zones.head(12)[show_cols].copy()
    rename_map = {"label": "Zone", "violations": "Viol.", "impact_score": "Impact",
                  "avg_pcu": "Avg PCU", "place_type": "Type"}
    show.columns = [rename_map.get(c, c) for c in show.columns]
    st.dataframe(show, hide_index=True, width="stretch", height=470,
                 column_config={"Impact": st.column_config.ProgressColumn(
                     "Impact", min_value=0, max_value=100, format="%.0f")})

# ---------------- insight + temporal ----------------
st.markdown("---")
st.subheader("When does enforcement actually happen?")
ins, heat = st.columns([1, 2], gap="large")
with ins:
    st.metric("Logged before 1 PM", f"{before1pm*100:.0f}%")
    st.markdown(
        f"**Observation-bias insight.** Roughly **{before1pm*100:.0f}%** of all violations are logged "
        "before 1 PM — the data reflects *when patrols are out*, not when illegal parking peaks. "
        "Evening commercial-hour parking is largely **uncaptured**.\n\n"
        "ParkSensei treats this honestly: we optimise **enforcement efficiency** on observed patterns "
        "and flag **coverage blind spots** for re-deployment.")
with heat:
    pivot = (df.groupby(["dow", "hour"]).size().reset_index(name="n"))
    pivot["Weekday"] = pivot["dow"].map(dict(enumerate(ui.DOW_NAMES)))
    fig = px.density_heatmap(pivot, x="hour", y="Weekday", z="n",
                             category_orders={"Weekday": ui.DOW_NAMES},
                             color_continuous_scale="Turbo", nbinsx=24)
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=10, b=0),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      coloraxis_colorbar=dict(title="Viol."), xaxis_title="Hour of day (IST)")
    st.plotly_chart(fig, width="stretch")

# ---------------- top enforcement actions preview ----------------
st.markdown("---")
st.subheader("🎯 Top enforcement actions")
st.caption("AI-generated recommendations for the highest-impact zones")

recs = ui.get_zone_recommendations(top_n=5)
rec_cols = st.columns(min(len(recs), 5))
for i, zr in enumerate(recs[:5]):
    with rec_cols[i]:
        z = zr["zone"]
        top_rec = zr["recommendations"][0] if zr["recommendations"] else None
        st.metric(z.get("label", "Zone")[:18],
                  f"Impact {z.get('impact_score', 0):.0f}")
        if top_rec:
            ui.render_recommendation_card(top_rec)

st.caption("Navigate ▸ Hotspot Explorer · Forecast & Patrol Planner · Enforcement Actions · Repeat-Offender Intelligence")
