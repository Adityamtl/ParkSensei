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
st.markdown("# ParkSensei")
st.markdown("#### Turning 298K parking-violation records into targeted enforcement for Bengaluru")
st.caption("Theme 1 — Poor Visibility on Parking-Induced Congestion — Gridlock Hackathon 2.0")

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
    f"**Summary** — ParkSensei's 7-factor Congestion Impact Score identifies "
    f"**{critical_zones} critical zones** (impact score 70+). "
    f"On average, **{junction_pct}%** of violations are junction-linked, "
    f"**{peak_pct}%** recur during peak hours, "
    f"and the dominant offence is **{top_violation.lower()}**. "
    f"Average vehicle obstruction weight is **{avg_pcu:.2f} PCU**."
)

# ---------------- PDF download ----------------
bt = ui.get_backtest()
pdf_bytes = ui.generate_pdf_brief(zones, backtest_result=bt)
if pdf_bytes:
    col_dl1, col_dl2, _ = st.columns([1, 1, 4])
    with col_dl1:
        st.download_button("Download PDF Brief", pdf_bytes,
                           file_name="parksensei_enforcement_brief.pdf",
                           mime="application/pdf")
    with col_dl2:
        csv_data = zones.head(30).to_csv(index=False).encode()
        st.download_button("Download Top Zones (CSV)", csv_data,
                           file_name="parksensei_top_zones.csv")

# ---------------- city map ----------------
left, right = st.columns([2, 1], gap="large")
with left:
    st.subheader("City-wide violation density")
    st.caption("3-D hex bins — height and colour represent violation volume. Drag to rotate.")
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
st.subheader("Top enforcement actions")
st.caption("Recommendations for the highest-impact zones")

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

st.caption("Navigate: Hotspot Explorer / Forecast & Patrol Planner / Enforcement Actions / Repeat-Offender Intelligence")

# ---------------- new feature previews (from GRIDLOCK2.0) ----------------
st.markdown("---")
st.subheader("Intelligence Modules")
st.caption("Advanced analytics from GRIDLOCK2.0 Prototype integration.")

p1, p2, p3, p4 = st.columns(4, gap="medium")

with p1:
    try:
        prop = ui.get_traffic_propagation(top_n=20)
        n_links = len(prop) if not prop.empty else 0
        very_high = prop[prop["propagation_risk"] == "Very High"].shape[0] if not prop.empty else 0
    except Exception:
        n_links, very_high = 0, 0
    st.markdown("""
    <div class="rec-card">
        <strong>🔗 Traffic Propagation</strong>
        <br/><span style="color:#8A93A6;font-size:0.82rem">Hotspot proximity network</span>
        <hr style="margin:8px 0;border-color:#1E2638"/>
        <div style="font-size:1.3rem;font-weight:700">{} links</div>
        <span style="color:#EF4444;font-size:0.8rem">{} very high risk</span>
    </div>
    """.format(n_links, very_high), unsafe_allow_html=True)

with p2:
    try:
        dna = ui.get_parking_dna()
        n_stations = len(dna) if not dna.empty else 0
        top_vehicle = dna.iloc[0]["dominant_vehicle"][:12] if not dna.empty else "—"
    except Exception:
        n_stations, top_vehicle = 0, "—"
    st.markdown("""
    <div class="rec-card">
        <strong>🧬 Parking DNA</strong>
        <br/><span style="color:#8A93A6;font-size:0.82rem">Station behavioural profiles</span>
        <hr style="margin:8px 0;border-color:#1E2638"/>
        <div style="font-size:1.3rem;font-weight:700">{} stations</div>
        <span style="color:#06B6D4;font-size:0.8rem">Top vehicle: {}</span>
    </div>
    """.format(n_stations, top_vehicle), unsafe_allow_html=True)

with p3:
    try:
        growth = ui.get_emerging_hotspots()
        emerging_cnt = len(growth[growth["trend"].isin(["Rapidly Emerging", "Emerging"])]) if not growth.empty else 0
        declining_cnt = len(growth[growth["trend"].isin(["Declining", "Rapidly Declining"])]) if not growth.empty else 0
    except Exception:
        emerging_cnt, declining_cnt = 0, 0
    st.markdown("""
    <div class="rec-card">
        <strong>📈 Emerging Hotspots</strong>
        <br/><span style="color:#8A93A6;font-size:0.82rem">Growth trend detection</span>
        <hr style="margin:8px 0;border-color:#1E2638"/>
        <div style="font-size:1.3rem;font-weight:700">{} emerging</div>
        <span style="color:#10B981;font-size:0.8rem">{} declining</span>
    </div>
    """.format(emerging_cnt, declining_cnt), unsafe_allow_html=True)

with p4:
    sim_preview = core.what_if_simulation(zones, target_idx=0, additional_officers=5)
    st.markdown("""
    <div class="rec-card">
        <strong>🎮 What-If Simulator</strong>
        <br/><span style="color:#8A93A6;font-size:0.82rem">Officer deployment scenarios</span>
        <hr style="margin:8px 0;border-color:#1E2638"/>
        <div style="font-size:1.3rem;font-weight:700">{:.0f} → {:.0f}</div>
        <span style="color:#10B981;font-size:0.8rem">+5 officers = -{:.0f}% impact</span>
    </div>
    """.format(sim_preview["current_impact"], sim_preview["new_impact"],
               sim_preview["impact_reduction_pct"]), unsafe_allow_html=True)

st.caption(
    "All modules: Analytics & Insights · Operations & Dispatch · Strategy & Review · "
    "Advanced Intelligence · Ask ParkSensei · Reports & Export"
)
