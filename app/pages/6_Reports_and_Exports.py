"""Reports & Exports - Unified download hub for all ParkSensei analysis outputs.
   Adapted from ParkSight AI's Reports page."""
import json
import numpy as np, pandas as pd
import streamlit as st
import ui, core

ui.page("Reports & Exports", "R")
ui.brand_sidebar()

st.markdown("## Reports & Exports")
st.caption("Download full analysis outputs: enforcement priorities, DBSCAN clusters, forecasts, recommendations, and more.")

df    = ui.load_data()
zones = ui.get_zones()

# ---------------- dashboard stats summary ----------------
st.subheader("Dashboard Statistics")

total = len(df)
n_zones = len(zones)
n_days = df["ymd"].nunique()
junctions = df.loc[df["has_junction"], "junction_name"].nunique() if "has_junction" in df.columns else 0
stations = df["police_station"].nunique() if "police_station" in df.columns else 0
vehicles = df["vehicle_number"].nunique() if "vehicle_number" in df.columns else 0

k = st.columns(6)
ui.kpi(k[0], "Total violations", f"{total:,}")
ui.kpi(k[1], "Scored zones", f"{n_zones}")
ui.kpi(k[2], "Active days", f"{n_days}")
ui.kpi(k[3], "Named junctions", f"{junctions}")
ui.kpi(k[4], "Police stations", f"{stations}")
ui.kpi(k[5], "Unique vehicles", f"{vehicles:,}")

# Date range
date_range = f"{df['ymd'].min()} to {df['ymd'].max()}"
st.caption(f"Data period: {date_range}")

# ---------------- key stats cards ----------------
st.markdown("---")
st.subheader("Key Findings")

s1, s2, s3, s4 = st.columns(4)
top_zone = zones.iloc[0] if len(zones) else {}
with s1:
    st.metric("Top hotspot", str(top_zone.get("label", "—")).split(" - ")[-1][:20])
    st.caption(f"Impact score: {top_zone.get('impact_score', 0):.0f}")
with s2:
    vc = core.vehicle_counts(df)
    repeat = vc[vc >= 2]
    st.metric("Repeat offender share", f"{repeat.sum() / total * 100:.1f}%")
    st.caption(f"{len(repeat):,} vehicles caught ≥2 times")
with s3:
    before1pm = (df["hour"] < 13).mean() * 100
    st.metric("Logged before 1 PM", f"{before1pm:.0f}%")
    st.caption("Morning enforcement concentration")
with s4:
    avg_pcu = df["pcu"].mean() if "pcu" in df.columns else 1.0
    st.metric("Avg PCU weight", f"{avg_pcu:.2f}")
    st.caption("Vehicle obstruction factor")

# Top junctions table
st.markdown("---")
st.subheader("Top Junctions by Volume")
if "junction_name" in df.columns:
    junc_counts = (df[df["junction_name"] != "No Junction"]["junction_name"]
                   .value_counts().head(10).reset_index())
    junc_counts.columns = ["Junction", "Violations"]
    st.dataframe(junc_counts, hide_index=True, width="stretch")

# Top stations table
if "police_station" in df.columns:
    station_counts = (df[df["police_station"].str.lower() != "nan"]["police_station"]
                      .value_counts().head(10).reset_index())
    station_counts.columns = ["Station", "Violations"]
    st.dataframe(station_counts, hide_index=True, width="stretch")

# ---------------- downloadable reports ----------------
st.markdown("---")
st.subheader("Downloadable Reports")
st.caption("All exports generated from live ParkSensei models.")

d1, d2, d3 = st.columns(3, gap="large")

with d1:
    st.markdown("**📊 Enforcement Priority (CSV)**")
    st.caption("Top 30 zones ranked by 7-factor Congestion Impact Score")
    show_cols = ["label", "violations", "impact_score", "avg_severity",
                 "junction_frac", "top_violation"]
    if "avg_pcu" in zones.columns:
        show_cols.append("avg_pcu")
    if "place_type" in zones.columns:
        show_cols.append("place_type")
    priority_csv = zones.head(30)[show_cols].to_csv(index=False)
    st.download_button("Download", priority_csv.encode(),
                       file_name="parksensei_enforcement_priority.csv",
                       mime="text/csv", use_container_width=True)

with d2:
    st.markdown("**🎯 DBSCAN Clusters (CSV)**")
    st.caption("Spatial hotspot clusters with impact scores")
    try:
        clusters = ui.get_dbscan_clusters()
        if clusters is not None and not clusters.empty:
            cluster_csv = clusters.to_csv(index=False)
            st.download_button("Download", cluster_csv.encode(),
                               file_name="parksensei_dbscan_clusters.csv",
                               mime="text/csv", use_container_width=True)
        else:
            st.info("No DBSCAN clusters available")
    except Exception as e:
        st.info(f"DBSCAN not available: {e}")

with d3:
    st.markdown("**📈 Next-Day Forecast (CSV)**")
    st.caption("7-day ahead predictions per junction with risk levels")
    try:
        forecast = ui.get_nextday_forecast()
        if isinstance(forecast, dict) and "error" not in forecast:
            rows = []
            for junc_name, junc_data in forecast.get("junctions", {}).items():
                for d in junc_data["days"]:
                    rows.append({
                        "junction": junc_data["shortName"],
                        "date": d["date"],
                        "day": d["dayName"],
                        "predicted": d["predicted"],
                        "ci_low": d["ciLow"],
                        "ci_high": d["ciHigh"],
                        "risk": d["risk"],
                        "recommendation": d["recommendation"],
                    })
            forecast_csv = pd.DataFrame(rows).to_csv(index=False)
            st.download_button("Download", forecast_csv.encode(),
                               file_name="parksensei_nextday_forecast.csv",
                               mime="text/csv", use_container_width=True)
        else:
            st.info("Forecast not available")
    except Exception as e:
        st.info(f"Forecast not available: {e}")

d4, d5, d6 = st.columns(3, gap="large")

with d4:
    st.markdown("**🔄 Repeat Offenders (CSV)**")
    st.caption("Top 50 most-cited vehicles with zones and areas")
    vc = core.vehicle_counts(df)
    top_off = vc.head(50).rename("violations").reset_index()
    top_off.columns = ["vehicle", "violations"]
    meta = (df[df["vehicle_number"].isin(top_off["vehicle"])]
            .groupby("vehicle_number")
            .agg(vehicle_type=("vehicle_type", lambda s: s.mode().iat[0]),
                 zones=("gh6", "nunique"),
                 top_area=("police_station", lambda s: s[s.str.lower() != "nan"].mode().iat[0]
                           if (s.str.lower() != "nan").any() else "—"))
            .reset_index().rename(columns={"vehicle_number": "vehicle"}))
    top_off = top_off.merge(meta, on="vehicle")
    st.download_button("Download", top_off.to_csv(index=False).encode(),
                       file_name="parksensei_repeat_offenders.csv",
                       mime="text/csv", use_container_width=True)

with d5:
    st.markdown("**🎬 Enforcement Actions (CSV)**")
    st.caption("AI-generated enforcement recommendations per zone")
    try:
        zone_recs = ui.get_zone_recommendations(top_n=20)
        rec_rows = []
        for zr in zone_recs:
            z = zr["zone"]
            for r in zr["recommendations"]:
                rec_rows.append({
                    "zone": z.get("label", ""),
                    "impact_score": z.get("impact_score", 0),
                    "violations": z.get("violations", 0),
                    "avg_pcu": z.get("avg_pcu", 1),
                    "place_type": z.get("place_type", ""),
                    "action": r["action"],
                    "priority": r["priority"],
                    "reason": r["reason"],
                    "window": r["window"],
                })
        recs_csv = pd.DataFrame(rec_rows).to_csv(index=False)
        st.download_button("Download", recs_csv.encode(),
                           file_name="parksensei_enforcement_actions.csv",
                           mime="text/csv", use_container_width=True)
    except Exception as e:
        st.info(f"Recommendations not available: {e}")

with d6:
    st.markdown("**📄 PDF Enforcement Brief**")
    st.caption("Full formatted PDF for field use")
    bt = ui.get_backtest()
    pdf_bytes = ui.generate_pdf_brief(zones, backtest_result=bt)
    if pdf_bytes:
        st.download_button("Download", pdf_bytes,
                           file_name="parksensei_enforcement_brief.pdf",
                           mime="application/pdf", use_container_width=True)
    else:
        st.info("PDF generation requires `fpdf2`: `pip install fpdf2`")

# ---------------- JSON stats export ----------------
st.markdown("---")
st.subheader("JSON Stats Export")
st.caption("Machine-readable dashboard statistics for integration with other systems.")

# Build stats JSON (ParkSight-compatible format)
hourly = {str(h): int(df[df["hour"] == h].shape[0]) for h in range(24)}
dow_counts = df["dow"].value_counts()
dow = {str(d): int(dow_counts.get(d, 0)) for d in range(7)}

# Vehicle types
veh_counts = df["vehicle_type"].value_counts() if "vehicle_type" in df.columns else pd.Series(dtype=int)
top6_names = ["SCOOTER", "CAR", "MOTOR CYCLE", "PASSENGER AUTO", "MAXI-CAB", "LGV"]
top6_veh = {name: int(veh_counts.get(name, 0)) for name in top6_names}
others_veh = int(veh_counts.sum()) - sum(top6_veh.values())
vehicle_types = {**top6_veh, "OTHERS": others_veh}

# Violation types
if "primary_type" in df.columns:
    vt = df["primary_type"].value_counts().head(8).to_dict()
    violation_types = {k: int(v) for k, v in vt.items()}
else:
    violation_types = {}

stats = {
    "total_violations": total,
    "date_range": {"start": df["ymd"].min(), "end": df["ymd"].max()},
    "hourly": hourly,
    "dow": dow,
    "vehicle_types": vehicle_types,
    "violation_types": violation_types,
    "top_zones": zones.head(10)[["label", "violations", "impact_score"]].to_dict("records"),
    "unique_stations": stations,
    "unique_junctions": junctions,
    "unique_vehicles": vehicles,
    "total_zones_scored": n_zones,
    "active_days": n_days,
}

stats_json = json.dumps(stats, indent=2, default=str)
st.code(stats_json[:500] + "\n...", language="json")
st.download_button("Download dashboard_stats.json",
                   stats_json.encode(),
                   file_name="parksensei_dashboard_stats.json",
                   mime="application/json")
