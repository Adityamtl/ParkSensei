"""Alerts — Real-time zone alerts derived from ML predictions.
   Adapted from ParkSight AI's Alerts page."""
import numpy as np, pandas as pd
import streamlit as st
import ui, core

ui.page("Alerts", "A")
ui.brand_sidebar()

st.markdown("## Active Alerts")
st.caption("Real-time violation zone notifications derived from ML predictions and DBSCAN cluster analysis.")

# ---------------- generate alerts from ML models ----------------
def _generate_alerts():
    """Build alerts from next-day forecast and DBSCAN clusters."""
    alerts = []
    alert_id = 0

    # 1. Alerts from next-day forecast (HIGH risk days)
    try:
        forecast = ui.get_nextday_forecast()
        if isinstance(forecast, dict) and "error" not in forecast:
            junctions = forecast.get("junctions", {})
            for junc_name, junc_data in junctions.items():
                for d in junc_data["days"]:
                    if d["risk"] == "HIGH":
                        alert_id += 1
                        alerts.append({
                            "id": alert_id,
                            "type": "critical",
                            "zone": f"{junc_data['shortName']} — {d['dayName']} {d['date']}",
                            "message": f"Predicted {d['predicted']} violations "
                                      f"(CI: {d['ciLow']}–{d['ciHigh']}). "
                                      f"{d['recommendation']}",
                            "time": f"Forecast for {d['date']}",
                            "source": "Next-Day RF Model",
                        })
                    elif d["risk"] == "MEDIUM":
                        alert_id += 1
                        alerts.append({
                            "id": alert_id,
                            "type": "medium",
                            "zone": f"{junc_data['shortName']} — {d['dayName']} {d['date']}",
                            "message": f"Predicted {d['predicted']} violations. "
                                      f"{d['recommendation']}",
                            "time": f"Forecast for {d['date']}",
                            "source": "Next-Day RF Model",
                        })

            # City-wide HIGH alerts
            cw = forecast.get("city_wide", {})
            for d in cw.get("forecast", []):
                if d["risk"] == "HIGH":
                    alert_id += 1
                    alerts.append({
                        "id": alert_id,
                        "type": "high",
                        "zone": f"City-wide — {d['dayName']} {d['date']}",
                        "message": f"City-wide predicted {d['predicted']:,} violations — "
                                  f"above high-congestion threshold.",
                        "time": f"Forecast for {d['date']}",
                        "source": "City-wide RF Model",
                    })
    except Exception:
        pass

    # 2. Alerts from DBSCAN mega-clusters
    try:
        clusters = ui.get_dbscan_clusters()
        if clusters is not None and not clusters.empty:
            mega = clusters[clusters["violation_count"] > 10000]
            for _, c in mega.iterrows():
                alert_id += 1
                alerts.append({
                    "id": alert_id,
                    "type": "high",
                    "zone": c["dominant_junction"],
                    "message": f"Mega-cluster: {c['violation_count']:,} violations in 300m radius. "
                              f"Impact score {c['impact_score']:.1f}. "
                              f"Consider permanent tow-away zone.",
                    "time": "Historical analysis",
                    "source": "DBSCAN Clustering",
                })

            # High-impact clusters
            high_impact = clusters[
                (clusters["impact_score"] >= 7) & (clusters["violation_count"] <= 10000)
            ].head(5)
            for _, c in high_impact.iterrows():
                alert_id += 1
                alerts.append({
                    "id": alert_id,
                    "type": "medium",
                    "zone": c["dominant_junction"],
                    "message": f"High-impact cluster: {c['violation_count']:,} violations, "
                              f"impact {c['impact_score']:.1f}. "
                              f"Peak-hour ratio: {c['peak_hour_ratio']*100:.0f}%.",
                    "time": "Historical analysis",
                    "source": "DBSCAN Clustering",
                })
    except Exception:
        pass

    # 3. Alerts from congestion model
    try:
        cong = ui.get_congestion_model()
        if isinstance(cong, dict) and "error" not in cong:
            if cong.get("f1_score", 0) < 0.6:
                alert_id += 1
                alerts.append({
                    "id": alert_id,
                    "type": "low",
                    "zone": "Congestion Model",
                    "message": f"Model F1 score is {cong['f1_score']:.2f} — consider retraining "
                              f"with more recent data or additional features.",
                    "time": "Model diagnostic",
                    "source": "Congestion RF Classifier",
                })
    except Exception:
        pass

    return alerts

# Priority ordering
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}
PRIORITY_CONFIG = {
    "critical": {"icon": "🔴", "label": "CRITICAL", "color": "#E2352B"},
    "high":     {"icon": "🟠", "label": "HIGH",     "color": "#E8923E"},
    "medium":   {"icon": "🟡", "label": "MEDIUM",   "color": "#F5CD5A"},
    "low":      {"icon": "🔵", "label": "LOW",      "color": "#4C8BF5"},
}

# Session state for dismissed alerts
if "dismissed_alerts" not in st.session_state:
    st.session_state.dismissed_alerts = set()

with st.spinner("Analysing predictions for alerts…"):
    all_alerts = _generate_alerts()

# Filter out dismissed
active_alerts = [a for a in all_alerts if a["id"] not in st.session_state.dismissed_alerts]
active_alerts.sort(key=lambda a: PRIORITY_ORDER.get(a["type"], 4))

# KPIs
k = st.columns(5)
ui.kpi(k[0], "Active alerts", f"{len(active_alerts)}")
ui.kpi(k[1], "Critical", f"{sum(1 for a in active_alerts if a['type'] == 'critical')}")
ui.kpi(k[2], "High", f"{sum(1 for a in active_alerts if a['type'] == 'high')}")
ui.kpi(k[3], "Medium", f"{sum(1 for a in active_alerts if a['type'] == 'medium')}")
ui.kpi(k[4], "Dismissed", f"{len(st.session_state.dismissed_alerts)}")

st.markdown("---")

# Clear all button
if active_alerts:
    c1, c2, _ = st.columns([1, 1, 4])
    with c1:
        if st.button("Clear all alerts", type="secondary"):
            st.session_state.dismissed_alerts.update(a["id"] for a in all_alerts)
            st.rerun()
    with c2:
        if st.session_state.dismissed_alerts:
            if st.button("Reset dismissed"):
                st.session_state.dismissed_alerts.clear()
                st.rerun()

# Alert cards
if not active_alerts:
    st.markdown(
        "<div style='text-align:center;padding:60px 0'>"
        "<div style='font-size:2.5rem;margin-bottom:12px'>✅</div>"
        "<div style='font-size:1rem;font-weight:600;color:#8A93A6'>No active alerts</div>"
        "<div style='font-size:0.8rem;color:#606878;margin-top:4px'>All zones operating normally</div>"
        "</div>",
        unsafe_allow_html=True
    )
else:
    for alert in active_alerts:
        config = PRIORITY_CONFIG.get(alert["type"], PRIORITY_CONFIG["low"])

        col_alert, col_dismiss = st.columns([20, 1])
        with col_alert:
            st.markdown(f"""
            <div class="rec-card">
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
                    <span class="rec-badge rec-badge-{alert['type'].upper()}"
                          style="background:{config['color']}">{config['label']}</span>
                    <strong>{alert['zone']}</strong>
                </div>
                <div style="color:#8A93A6;font-size:0.85rem">{alert['message']}</div>
                <div style="display:flex;justify-content:space-between;margin-top:8px">
                    <span style="color:#606878;font-size:0.75rem">{alert['time']}</span>
                    <span style="color:#606878;font-size:0.75rem">Source: {alert['source']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with col_dismiss:
            if st.button("✕", key=f"dismiss_{alert['id']}", help="Dismiss this alert"):
                st.session_state.dismissed_alerts.add(alert["id"])
                st.rerun()
