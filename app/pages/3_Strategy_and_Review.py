"""Strategy & Review — Enforcement Actions, Repeat Offenders, Coverage & ROI."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import ui, core

ui.page("Strategy & Review", "S")
ui.brand_sidebar()


# =====================================================================
# TAB 1: ENFORCEMENT ACTIONS
# =====================================================================
def _render_enforcement_actions():
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


# =====================================================================
# TAB 2: REPEAT OFFENDERS
# =====================================================================
def _render_repeat_offenders():
    df = ui.load_data()

    st.markdown("## Repeat-Offender Intelligence")
    st.caption("15% of vehicles account for a third of all violations. Identify and target chronic offenders.")

    vc = core.vehicle_counts(df)
    repeat = vc[vc >= 2]
    rep_share = repeat.sum() / len(df)

    k = st.columns(4)
    ui.kpi(k[0], "Unique vehicles", f"{len(vc):,}")
    ui.kpi(k[1], "Repeat offenders", f"{len(repeat):,}", help="caught ≥ 2 times")
    ui.kpi(k[2], "Their share of violations", f"{rep_share*100:.0f}%")
    ui.kpi(k[3], "Worst offender",
           f"{int(vc.iloc[0])}×" if len(vc) else "—",
           help=f"vehicle {vc.index[0]}" if len(vc) else None)

    st.markdown("---")
    left, right = st.columns([1.3, 1], gap="large")

    with left:
        st.markdown("**Most-cited vehicles**")
        top = vc.head(25).rename("violations").reset_index()
        top.columns = ["vehicle", "violations"]
        meta = (df[df["vehicle_number"].isin(top["vehicle"])]
                .groupby("vehicle_number")
                .agg(vehicle_type=("vehicle_type", lambda s: s.mode().iat[0]),
                     zones=("gh6", "nunique"),
                     top_area=("police_station", lambda s: s[s.str.lower() != "nan"].mode().iat[0]
                               if (s.str.lower() != "nan").any() else "—"))
                .reset_index().rename(columns={"vehicle_number": "vehicle"}))
        top = top.merge(meta, on="vehicle")
        top.columns = ["Vehicle", "Times caught", "Type", "Distinct zones", "Main area"]
        st.dataframe(top, hide_index=True, width="stretch", height=520)

    with right:
        st.markdown("**How repeat behaviour is distributed**")
        bins = pd.cut(vc, [0, 1, 2, 3, 5, 10, 1000],
                      labels=["1", "2", "3", "4–5", "6–10", "10+"])
        dist = vc.groupby(bins, observed=True).agg(vehicles="count", violations="sum").reset_index()
        dist.columns = ["times_caught", "vehicles", "violations"]
        fig = px.bar(dist, x="times_caught", y="violations", color="violations",
                     color_continuous_scale="Reds", text="vehicles")
        fig.update_traces(texttemplate="%{text:,} veh", textposition="outside")
        fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="times caught", yaxis_title="violations",
                          coloraxis_showscale=False,
                          paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig, width="stretch")
        st.info("**Action:** vehicles in the 6+ buckets are prime candidates for escalated penalties, "
                "towing priority, or registered-owner notices — a small list with outsized impact.")


# =====================================================================
# TAB 3: COVERAGE & ROI
# =====================================================================
def _render_coverage_roi():
    df = ui.load_data()
    fc = ui.get_forecaster()

    st.markdown("## Coverage & ROI")
    st.caption("Why targeted deployment outperforms even coverage, and where enforcement gaps exist.")

    # ============================ ROI / targeting ============================
    st.subheader("Enforcement ROI — the power of targeting")
    c1, c2, c3 = st.columns([1.2, 1.6, 1])
    day  = c1.selectbox("Weekday", ui.DOW_NAMES, index=5, key="roi_weekday")
    dow  = ui.DOW_NAMES.index(day)
    win  = c2.slider("Shift window (IST)", 0, 23, (9, 13), key="roi_window")
    teams = c3.slider("Patrol teams", 3, 20, 8, key="roi_teams")
    hours = list(range(win[0], win[1] + 1))

    pred = core.predict_load(fc, dow, hours)
    roi  = core.roi_curve(pred, 20)
    row  = roi[roi["teams"] == teams].iloc[0]
    n_active = len(pred)

    over50 = roi[roi["optimal"] >= 0.5]
    k50 = f"{int(over50['teams'].iloc[0])}" if len(over50) else ">20"
    slow = roi[(roi["teams"] >= 3) & (roi["marginal"] < 0.01)]
    k_sweet = int(slow["teams"].iloc[0]) if len(slow) else int(roi["teams"].iloc[-1])

    k = st.columns(3)
    ui.kpi(k[0], f"Covered by {teams} teams", f"{row['optimal']*100:.0f}%",
           help=f"share of predicted violations at the {teams} highest-impact zones (of {n_active} active)")
    ui.kpi(k[1], "Half the violations sit in", f"{k50} zones",
           help=f"out of {n_active} active zones this shift — extreme concentration")
    ui.kpi(k[2], "Staffing sweet spot", f"~{k_sweet} teams",
           help="beyond this, each extra team adds < 1 percentage point of coverage")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=roi["teams"], y=roi["optimal"]*100, name="ParkSensei targeting",
                             line=dict(color=ui.ACCENT, width=3),
                             fill="tozeroy", fillcolor="rgba(76,139,245,0.12)"))
    fig.add_trace(go.Scatter(x=roi["teams"], y=roi["even"]*100, name="Untargeted / even spread",
                             line=dict(color="#777", width=2, dash="dash")))
    fig.add_vline(x=teams, line_color="#bbb", line_dash="dot")
    fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0),
                      xaxis_title="patrol teams", yaxis_title="% of predicted violations covered",
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      legend=dict(orientation="h", y=1.15, title=""))
    st.plotly_chart(fig, width="stretch")
    st.caption(
        f"**{teams} teams** placed by ParkSensei cover **{row['optimal']*100:.0f}%** of predicted violations "
        f"this shift — because half of all activity is concentrated in just **{k50} zones**. Targeting isn't a "
        f"nicety, it's the whole game. And past **~{k_sweet} teams** each extra team adds little — that's the "
        "data-driven staffing sweet spot.")

    # ============================ coverage blind spots ============================
    st.markdown("---")
    st.subheader("Coverage blind spots — when enforcement isn't looking")
    cov = core.coverage_by_hour(df)
    before1pm = cov[cov["hour"] < 13]["share"].sum()
    evening   = cov[cov["hour"].between(17, 21)]["share"].sum()

    g1, g2 = st.columns([1, 2], gap="large")
    with g1:
        st.metric("Logged before 1 PM", f"{before1pm*100:.0f}%")
        st.metric("Logged 5–9 PM (evening peak)", f"{evening*100:.1f}%")
        st.markdown(
            "Evenings are a **near-total blind spot.** Commercial-hour parking after work goes almost entirely "
            "uncaught. This is the single clearest opportunity: a **targeted evening shift** would surface "
            "violations the current pattern never sees.")
    with g2:
        cov2 = cov.copy()
        cov2["band"] = np.where(cov2["hour"].between(17, 21), "Evening blind spot (5–9 PM)",
                       np.where(cov2["hour"] < 13, "Current focus (before 1 PM)", "Other hours"))
        fig2 = px.bar(cov2, x="hour", y="share", color="band",
                      color_discrete_map={"Current focus (before 1 PM)": ui.ACCENT,
                                          "Evening blind spot (5–9 PM)": "#E2352B",
                                          "Other hours": "#3a4254"})
        fig2.update_layout(height=330, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_title="hour of day (IST)", yaxis_title="share of enforcement",
                           yaxis_tickformat=".0%", paper_bgcolor="rgba(0,0,0,0)",
                           plot_bgcolor="rgba(0,0,0,0)", legend=dict(orientation="h", y=1.18, title=""))
        st.plotly_chart(fig2, width="stretch")


# =====================================================================
# RENDER TABS
# =====================================================================
tab1, tab2, tab3 = st.tabs(["⚡ Enforcement Actions", "🔄 Repeat Offenders", "📊 Coverage & ROI"])

with tab1:
    _render_enforcement_actions()
with tab2:
    _render_repeat_offenders()
with tab3:
    _render_coverage_roi()
