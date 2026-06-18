"""Coverage & ROI — quantify the payoff of targeted deployment and expose enforcement blind spots."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import ui, core

ui.page("Coverage & ROI", "\U0001F3AF")
ui.brand_sidebar()
df = ui.load_data()
fc = ui.get_forecaster()

st.markdown("## \U0001F3AF Coverage & ROI")
st.caption("Why targeted deployment beats spreading thin — and where enforcement isn't looking.")

# ============================ ROI / targeting ============================
st.subheader("Enforcement ROI — the power of targeting")
c1, c2, c3 = st.columns([1.2, 1.6, 1])
day  = c1.selectbox("Weekday", ui.DOW_NAMES, index=5)
dow  = ui.DOW_NAMES.index(day)
win  = c2.slider("Shift window (IST)", 0, 23, (9, 13))
teams = c3.slider("Patrol teams", 3, 20, 8)
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
