"""Repeat-Offender Intelligence — the chronic minority driving a third of violations."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import ui, core

ui.page("Repeat Offenders", "\U0001F501")
ui.brand_sidebar()
df = ui.load_data()

st.markdown("## \U0001F501 Repeat-Offender Intelligence")
st.caption("15% of vehicles cause a third of all violations — find and target the chronic ones.")

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
