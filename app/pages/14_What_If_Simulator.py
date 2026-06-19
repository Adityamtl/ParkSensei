"""What-If Simulator — Estimate risk reduction from adding enforcement officers.
   Adapted from GRIDLOCK2.0 Prototype script 13_what_if_simulation.py."""
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import ui, core

ui.page("What-If Simulator", "W")
ui.brand_sidebar()

st.markdown("## What-If Enforcement Simulator")
st.caption("Estimate the impact of adding officers to specific zones — risk, congestion, and propagation reduction.")

zones = ui.get_zones()

# ===================== SINGLE ZONE SIMULATION =====================
st.subheader("Single Zone Simulation")

zone_labels = zones.head(30)["label"].tolist()
c1, c2, _ = st.columns([2, 1, 1])
selected_zone = c1.selectbox("Target zone", zone_labels, index=0)
additional = c2.slider("Additional officers", 1, 25, 5)

target_idx = zones[zones["label"] == selected_zone].index[0]
sim = core.what_if_simulation(zones, target_idx=target_idx, additional_officers=additional)

if "error" in sim:
    st.error(sim["error"])
    st.stop()

# Before / After gauges
st.markdown("---")
g1, g2, g3, g4 = st.columns(4)

with g1:
    st.markdown("""
    <div class="rec-card" style="text-align:center">
        <span style="color:#606878;font-size:0.75rem;text-transform:uppercase">Current Impact</span>
        <div style="font-size:2rem;font-weight:700;color:#EF4444">{:.0f}</div>
        <span style="color:#606878;font-size:0.75rem">{}</span>
    </div>
    """.format(sim["current_impact"], sim["zone_label"].split(" - ")[-1][:20]),
    unsafe_allow_html=True)

with g2:
    new_color = "#10B981" if sim["new_impact"] < 50 else "#F59E0B" if sim["new_impact"] < 70 else "#EF4444"
    st.markdown("""
    <div class="rec-card" style="text-align:center">
        <span style="color:#606878;font-size:0.75rem;text-transform:uppercase">New Impact</span>
        <div style="font-size:2rem;font-weight:700;color:{color}">{score:.0f}</div>
        <span style="color:#10B981;font-size:0.85rem;font-weight:600">▼ {reduction:.0f} pts ({pct:.0f}%)</span>
    </div>
    """.format(color=new_color, score=sim["new_impact"],
               reduction=sim["impact_reduction"], pct=sim["impact_reduction_pct"]),
    unsafe_allow_html=True)

with g3:
    st.markdown("""
    <div class="rec-card" style="text-align:center">
        <span style="color:#606878;font-size:0.75rem;text-transform:uppercase">Congestion Reduction</span>
        <div style="font-size:2rem;font-weight:700;color:#06B6D4">{:.1f}%</div>
        <span style="color:#606878;font-size:0.75rem">Estimated flow improvement</span>
    </div>
    """.format(sim["congestion_reduction"]),
    unsafe_allow_html=True)

with g4:
    st.markdown("""
    <div class="rec-card" style="text-align:center">
        <span style="color:#606878;font-size:0.75rem;text-transform:uppercase">Propagation Reduction</span>
        <div style="font-size:2rem;font-weight:700;color:#8B5CF6">{:.1f}%</div>
        <span style="color:#606878;font-size:0.75rem">Spill-over risk decrease</span>
    </div>
    """.format(sim["propagation_reduction"]),
    unsafe_allow_html=True)

# Visual before/after comparison
st.markdown("---")
bl, br = st.columns([2, 1], gap="large")

with bl:
    st.markdown("**Impact Score Gauge — Before vs After**")

    fig = go.Figure()
    # Before gauge
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=sim["new_impact"],
        delta={"reference": sim["current_impact"], "decreasing": {"color": "#10B981"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar": {"color": "#4C8BF5"},
            "steps": [
                {"range": [0, 35], "color": "#1A3A1A"},
                {"range": [35, 70], "color": "#3A2A1A"},
                {"range": [70, 100], "color": "#3A1A1A"},
            ],
            "threshold": {
                "line": {"color": "#EF4444", "width": 3},
                "thickness": 0.8,
                "value": sim["current_impact"],
            },
        },
        title={"text": f"Impact Score — {sim['zone_label'].split(' - ')[-1][:22]}"},
        domain={"x": [0, 1], "y": [0, 1]},
    ))
    fig.update_layout(height=280, margin=dict(l=30, r=30, t=60, b=10),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch")

with br:
    st.markdown("**Zone Details**")
    st.metric("Violations", f"{sim['violations']:,}")
    st.metric("Avg severity", f"{sim['avg_severity']:.2f}")
    st.metric("Junction exposure", f"{sim['junction_frac']:.0f}%")
    st.metric("Officers added", f"+{sim['additional_officers']}")

    # Recommendation for the zone
    recs = core.generate_recommendations(zones.iloc[target_idx])
    if recs:
        st.markdown("**Top recommendation:**")
        ui.render_recommendation_card(recs[0])

# ===================== MULTI-ZONE COMPARISON =====================
st.markdown("---")
st.subheader("Multi-Zone Strategy Comparison")
st.caption("Compare the effect of adding officers across multiple zones to find the best investment.")

mc1, mc2, _ = st.columns([2, 1, 1])
num_zones = mc1.slider("Compare top N zones", 5, 20, 10)
officers_each = mc2.slider("Officers per zone", 1, 20, 5, key="multi_officers")

target_indices = list(range(num_zones))
batch = core.what_if_batch(zones, target_indices, officers_each)
batch_df = pd.DataFrame(batch)

# Sort by impact reduction (descending)
batch_df = batch_df.sort_values("impact_reduction", ascending=False).reset_index(drop=True)
batch_df["label_short"] = batch_df["zone_label"].str.split(" - ").str[-1].str[:22]

# Comparison chart
comp_l, comp_r = st.columns([2, 1], gap="large")

with comp_l:
    st.markdown("**Impact Reduction per Zone**")
    st.caption(f"Adding {officers_each} officers to each zone. Sorted by effectiveness.")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=batch_df["label_short"],
        x=batch_df["current_impact"],
        name="Current Impact",
        orientation="h",
        marker_color="rgba(239, 68, 68, 0.3)",
    ))
    fig.add_trace(go.Bar(
        y=batch_df["label_short"],
        x=batch_df["new_impact"],
        name="New Impact",
        orientation="h",
        marker_color="#4C8BF5",
    ))
    fig.update_layout(
        height=max(350, len(batch_df) * 30),
        margin=dict(l=0, r=0, t=10, b=0),
        xaxis_title="Impact Score",
        barmode="overlay",
        yaxis={"categoryorder": "total ascending"},
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.1),
    )
    st.plotly_chart(fig, width="stretch")

with comp_r:
    st.markdown("**Best Investment Zones**")
    st.caption("Zones where adding officers has the highest effect.")

    top3 = batch_df.head(3)
    for _, row in top3.iterrows():
        pct = row["impact_reduction_pct"]
        badge = "🟢" if pct >= 15 else "🟡" if pct >= 8 else "🔵"
        st.markdown(
            f"{badge} **{row['label_short']}** — "
            f"{row['impact_reduction']:.0f} pt reduction "
            f"({pct:.0f}%)"
        )

    st.markdown("---")
    st.markdown("**Lowest ROI Zones**")
    bottom3 = batch_df.tail(3)
    for _, row in bottom3.iterrows():
        st.markdown(
            f"⚪ **{row['label_short']}** — "
            f"{row['impact_reduction']:.0f} pt "
            f"({row['impact_reduction_pct']:.0f}%)"
        )

# Comparison table
with st.expander("Full simulation results"):
    show_batch = batch_df[["zone_label", "current_impact", "new_impact",
                            "impact_reduction", "impact_reduction_pct",
                            "congestion_reduction", "propagation_reduction",
                            "violations"]].copy()
    show_batch.columns = ["Zone", "Current", "After", "Δ Impact",
                           "Δ %", "Congest. Δ%", "Propag. Δ%", "Violations"]
    st.dataframe(show_batch, hide_index=True, width="stretch", height=400)

    st.download_button("Download simulation results (CSV)",
                       batch_df.to_csv(index=False).encode(),
                       file_name="parksensei_whatif_simulation.csv",
                       mime="text/csv")

with st.expander("How the simulator works"):
    st.markdown(
        "**Formula (from GRIDLOCK2.0 Prototype):**\n\n"
        "- Each additional officer reduces the zone's impact score by **2 points**\n"
        "- Congestion reduction = impact change × **0.8** (80% of risk reduction flows to traffic)\n"
        "- Propagation reduction = impact change × **0.6** (60% reduces spill-over to neighbours)\n\n"
        "**Limitations:**\n"
        "- Linear estimation — real-world effects are non-linear with diminishing returns\n"
        "- Does not account for officer fatigue, shift patterns, or inter-zone coordination\n"
        "- Best used for comparative analysis (which zones benefit most) rather than absolute numbers"
    )
