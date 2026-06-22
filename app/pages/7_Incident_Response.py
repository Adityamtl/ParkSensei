"""Incident Response - Dynamic Routing, Emergency Corridors, Police Dispatch, Transit Advisor,
   KNN Intelligence, Road Closure Predictor, After-Action Feedback Loop."""
import math
from datetime import datetime
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import ui, core, traffic_network as tn, incident_db

ui.page("Incident Response", "R")
ui.brand_sidebar()

# Build the road network graph once
@st.cache_resource
def _build_graph():
    return tn.create_bengaluru_graph()

G = _build_graph()

# ===================== SHARED CONTROLS =====================
ZONE_NAMES = sorted(list(tn.NODE_COORDS.keys()))
HOSPITAL_NAMES = list(tn.HOSPITAL_COORDS.keys())
RISK_PRESETS = {
    "Minor Traffic Jam": 25,
    "Moderate Congestion": 55,
    "Major Accident": 80,
    "Severe Gridlock / VIP Movement": 95,
}

st.markdown("## Incident Response")
st.caption("Real-time incident response: routing diversions, emergency corridors, police dispatch, and transit advisories "
           "powered by Dijkstra's algorithm on Bengaluru's 20-node road network.")

# Controls row
ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([1.5, 1.5, 1.2, 1])
with ctrl1:
    incident_zone = st.selectbox("Incident zone", ZONE_NAMES,
                                  index=ZONE_NAMES.index("Central Zone 1"),
                                  help="Select the zone where the incident occurred")
with ctrl2:
    scenario = st.selectbox("Scenario severity", list(RISK_PRESETS.keys()),
                             index=1, help="Preset risk score based on incident type")
with ctrl3:
    risk_score = st.slider("Risk score override", 0, 100,
                            RISK_PRESETS[scenario],
                            help="Adjust the risk score manually for fine-grained simulation")
with ctrl4:
    risk_level = "Severe" if risk_score >= 75 else "Moderate" if risk_score >= 45 else "Minor"
    risk_color_hex = {"Minor": "#10B981", "Moderate": "#F59E0B", "Severe": "#EF4444"}[risk_level]
    st.markdown(f"""
    <div style="text-align:center;padding:8px;border-radius:10px;border:1px solid #1E2638;
                background:linear-gradient(180deg, #151A25 0%, #121620 100%);
                box-shadow:0 1px 4px rgba(0,0,0,0.18)">
        <div style="font-size:0.75rem;color:#8A93A6;text-transform:uppercase;letter-spacing:0.04em;font-weight:500">Risk Level</div>
        <div style="font-size:1.5rem;font-weight:700;color:{risk_color_hex};margin-top:2px">{risk_level.upper()}</div>
        <div style="font-size:0.8rem;color:#606878">{risk_score}/100</div>
    </div>
    """, unsafe_allow_html=True)

# KPIs row
inc_lat, inc_lon = tn.NODE_COORDS[incident_zone]
penalty = 1.0 + (risk_score / 20.0)

k = st.columns(5)
ui.kpi(k[0], "Incident zone", incident_zone.split()[-1][:16])
ui.kpi(k[1], "Risk score", f"{risk_score}/100")
ui.kpi(k[2], "Congestion multiplier", f"{penalty:.1f}x", help="Speed reduction on affected road edges")
ui.kpi(k[3], "Network nodes", f"{G.number_of_nodes()}")
ui.kpi(k[4], "Road segments", f"{G.number_of_edges()}")


# =====================================================================
# HELPER: PyDeck base layers for the road network
# =====================================================================
def _network_base_layers():
    """Background road graph."""
    node_data = pd.DataFrame([
        {"name": name, "lat": coords[0], "lon": coords[1]}
        for name, coords in tn.NODE_COORDS.items()
    ])
    node_layer = pdk.Layer(
        "ScatterplotLayer", data=node_data,
        get_position=["lon", "lat"], get_radius=250,
        get_fill_color=[160, 170, 190, 120], pickable=True, auto_highlight=True,
    )
    edge_data = []
    for u, v in tn.ROAD_EDGES:
        c1 = tn.NODE_COORDS[u]
        c2 = tn.NODE_COORDS[v]
        edge_data.append({"path": [[c1[1], c1[0]], [c2[1], c2[0]]], "name": f"{u} - {v}"})
    edge_layer = pdk.Layer(
        "PathLayer", data=pd.DataFrame(edge_data),
        get_path="path", get_color=[47, 49, 73, 150],
        width_min_pixels=1.5, get_width=3,
    )
    return [edge_layer, node_layer]


def _incident_marker_layer():
    """Incident zone marker."""
    inc_df = pd.DataFrame([{
        "lat": inc_lat, "lon": inc_lon, "name": incident_zone,
        "risk": risk_level, "score": risk_score
    }])
    fill = ([226, 53, 43, 180] if risk_level == "Severe"
            else [245, 158, 65, 160] if risk_level == "Moderate"
            else [16, 185, 129, 140])
    return pdk.Layer(
        "ScatterplotLayer", data=inc_df,
        get_position=["lon", "lat"], get_radius=600,
        get_fill_color=fill, stroked=True, get_line_color=[255, 255, 255, 200],
        line_width_min_pixels=2, pickable=True,
    )


# =====================================================================
# TAB 1: DYNAMIC ROUTING
# =====================================================================
def _render_dynamic_routing():
    st.markdown("## Dynamic Diversion Routing")
    st.caption("Simulates a commuter journey traversing the city through the affected zone. "
               "Dijkstra's algorithm computes the optimal detour path and delay savings.")

    rc1, rc2, _ = st.columns([1, 1, 2])
    with rc1:
        origin = st.selectbox("Commuter origin", ZONE_NAMES,
                               index=ZONE_NAMES.index("West Zone 1"), key="route_origin")
    with rc2:
        dest = st.selectbox("Commuter destination", ZONE_NAMES,
                             index=ZONE_NAMES.index("East Zone 2"), key="route_dest")

    if origin == dest:
        st.error("Origin and destination must be different zones.")
        return

    routing = tn.get_routing_scenarios(G, source=origin, target=dest,
                                        incident_node=incident_zone, risk_score=risk_score)
    if not routing:
        st.error("Could not compute routing — zones may not be connected.")
        return

    # KPIs
    rk = st.columns(5)
    ui.kpi(rk[0], "Standard time", f"{routing['std_time_mins']} min",
           help="Normal travel time without congestion")
    ui.kpi(rk[1], "Stuck time", f"{routing['stuck_time_mins']} min",
           help="Time if you take the standard route through congestion")
    ui.kpi(rk[2], "Diversion time", f"{routing['congested_time_mins']} min",
           help="Time via optimal bypass route")
    savings = routing['savings_mins']
    ui.kpi(rk[3], "Time saved", f"{savings} min",
           help="Delay saved by taking the diversion")
    ui.kpi(rk[4], "Diversion distance", f"{routing['congested_distance']:.1f} km")

    st.markdown("---")
    map_col, info_col = st.columns([2, 1], gap="large")

    with map_col:
        st.subheader("Route Comparison Map")
        st.caption("Red = standard route (gridlocked) · Green = optimal diversion bypass")

        layers = _network_base_layers() + [_incident_marker_layer()]

        # Standard route (red)
        std_coords = [[tn.NODE_COORDS[n][1], tn.NODE_COORDS[n][0]] for n in routing["std_path"]]
        layers.append(pdk.Layer(
            "PathLayer", data=pd.DataFrame([{"path": std_coords, "name": "Standard Route (Gridlocked)"}]),
            get_path="path", get_color=[229, 83, 83, 200], width_min_pixels=3, get_width=6,
        ))

        # Diversion route (green)
        div_coords = [[tn.NODE_COORDS[n][1], tn.NODE_COORDS[n][0]] for n in routing["congested_path"]]
        layers.append(pdk.Layer(
            "PathLayer", data=pd.DataFrame([{"path": div_coords, "name": "Diversion Bypass Route"}]),
            get_path="path", get_color=[46, 204, 113, 220], width_min_pixels=4, get_width=8,
        ))

        # Origin & destination markers
        markers = pd.DataFrame([
            {"lat": tn.NODE_COORDS[origin][0], "lon": tn.NODE_COORDS[origin][1],
             "name": f"Origin: {origin}", "color": [16, 185, 129]},
            {"lat": tn.NODE_COORDS[dest][0], "lon": tn.NODE_COORDS[dest][1],
             "name": f"Destination: {dest}", "color": [76, 139, 245]},
        ])
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=markers, get_position=["lon", "lat"],
            get_radius=400, get_fill_color="color", stroked=True,
            get_line_color=[255, 255, 255], line_width_min_pixels=2, pickable=True,
        ))

        tip = {"html": "<b>{name}</b>",
               "style": {"backgroundColor": "#161B26", "color": "#E6E9EF", "fontSize": "12px"}}
        st.pydeck_chart(ui.deck(layers, ui.view(inc_lat, inc_lon, zoom=11.5, pitch=30), tip),
                        width="stretch")

    with info_col:
        st.subheader("Routing Details")

        st.markdown("**Standard path:**")
        st.markdown(f"<div style='color:#E55353;font-size:0.85rem'>"
                    f"{'  →  '.join(routing['std_path'])}</div>", unsafe_allow_html=True)
        st.metric("Distance", f"{routing['std_distance']:.1f} km")

        st.markdown("---")
        st.markdown("**Diversion path:**")
        st.markdown(f"<div style='color:#2ECC71;font-size:0.85rem'>"
                    f"{'  →  '.join(routing['congested_path'])}</div>", unsafe_allow_html=True)
        st.metric("Distance", f"{routing['congested_distance']:.1f} km")

        st.markdown("---")
        if savings > 0:
            st.success(f"Diversion saves ~{savings} minutes versus sitting in gridlock.")
        else:
            st.info("Standard path is still optimal — congestion is not severe enough to warrant a detour.")

        st.markdown("**Congestion multiplier**")
        st.markdown(f"""
        <div class="rec-card" style="text-align:center">
            <div style="font-size:2rem;font-weight:700;color:{risk_color_hex}">{penalty:.1f}x</div>
            <div style="font-size:0.75rem;color:#8A93A6">Speed reduction on affected edges</div>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# TAB 2: EMERGENCY GREEN CORRIDOR
# =====================================================================
def _render_emergency_corridor():
    st.markdown("## Emergency Green Corridor Planner")
    st.caption("Plan an ambulance route from the incident zone to the nearest hospital. "
               "Auto-generates a signal preemption schedule for traffic controllers.")

    ec1, ec2, _ = st.columns([1.5, 1, 2])
    with ec1:
        closest_hospital = min(HOSPITAL_NAMES,
                                key=lambda h: tn.haversine_distance(
                                    (inc_lat, inc_lon), tn.HOSPITAL_COORDS[h]))
        target_hospital = st.selectbox("Target hospital", HOSPITAL_NAMES,
                                        index=HOSPITAL_NAMES.index(closest_hospital),
                                        help="System auto-recommends closest facility")
    with ec2:
        dist_to_hosp = tn.haversine_distance((inc_lat, inc_lon),
                                              tn.HOSPITAL_COORDS[target_hospital])
        st.markdown(f"""
        <div class="rec-card" style="text-align:center;margin-top:24px">
            <div style="font-size:0.75rem;color:#8A93A6;text-transform:uppercase;letter-spacing:0.04em">Direct distance</div>
            <div style="font-size:1.3rem;font-weight:700;color:{ui.ACCENT}">{dist_to_hosp:.1f} km</div>
        </div>
        """, unsafe_allow_html=True)

    corridor = tn.get_emergency_corridor(G, incident_zone=incident_zone,
                                          hospital_name=target_hospital, risk_score=risk_score)
    if not corridor:
        st.error("Could not compute emergency corridor route.")
        return

    normal_eta = corridor['distance_km'] * 2.0  # 30 km/h
    time_saved = normal_eta - corridor['eta_mins']

    # KPIs
    ek = st.columns(4)
    ui.kpi(ek[0], "Corridor distance", f"{corridor['distance_km']:.1f} km")
    ui.kpi(ek[1], "ETA (emergency speed)", f"{corridor['eta_mins']:.0f} min",
           help="Emergency vehicle at 50 km/h with green corridor")
    ui.kpi(ek[2], "Intersections to clear", f"{len(corridor['schedule']) - 1}",
           help="Signal preemption points along the route")
    ui.kpi(ek[3], "Normal ETA (no corridor)", f"{normal_eta:.0f} min",
           help="Without green corridor, normal traffic speed")

    st.markdown("---")
    map_col, detail_col = st.columns([2, 1], gap="large")

    with map_col:
        st.subheader("Emergency Corridor Route")
        st.caption("Blue line = green corridor path. Yellow dots = signal preemption intersections.")

        layers = _network_base_layers() + [_incident_marker_layer()]

        # Corridor path (blue)
        corridor_coords = [[c[1], c[0]] for c in corridor["coords_path"]]
        layers.append(pdk.Layer(
            "PathLayer", data=pd.DataFrame([{"path": corridor_coords, "name": "Emergency Green Corridor"}]),
            get_path="path", get_color=[51, 153, 255, 230], width_min_pixels=5, get_width=10,
        ))

        # Hospital marker
        hosp_coords = tn.HOSPITAL_COORDS[target_hospital]
        hosp_df = pd.DataFrame([{
            "lat": hosp_coords[0], "lon": hosp_coords[1],
            "name": target_hospital, "eta": f"{corridor['eta_mins']:.0f} min"
        }])
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=hosp_df, get_position=["lon", "lat"],
            get_radius=500, get_fill_color=[51, 153, 255, 200],
            stroked=True, get_line_color=[255, 255, 255], line_width_min_pixels=2, pickable=True,
        ))

        # Signal preemption point markers (yellow)
        sched_points = []
        for s in corridor["schedule"][:-1]:
            if s["node"] in tn.NODE_COORDS:
                coords = tn.NODE_COORDS[s["node"]]
                sched_points.append({
                    "lat": coords[0], "lon": coords[1],
                    "name": s["node"], "eta": s["eta_str"], "window": s["preempt_window"],
                })
        if sched_points:
            layers.append(pdk.Layer(
                "ScatterplotLayer", data=pd.DataFrame(sched_points), get_position=["lon", "lat"],
                get_radius=300, get_fill_color=[245, 205, 90, 200],
                stroked=True, get_line_color=[255, 255, 255, 150],
                line_width_min_pixels=1, pickable=True,
            ))

        tip = {"html": "<b>{name}</b><br/>ETA: {eta}",
               "style": {"backgroundColor": "#161B26", "color": "#E6E9EF", "fontSize": "12px"}}
        center_lat = np.mean([c[0] for c in corridor["coords_path"]])
        center_lon = np.mean([c[1] for c in corridor["coords_path"]])
        st.pydeck_chart(ui.deck(layers, ui.view(center_lat, center_lon, zoom=12, pitch=25), tip),
                        width="stretch")

    with detail_col:
        st.subheader("Signal Preemption Schedule")
        st.caption("Traffic controllers must force green state at each intersection before the ETA window.")

        schedule_df = pd.DataFrame(corridor["schedule"])
        schedule_df.columns = ["Intersection", "Distance (km)", "ETA", "Override Window"]
        st.dataframe(schedule_df, hide_index=True, width="stretch", height=350)

        st.markdown("---")
        st.markdown("**Controller Instructions**")
        st.markdown("""
        <div class="rec-card">
            <div style="font-size:0.85rem;color:#E6E9EF">
                <strong>1.</strong> Broadcast preemptive signals to all roadside controllers.<br/>
                <strong>2.</strong> Force GREEN state at intersections matching the schedule.<br/>
                <strong>3.</strong> Clear central lane approaches <strong>2 min</strong> before ETA window.<br/>
                <strong>4.</strong> Resume normal signal cycle after ambulance passes.
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.metric("Time saved with corridor", f"{time_saved:.0f} min",
                  delta=f"{normal_eta:.0f} min → {corridor['eta_mins']:.0f} min",
                  delta_color="inverse")


# =====================================================================
# TAB 3: POLICE DISPATCH OPTIMIZER
# =====================================================================
def _render_police_dispatch():
    st.markdown("## Police Dispatch Optimizer")
    st.caption("Allocate officers and patrol vehicles from the nearest depots with available capacity. "
               "Travel times calculated via shortest-path on the road network.")

    dc1, dc2, _ = st.columns([1, 1, 2])
    with dc1:
        req_officers = st.slider("Required officers", 2, 50, 15,
                                  help="Total officers needed at incident site")
    with dc2:
        req_cars = st.slider("Required patrol vehicles", 1, 15, 4,
                              help="Total patrol cars needed")

    dispatch_res, unmet_off, unmet_cars = tn.optimize_police_dispatch(
        required_officers=req_officers, required_cars=req_cars,
        incident_zone=incident_zone, G=G
    )

    total_dispatched_off = sum(d["officers_dispatched"] for d in dispatch_res)
    total_dispatched_cars = sum(d["cars_dispatched"] for d in dispatch_res)

    # KPIs
    dk = st.columns(5)
    ui.kpi(dk[0], "Officers dispatched", f"{total_dispatched_off}/{req_officers}")
    ui.kpi(dk[1], "Vehicles dispatched", f"{total_dispatched_cars}/{req_cars}")
    ui.kpi(dk[2], "Stations activated", f"{len(dispatch_res)}")
    if dispatch_res:
        ui.kpi(dk[3], "Fastest arrival", f"{dispatch_res[0]['travel_time_mins']} min",
               help=dispatch_res[0]["station"])
        ui.kpi(dk[4], "Slowest arrival", f"{dispatch_res[-1]['travel_time_mins']} min",
               help=dispatch_res[-1]["station"])
    else:
        ui.kpi(dk[3], "Fastest arrival", "—")
        ui.kpi(dk[4], "Slowest arrival", "—")

    if unmet_off > 0 or unmet_cars > 0:
        st.error(f"Resource shortfall: {unmet_off} officers and {unmet_cars} patrol cars "
                 f"could not be sourced from nearby stations. Initiate regional mutual aid.")
    else:
        st.success("Target resource requirements fully covered by adjacent police stations.")

    st.markdown("---")
    map_col, table_col = st.columns([2, 1], gap="large")

    with map_col:
        st.subheader("Dispatch Network Map")
        st.caption("Red = incident zone. Blue = police station depots. Lines = dispatch routes.")

        layers = _network_base_layers() + [_incident_marker_layer()]

        for d in dispatch_res:
            st_coords = tn.POLICE_STATIONS[d["station"]]["coords"]
            # Dispatch line
            layers.append(pdk.Layer(
                "PathLayer",
                data=pd.DataFrame([{"path": [[st_coords[1], st_coords[0]], [inc_lon, inc_lat]],
                                     "name": f"{d['station']} dispatch"}]),
                get_path="path", get_color=[76, 139, 245, 180], width_min_pixels=2, get_width=4,
            ))
            # Station marker
            layers.append(pdk.Layer(
                "ScatterplotLayer",
                data=pd.DataFrame([{"lat": st_coords[0], "lon": st_coords[1],
                                     "name": d['station'],
                                     "officers": d["officers_dispatched"],
                                     "cars": d["cars_dispatched"],
                                     "eta": f"{d['travel_time_mins']} min"}]),
                get_position=["lon", "lat"], get_radius=350,
                get_fill_color=[76, 139, 245, 220], stroked=True,
                get_line_color=[255, 255, 255], line_width_min_pixels=2, pickable=True,
            ))

        # Standby stations (faded)
        for station_name, info in tn.POLICE_STATIONS.items():
            if not any(d["station"] == station_name for d in dispatch_res):
                layers.append(pdk.Layer(
                    "ScatterplotLayer",
                    data=pd.DataFrame([{"lat": info["coords"][0], "lon": info["coords"][1],
                                         "name": f"{station_name} (standby)",
                                         "officers": 0, "cars": 0, "eta": "—"}]),
                    get_position=["lon", "lat"], get_radius=250,
                    get_fill_color=[100, 110, 130, 100], pickable=True,
                ))

        tip = {"html": "<b>{name}</b><br/>Officers: {officers}<br/>Cars: {cars}<br/>ETA: {eta}",
               "style": {"backgroundColor": "#161B26", "color": "#E6E9EF", "fontSize": "12px"}}
        st.pydeck_chart(ui.deck(layers, ui.view(inc_lat, inc_lon, zoom=11.5, pitch=30), tip),
                        width="stretch")

    with table_col:
        st.subheader("Dispatch Schedule")
        st.caption("Sorted by travel time — closest stations dispatched first.")

        if dispatch_res:
            for d in dispatch_res:
                eta = d["travel_time_mins"]
                badge_color = "#10B981" if eta <= 10 else "#F59E0B" if eta <= 20 else "#EF4444"
                st.markdown(f"""
                <div class="rec-card" style="margin-bottom:10px">
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <strong>{d['station']}</strong>
                        <span class="rec-badge" style="background:{badge_color}">{eta} min</span>
                    </div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:4px;margin-top:8px;font-size:0.8rem">
                        <div><span style="color:#606878">Officers</span><br/><strong>{d['officers_dispatched']}</strong></div>
                        <div><span style="color:#606878">Vehicles</span><br/><strong>{d['cars_dispatched']}</strong></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.warning("No police stations could be reached from this incident zone.")

        st.markdown("---")
        st.markdown("**Station Capacity Overview**")
        cap_rows = []
        for name, info in tn.POLICE_STATIONS.items():
            dispatched = next((d for d in dispatch_res if d["station"] == name), None)
            cap_rows.append({
                "Station": name.replace(" Police Station", "").replace(" Traffic Station", ""),
                "Officers": info["officers"],
                "Cars": info["cars"],
                "Status": "Dispatched" if dispatched else "Standby"
            })
        st.dataframe(pd.DataFrame(cap_rows), hide_index=True, width="stretch")


# =====================================================================
# TAB 4: BMTC TRANSIT ADVISOR
# =====================================================================
def _render_transit_advisor():
    st.markdown("## BMTC Transit Advisory")
    st.caption("Detect which public bus routes intersect the incident zone, calculate diversion delays, "
               "and generate commuter advisories with shifted boarding stops.")

    transit_res = tn.check_bmtc_transit(G, incident_node=incident_zone, risk_score=risk_score)

    # KPIs
    tk = st.columns(4)
    ui.kpi(tk[0], "Routes analysed", f"{len(tn.BMTC_ROUTES)}")
    ui.kpi(tk[1], "Routes disrupted", f"{len(transit_res)}",
           help="Bus routes passing through or near the incident zone")
    total_delay = sum(r["estimated_delay_mins"] for r in transit_res) if transit_res else 0
    ui.kpi(tk[2], "Total delay impact", f"{total_delay} min")
    rerouted = sum(1 for r in transit_res if r["status"] == "Delayed & Rerouted") if transit_res else 0
    ui.kpi(tk[3], "Rerouted", f"{rerouted}")

    if not transit_res:
        st.success("No active BMTC transit routes are disrupted by this incident. Normal operations.")
        st.markdown("---")
        st.markdown("**All monitored BMTC routes**")
        for route_name, path in tn.BMTC_ROUTES.items():
            st.markdown(f"- **{route_name}**: {' → '.join(path)}")
        return

    st.markdown("---")

    # Affected route cards
    for route in transit_res:
        status_color = "#EF4444" if route["status"] == "Delayed & Rerouted" else "#F59E0B"
        delay = route["estimated_delay_mins"]

        st.markdown(f"""
        <div class="rec-card" style="border-left:3px solid {status_color};margin-bottom:14px">
            <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
                <strong style="font-size:0.95rem">{route['name']}</strong>
                <span class="rec-badge" style="background:{status_color}">{route['status']}</span>
            </div>
            <div style="margin-bottom:6px">
                <span style="color:#606878;font-size:0.8rem">Standard path:</span>
                <span style="font-size:0.8rem;color:#E6E9EF"> {route['standard_stops']}</span>
            </div>
            <div style="margin-bottom:6px">
                <span style="color:#606878;font-size:0.8rem">Diverted path:</span>
                <span style="font-size:0.8rem;color:#2ECC71;font-weight:500"> {route['diverted_stops']}</span>
            </div>
            <div style="margin-top:8px">
                <span style="color:#606878;font-size:0.75rem">Estimated delay</span><br/>
                <strong style="color:{status_color}">+{delay} minutes</strong>
            </div>
            <div style="margin-top:10px;background:#0D1520;border-radius:6px;padding:10px 12px">
                <span style="color:#F5CD5A;font-weight:600;font-size:0.8rem">Commuter advisory — </span>
                <span style="color:#E6E9EF;font-size:0.8rem">{route['shifted_stop_advise']}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Route disruption map
    st.markdown("---")
    st.subheader("Disruption Impact Map")
    st.caption("Bus routes passing through the incident zone. Coloured lines = disrupted routes.")

    layers = _network_base_layers() + [_incident_marker_layer()]
    route_colors = [[229, 83, 83, 180], [245, 158, 65, 180], [139, 92, 246, 180], [6, 182, 212, 180]]
    for i, route in enumerate(transit_res):
        stops = [s.strip() for s in route["standard_stops"].split("->")]
        route_coords = []
        for stop in stops:
            if stop in tn.NODE_COORDS:
                c = tn.NODE_COORDS[stop]
                route_coords.append([c[1], c[0]])
        if len(route_coords) >= 2:
            layers.append(pdk.Layer(
                "PathLayer",
                data=pd.DataFrame([{"path": route_coords, "name": route["name"]}]),
                get_path="path", get_color=route_colors[i % len(route_colors)],
                width_min_pixels=3, get_width=6,
            ))

    tip = {"html": "<b>{name}</b>",
           "style": {"backgroundColor": "#161B26", "color": "#E6E9EF", "fontSize": "12px"}}
    st.pydeck_chart(ui.deck(layers, ui.view(inc_lat, inc_lon, zoom=11, pitch=25), tip),
                    width="stretch")

    # Summary table
    st.markdown("---")
    summary_rows = []
    for r in transit_res:
        summary_rows.append({
            "Route": r["name"],
            "Status": r["status"],
            "Delay (min)": r["estimated_delay_mins"],
            "Advisory": r["shifted_stop_advise"],
        })
    st.dataframe(pd.DataFrame(summary_rows), hide_index=True, width="stretch")


# =====================================================================
# TAB 5: KNN SIMILAR INCIDENT FINDER
# =====================================================================
@st.cache_resource
def _build_knn():
    """Train a KNN model on the violations dataset for similar incident retrieval."""
    df = ui.load_data()
    features = df[["lat", "lon", "hour", "severity"]].copy()
    if "pcu" in df.columns:
        features["pcu"] = df["pcu"]
    else:
        features["pcu"] = 1.0
    features = features.dropna()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features.values)
    knn = NearestNeighbors(n_neighbors=20, metric="euclidean", algorithm="ball_tree")
    knn.fit(X_scaled)
    return knn, scaler, features, df.loc[features.index]


def _render_knn_intelligence():
    st.markdown("## Similar Incident Intelligence")
    st.caption("K-Nearest Neighbors search on 298K historical violations. Finds the top-K most similar "
               "past incidents based on coordinates, time-of-day, severity, and vehicle weight.")

    knn, scaler, feat_df, source_df = _build_knn()

    # Controls
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        query_hour = st.slider("Incident hour", 0, 23, 10, key="knn_hour",
                                help="Hour of day for the incident")
    with c2:
        query_severity = st.slider("Severity", 0.0, 1.0, 0.5, 0.05, key="knn_sev",
                                    help="Severity of the incident (0 = minor, 1 = severe)")
    with c3:
        top_k = st.slider("Results to show", 3, 20, 5, key="knn_k")

    # Build query vector using the incident zone coordinates
    query = np.array([[inc_lat, inc_lon, query_hour, query_severity, 1.0]])
    query_scaled = scaler.transform(query)
    distances, indices = knn.kneighbors(query_scaled, n_neighbors=min(top_k, len(feat_df)))

    # Build results
    results = []
    for rank, (idx, dist) in enumerate(zip(indices[0], distances[0]), start=1):
        row = source_df.iloc[idx]
        results.append({
            "Rank": rank,
            "Similarity": f"{max(0, 100 - dist * 10):.0f}%",
            "Distance": round(float(dist), 3),
            "Zone": row.get("police_station", "—") if "police_station" in row.index else "—",
            "Type": row.get("primary_type", "—") if "primary_type" in row.index else "—",
            "Hour": int(row.get("hour", 0)),
            "Severity": round(float(row.get("severity", 0)), 3),
            "Vehicle": row.get("vehicle_type", "—") if "vehicle_type" in row.index else "—",
            "lat": float(row["lat"]),
            "lon": float(row["lon"]),
        })

    results_df = pd.DataFrame(results)

    # KPIs
    ik = st.columns(4)
    ui.kpi(ik[0], "Incidents searched", f"{len(feat_df):,}")
    ui.kpi(ik[1], "Results returned", f"{len(results)}")
    avg_dist = results_df["Distance"].mean()
    ui.kpi(ik[2], "Avg similarity distance", f"{avg_dist:.2f}")
    top_zone = core._first_mode(results_df["Zone"], "Unknown") if not results_df.empty else "Unknown"
    ui.kpi(ik[3], "Most common zone", str(top_zone)[:18])

    st.markdown("---")
    map_col, table_col = st.columns([2, 1], gap="large")

    with map_col:
        st.subheader("Similar Incidents Map")
        st.caption("Orange dots = matched historical incidents. Red = current incident zone.")

        layers = _network_base_layers() + [_incident_marker_layer()]

        match_df = results_df.copy()
        match_df["name"] = match_df.apply(
            lambda r: f"#{r['Rank']} - {r['Type'][:20]} ({r['Similarity']})", axis=1)
        layers.append(pdk.Layer(
            "ScatterplotLayer", data=match_df,
            get_position=["lon", "lat"], get_radius=350,
            get_fill_color=[245, 158, 65, 200], stroked=True,
            get_line_color=[255, 255, 255, 180], line_width_min_pixels=1,
            pickable=True, auto_highlight=True,
        ))

        tip = {"html": "<b>{name}</b><br/>Hour: {Hour}<br/>Severity: {Severity}",
               "style": {"backgroundColor": "#161B26", "color": "#E6E9EF", "fontSize": "12px"}}
        st.pydeck_chart(ui.deck(layers, ui.view(inc_lat, inc_lon, zoom=11, pitch=25), tip),
                        width="stretch")

    with table_col:
        st.subheader("Matched Incidents")
        st.caption("Ranked by KNN Euclidean distance (lower = more similar).")

        show_cols = ["Rank", "Similarity", "Zone", "Type", "Hour", "Severity", "Vehicle"]
        st.dataframe(results_df[show_cols], hide_index=True, width="stretch", height=400)

    # Similarity breakdown chart
    st.markdown("---")
    bl, br = st.columns([2, 1], gap="large")
    with bl:
        st.markdown("**Similarity Distance Distribution**")
        fig = go.Figure(go.Bar(
            x=[f"#{r['Rank']}" for _, r in results_df.iterrows()],
            y=results_df["Distance"],
            marker_color=[ui.ACCENT if d < avg_dist else "#3a4254" for d in results_df["Distance"]],
            text=[f"{d:.2f}" for d in results_df["Distance"]],
            textposition="outside",
        ))
        fig.update_layout(
            height=280, margin=dict(l=0, r=20, t=10, b=0),
            xaxis_title="Match rank", yaxis_title="Distance",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, width="stretch")

    with br:
        st.markdown("**Violation type breakdown**")
        if "Type" in results_df.columns:
            type_counts = results_df["Type"].value_counts().head(5).reset_index()
            type_counts.columns = ["type", "count"]
            fig = px.pie(type_counts, values="count", names="type", hole=0.4,
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=250, margin=dict(l=0, r=0, t=10, b=0),
                              paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig, width="stretch")


# =====================================================================
# TAB 6: ROAD CLOSURE PREDICTOR
# =====================================================================
RISK_RULES = [
    (lambda i: i["risk_score"] >= 75,  25, "Severe risk score (75+)"),
    (lambda i: i["risk_score"] >= 45,  10, "Moderate risk score (45+)"),
    (lambda i: i["hour"] in list(range(8, 11)) + list(range(17, 20)),
     15, "Peak traffic hour (8-10 AM or 5-7 PM)"),
    (lambda i: i["hour"] in list(range(22, 24)) + list(range(0, 5)),
     10, "Night-time incident (reduced visibility)"),
    (lambda i: i.get("lanes_blocked", 0) >= 2,
     20, "Multiple lanes blocked"),
    (lambda i: i.get("lanes_blocked", 0) >= 1,
     10, "At least one lane blocked"),
    (lambda i: i.get("crowd_size", 0) > 10000,
     15, "Large crowd (10K+)"),
    (lambda i: i.get("rain", False),
     10, "Adverse weather conditions"),
]

RESOURCE_TABLE = {
    "CRITICAL": {"officers": 8, "barricades": 6, "escalation": "ACP / Joint Commissioner",
                 "notes": ["Deploy traffic diversion team immediately",
                           "Alert nearby hospitals and fire brigade",
                           "Set up temporary traffic signals",
                           "Notify senior officers and media cell"]},
    "HIGH":     {"officers": 6, "barricades": 4, "escalation": "ACP",
                 "notes": ["Deploy traffic diversion team",
                           "Alert nearby hospitals",
                           "Set up temporary traffic signals"]},
    "MEDIUM":   {"officers": 4, "barricades": 2, "escalation": "Inspector",
                 "notes": ["Standard diversion protocol",
                           "Monitor traffic flow actively"]},
    "LOW":      {"officers": 2, "barricades": 0, "escalation": "Inspector",
                 "notes": ["Standard response protocol",
                           "Monitor traffic flow",
                           "File standard report"]},
}


def _compute_risk_assessment(params):
    """Rule-based operational risk assessment like ARES."""
    score = 0
    factors = []
    for cond_fn, points, desc in RISK_RULES:
        try:
            if cond_fn(params):
                score += points
                factors.append((desc, points))
        except (KeyError, TypeError):
            continue
    score = min(score, 100)
    if score >= 75:
        level = "CRITICAL"
    elif score >= 50:
        level = "HIGH"
    elif score >= 25:
        level = "MEDIUM"
    else:
        level = "LOW"
    return score, level, factors


def _render_road_closure():
    st.markdown("## Road Closure Predictor")
    st.caption("Rule-based operational risk assessment and resource recommendation engine. "
               "Adapted from the ARES incident response copilot.")

    # Inputs
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        rc_hour = st.selectbox("Incident hour", list(range(24)),
                                index=10, key="rc_hour",
                                format_func=lambda x: f"{x:02d}:00")
    with c2:
        rc_lanes = st.selectbox("Lanes blocked", [0, 1, 2, 3, 4], index=1, key="rc_lanes")
    with c3:
        rc_crowd = st.number_input("Affected population", 100, 100000, 5000, 500, key="rc_crowd")
    with c4:
        rc_rain = st.checkbox("Adverse weather", key="rc_rain")

    params = {
        "risk_score": risk_score,
        "hour": rc_hour,
        "lanes_blocked": rc_lanes,
        "crowd_size": rc_crowd,
        "rain": rc_rain,
        "zone": incident_zone,
    }

    op_score, op_level, factors = _compute_risk_assessment(params)

    # Road closure decision
    closure_needed = op_score >= 50 or rc_lanes >= 2
    closure_confidence = min(100, op_score + (20 if rc_lanes >= 2 else 0)) / 100.0

    # Resource recommendation
    rec = RESOURCE_TABLE.get(op_level, RESOURCE_TABLE["LOW"])

    # KPIs
    rk = st.columns(5)
    level_color = {"CRITICAL": "#EF4444", "HIGH": "#F59E0B", "MEDIUM": "#4C8BF5", "LOW": "#10B981"}
    ui.kpi(rk[0], "Operational risk", f"{op_score}/100")
    ui.kpi(rk[1], "Risk level", op_level)
    ui.kpi(rk[2], "Road closure", "Required" if closure_needed else "Not required")
    ui.kpi(rk[3], "Closure confidence", f"{closure_confidence:.0%}")
    ui.kpi(rk[4], "Escalation contact", rec["escalation"])

    st.markdown("---")
    left_col, right_col = st.columns([1.5, 1], gap="large")

    with left_col:
        # Risk factor breakdown
        st.subheader("Risk Factor Breakdown")
        st.caption("Each rule is evaluated independently. Points are summed and capped at 100.")

        if factors:
            factor_df = pd.DataFrame(factors, columns=["Factor", "Points"])
            fig = go.Figure(go.Bar(
                x=factor_df["Points"], y=factor_df["Factor"], orientation="h",
                marker_color=[ui.ACCENT if p >= 15 else "#3a4254" for p in factor_df["Points"]],
                text=[f"+{p}" for p in factor_df["Points"]],
                textposition="outside",
            ))
            fig.update_layout(
                height=max(200, len(factors) * 40),
                margin=dict(l=0, r=40, t=10, b=0),
                xaxis_title="Risk points",
                yaxis={"categoryorder": "total ascending"},
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("No risk factors triggered at current settings.")

        # Road closure verdict card
        verdict_color = "#EF4444" if closure_needed else "#10B981"
        st.markdown(f"""
        <div class="rec-card" style="border-left:4px solid {verdict_color}">
            <div style="display:flex;justify-content:space-between;align-items:center">
                <strong style="font-size:1rem">Road Closure Verdict</strong>
                <span class="rec-badge" style="background:{verdict_color}">
                    {"CLOSURE REQUIRED" if closure_needed else "NO CLOSURE"}
                </span>
            </div>
            <div style="margin-top:8px;font-size:0.85rem;color:#8A93A6">
                Confidence: {closure_confidence:.0%} — based on {len(factors)} active risk factors
                scoring {op_score} total points.
            </div>
        </div>
        """, unsafe_allow_html=True)

    with right_col:
        # Resource recommendation
        st.subheader("Resource Recommendation")
        st.caption(f"Based on {op_level} risk level assessment.")

        st.markdown(f"""
        <div class="rec-card">
            <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px;font-size:0.85rem">
                <div>
                    <span style="color:#606878">Officers needed</span><br/>
                    <strong style="font-size:1.2rem">{rec['officers']}</strong>
                </div>
                <div>
                    <span style="color:#606878">Barricades needed</span><br/>
                    <strong style="font-size:1.2rem">{rec['barricades']}</strong>
                </div>
                <div>
                    <span style="color:#606878">Escalation</span><br/>
                    <strong>{rec['escalation']}</strong>
                </div>
                <div>
                    <span style="color:#606878">Risk level</span><br/>
                    <strong style="color:{level_color.get(op_level, '#8A93A6')}">{op_level}</strong>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")
        st.markdown("**Action checklist**")
        for note in rec["notes"]:
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;gap:8px;margin-bottom:6px">
                <div style="width:6px;height:6px;border-radius:50%;background:{ui.ACCENT};
                            margin-top:7px;flex-shrink:0"></div>
                <span style="font-size:0.85rem;color:#E6E9EF">{note}</span>
            </div>
            """, unsafe_allow_html=True)

        if closure_needed and rc_lanes >= 2:
            st.markdown("""
            <div class="rec-card" style="border-left:3px solid #F59E0B;margin-top:12px">
                <div style="font-size:0.85rem;color:#E6E9EF">
                    <strong>Additional:</strong> Request crane/heavy vehicle recovery unit.
                    Prepare road closure equipment and signage.
                </div>
            </div>
            """, unsafe_allow_html=True)


# =====================================================================
# TAB 7: AFTER-ACTION LOG (FEEDBACK LOOP)
# =====================================================================
def _render_after_action():
    st.markdown("## After-Action Log")
    st.caption("Log incidents, submit post-resolution feedback, and build a learning loop. "
               "Actual vs. predicted resource usage is tracked for future model improvement.")

    counts = incident_db.count_incidents()

    # KPIs
    ak = st.columns(4)
    ui.kpi(ak[0], "Total incidents logged", f"{counts['total']}")
    ui.kpi(ak[1], "Active incidents", f"{counts['active']}")
    ui.kpi(ak[2], "Resolved incidents", f"{counts['resolved']}")
    resolved_pct = (counts["resolved"] / counts["total"] * 100) if counts["total"] > 0 else 0
    ui.kpi(ak[3], "Resolution rate", f"{resolved_pct:.0f}%")

    st.markdown("---")
    log_tab, feedback_tab, history_tab = st.tabs([
        "Log New Incident", "Submit Feedback", "Incident History"
    ])

    # -------- Log New Incident --------
    with log_tab:
        st.subheader("Log Current Scenario as Incident")
        st.caption("Saves the current simulation parameters as an active incident record.")

        lc1, lc2 = st.columns([2, 1], gap="large")
        with lc1:
            st.markdown(f"""
            <div class="rec-card">
                <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.85rem">
                    <div><span style="color:#606878">Zone</span><br/><strong>{incident_zone}</strong></div>
                    <div><span style="color:#606878">Scenario</span><br/><strong>{scenario}</strong></div>
                    <div><span style="color:#606878">Risk score</span><br/><strong>{risk_score}/100</strong></div>
                    <div><span style="color:#606878">Risk level</span><br/>
                         <strong style="color:{risk_color_hex}">{risk_level}</strong></div>
                    <div><span style="color:#606878">Coordinates</span><br/>
                         <strong>{inc_lat:.4f}, {inc_lon:.4f}</strong></div>
                    <div><span style="color:#606878">Timestamp</span><br/>
                         <strong>{datetime.now().strftime('%Y-%m-%d %H:%M')}</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            log_notes = st.text_area("Additional notes (optional)", key="log_notes",
                                      placeholder="E.g. VIP convoy, waterlogging on service road...",
                                      height=80)

        with lc2:
            st.markdown("**Incident will be saved with:**")
            st.markdown(f"- Zone: **{incident_zone}**")
            st.markdown(f"- Scenario: **{scenario}**")
            st.markdown(f"- Risk: **{risk_level}** ({risk_score}/100)")

            if st.button("Log incident", type="primary", use_container_width=True, key="log_btn"):
                inc_id = incident_db.save_incident({
                    "zone": incident_zone,
                    "scenario": scenario,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "latitude": inc_lat,
                    "longitude": inc_lon,
                    "notes": log_notes,
                })
                st.success(f"Incident logged: **{inc_id}**")
                st.rerun()

    # -------- Submit Feedback --------
    with feedback_tab:
        st.subheader("Post-Incident Feedback")
        st.caption("After an incident is resolved, submit what actually happened vs. what was predicted. "
                   "This closes the learning loop for future model improvement.")

        active = incident_db.get_active_incidents()
        if not active:
            st.info("No active incidents to submit feedback for. Log an incident first.")
        else:
            selected_id = st.selectbox("Select active incident",
                                        [f"{a['id']} — {a['zone']} ({a['risk_level']})" for a in active],
                                        key="fb_select")
            selected_inc = active[[f"{a['id']} — {a['zone']} ({a['risk_level']})" for a in active].index(selected_id)]

            st.markdown(f"""
            <div class="rec-card" style="margin-bottom:16px">
                <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:8px;font-size:0.85rem">
                    <div><span style="color:#606878">ID</span><br/><strong>{selected_inc['id']}</strong></div>
                    <div><span style="color:#606878">Zone</span><br/><strong>{selected_inc['zone']}</strong></div>
                    <div><span style="color:#606878">Predicted risk</span><br/>
                         <strong>{selected_inc['risk_level']} ({selected_inc['risk_score']})</strong></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            fc1, fc2 = st.columns(2)
            with fc1:
                actual_off = st.number_input("Actual officers deployed", 0, 100, 4, key="fb_off")
                actual_barr = st.number_input("Actual barricades used", 0, 50, 2, key="fb_barr")
            with fc2:
                closure_used = st.selectbox("Road closure used?", ["No", "Yes"], key="fb_closure")
                actual_duration = st.number_input("Resolution time (minutes)", 0, 600, 60, key="fb_dur")

            res_notes = st.text_area("Resolution notes", key="fb_notes",
                                      placeholder="What actually happened? Any lessons learned?",
                                      height=80)

            if st.button("Submit feedback and resolve", type="primary",
                          use_container_width=True, key="fb_submit"):
                incident_db.save_feedback({
                    "incident_id": selected_inc["id"],
                    "actual_officers": actual_off,
                    "actual_barricades": actual_barr,
                    "road_closure_used": 1 if closure_used == "Yes" else 0,
                    "actual_duration_min": actual_duration,
                    "resolution_notes": res_notes,
                })
                st.success(f"Feedback saved. Incident **{selected_inc['id']}** marked as RESOLVED.")
                st.rerun()

    # -------- Incident History --------
    with history_tab:
        st.subheader("Incident History")
        st.caption("Full log of all incidents with their resolution feedback.")

        all_incidents = incident_db.get_all_incidents()
        if not all_incidents:
            st.info("No incidents logged yet. Use the 'Log New Incident' tab to create one.")
        else:
            # Active vs resolved split
            active_list = [i for i in all_incidents if i["status"] == "ACTIVE"]
            resolved_list = [i for i in all_incidents if i["status"] == "RESOLVED"]

            if active_list:
                st.markdown(f"**Active incidents ({len(active_list)})**")
                active_df = pd.DataFrame(active_list)[
                    ["id", "created_at", "zone", "scenario", "risk_score", "risk_level", "status"]
                ].copy()
                active_df.columns = ["ID", "Created", "Zone", "Scenario", "Score", "Level", "Status"]
                st.dataframe(active_df, hide_index=True, width="stretch")

            if resolved_list:
                st.markdown("---")
                st.markdown(f"**Resolved incidents ({len(resolved_list)})**")
                resolved_with_fb = incident_db.get_incidents_with_feedback()

                if resolved_with_fb:
                    res_rows = []
                    for r in resolved_with_fb:
                        res_rows.append({
                            "ID": r["id"],
                            "Zone": r["zone"],
                            "Predicted Risk": f"{r['risk_level']} ({r['risk_score']})",
                            "Actual Officers": r.get("actual_officers", "—"),
                            "Actual Barricades": r.get("actual_barricades", "—"),
                            "Road Closure": "Yes" if r.get("road_closure_used") else "No",
                            "Duration (min)": r.get("actual_duration_min", "—"),
                            "Notes": (r.get("resolution_notes", "") or "")[:40],
                        })
                    st.dataframe(pd.DataFrame(res_rows), hide_index=True, width="stretch", height=350)

                    # Predicted vs Actual chart
                    st.markdown("---")
                    st.markdown("**Predicted vs Actual Resource Usage**")
                    st.caption("Compare AI recommendations against field reality — key metric for model retraining.")

                    pred_off = [r.get("officers_req", 0) or 0 for r in resolved_with_fb]
                    actual_off_list = [r.get("actual_officers", 0) or 0 for r in resolved_with_fb]
                    labels = [r["id"][-8:] for r in resolved_with_fb]

                    if any(v > 0 for v in pred_off + actual_off_list):
                        fig = go.Figure()
                        fig.add_trace(go.Bar(name="Predicted", x=labels, y=pred_off,
                                             marker_color=ui.ACCENT))
                        fig.add_trace(go.Bar(name="Actual", x=labels, y=actual_off_list,
                                             marker_color="#F59E0B"))
                        fig.update_layout(
                            barmode="group", height=280,
                            margin=dict(l=0, r=0, t=10, b=0),
                            xaxis_title="Incident", yaxis_title="Officers",
                            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                            legend=dict(orientation="h", y=1.1),
                        )
                        st.plotly_chart(fig, width="stretch")

            # Download all incidents
            if all_incidents:
                st.markdown("---")
                csv_data = pd.DataFrame(all_incidents).to_csv(index=False)
                st.download_button("Download incident log (CSV)", csv_data.encode(),
                                    file_name="parksensei_incident_log.csv", mime="text/csv")


# =====================================================================
# RENDER TABS
# =====================================================================
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dynamic Routing",
    "Emergency Corridor",
    "Police Dispatch",
    "Transit Advisor",
    "Similar Incidents",
    "Road Closure",
    "After-Action Log"
])

with tab1:
    _render_dynamic_routing()
with tab2:
    _render_emergency_corridor()
with tab3:
    _render_police_dispatch()
with tab4:
    _render_transit_advisor()
with tab5:
    _render_knn_intelligence()
with tab6:
    _render_road_closure()
with tab7:
    _render_after_action()

