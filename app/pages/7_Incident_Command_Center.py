"""Incident Command Center — Dynamic Routing, Emergency Corridors, Police Dispatch, Transit Advisor."""
import math
import numpy as np, pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import ui, core, traffic_network as tn

ui.page("Incident Command Center", "C")
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

st.markdown("## Incident Command Center")
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
# RENDER TABS
# =====================================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Dynamic Routing",
    "Emergency Corridor",
    "Police Dispatch",
    "Transit Advisor"
])

with tab1:
    _render_dynamic_routing()
with tab2:
    _render_emergency_corridor()
with tab3:
    _render_police_dispatch()
with tab4:
    _render_transit_advisor()
