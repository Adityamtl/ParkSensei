"""
ui.py - shared Streamlit helpers: cached data loaders, colour ramps, pydeck/plotly
builders, PDF report generation, and brand constants. Pages stay thin and import from here.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))   # allow `import core`

import io
import numpy as np, pandas as pd
import streamlit as st
import pydeck as pdk
import core

# ---------------------------------------------------------------- brand
ACCENT  = "#4C8BF5"
BASEMAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
BLR     = {"lat": 12.9716, "lon": 77.5946}
DOW_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
# blue -> amber -> red, used for impact / density everywhere
COLOR_RANGE = [[46,134,222],[92,160,180],[245,205,90],[245,158,65],[238,110,55],[226,53,43]]

# recommendation priority colors
REC_COLORS = {
    "CRITICAL": "#E2352B",
    "HIGH":     "#E8923E",
    "MEDIUM":   "#F5CD5A",
    "LOW":      "#4C8BF5",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"], [data-testid="stMarkdownContainer"] { font-family: 'Inter', sans-serif; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent; }
[data-testid="stToolbar"] { visibility: visible; }
[data-testid="stExpandSidebarButton"],
[data-testid="stSidebarCollapseButton"] {
    visibility: visible !important;
    opacity: 1 !important;
}
.block-container { padding-top: 2rem; padding-bottom: 2rem; max-width: 1480px; }
[data-testid="stMetric"] {
    background: linear-gradient(180deg, #151A25 0%, #121620 100%);
    border: 1px solid #1E2638; border-radius: 10px; padding: 14px 18px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.18);
    transition: border-color 0.2s ease;
}
[data-testid="stMetric"]:hover { border-color: #2E3A52; }
[data-testid="stMetricValue"] { font-weight: 700; font-size: 1.5rem; }
[data-testid="stMetricLabel"] p { color: #8A93A6; font-weight: 500; font-size: 0.82rem; text-transform: uppercase; letter-spacing: 0.04em; }
section[data-testid="stSidebar"] {
    background: #0A0E16; border-right: 1px solid #171D2A;
}
section[data-testid="stSidebar"] > div { padding-top: 1.2rem; }
h1 { letter-spacing: -0.02em; font-weight: 700; font-size: 1.8rem; }
h2 { letter-spacing: -0.015em; font-weight: 700; font-size: 1.35rem; }
h3 { letter-spacing: -0.01em; font-weight: 600; font-size: 1.1rem; }
.stButton button, [data-testid="stDownloadButton"] button {
    border-radius: 8px; font-weight: 600; border: 1px solid #252E40;
    transition: border-color 0.15s ease, background-color 0.15s ease;
}
.stButton button:hover, [data-testid="stDownloadButton"] button:hover {
    border-color: #3A4766;
}
hr { margin: 1.2rem 0; border-color: #1A2030; }
.rec-card {
    background: linear-gradient(180deg, #151A25 0%, #121620 100%);
    border: 1px solid #1E2638; border-radius: 10px; padding: 14px 16px; margin-bottom: 8px;
    transition: border-color 0.2s ease;
}
.rec-card:hover { border-color: #2E3A52; }
.rec-badge { display: inline-block; padding: 3px 9px; border-radius: 4px;
    font-weight: 700; font-size: 0.7rem; color: #fff; letter-spacing: 0.03em; text-transform: uppercase; }
.rec-badge-CRITICAL { background: #C4302B; }
.rec-badge-HIGH { background: #CC7E35; }
.rec-badge-MEDIUM { background: #A68D2E; }
.rec-badge-LOW { background: #3D73CC; }
.sidebar-brand { font-size: 1.15rem; font-weight: 700; letter-spacing: -0.01em; color: #E0E4EC; }
.sidebar-tagline { font-size: 0.78rem; color: #6B7384; line-height: 1.4; margin-top: 2px; }
</style>
"""

def page(title, icon="P"):
    st.set_page_config(page_title=f"{title} | ParkSensei", page_icon=icon, layout="wide",
                       initial_sidebar_state="expanded")
    st.markdown(_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------- cached loaders
@st.cache_resource(show_spinner="Loading violations...", max_entries=1, ttl=3600)
def load_data():
    return core.load_clean()

@st.cache_data(show_spinner="Scoring zones…", max_entries=1, ttl=3600)
def get_zones():
    return core.add_impact(core.build_zones(load_data()))

@st.cache_data(show_spinner="Building grid…", max_entries=1, ttl=3600)
def get_grid():
    df = load_data()
    return (df.groupby("gh7")
              .agg(lat=("lat","median"), lon=("lon","median"),
                   n=("lat","size"), sev=("severity","mean"))
              .reset_index())

@st.cache_resource(show_spinner="Training forecaster…", max_entries=1, ttl=3600)
def get_forecaster():
    return core.build_forecaster(load_data())

@st.cache_data(show_spinner=False, max_entries=1, ttl=3600)
def get_backtest():
    return core.backtest(load_data())

@st.cache_data(show_spinner="Generating recommendations…", max_entries=1, ttl=3600)
def get_zone_recommendations(top_n=20):
    zones = get_zones()
    return core.zone_recommendations(zones, top_n)

@st.cache_data(show_spinner="Running DBSCAN clustering…", max_entries=1, ttl=3600)
def get_dbscan_clusters():
    return core.dbscan_clusters(load_data())

@st.cache_data(show_spinner="Computing cluster quality metrics…", max_entries=1, ttl=3600)
def get_cluster_quality():
    return core.cluster_quality_metrics(load_data())

@st.cache_resource(show_spinner="Getting trained model details…", max_entries=1, ttl=3600)
def get_nextday_trained():
    """Return the raw trained model dict (for feature importance etc.)."""
    df = load_data()
    return core.train_nextday_models(df)

@st.cache_data(show_spinner="Training next-day prediction models…", max_entries=1, ttl=3600)
def get_nextday_forecast():
    df = load_data()
    trained = get_nextday_trained()
    if "error" in trained:
        return trained
    return core.predict_next_7days(trained, df)

@st.cache_resource(show_spinner="Training congestion model…", max_entries=1, ttl=3600)
def get_congestion_model():
    return core.train_congestion_model(load_data())

@st.cache_data(show_spinner="Analysing traffic propagation…", max_entries=1, ttl=3600)
def get_traffic_propagation(top_n=30):
    """Run propagation analysis on top N zones (limited for performance)."""
    zones = get_zones()
    return core.traffic_propagation(zones.head(top_n))

@st.cache_data(show_spinner="Building parking DNA profiles…", max_entries=1, ttl=3600)
def get_parking_dna():
    return core.parking_dna_profiles(load_data())

@st.cache_data(show_spinner="Detecting emerging hotspots…", max_entries=1, ttl=3600)
def get_emerging_hotspots():
    return core.emerging_hotspot_analysis(load_data())

@st.cache_data(show_spinner="Computing officer allocation…", max_entries=1, ttl=3600)
def get_officer_allocation(total_officers=100, top_n=20):
    zones = get_zones()
    return core.officer_allocation(zones, total_officers, top_n)


# ---------------------------------------------------------------- colour
def impact_color(score):
    """0..100 -> [r,g,b] along blue->amber->red."""
    stops = [(0,(46,134,222)), (45,(245,205,90)), (70,(245,158,65)), (100,(226,53,43))]
    score = max(0, min(100, float(score)))
    for (a,ca),(b,cb) in zip(stops, stops[1:]):
        if score <= b:
            t = 0 if b==a else (score-a)/(b-a)
            return [int(ca[i]+(cb[i]-ca[i])*t) for i in range(3)]
    return list(stops[-1][1])

def rec_color(priority: str) -> str:
    """Priority string -> hex color."""
    return REC_COLORS.get(priority, "#4C8BF5")

# ---------------------------------------------------------------- map builders
def view(lat=None, lon=None, zoom=10.6, pitch=45):
    return pdk.ViewState(latitude=BLR["lat"] if lat is None else lat,
                         longitude=BLR["lon"] if lon is None else lon,
                         zoom=zoom, pitch=pitch, bearing=0)

def deck(layers, viewstate, tooltip=None):
    return pdk.Deck(layers=layers, initial_view_state=viewstate, map_style=BASEMAP,
                    tooltip=tooltip or True)

def hex_layer(grid, radius=160, elev=18):
    return pdk.Layer(
        "HexagonLayer", data=grid, get_position=["lon","lat"],
        get_elevation_weight="n", get_color_weight="n",
        elevation_aggregation="SUM", color_aggregation="SUM",
        radius=radius, elevation_scale=elev, elevation_range=[0,2400],
        extruded=True, coverage=0.92, pickable=True, auto_highlight=True,
        color_range=COLOR_RANGE)

def zone_layer(zones):
    z = zones.copy()
    z["color"] = z["impact_score"].map(impact_color)
    z["radius"] = (np.sqrt(z["violations"]) * 7).clip(60, 900)
    return pdk.Layer(
        "ScatterplotLayer", data=z, get_position=["lon","lat"],
        get_radius="radius", get_fill_color="color",
        opacity=0.65, stroked=True, get_line_color=[255,255,255,60],
        line_width_min_pixels=0.5, pickable=True, auto_highlight=True)

def plan_layers(plan):
    halo = pdk.Layer("ScatterplotLayer", data=plan, get_position=["lon","lat"],
                     get_radius=420, get_fill_color=[76,139,245,55], pickable=False)
    pts  = pdk.Layer("ScatterplotLayer", data=plan, get_position=["lon","lat"],
                     get_radius=120, get_fill_color=[76,139,245], stroked=True,
                     get_line_color=[255,255,255], line_width_min_pixels=2, pickable=True)
    txt  = pdk.Layer("TextLayer", data=plan, get_position=["lon","lat"], get_text="team",
                     get_size=13, get_color=[255,255,255], get_pixel_offset=[0,-22],
                     get_alignment_baseline="'bottom'")
    return [halo, pts, txt]

TIP_ZONE = {"html": "<b>{label}</b><br/>Violations: {violations}<br/>"
                    "Impact score: {impact_score}<br/>Top: {top_violation}",
            "style": {"backgroundColor": "#161B26", "color": "#E6E9EF", "fontSize": "12px"}}
TIP_PLAN = {"html": "<b>{team} → {label}</b><br/>Predicted catches: {pred_load}<br/>"
                    "Impact score: {impact_score}",
            "style": {"backgroundColor": "#161B26", "color": "#E6E9EF", "fontSize": "12px"}}

# ---------------------------------------------------------------- PDF generation
def _pdf_safe(text):
    """Replace Unicode characters that Helvetica can't encode with ASCII equivalents."""
    return (str(text)
            .replace("\u2014", "-")    # em-dash
            .replace("\u2013", "-")    # en-dash
            .replace("\u2019", "'")    # right single quote
            .replace("\u2018", "'")    # left single quote
            .replace("\u201c", '"')    # left double quote
            .replace("\u201d", '"')    # right double quote
            .replace("\u2026", "...")  # ellipsis
            .replace("\u00d7", "x")   # multiplication sign
            .replace("\u2192", "->")  # right arrow
            .replace("\u2022", "-")   # bullet
            .replace("\u2265", ">=")  # greater-than-or-equal
            .replace("\u2264", "<=")  # less-than-or-equal
            )

def generate_pdf_brief(zones, plan=None, backtest_result=None):
    """Generate a PDF enforcement brief using fpdf2.
       Returns bytes of the PDF document."""
    try:
        from fpdf import FPDF
    except ImportError:
        return None

    class ParkSenseiPDF(FPDF):
        def header(self):
            self.set_fill_color(76, 139, 245)
            self.rect(0, 0, 210, 12, 'F')
            self.set_font("Helvetica", "B", 8)
            self.set_text_color(255, 255, 255)
            self.cell(0, 10, "ParkSensei - Parking Enforcement Intelligence Brief", align="C")
            self.ln(14)

        def footer(self):
            self.set_y(-15)
            self.set_font("Helvetica", "I", 7)
            self.set_text_color(140, 140, 140)
            self.cell(0, 10, f"ParkSensei - Page {self.page_no()}/{{nb}}", align="C")

    pdf = ParkSenseiPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # Title
    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(30, 40, 55)
    pdf.cell(0, 12, "ParkSensei Enforcement Brief", ln=True)
    pdf.set_font("Helvetica", "", 10)
    pdf.set_text_color(100, 110, 130)
    total = int(zones["violations"].sum()) if "violations" in zones.columns else 0
    pdf.cell(0, 8, _pdf_safe(f"298K violations analysed  |  {len(zones)} zones scored  |  Bengaluru Traffic Police"), ln=True)
    pdf.ln(4)

    # Executive summary
    pdf.set_fill_color(235, 240, 250)
    pdf.rect(10, pdf.get_y(), 190, 30, 'F')
    y0 = pdf.get_y() + 4
    pdf.set_xy(14, y0)
    pdf.set_font("Helvetica", "B", 11)
    pdf.set_text_color(30, 40, 55)
    pdf.cell(0, 6, "Executive Signal", ln=True)
    pdf.set_x(14)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(70, 80, 100)
    top = zones.iloc[0] if len(zones) else {}
    brief = _pdf_safe(
        f"Top priority: {top.get('label', 'N/A')} "
        f"(impact {top.get('impact_score', 0):.0f}). "
        f"PCU obstruction avg {top.get('avg_pcu', 1):.2f}. "
        f"Enhanced 7-factor scoring ranks zones by congestion impact, "
        f"not just violation volume."
    )
    pdf.multi_cell(182, 5, brief)
    pdf.ln(8)

    # KPIs
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(30, 40, 55)
    kpis = [
        ("Zones scored", str(len(zones))),
        ("Critical zones", str(len(zones[zones["impact_score"] >= 70])) if "impact_score" in zones.columns else "-"),
        ("Avg PCU weight", f"{zones['avg_pcu'].mean():.2f}" if "avg_pcu" in zones.columns else "-"),
    ]
    if backtest_result:
        kpis.append(("Forecast accuracy", f"r = {backtest_result.get('pearson_r', '-')}"))

    for label, value in kpis:
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(100, 110, 130)
        pdf.cell(47, 5, _pdf_safe(label.upper()))
    pdf.ln()
    for label, value in kpis:
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 40, 55)
        pdf.cell(47, 7, _pdf_safe(value))
    pdf.ln(10)

    # Top zones table
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 40, 55)
    pdf.cell(0, 8, "Top Enforcement Zones by Congestion Impact", ln=True)
    pdf.ln(2)

    # Table header
    pdf.set_fill_color(76, 139, 245)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 8)
    widths = [8, 50, 20, 18, 18, 15, 61]
    headers = ["#", "Zone", "Viol.", "Impact", "PCU", "Peak%", "Recommended Action"]
    for w, h in zip(widths, headers):
        pdf.cell(w, 7, h, border=0, fill=True, align="C")
    pdf.ln()

    # Table rows
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(50, 60, 75)
    for i, (_, z) in enumerate(zones.head(15).iterrows()):
        if pdf.get_y() > 260:
            pdf.add_page()
        bg = (245, 247, 252) if i % 2 == 0 else (255, 255, 255)
        pdf.set_fill_color(*bg)
        recs = core.generate_recommendations(z)
        top_action = recs[0]["action"] if recs else "Routine Patrol"
        row_data = [
            str(i + 1),
            _pdf_safe(str(z.get("label", ""))[:28]),
            str(int(z.get("violations", 0))),
            f"{z.get('impact_score', 0):.0f}",
            f"{z.get('avg_pcu', 1):.2f}",
            f"{z.get('peak_share', 0)*100:.0f}",
            _pdf_safe(top_action),
        ]
        for w, val in zip(widths, row_data):
            pdf.cell(w, 6, val, border=0, fill=True, align="C" if w < 30 else "L")
        pdf.ln()

    # Patrol plan (if provided)
    if plan is not None and len(plan):
        pdf.ln(6)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(30, 40, 55)
        pdf.cell(0, 8, "Deployment Plan", ln=True)
        pdf.ln(2)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(50, 60, 75)
        for _, p in plan.iterrows():
            if pdf.get_y() > 265:
                pdf.add_page()
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 6, _pdf_safe(
                f"{p.get('team', '')} -> {p.get('label', '')} "
                f"(exp. catches: {p.get('pred_load', 0):.1f}, "
                f"impact: {p.get('impact_score', 0):.0f})"
            ), ln=True)
            if "recommended_action" in p.index:
                pdf.set_font("Helvetica", "", 8)
                pdf.set_text_color(100, 110, 130)
                pdf.cell(0, 5, _pdf_safe(f"  Action: {p['recommended_action']}"), ln=True)
                pdf.set_text_color(50, 60, 75)

    # Method note
    pdf.ln(6)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(30, 40, 55)
    pdf.cell(0, 8, "Scoring Method", ln=True)
    pdf.set_font("Helvetica", "", 8)
    pdf.set_text_color(70, 80, 100)
    method = _pdf_safe(
        "Congestion Impact Score (0-100) = "
        "0.30 x obstruction + 0.18 x density + 0.15 x junction + 0.13 x arterial "
        "+ 0.10 x peak + 0.08 x recurrence + 0.06 x severity. "
        "Obstruction uses PCU vehicle weights (bus=2.8, car=1.0, motorcycle=0.35). "
        "All factors are log-normalised and monotone."
    )
    pdf.multi_cell(0, 5, method)

    buf = io.BytesIO()
    pdf.output(buf)
    return buf.getvalue()

# ---------------------------------------------------------------- recommendation rendering
def render_recommendation_card(rec: dict):
    """Render a single recommendation as styled HTML in Streamlit."""
    p = rec["priority"]
    st.markdown(f"""
    <div class="rec-card">
        <span class="rec-badge rec-badge-{p}">{p}</span>
        &nbsp; <strong>{rec['action']}</strong>
        <br/><span style="color:#8A93A6; font-size:0.85rem;">{rec['reason']}</span>
        <br/><span style="color:#606878; font-size:0.78rem;">Window: {rec['window']}</span>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------- misc
def kpi(col, label, value, help=None):
    col.metric(label, value, help=help)

def brand_sidebar():
    st.sidebar.markdown('<div class="sidebar-brand">ParkSensei</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<div class="sidebar-tagline">Parking Enforcement Intelligence<br/>Bengaluru Traffic Police</div>', unsafe_allow_html=True)
    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "PCU vehicle weights, "
        "7-factor impact scoring, and "
        "actionable enforcement recommendations."
    )
    st.sidebar.markdown("---")

