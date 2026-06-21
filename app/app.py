"""ParkSensei navigation entry point.
Run locally with: streamlit run app/app.py
"""
from pathlib import Path
import sys

import streamlit as st


APP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(APP_DIR))

pages = [
    st.Page(APP_DIR / "Dashboard.py", title="Dashboard", default=True),
    st.Page(APP_DIR / "pages" / "1_Hotspot_Analytics.py", title="Hotspot Analytics"),
    st.Page(APP_DIR / "pages" / "2_Patrol_Planning.py", title="Patrol Planning"),
    st.Page(APP_DIR / "pages" / "3_Enforcement_Strategy.py", title="Enforcement Strategy"),
    st.Page(APP_DIR / "pages" / "4_Intelligence_Lab.py", title="Intelligence Lab"),
    st.Page(APP_DIR / "pages" / "7_Incident_Response.py", title="Incident Response"),
    st.Page(APP_DIR / "pages" / "5_Ask_ParkSensei.py", title="Ask ParkSensei"),
    st.Page(APP_DIR / "pages" / "6_Reports_and_Exports.py", title="Reports & Exports"),
]

st.navigation(pages, position="sidebar", expanded=True).run()
