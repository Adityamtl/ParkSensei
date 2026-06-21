# ParkSensei

Parking and incident intelligence for Bengaluru Traffic Police.

ParkSensei turns 298,445 Bengaluru parking-violation records into a command system that tells officers where illegal parking hurts traffic, when it is likely to recur, how to deploy patrol teams, which repeat offenders to target, and how to respond when a traffic incident needs diversion, dispatch, or an emergency corridor.

Built for Gridlock Hackathon 2.0, Round 2, Theme 1: Poor Visibility on Parking-Induced Congestion.


## Why ParkSensei

Illegal parking does not just create tickets. It blocks live lanes, narrows junctions, slows buses, delays emergency vehicles, and creates spillover congestion around markets, metro stations, hospitals, and commercial roads. Today, enforcement is often reactive: officers know many hotspots from experience, but they do not have a live, city-wide decision system that ranks zones by traffic impact and converts that ranking into an operational plan.

ParkSensei closes that gap with a complete loop:

```text
Detect hotspots -> Score traffic impact -> Forecast demand -> Deploy teams -> Target offenders -> Respond to incidents -> Learn from feedback
```

## Data Used

The current prototype uses the provided Theme 1 BTP parking-violation data.

| Metric | Value |
|---|---:|
| Violation records | 298,445 |
| Data period | 2023-11-10 to 2024-04-08 |
| Active days | 151 |
| Scored hotspot zones | 802 |
| Named junctions | 168 |
| Police stations | 54 |
| Unique vehicles | 231,890 |
| Repeat-offender violation load | 34.2% |
| Logged before 1 PM | 92.8% |
| Logged during 5-9 PM | 0.25% |
| Forecast backtest | Pearson r = 0.685, MAE = 2.078 |

Important honesty note: the dataset records when violations were enforced, not the full ground truth of when illegal parking existed. ParkSensei therefore optimizes observed enforcement efficiency and clearly flags blind spots, especially evening coverage.

## Main Features

### 1. Command Center

The landing dashboard gives the officer the city-wide picture:

- Total violations, hotspot zones, average daily load, repeat-offender load, average PCU obstruction weight, and top hotspot.
- A summary of critical zones using the 7-factor Congestion Impact Score.
- Downloadable PDF enforcement brief and top-zone CSV.
- 3-D city-wide violation density map.
- Top impact zones ranked by impact score, PCU weight, and place type.
- Weekday-by-hour heatmap showing the morning-heavy enforcement pattern.
- Top enforcement action previews for the highest-impact zones.
- Previews for traffic propagation, parking DNA, emerging hotspots, and what-if simulation modules.

### 2. Analytics and Insights

This page explains where the violations happen and why certain locations matter.

Tabs:

- Hotspot Explorer: filter by weekday, hour, violation type, and police station; see maps, ranked zones, 7-factor impact breakdowns, recommended actions, violation mix, daily rhythm, vehicle type mix, and place-type distribution.
- Parking DNA: creates behavioral fingerprints for each police station, including dominant vehicle type, top violation, peak hour, weekend ratio, severity, and junction exposure. It also shows emerging and declining hotspot trends.
- Traffic Propagation: finds nearby high-impact zones that can affect each other, shows propagation links on a map, classifies risk by distance, and gives blast-radius analysis for selected hotspots.

### 3. Operations and Dispatch

This page converts analytics into field deployment.

Tabs:

- Forecast and Patrol Planner: predicts expected violations per zone, weekday, and hour; allocates patrol teams with minimum spacing; shows map pins, expected catches, recommended actions, CSV download, and PDF brief.
- Officer Allocation and Route Optimizer: distributes a given officer pool across top zones proportional to impact score, then builds an optimized patrol route through selected hotspots.
- Next-Day Violation Forecast: trains city-wide and per-junction models to forecast the next 7 days, classifies risk as high/medium/low, shows confidence intervals, model performance, and downloadable forecast tables.

### 4. Strategy and Review

This page supports policy decisions and enforcement review.

Tabs:

- Enforcement Actions: rule-based recommendations for high-impact zones, including tow-away zones, peak-hour enforcement, CCTV/ANPR monitoring, repeat-offender escalation, heavy-vehicle restrictions, signage audits, precinct escalation, and routine patrol.
- Repeat-Offender Intelligence: identifies vehicles caught multiple times, shows the repeat-offender share of total violations, lists the most-cited vehicles, and shows how repeat behavior is distributed.
- Coverage and ROI: compares targeted deployment against even coverage, finds staffing sweet spots, and highlights coverage blind spots such as the near-empty 5-9 PM enforcement window.

### 5. Advanced Intelligence

This page contains the model diagnostics and scenario simulator.

Tabs:

- ML Impact Analysis: DBSCAN spatial clustering, cluster quality metrics, cluster map, impact distribution, congestion probability classifier, feature importance, and next-day model performance.
- What-If Enforcement Simulator: estimates how adding officers to a zone reduces impact, congestion risk, and propagation risk. It supports both single-zone simulation and multi-zone comparison.
- Alerts: generates active alerts from high-risk next-day forecasts, DBSCAN mega-clusters, high-impact clusters, and model diagnostics. Alerts can be dismissed or reset.

### 6. Ask ParkSensei

A natural-language copilot for enforcement planning.

- Accepts plain English, Hindi, or Kannada style prompts.
- Examples: "Where should I send 6 teams on Friday evening?", "Plan 8 teams for Saturday morning around KR Market", "What are the 5 worst hotspots?", "How bad is our evening coverage?"
- Uses Gemini function calling over the real ParkSensei computation layer. The model does not invent the patrol plan; it calls the same forecaster, optimizer, hotspot, coverage, and repeat-offender functions used by the dashboard.

Set `GEMINI_API_KEY` in `.streamlit/secrets.toml` or Streamlit Cloud secrets to enable it.

### 7. Reports and Export

A single download hub for officers, judges, and downstream systems.

Exports:

- Enforcement priority CSV.
- DBSCAN clusters CSV.
- Next-day forecast CSV.
- Repeat offenders CSV.
- Enforcement actions CSV.
- PDF enforcement brief.
- Machine-readable dashboard statistics JSON.

### 8. Incident Command Center

An operational response layer built on a simplified Bengaluru road graph with NetworkX.

Tabs:

- Dynamic Routing: compares the standard route against the diversion route under a selected incident severity, showing travel time, stuck time, diversion time, and savings.
- Emergency Corridor: plans an ambulance route from the incident zone to a hospital and generates a signal-preemption schedule.
- Police Dispatch: allocates officers and patrol vehicles from nearby police stations based on capacity and travel time.
- Transit Advisor: checks affected BMTC routes, estimates delay, and suggests shifted bus-stop advice.
- Similar Incidents: finds historical incidents similar to the selected scenario using KNN-style feature similarity.
- Road Closure: rule-based risk assessment that recommends closure, escalation contact, officers, barricades, and action checklist.
- After-Action Log: records incident scenarios, captures post-resolution feedback, and compares predicted versus actual resource use.

## Core Algorithms

### 7-Factor Congestion Impact Score

Each zone receives a transparent 0-100 score:

```text
impact = 100 * (
  0.30 * obstruction
  + 0.18 * density
  + 0.15 * junction
  + 0.13 * arterial
  + 0.10 * peak
  + 0.08 * recurrence
  + 0.06 * severity
)
```

The factors are normalized and monotonic:

- Obstruction: PCU-weighted violation pressure.
- Density: log-scaled violation count.
- Junction: share of violations linked to junctions.
- Arterial: main-road or arterial exposure.
- Peak: peak-hour recurrence.
- Recurrence: active-day recurrence.
- Severity: average violation severity.

### Forecasting

ParkSensei predicts enforcement load per zone, weekday, and hour using Bayesian-shrunk historical rates:

```text
rate(zone, weekday, hour) =
  (count + alpha * zone_hour_backoff) / (weekday_observations + alpha)
```

This keeps sparse weekday-hour cells stable while preserving strong historical signals.

Validation uses a strict time split:

- Train on the first 80% of dates.
- Test on the held-out tail after 2024-03-09.
- Result: Pearson r = 0.685, MAE = 2.078 across 2,348 evaluated cells.

### Patrol Allocation

The patrol planner sorts predicted zones by load and greedily selects high-value zones while enforcing a minimum spacing constraint, usually 600 m. This prevents all teams from being stacked around the same street.

### Incident Response

The Incident Command Center uses:

- NetworkX road graph for shortest paths.
- Dijkstra-style routing with risk-based congestion penalties.
- Capacity-aware police dispatch.
- Rule-based closure and resource recommendations.
- SQLite-backed after-action feedback logging.

## Tech Stack

- Python
- Streamlit
- pandas and NumPy
- Plotly
- PyDeck / deck.gl
- scikit-learn
- NetworkX
- Google Gemini API for the copilot
- fpdf2 for PDF briefs
- SQLite for incident logs

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

Open:

```text
http://localhost:8501
```

If `data/clean.pkl` is already present, you can run the app directly. If you need to rebuild the cleaned data layer from the original raw CSV bundle:

```bash
python app/prep.py
```

## Optional Copilot Setup

Create `.streamlit/secrets.toml`:

```toml
GEMINI_API_KEY = "your_google_ai_studio_key"
COPILOT_MODEL = "gemini-2.5-flash"
```

Then reload the Ask ParkSensei page.

## Project Structure

```text
.
|-- app/
|   |-- app.py                         # Command Center
|   |-- core.py                        # scoring, forecasting, allocation, recommendations
|   |-- copilot.py                     # Gemini tool-calling agent
|   |-- incident_db.py                 # after-action SQLite helpers
|   |-- prep.py                        # raw data cleaning pipeline
|   |-- traffic_network.py             # Bengaluru road graph and routing helpers
|   |-- ui.py                          # cached loaders, maps, charts, PDF generation
|   `-- pages/
|       |-- 1_Analytics_and_Insights.py
|       |-- 2_Operations_and_Dispatch.py
|       |-- 3_Strategy_and_Review.py
|       |-- 4_Advanced_Intelligence.py
|       |-- 5_Ask_ParkSensei.py
|       |-- 6_Reports_and_Export.py
|       `-- 7_Incident_Command_Center.py
|-- data/
|   |-- clean.pkl
|   |-- junctions.pkl
|   `-- incidents.db
|-- pitch/
|   |-- VIDEO_SCRIPT.md
|   |-- PITCH_DECK.md
|   `-- GAMMA_DECK.md
|-- README.md
|-- SOLUTION.md
|-- WALKTHROUGH.md
`-- requirements.txt
```

## One-Line Pitch

ParkSensei turns 298,445 rows of what already happened into where to stand tomorrow, what to fix next, and how to respond when the city grid starts locking up.
