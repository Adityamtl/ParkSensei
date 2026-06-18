# 🚦 ParkSensei — Parking Enforcement Intelligence for Bengaluru

**Gridlock Hackathon 2.0 · Round 2 Prototype · Theme 1 — Poor Visibility on Parking-Induced Congestion**

> 🔴 **Live demo:** https://flipkart-gridlock-20-r2-rinxb9yibzp8knp6quhfew.streamlit.app

ParkSensei turns **298,445 real BTP parking-violation records** (Nov 2023 – Apr 2024) into a
decision-support tool that tells Bengaluru Traffic Police **where** illegal parking chokes traffic,
**how much** each hotspot matters, **when** violations will recur, and **how** to deploy limited
patrol teams for maximum impact.

It is not a static dashboard — it is a closed loop: **detect → score → forecast → deploy → target.**

> 📖 **Full solution write-up — the idea, logic and algorithms behind every module (for the team *and* the judges):** see **[SOLUTION.md](SOLUTION.md)**. For a from-scratch walkthrough + judge-Q&A prep (to present and defend it), see **[WALKTHROUGH.md](WALKTHROUGH.md)**.

---

## What it does (6 modules)

| **Command Center** | City-wide 3-D hotspot map, KPIs (incl. PCU weight), Judge Brief, PDF download, enforcement action previews |
| **Hotspot Explorer** | Slice 298K records live by time / weekday / violation type / police station; 7-factor impact breakdown, place-type badges |
| **Forecast & Patrol Planner** | Predicts enforcement load per *zone × weekday × hour*, auto-generates deployment plan with **recommended actions** & PDF brief |
| **Coverage & ROI** | Quantifies the targeting payoff (Pareto + data-driven staffing sweet-spot) and surfaces the evening enforcement blind spot |
| **Repeat-Offender Intelligence** | Surfaces the chronic 15% of vehicles responsible for ~34% of violations |
| **Enforcement Actions** 🆕 | AI-generated enforcement recommendations per zone: tow-away, peak-hour, camera/ANPR, signage audits — with explainable impact breakdown |
| **Ask ParkSensei** 🤖 | Natural-language co-pilot (English / Hindi / Kannada) — **Gemini** function-calling over the *real* models returns patrol plans & zone recommendations |

**Congestion Impact Score (0–100) — enhanced 7-factor formula:**
```
impact = 100 × (
  0.30 × weighted_obstruction + 0.18 × density + 0.15 × junction
  + 0.13 × arterial + 0.10 × peak + 0.08 × recurrence + 0.06 × severity
)
```
PCU vehicle weights: motorcycle=0.35, car=1.0, bus=2.8, tanker=3.2. Transparent, monotone, auditable.

**Forecast:** Bayesian-shrunk historical rates per zone×weekday×hour; sparse weekday cells borrow
strength from the zone's overall hour profile. Validated by an honest time-split backtest
(train on the first 80% of the calendar, predict the held-out tail) → **Pearson r ≈ 0.69**.

**Honesty note:** the timestamps record *when enforcement happened* (patrols are morning-heavy), not
ground-truth parking demand. We treat this explicitly — optimising **enforcement efficiency** on the
observed signal and flagging evening **coverage blind spots** rather than over-claiming.

---

## Run it locally

```bash
# 1. install
pip install -r requirements.txt

# 2. build the cleaned data layer from the raw CSV
#    (expects: "dataset theme 1/jan to may police violation_anonymized*.csv")
python app/prep.py

# 3. launch
streamlit run app/app.py
```

Then open **http://localhost:8501**.

> If `data/clean.pkl` is already present (shipped in the source bundle), skip step 2.

## Project layout

```
round2/
├─ app/
│  ├─ app.py                     # Command Center (entry)
│  ├─ core.py                    # intelligence layer (zones, impact, forecaster, patrol optimiser) — pure pandas, unit-testable
│  ├─ ui.py                      # cached loaders, colour ramps, pydeck/plotly builders
│  ├─ prep.py                    # raw CSV → data/clean.pkl + junctions.pkl
│  └─ pages/
│     ├─ 1_Hotspot_Explorer.py
│     ├─ 2_Forecast_and_Patrol.py
│     └─ 3_Repeat_Offenders.py
├─ data/                         # generated: clean.pkl, junctions.pkl
├─ .streamlit/config.toml        # dark theme
└─ requirements.txt
```

`python app/core.py` runs a standalone smoke test of the intelligence layer (zones, backtest, sample patrol plan).

---

*Built for Bengaluru Traffic Police (ASTraM) · data provided by HackerEarth · uses only the provided Theme-1 dataset.*
