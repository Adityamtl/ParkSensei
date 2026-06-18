# 🎬 ParkSensei — Demo Video Script (line by line)

**Target length:** ~2 min 45 sec · **Format:** screen-record the running app + voiceover.
Read each **SAY** line aloud verbatim; do the **SCREEN** action as you say it.

### Before you record
- Run the app: `streamlit run app/app.py` → full-screen the browser at **1920×1080**.
- Pre-open all 4 pages once so they're cached (no loading spinners on camera).
- On the Forecast page, pre-set **Saturday · 9–13 · 8 teams** so it's ready.
- Use a quiet room; speak slightly slower than feels natural.

---

## 0:00–0:20 · Hook (Command Center on screen, map idle)

**1. [SCREEN: Command Center, 3-D hotspot map visible]**
**SAY:** "Every day in Bengaluru, illegally parked vehicles block live lanes and junctions — choking traffic at markets, metro stations, and commercial hubs."

**2. [SCREEN: slowly drag-rotate the 3-D hex map]**
**SAY:** "But Traffic Police enforce this reactively — on instinct. There's no map of where parking actually hurts traffic, and no way to prioritize."

**3. [SCREEN: hold on the rotated map]**
**SAY:** "We built ParkSensei to fix that — using the Traffic Police's own data."

## 0:20–0:45 · What it is (stay on Command Center)

**4. [SCREEN: point cursor across the KPI row]**
**SAY:** "It's built on 298,000 real violation records over five months. ParkSensei is a closed loop: detect hotspots, score their traffic impact, forecast them, and deploy patrols."

**5. [SCREEN: scroll to the weekday × hour heatmap]**
**SAY:** "And we're honest about the data: 93% of violations are logged before 1 PM — this is when patrols are out, not when parking peaks."

**6. [SCREEN: hover the empty evening columns of the heatmap]**
**SAY:** "So ParkSensei optimizes enforcement efficiency, and flags the evening blind spots that current patrols miss."

## 0:45–1:15 · Detect & Score (go to Hotspot Explorer)

**7. [SCREEN: click "Hotspot Explorer"; full map of bubbles]**
**SAY:** "Here's the city. Every bubble is a hotspot — sized by violations, colored by impact. The worst 1% of locations carry a third of all violations."

**8. [SCREEN: change the violation-type filter to "PARKING IN A MAIN ROAD"]**
**SAY:** "We don't just count tickets — we score them. A car blocking a main road matters more than one on a footpath…"

**9. [SCREEN: watch the map and the ranked table re-rank live]**
**SAY:** "…so our Congestion Impact Score weights each violation by how badly it blocks the carriageway and whether it's at a junction. The ranking updates instantly."

**10. [SCREEN: set the hour slider to the evening, e.g. 17–21]**
**SAY:** "Slice it by any time, day, or station — and the hotspots re-rank in real time."

## 1:15–2:05 · Forecast & Deploy (go to Forecast & Patrol — the hero)

**11. [SCREEN: click "Forecast & Patrol Planner"]**
**SAY:** "Now the payoff. ParkSensei forecasts expected violations for every zone, by weekday and hour."

**12. [SCREEN: point at the "Forecast accuracy r = 0.69" KPI]**
**SAY:** "And we validate it honestly — trained on past weeks, tested on unseen ones: correlation 0.69. Real signal, not an overfit number."

**13. [SCREEN: show the controls set to Saturday · 9–13 · 8 teams]**
**SAY:** "A supervisor just picks a shift — Saturday, 9 to 1 — and the number of teams. Say eight."

**14. [SCREEN: the map redraws with 8 blue team pins; the plan table on the right]**
**SAY:** "Instantly, ParkSensei places eight teams — spread across the city so they don't bunch up — each at a high-impact zone, with the violations they'll likely encounter."

**15. [SCREEN: click the "Download plan (CSV)" button]**
**SAY:** "One click downloads the deployment plan. The supervisor briefs the teams — and every team starts exactly where congestion will build, before it does."

## 2:05–2:25 · Target offenders (go to Repeat Offenders)

**16. [SCREEN: click "Repeat-Offender Intelligence"; KPIs + chart]**
**SAY:** "Finally, ParkSensei targets the chronic offenders — just 15% of vehicles cause 34% of all violations."

**17. [SCREEN: scroll the top-offenders table]**
**SAY:** "It hands the Traffic Police a short, high-value list for notices and escalated penalties."

## 2:25–2:45 · Close (back to Command Center map)

**18. [SCREEN: return to Command Center hotspot map]**
**SAY:** "ParkSensei needs no new sensors — only the data BTP already collects. It runs today, and scales to a live feed."

**19. [SCREEN: hold on the map; optionally fade to logo + demo URL]**
**SAY:** "It turns 298,000 rows of what already happened into where to stand tomorrow. That's ParkSensei — thank you."

---

### Timing cheat-sheet
| Segment | Time | Page |
|---|---|---|
| Hook + what it is | 0:00–0:45 | Command Center |
| Detect & Score | 0:45–1:15 | Hotspot Explorer |
| Forecast & Deploy | 1:15–2:05 | Forecast & Patrol |
| Target offenders | 2:05–2:25 | Repeat Offenders |
| Close | 2:25–2:45 | Command Center |

> Keep it under 3 minutes. If you need to trim, shorten the Hook (lines 1–3) — the Forecast & Deploy
> segment (lines 11–15) is the one that wins it; never cut that.
