# 🎤 ParkSensei — Pitch Deck (blueprint with speaker notes)

**Use this to build/refine the deck.** Each slide has: what appears **On slide**, a **Visual** suggestion,
and **Say** (your spoken talking points). The clean text-only version for Gamma is in `GAMMA_DECK.md`.
Target length: **11–12 slides, ~2.5–3 min.** Replace `[Team Name]` before presenting (the live demo link is already filled in).

---

## SLIDE 1 — Title
- **On slide:** **ParkSensei** · *"From 298K parking tickets to where to stand tomorrow."* · Theme 1 — Poor Visibility on Parking-Induced Congestion · Gridlock Hackathon 2.0 · [Team Name]
- **Visual:** Bold title on dark; faint Bengaluru hotspot map as background; 🚦 mark.
- **Say:** "We're [Team Name]. ParkSensei turns Bengaluru Traffic Police's own parking data into a tool that tells officers exactly *where* to enforce — and *when*."

## SLIDE 2 — The problem
- **On slide:** Illegal parking chokes Bengaluru's busiest roads — markets, metro stations, commercial hubs. Vehicles block live lanes and junctions every day.
- **Visual:** Photo of an illegally-parked, congested Bengaluru street.
- **Say:** "On-street illegal parking near markets and metro stations chokes carriageways and intersections every single day — one of the city's most visible, most complained-about congestion sources."

## SLIDE 3 — Why it's hard today
- **On slide:** ① Enforcement is reactive & patrol-based — officers go on instinct. ② No heatmap of where violations actually hurt traffic. ③ No way to prioritize zones or schedule teams.
- **Visual:** Three icons; a "today → needed" contrast.
- **Say:** "Today, enforcement is reactive. There's no city-wide heatmap, no way to rank which violations actually hurt traffic, and no data to decide where to send limited teams. It's instinct — not intelligence."

## SLIDE 4 — Meet ParkSensei
- **On slide:** A closed decision loop: **DETECT → SCORE → FORECAST → DEPLOY → TARGET.** *"Not a dashboard. A decision engine."*
- **Visual:** The 5-stage loop diagram (circular, feeds back).
- **Say:** "ParkSensei closes that gap with a closed loop — detect the hotspots, score them by real traffic impact, forecast when they'll flare up, generate a patrol plan, and target repeat offenders."

## SLIDE 5 — Built on real data, read honestly
- **On slide:** 298,445 real BTP records · 150 days · zero missing fields. **Honest insight:** this is *enforcement time, not demand* — ~93% logged before 1 PM. → We optimize **efficiency** & flag **blind spots**; we don't over-claim.
- **Visual:** The weekday × hour heatmap screenshot (shows the morning concentration).
- **Say:** "It's built on 298,000 real records — five months, clean. And we read them honestly: this shows when patrols wrote tickets, not true demand — 93% before 1 PM. So we optimize enforcement *efficiency* and flag the *evening blind spots*, instead of pretending to predict demand we can't see. The judges are BTP — they know their own data, and we respect that."

## SLIDE 6 — Detect & Score
- **On slide:** City-wide 3-D hotspot map. **Congestion Impact Score (0–100)** = volume × severity × flow-criticality. Top **1%** of locations = **33%** of violations.
- **Visual:** Command Center hotspot map screenshot + zoom on the impact-ranked table.
- **Say:** "First we detect — every violation mapped; the worst 1% of locations carry a third of them. But raw counts mislead, so we built a Congestion Impact Score: it weights volume by how badly each violation blocks traffic and whether it's at a junction. One transparent 0-to-100 number to rank every zone."

## SLIDE 7 — Forecast
- **On slide:** Predict violations per **zone × weekday × hour.** Empirical-Bayes shrinkage → stable even for sparse cells. **Honest backtest: r ≈ 0.69** on held-out weeks.
- **Visual:** Forecast/Patrol page screenshot + the "How the forecast works" panel.
- **Say:** "Then we forecast — expected violations for any zone, any weekday, any hour. Bayesian shrinkage keeps sparse cells stable. And we validate honestly: trained on the first 80% of the calendar, tested on unseen weeks — correlation 0.69. Real signal, not an overfit 0.99."

## SLIDE 8 — Deploy *(the hero slide)*
- **On slide:** Pick a shift + #teams → an instant, **spatially-spread patrol plan.** Expected catches per team · downloadable CSV. *"Every team starts where congestion will build — before it does."*
- **Visual:** Patrol Planner screenshot — team pins on the map + the deployment table.
- **Say:** "Here's the payoff. A supervisor picks Saturday, 9 to 1, eight teams — and in two seconds gets eight deployment points, spread so teams don't bunch up, each with the violations they'll likely encounter. Download the CSV, brief the teams. Reactive enforcement becomes proactive."

## SLIDE 8b — Coverage & ROI *(targeting pays)*
- **On slide:** A few zones hold most violations → a few well-placed teams capture a big share. **Staffing sweet spot** (extra teams add < 1%). **Blind spot:** ~93% before 1 PM, near-zero evenings.
- **Visual:** The ROI coverage curve + the evening blind-spot bar chart (Coverage & ROI page).
- **Say:** "Targeting isn't a nicety — it's the whole game. A few zones hold most of the violations, so a handful of well-placed teams see a big share. We even show the staffing sweet spot, where extra teams stop helping — and we flag the evening blind spot you're not covering today."

## SLIDE 9 — Target repeat offenders
- **On slide:** **15%** of vehicles cause **34%** of violations. Auto-generated target list → owner notices / escalated penalties.
- **Visual:** Repeat-offender distribution chart + top-offenders table.
- **Say:** "And we target the chronic minority — 15% of vehicles cause a third of all violations. ParkSensei hands BTP that short, high-value list for notices and escalated action."

## SLIDE 9b — Ask ParkSensei 🤖 *(the surprise)*
- **On slide:** Ask in **English / Hindi / Kannada** — *"Plan 6 teams for Friday evening around KR Market"* → a **real** plan + map, instantly. Powered by **Gemini function-calling** over the actual models.
- **Visual:** The chat — a typed question → the returned deployment table + map.
- **Say:** "And here's the surprise. An officer can just ask — in English, Hindi or Kannada — 'plan six teams for Friday evening around KR Market.' Gemini routes that to our real forecaster and optimiser and hands back an actual plan, on the map, in seconds. It's not a chatbot — it's an agent driving our models. No dashboard training; you just ask."

## SLIDE 10 — Impact & rollout
- **On slide:** **BTP gains:** visibility · prioritization · proactive shifts · offender targeting. **Rollout:** ① Retrospective (today) → ② Nightly refresh → ③ Live ASTraM feed. *Uses only data BTP already collects.*
- **Visual:** 3-phase arrow timeline.
- **Say:** "The impact: the visibility the brief says is missing, prioritized zones, proactive scheduling, offender targeting — using only data BTP already has. It runs today on historical exports, scales to a nightly refresh, and ultimately a live ASTraM feed."

## SLIDE 11 — Why ParkSensei wins
- **On slide:** **Feasible** — works today, on a laptop. **Innovative** — a decision engine, not a chart. **Explainable** — every score justifiable. **Scalable** — geohash engine; any city, any violation type. **Honest** — credible with BTP.
- **Visual:** Five checkmarks (mirrors the judging criteria).
- **Say:** "It's feasible — running now. Innovative — it outputs a decision, not a chart. Explainable — every number can be justified to a citizen or a court. Scalable — the engine is city-agnostic. And honest about its data. That's ParkSensei."

## SLIDE 12 — Close
- **On slide:** **ParkSensei** — turning *"what already happened"* into *"where to stand tomorrow."* · https://flipkart-gridlock-20-r2-rinxb9yibzp8knp6quhfew.streamlit.app · [Team Name] · Thank you.
- **Visual:** Logo + live demo URL + Bengaluru map backdrop.
- **Say:** "ParkSensei turns 298,000 rows of what already happened into where to stand tomorrow. Thank you — we'd love to put this in front of Bengaluru Traffic Police."

---

### Design notes
- **Palette:** dark slate background, blue `#4C8BF5` accent, amber→red for "impact/severity" — matches the app, so screenshots blend in.
- **One idea per slide.** Let the screenshots carry the weight; keep text to the bullets above.
- **Screenshots to capture** (from the running app): Command Center hex map · Hotspot Explorer (filtered) · Forecast & Patrol plan · Repeat-Offender chart.
- **Order = the loop.** Slides 6→9 walk Detect→Score→Forecast→Deploy→Target in the same order as the demo, so the deck and the video reinforce each other.
