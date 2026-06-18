# ParkSensei — Gamma-ready deck text

**HOW TO USE (don't paste this part):**
1. Go to **gamma.app** → **Create new** → **Paste in text**.
2. Copy **everything below the `=== PASTE BELOW ===` line** and paste it in.
3. Choose **"Cards" / one card per `---`**, set tone to *Professional*, theme *dark*.
4. Generate. Then drop in the app screenshots (Command Center map, Patrol plan, Repeat-offender chart).
5. Replace `[Team Name]` with your team name (the live demo link is already filled in).

=== PASTE BELOW ===

# ParkSensei
### From 298K parking tickets to where to stand tomorrow.
Theme 1 — Poor Visibility on Parking-Induced Congestion
Gridlock Hackathon 2.0 · [Team Name]

---

# The problem
Illegal parking chokes Bengaluru's busiest roads.
- Markets, metro stations, commercial hubs — vehicles block live lanes and junctions every day.
- One of the city's most visible and most complained-about congestion sources.

---

# Why it's hard today
- Enforcement is **reactive** and patrol-based — officers go on instinct.
- **No heatmap** of where violations actually hurt traffic.
- **No way to prioritize** zones or schedule limited teams.

---

# Meet ParkSensei
**A closed decision loop — not a dashboard.**
**Detect → Score → Forecast → Deploy → Target**
- Detect the hotspots, score them by real traffic impact, forecast when they recur, generate a patrol plan, and target repeat offenders.

---

# Built on real data — read honestly
- **298,445** real BTP violation records · 150 days · zero missing fields.
- **Honest insight:** this is *enforcement time, not demand* — ~93% logged before 1 PM.
- So we optimize **enforcement efficiency** and flag **evening blind spots** — we don't over-claim.

---

# Detect & Score
- City-wide **3-D hotspot map** — the top **1%** of locations hold **33%** of all violations.
- **Congestion Impact Score (0–100)** = volume × severity × flow-criticality.
- One transparent number to rank every zone — junction-blocking and main-road violations weigh most.

---

# Forecast
- Predict violations per **zone × weekday × hour.**
- Empirical-Bayes shrinkage keeps even sparse cells stable.
- **Honest backtest: r ≈ 0.69** on held-out weeks — real signal, not an overfit 0.99.

---

# Deploy — the patrol planner
- Pick a shift window + number of teams → an instant, **spatially-spread deployment plan.**
- Each team gets a zone with its **expected catches**; download as CSV and brief the teams.
- **Every team starts where congestion will build — before it does.**

---

# Coverage & ROI — targeting pays
- A handful of zones hold most of the action — a few well-placed teams capture a large share of violations.
- We surface the **staffing sweet spot**: past it, each extra team adds < 1 percentage point of coverage.
- And the **blind spot**: ~93% of enforcement is before 1 PM, almost none in the 5–9 PM evening peak.

---

# Target repeat offenders
- **15%** of vehicles cause **34%** of all violations.
- Auto-generated target list → owner notices and escalated penalties.
- A short, high-value action list with outsized impact.

---

# Ask ParkSensei — the AI co-pilot 🤖
- Ask in plain **English, Hindi or Kannada**: *"Plan 6 teams for Friday evening around KR Market."*
- **Gemini function-calling** runs our *real* forecaster + optimiser and returns an actual plan + map.
- Not a chatbot — a true agent over the models. A field officer just **asks**; no dashboard training needed.

---

# Impact & rollout
- **BTP gains:** city-wide visibility · prioritized zones · proactive shifts · offender targeting.
- **Rollout:** ① Retrospective (works today) → ② Nightly refresh → ③ Live ASTraM feed.
- Uses **only data BTP already collects** — no new sensors, runs on a laptop.

---

# Why ParkSensei wins
- **Feasible** — working software, today.
- **Innovative** — outputs a decision, not a chart.
- **Explainable** — every score is justifiable.
- **Scalable** — geohash engine; any city, any violation type.
- **Honest** — credible with the people who own the data.

---

# ParkSensei
### Turning "what already happened" into "where to stand tomorrow."
Live demo: https://flipkart-gridlock-20-r2-rinxb9yibzp8knp6quhfew.streamlit.app
[Team Name] · Thank you
