# ParkSensei — Next.js frontend

A polished, client-side rebuild of the ParkSensei dashboard. The Python intelligence (`../app/core.py`) is
**precomputed to static JSON**; this app loads that JSON and does the light interactive math in the browser.
Result: a fast, beautiful app that deploys to **Vercel with no always-on backend**.

```bash
npm install
npm run dev      # http://localhost:3000
```

> Regenerate the data after any change to the models: `python ../app/export_json.py`
> (writes `public/data/*.json`).

---

## How this differs from the Streamlit version — what's precomputed vs live

The Streamlit app runs **everything in Python at request time** (cached per session). This app splits the work
three ways:

| Work | Streamlit | Next.js |
|---|---|---|
| Clean data, build zones, **impact score** | Python, live (cached) | **Precomputed** → `zones.json` |
| City density (gh7 hexes) | Python, live | **Precomputed** → `grid.json` |
| **Forecaster** (zone × weekday × hour rates) | Python, live (cached) | **Precomputed** → `forecast.json` (23k cells) |
| Coverage curve, repeat-offender + backtest stats | Python, live | **Precomputed** → `meta.json`, `offenders.json` |
| **Filter** hotspots by time/day/type | Python, live | **Browser (JS)** over the JSON |
| **predict_load** (sum rates for a shift) | Python, live | **Browser (JS)** — `lib/ParkSensei.ts` |
| **allocate_patrols** (greedy + spacing) | Python, live | **Browser (JS)** — ported to TS |
| **ROI / sweet-spot** | Python, live | **Browser (JS)** |
| **Ask ParkSensei** (Gemini + tools) | Python, live | **Serverless route** (the one piece that needs a server — to keep the API key off the client) |

**So: we precompute the heavy, static intelligence; we compute the light, interactive bits in the browser;
and only the AI co-pilot needs a tiny serverless function.** No database, no always-on Python — the whole thing
is static files + one edge/serverless route.

Why this is the right call: the forecaster, zones and impact scores **don't change between requests** (the data
is a fixed 5-month export), so computing them per-request is wasted work. Precomputing once → instant loads.
The only genuinely dynamic things (filter, predict, allocate, ROI) are cheap and run fine on the client.

---

## What's built vs what's left (migration status)

**Built (this milestone):**
- ✅ Precompute pipeline (`../app/export_json.py` → `public/data/`)
- ✅ Patrol allocator + forecaster math ported to TypeScript (`src/lib/ParkSensei.ts`)
- ✅ **Command Center** — deck.gl 3-D hotspot map, KPIs, coverage chart, top hotspots

**To fully shift off Streamlit, port these 4 pages + the co-pilot route:**
1. **Hotspot Explorer** — filters + zone map + table. ⚠️ *Filtering by violation-type/station needs data the
   current export doesn't carry.* Pick one: (a) precompute extra cuts (e.g. zone×hour, zone×type aggregates),
   (b) ship a slimmed per-violation file and filter client-side, or (c) a serverless aggregate endpoint.
   Simplest for the prototype: precompute a `zone × hour × weekday` table (small) and filter on that.
2. **Forecast & Patrol** — weekday/hours/teams → `predictLoad` + `allocatePatrols` (already in TS) → map with
   team pins + CSV download. Pure client-side. *Easy.*
3. **Coverage & ROI** — port `roi_curve` to TS (~15 lines) + coverage from `meta.json` + charts. *Easy.*
4. **Repeat Offenders** — from `offenders.json` + `meta.json`. *Easy.*
5. **Ask ParkSensei** — add `src/app/api/copilot/route.ts` holding `GEMINI_API_KEY` (server env), using the
   JS Gemini SDK (`@google/genai`) with function-calling that runs the TS tool functions server-side. This is
   the only part that needs server code.

**Also needed:** shared components (Map variants, KPI cards, a chart lib like `recharts` or hand-rolled SVG,
nav/sidebar), app-router pages (`/explorer`, `/forecast`, `/coverage`, `/offenders`, `/ask`), and consistent
Tailwind styling.

**Rough effort:** ~1 focused day for a strong, complete port.

---

## Deploy to Vercel
- Import the repo, set **Root Directory = `web`**.
- For the co-pilot route, add env var `GEMINI_API_KEY`.
- Deploy. The static JSON ships in `public/`; everything else is static or edge.

> ⚠️ This is **Next.js 16** — see `AGENTS.md`; some APIs differ from older versions. Check
> `node_modules/next/dist/docs/` before writing new patterns.
