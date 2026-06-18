"""
export_json.py — precompute ParkSensei intelligence to static JSON for the Next.js frontend.
Run:  python app/export_json.py   → writes web/public/data/*.json
This is what lets the Next.js app run fully client-side on Vercel (no live Python backend).
"""
import json, pathlib
import core

OUT = pathlib.Path(__file__).resolve().parent.parent / "web" / "public" / "data"
OUT.mkdir(parents=True, exist_ok=True)

df    = core.load_clean()
zones = core.add_impact(core.build_zones(df))
fc    = core.build_forecaster(df)
cov   = core.coverage_by_hour(df)
vc    = core.vehicle_counts(df); rep = vc[vc >= 2]
bt    = core.backtest(df)

# --- zones (for the impact map + patrol allocation, done client-side) ---
zcols = ["gh6", "lat", "lon", "label", "violations", "impact_score",
         "avg_severity", "junction_frac", "main_road_frac", "top_violation"]
zout = zones[zcols].round({"lat": 5, "lon": 5, "avg_severity": 3,
                           "junction_frac": 3, "main_road_frac": 3})
zout.to_json(OUT / "zones.json", orient="records")

# --- gh7 density grid (for the hex map) ---
grid = (df.groupby("gh7").agg(lat=("lat", "median"), lon=("lon", "median"), n=("lat", "size"))
          .reset_index().round({"lat": 5, "lon": 5}))
grid[["lat", "lon", "n"]].to_json(OUT / "grid.json", orient="records")

# --- forecast rate table (zone x weekday x hour); drop negligible cells to shrink ---
rate = fc["rate_zdh"][["gh6", "dow", "hour", "rate"]].copy()
rate["rate"] = rate["rate"].round(3)
rate = rate[rate["rate"] > 0.005]
rate.to_json(OUT / "forecast.json", orient="records")

# --- top repeat offenders (for that page) ---
top_off = (vc.head(25).rename("violations").reset_index())
top_off.columns = ["vehicle", "violations"]
top_off.to_json(OUT / "offenders.json", orient="records")

# --- meta: headline stats, coverage curve, backtest, hotspots ---
meta = {
    "totals": {"violations": int(len(df)), "zones": int(len(zones)),
               "days": int(df["ymd"].nunique()),
               "junctions": int(df.loc[df.has_junction, "junction_name"].nunique())},
    "coverage": cov.assign(share=cov["share"].round(4))[["hour", "share"]].to_dict("records"),
    "coverage_summary": {"before_1pm": round(cov[cov.hour < 13]["share"].sum() * 100, 1),
                         "evening_5_9": round(cov[cov.hour.between(17, 21)]["share"].sum() * 100, 2)},
    "repeat": {"unique": int(len(vc)), "repeat_vehicles": int(len(rep)),
               "share_pct": round(rep.sum() / len(df) * 100, 1),
               "worst": int(vc.iloc[0]) if len(vc) else 0},
    "backtest": bt,
    "top_hotspots": zones.head(10)[["label", "violations", "impact_score"]].to_dict("records"),
    "dow_names": ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
}
(OUT / "meta.json").write_text(json.dumps(meta, default=str), encoding="utf-8")

print("wrote to", OUT)
for f in sorted(OUT.glob("*.json")):
    print(f"  {f.name:16s} {f.stat().st_size/1024:8.1f} KB")
print(f"rows -> zones {len(zout)}, grid {len(grid)}, forecast {len(rate)}, offenders {len(top_off)}")
