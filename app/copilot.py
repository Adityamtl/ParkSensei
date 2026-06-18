"""
copilot.py — the "Ask ParkSensei" agent (Google Gemini automatic function calling).
Maps a natural-language question to REAL ParkSensei computations over the 298K-record models, so every
answer is grounded (not generated prose). The Gemini SDK runs the tool loop automatically. No Streamlit.

Enhanced with enforcement recommendation capabilities.
"""
import json
import core

DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
_DAYS = {"mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6}

def _dow_index(name):
    return _DAYS.get(str(name or "sat").strip().lower()[:3], 5)

def _clean(o):
    """Coerce numpy scalars to native types so the tool result is JSON-serialisable for Gemini."""
    return json.loads(json.dumps(o, default=lambda x: x.item() if hasattr(x, "item") else str(x)))

SYSTEM = (
    "You are ParkSensei's enforcement co-pilot for the Bengaluru Traffic Police. You help officers decide "
    "where and when to deploy parking-enforcement teams, using ONLY the tools provided — which run on "
    "298,000 real violation records. Talk like a sharp duty officer: concise, concrete, operational. "
    "For any 'where should I deploy / plan teams' request, call make_patrol_plan. "
    "For 'what should we do about [zone]' questions, call get_zone_recommendations. "
    "Ground every number in a tool result; never invent figures. "
    "If the weekday, time window or team count is missing, assume a sensible default and say so. "
    "You may answer in English, Hindi or Kannada to match the user. "
    "When recommending actions, mention the impact score, PCU weight, and specific enforcement actions."
)

def _make_tools(ctx):
    """Build the tool callables as closures over ctx (df, zones, fc). Gemini introspects their
       type hints + docstrings to build the schema, and executes them via automatic function calling."""
    df, zones, fc = ctx["df"], ctx["zones"], ctx["fc"]

    def make_patrol_plan(weekday: str, start_hour: int, end_hour: int,
                         teams: int, area: str = "") -> dict:
        """Generate an optimal, spatially-spread patrol deployment plan and return the deployments.

        Args:
            weekday: e.g. 'Saturday'.
            start_hour: shift start hour, 0-23 IST.
            end_hour: shift end hour, 0-23 IST.
            teams: number of patrol teams to place.
            area: optional; restrict to zones whose name contains this text, e.g. 'KR Market'.
        """
        dow = _dow_index(weekday)
        h0, h1 = max(0, min(23, int(start_hour))), max(0, min(23, int(end_hour)))
        if h1 < h0:
            h0, h1 = h1, h0
        teams = max(1, int(teams))
        zsub = zones
        if area and area.strip():
            mask = zones["label"].str.contains(area.strip(), case=False, na=False)
            if mask.any():
                zsub = zones[mask]
        pred = core.predict_load(fc, dow, range(h0, h1 + 1))
        plan = core.allocate_patrols(zsub, pred, k=teams)
        ctx["plan"] = plan
        disp_cols = ["team", "label", "pred_load", "impact_score"]
        if "recommended_action" in plan.columns:
            disp_cols.append("recommended_action")
        return _clean({"weekday": DOW[dow], "window": f"{h0:02d}:00-{h1:02d}:59",
                       "teams": int(len(plan)), "scope": f"area: {area}" if area else "city-wide",
                       "deployments": plan[disp_cols].head(teams).to_dict("records")})

    def top_hotspots(n: int = 10) -> dict:
        """List the highest Congestion-Impact-Score parking hotspots city-wide.

        Args:
            n: how many hotspots to return.
        """
        n = max(1, int(n))
        cols = ["label", "violations", "impact_score", "top_violation"]
        if "avg_pcu" in zones.columns:
            cols.append("avg_pcu")
        if "place_type" in zones.columns:
            cols.append("place_type")
        return _clean({"hotspots": zones.head(n)[cols].to_dict("records")})

    def coverage_stats() -> dict:
        """Enforcement coverage by time of day, incl. the morning concentration and evening blind spot."""
        cov = core.coverage_by_hour(df)
        return _clean({"pct_before_1pm": round(cov[cov.hour < 13]["share"].sum() * 100, 1),
                       "pct_evening_5_9pm": round(cov[cov.hour.between(17, 21)]["share"].sum() * 100, 2),
                       "peak_hour_ist": int(cov.loc[cov["share"].idxmax(), "hour"])})

    def repeat_offenders() -> dict:
        """Statistics on chronic repeat-offender vehicles."""
        vc = core.vehicle_counts(df); rep = vc[vc >= 2]
        return _clean({"unique_vehicles": int(len(vc)), "repeat_vehicles": int(len(rep)),
                       "repeat_share_pct": round(rep.sum() / len(df) * 100, 1),
                       "worst_offender_count": int(vc.iloc[0]) if len(vc) else 0})

    def get_zone_recommendations(zone_name: str) -> dict:
        """Get enforcement recommendations for a specific zone or the top zones.

        Args:
            zone_name: name or partial name of the zone, e.g. 'KR Market', 'Silk Board'.
                       If 'top' or empty, returns recommendations for the top 5 zones.
        """
        if not zone_name or zone_name.strip().lower() in ("top", "all", "best", "worst"):
            # Return top 5
            results = core.zone_recommendations(zones, top_n=5)
        else:
            mask = zones["label"].str.contains(zone_name.strip(), case=False, na=False)
            if mask.any():
                matched = zones[mask]
                results = core.zone_recommendations(matched, top_n=3)
            else:
                return _clean({"error": f"No zone found matching '{zone_name}'",
                               "available_zones": zones["label"].head(10).tolist()})

        formatted = []
        for zr in results:
            z = zr["zone"]
            formatted.append({
                "zone": z.get("label", "Unknown"),
                "impact_score": z.get("impact_score", 0),
                "violations": z.get("violations", 0),
                "avg_pcu": z.get("avg_pcu", 1.0),
                "place_type": z.get("place_type", ""),
                "junction_frac": z.get("junction_frac", 0),
                "peak_share": z.get("peak_share", 0),
                "repeat_share": z.get("repeat_vehicle_share", 0),
                "recommendations": [
                    {"action": r["action"], "priority": r["priority"],
                     "reason": r["reason"], "window": r["window"]}
                    for r in zr["recommendations"]
                ]
            })
        return _clean({"zone_recommendations": formatted})

    return [make_patrol_plan, top_hotspots, coverage_stats, repeat_offenders, get_zone_recommendations]

def run_agent(client, query, ctx, model="gemini-2.5-flash"):
    """Returns (answer_text, plan_DataFrame_or_None). `client` is a google.genai.Client.
       Uses Gemini automatic function calling — the SDK runs the tool loop and returns the final text."""
    from google.genai import types
    ctx = dict(ctx); ctx["plan"] = None
    config = types.GenerateContentConfig(
        system_instruction=SYSTEM, tools=_make_tools(ctx), temperature=0.3)
    resp = client.models.generate_content(model=model, contents=query, config=config)
    return (resp.text or "I couldn't find an answer — try rephrasing."), ctx.get("plan")
