"""
core.py - the intelligence layer for Parking Enforcement Intelligence (Theme 1).
Pure pandas/numpy (no Streamlit) so it is unit-testable standalone:
    python app/core.py
Provides:
    load_clean()                    -> cleaned violations DataFrame
    build_zones(df)                 -> per-neighbourhood (gh6) summary + labels
    add_impact(zones)               -> Congestion Impact Score (0-100), 7-factor
    impact_breakdown(zones)         -> explainable per-factor scores
    generate_recommendations(zone)  -> actionable enforcement recommendations
    build_forecaster(df)            -> shrunk expected-load table per (zone, dow, hour)
    predict_load(fc, dow, hours)    -> predicted enforcement load per zone for a window
    backtest(df)                    -> time-split validation (Pearson r, MAE)
    allocate_patrols(...)           -> spatially-spread top-K deployment plan
"""
import numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

# --------------------------------------------------------------------------
def load_clean() -> pd.DataFrame:
    return pd.read_pickle(DATA / "clean.pkl")

def load_junctions() -> pd.DataFrame:
    return pd.read_pickle(DATA / "junctions.pkl")

def vehicle_counts(df: pd.DataFrame) -> pd.Series:
    """Violations per vehicle (descending), excluding unrecorded ('nan') plates.
       Single source of truth for repeat-offender stats across pages."""
    vc = df["vehicle_number"].value_counts()
    return vc[vc.index.str.lower() != "nan"]

# --------------------------------------------------------------------------
def _label_for(group: pd.DataFrame) -> str:
    """Human-readable label for a zone: dominant named junction else police station."""
    j = group.loc[group["has_junction"], "junction_name"]
    if len(j):
        return j.mode().iat[0]
    s = group["police_station"]
    s = s[s.str.lower() != "nan"]
    return (s.mode().iat[0] if len(s) else "Unnamed area")

def _safe_col_mean(df, col, default=0.0):
    """Safely compute mean of a column, returning default if column missing."""
    if col in df.columns:
        return df[col].mean()
    return default

def build_zones(df: pd.DataFrame, key: str = "gh6") -> pd.DataFrame:
    """Collapse the point cloud into ~800 neighbourhood zones with rich stats."""
    g = df.groupby(key)

    # --- base aggregations (always available) ---
    zones = g.agg(
        lat=("lat", "median"),
        lon=("lon", "median"),
        violations=("lat", "size"),
        avg_severity=("severity", "mean"),
        junction_frac=("has_junction", "mean"),
        main_road_frac=("primary_type", lambda s: (s == "PARKING IN A MAIN ROAD").mean()),
    ).reset_index()

    # --- new aggregations (from ParkSight/ParkWatch integration) ---
    if "pcu" in df.columns:
        pcu_agg = g["pcu"].agg(avg_pcu="mean", total_pcu="sum").reset_index()
        zones = zones.merge(pcu_agg, on=key)
    else:
        zones["avg_pcu"] = 1.0
        zones["total_pcu"] = zones["violations"].astype(float)

    if "obstruction_weight" in df.columns:
        obs_agg = g["obstruction_weight"].agg(total_obstruction="sum",
                                               avg_obstruction="mean").reset_index()
        zones = zones.merge(obs_agg, on=key)
    else:
        zones["total_obstruction"] = zones["violations"] * zones["avg_severity"]
        zones["avg_obstruction"] = zones["avg_severity"]

    if "is_peak_hour" in df.columns:
        peak_agg = g["is_peak_hour"].mean().rename("peak_share").reset_index()
        zones = zones.merge(peak_agg, on=key)
    else:
        zones["peak_share"] = 0.5

    if "is_arterial" in df.columns:
        art_agg = g["is_arterial"].mean().rename("arterial_share").reset_index()
        zones = zones.merge(art_agg, on=key)
    else:
        zones["arterial_share"] = zones["main_road_frac"]

    if "action_delay_mins" in df.columns:
        delay_agg = g["action_delay_mins"].agg(avg_delay_mins="mean",
                                                median_delay_mins="median").reset_index()
        zones = zones.merge(delay_agg, on=key)
    else:
        zones["avg_delay_mins"] = np.nan
        zones["median_delay_mins"] = np.nan

    # active days (recurrence)
    active_days = g["ymd"].nunique().rename("active_days").reset_index()
    zones = zones.merge(active_days, on=key)

    # place type (dominant)
    if "place_type" in df.columns:
        place_agg = g["place_type"].agg(
            lambda s: s.mode().iat[0] if len(s.mode()) else "Street segment"
        ).rename("place_type").reset_index()
        zones = zones.merge(place_agg, on=key)
    else:
        zones["place_type"] = "Street segment"

    # repeat vehicle share
    vc = df["vehicle_number"].map(df["vehicle_number"].value_counts())
    df_temp = df.copy()
    df_temp["_is_repeat"] = (vc >= 2) & (df["vehicle_number"].str.lower() != "nan")
    rep_agg = df_temp.groupby(key)["_is_repeat"].mean().rename("repeat_vehicle_share").reset_index()
    zones = zones.merge(rep_agg, on=key)

    # labels and top violation
    labels = g.apply(_label_for, include_groups=False).rename("label").reset_index()
    top_type = g["primary_type"].agg(lambda s: s.mode().iat[0]).rename("top_violation").reset_index()
    zones = zones.merge(labels, on=key).merge(top_type, on=key)

    return zones

# --------------------------------------------------------------------------
# Impact score weights (7-factor, inspired by ParkSight AI)
_W = {
    "obstruction":  0.30,   # PCU-weighted violation pressure
    "density":      0.18,   # log-scaled violation count
    "junction":     0.15,   # junction proximity fraction
    "arterial":     0.13,   # main-road / crossing share
    "peak":         0.10,   # peak-hour recurrence
    "recurrence":   0.08,   # active-day recurrence
    "severity":     0.06,   # mean violation severity
}

def add_impact(zones: pd.DataFrame) -> pd.DataFrame:
    """Congestion Impact Score (0-100), 7-factor, transparent & monotone.
       Enhanced version fusing ParkSensei's original flow-multiplier approach with
       ParkSight AI's multi-factor weighted index and PCU obstruction weights."""
    z = zones.copy()

    # --- individual factor scores (all normalized to 0..1) ---
    # 1. Weighted obstruction (PCU × severity × boosts) — log-scaled
    log_obs = np.log1p(z["total_obstruction"])
    z["_f_obstruction"] = log_obs / log_obs.max() if log_obs.max() > 0 else 0.0

    # 2. Density (violation count) — log-scaled
    log_n = np.log1p(z["violations"])
    z["_f_density"] = log_n / log_n.max() if log_n.max() > 0 else 0.0

    # 3. Junction exposure
    z["_f_junction"] = z["junction_frac"]

    # 4. Arterial / main-road share
    z["_f_arterial"] = z["arterial_share"]

    # 5. Peak-hour recurrence
    z["_f_peak"] = z["peak_share"]

    # 6. Active-day recurrence (normalized against ~150 days in the dataset)
    total_days = max(z["active_days"].max(), 1)
    z["_f_recurrence"] = (z["active_days"] / total_days).clip(0, 1)

    # 7. Average severity (normalized: severity ranges ~0.1 to 1.0)
    z["_f_severity"] = z["avg_severity"].clip(0, 1)

    # --- composite score ---
    z["impact_score"] = (100 * (
        _W["obstruction"] * z["_f_obstruction"]
        + _W["density"]   * z["_f_density"]
        + _W["junction"]  * z["_f_junction"]
        + _W["arterial"]  * z["_f_arterial"]
        + _W["peak"]      * z["_f_peak"]
        + _W["recurrence"]* z["_f_recurrence"]
        + _W["severity"]  * z["_f_severity"]
    )).round(1)

    # keep factor scores for breakdown display
    for factor in _W:
        z[f"factor_{factor}"] = z[f"_f_{factor}"].round(3)
    z.drop(columns=[c for c in z.columns if c.startswith("_f_")], inplace=True)

    return z.sort_values("impact_score", ascending=False).reset_index(drop=True)

def impact_breakdown(zone_row) -> dict:
    """Return explainable component scores for a single zone (for display)."""
    result = {}
    for factor, weight in _W.items():
        col = f"factor_{factor}"
        raw = float(zone_row.get(col, 0))
        result[factor] = {
            "raw_score": round(raw, 3),
            "weighted":  round(raw * weight * 100, 1),
            "weight":    weight,
            "label":     factor.replace("_", " ").title(),
        }
    return result

# --------------------------------------------------------------------------
# Recommendation engine (inspired by ParkWatch AI)
def generate_recommendations(zone) -> list:
    """Rule-based enforcement recommendation engine.
       Takes a zone row (Series or dict) and returns a list of action dicts.
       Inspired by ParkWatch AI's 8-action recommendation engine."""
    recs = []
    score = float(zone.get("impact_score", 0))
    junction_frac = float(zone.get("junction_frac", 0))
    peak_share = float(zone.get("peak_share", 0))
    arterial_share = float(zone.get("arterial_share", 0))
    repeat_share = float(zone.get("repeat_vehicle_share", 0))
    avg_delay = float(zone.get("avg_delay_mins", 0)) if pd.notna(zone.get("avg_delay_mins")) else 0
    avg_pcu = float(zone.get("avg_pcu", 1.0))

    # 1. Tow Away Zone — high impact + junction/arterial blockage
    if score >= 70 and (junction_frac >= 0.35 or arterial_share >= 0.30):
        recs.append({
            "action": "Tow-Away Zone",
            "priority": "CRITICAL",
            "reason": f"Impact {score:.0f} with {junction_frac*100:.0f}% junction exposure and "
                      f"{arterial_share*100:.0f}% arterial obstruction — vehicles are directly blocking "
                      f"moving lanes and intersections.",
            "window": "24/7 continuous tow presence",
            "icon": "🚛",
        })

    # 2. Peak Hour Enforcement — violation clustering during rush hours
    if peak_share >= 0.50 and score >= 40:
        recs.append({
            "action": "Peak Hour Enforcement",
            "priority": "HIGH" if peak_share >= 0.65 else "MEDIUM",
            "reason": f"{peak_share*100:.0f}% of violations during peak hours — targeted enforcement "
                      f"during rush windows will catch maximum offenders.",
            "window": "08:00–11:00 & 17:00–21:00 IST",
            "icon": "⏰",
        })

    # 3. Camera / ANPR Monitoring — high delay or high impact unreachable by patrol
    if (avg_delay >= 45 and score >= 45) or (score >= 75 and avg_delay >= 30):
        recs.append({
            "action": "CCTV & ANPR Monitoring",
            "priority": "HIGH",
            "reason": f"Average enforcement delay {avg_delay:.0f} mins — manual patrols are too slow. "
                      f"Automated camera enforcement recommended.",
            "window": "Continuous (24/7 automated)",
            "icon": "📷",
        })

    # 4. Repeat Offender Escalation — chronic violators
    if repeat_share >= 0.25:
        recs.append({
            "action": "Repeat Offender Escalation",
            "priority": "HIGH" if repeat_share >= 0.40 else "MEDIUM",
            "reason": f"{repeat_share*100:.0f}% of violations from repeat offenders — standard fines "
                      f"aren't deterring. Escalate to towing priority / registration flags.",
            "window": "Target arrival times of repeat plates",
            "icon": "🔁",
        })

    # 5. Heavy Vehicle Restriction — high PCU weight zone
    if avg_pcu >= 1.5 and score >= 35:
        recs.append({
            "action": "Heavy Vehicle Restriction",
            "priority": "MEDIUM",
            "reason": f"Average PCU {avg_pcu:.1f} — large vehicles (buses, trucks, tempos) dominate "
                      f"violations here. Consider time-based heavy-vehicle entry restrictions.",
            "window": "Peak hours (8–11 AM, 5–9 PM)",
            "icon": "🚚",
        })

    # 6. Signage & Road Marking Audit — high violations but few repeat offenders
    if repeat_share < 0.15 and score >= 35:
        recs.append({
            "action": "Signage & Marking Audit",
            "priority": "MEDIUM",
            "reason": f"Low repeat ratio ({repeat_share*100:.0f}%) suggests many first-time violators — "
                      f"regulations may be poorly marked. Clear signage could reduce violations 20–30%.",
            "window": "Infrastructure rollout within 14 days",
            "icon": "🪧",
        })

    # 7. Precinct Escalation — critical risk
    if score >= 80:
        recs.append({
            "action": "Precinct Command Escalation",
            "priority": "CRITICAL",
            "reason": f"Critical impact score ({score:.0f}) — zone requires jurisdictional-level "
                      f"coordination, dedicated towing, and coordinated police sweeps.",
            "window": "Immediate dispatch",
            "icon": "🚨",
        })

    # 8. Routine Patrol — fallback for lower-risk zones
    if not recs or (score < 35 and peak_share < 0.40):
        recs.append({
            "action": "Routine Patrol",
            "priority": "LOW",
            "reason": "Zone within safe limits. Periodic check-ins maintain compliance and gather data.",
            "window": "Weekly daytime rotation",
            "icon": "👮",
        })

    return recs

def _priority_rank(p):
    """Sort helper for recommendations."""
    return {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}.get(p, 4)

def zone_recommendations(zones: pd.DataFrame, top_n: int = 20) -> list:
    """Generate recommendations for the top N zones. Returns list of dicts with zone info + recs."""
    results = []
    for _, zone in zones.head(top_n).iterrows():
        recs = generate_recommendations(zone)
        recs.sort(key=lambda r: _priority_rank(r["priority"]))
        results.append({
            "zone": zone.to_dict(),
            "recommendations": recs,
            "top_action": recs[0]["action"] if recs else "Routine Patrol",
            "top_priority": recs[0]["priority"] if recs else "LOW",
        })
    return results

# --------------------------------------------------------------------------
def build_forecaster(df: pd.DataFrame, alpha: float = 5.0) -> dict:
    """Expected enforcement load per (zone, weekday, hour), Bayesian-shrunk toward
       the zone-hour average so sparse weekday cells stay stable.
       rate = (count_zdh + alpha * r_zh) / (n_dow + alpha)   (per matching calendar day)."""
    n_dow = df.groupby("dow")["ymd"].nunique()                 # # of each weekday in span
    n_dates = df["ymd"].nunique()
    czdh = df.groupby(["gh6", "dow", "hour"]).size().rename("c").reset_index()
    # zone-hour backoff rate (avg violations per day at that hour, any weekday)
    r_zh = (df.groupby(["gh6", "hour"]).size() / n_dates).rename("r_zh").reset_index()
    czdh = czdh.merge(r_zh, on=["gh6", "hour"], how="left")
    czdh["n_dow"] = czdh["dow"].map(n_dow)
    czdh["rate"] = (czdh["c"] + alpha * czdh["r_zh"]) / (czdh["n_dow"] + alpha)
    return {"rate_zdh": czdh[["gh6", "dow", "hour", "rate", "c"]],
            "r_zh": r_zh, "n_dow": n_dow, "n_dates": n_dates}

def predict_load(fc: dict, dow: int, hours) -> pd.DataFrame:
    """Predicted enforcement load per zone for weekday `dow` summed over `hours`."""
    hours = list(hours)
    t = fc["rate_zdh"]
    sub = t[(t["dow"] == dow) & (t["hour"].isin(hours))]
    return (sub.groupby("gh6")["rate"].sum()
            .rename("pred_load").reset_index()
            .sort_values("pred_load", ascending=False))

# --------------------------------------------------------------------------
def backtest(df: pd.DataFrame, train_frac: float = 0.8, min_count: int = 20) -> dict:
    """Honest time-split: fit rates on the first `train_frac` of the calendar,
       predict the held-out tail, score per (zone,dow,hour) cell."""
    dates = np.sort(df["ymd"].unique())
    cut = dates[int(len(dates) * train_frac)]
    tr, te = df[df["ymd"] < cut], df[df["ymd"] >= cut]
    fc = build_forecaster(tr)
    pred = fc["rate_zdh"].rename(columns={"rate": "pred"})
    # actual per-day rate in the test window
    n_dow_te = te.groupby("dow")["ymd"].nunique()
    act = te.groupby(["gh6", "dow", "hour"]).size().rename("c_te").reset_index()
    act["actual"] = act["c_te"] / act["dow"].map(n_dow_te)
    m = pred.merge(act, on=["gh6", "dow", "hour"], how="inner")
    m = m[m["c"] >= min_count * train_frac]            # evaluate on cells with signal
    r = m["pred"].corr(m["actual"])
    mae = (m["pred"] - m["actual"]).abs().mean()
    return {"pearson_r": round(float(r), 3), "mae": round(float(mae), 3),
            "cells": int(len(m)), "cutoff": str(cut)}

# --------------------------------------------------------------------------
def _haversine(lat1, lon1, lat2, lon2):
    R = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = np.radians(lat2 - lat1), np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))

def allocate_patrols(zones: pd.DataFrame, pred: pd.DataFrame, k: int = 10,
                     min_sep_m: float = 600.0) -> pd.DataFrame:
    """Greedy deploy K teams to the highest predicted-load zones while keeping
       them at least `min_sep_m` apart (avoid stacking teams on one street)."""
    cols = ["gh6", "lat", "lon", "label", "impact_score", "avg_severity",
            "top_violation", "place_type", "junction_frac", "peak_share",
            "arterial_share", "repeat_vehicle_share", "avg_pcu", "avg_delay_mins"]
    available_cols = [c for c in cols if c in zones.columns]
    cand = (pred.merge(zones[available_cols], on="gh6")
            .sort_values("pred_load", ascending=False).reset_index(drop=True))
    chosen = []
    for _, row in cand.iterrows():
        if len(chosen) >= k:
            break
        if chosen:
            d = _haversine(row["lat"], row["lon"],
                           np.array([c["lat"] for c in chosen]),
                           np.array([c["lon"] for c in chosen]))
            if d.min() < min_sep_m:
                continue
        chosen.append(row.to_dict())
    if not chosen:                                   # always return the expected columns
        return pd.DataFrame(columns=["team", "pred_load"] + available_cols)
    out = pd.DataFrame(chosen)
    out.insert(0, "team", [f"Team {i+1}" for i in range(len(out))])
    out["pred_load"] = out["pred_load"].round(2)

    # --- add recommendations for each patrol zone ---
    if "impact_score" in out.columns:
        out["recommended_action"] = out.apply(
            lambda r: generate_recommendations(r)[0]["action"] if len(generate_recommendations(r)) else "Routine Patrol",
            axis=1
        )
    return out

# --------------------------------------------------------------------------
def coverage_by_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Share of enforcement by hour-of-day — exposes the evening coverage gap."""
    c = df.groupby("hour").size()
    return pd.DataFrame({"hour": c.index.astype(int),
                         "violations": c.values,
                         "share": (c / c.sum()).values})

def roi_curve(pred: pd.DataFrame, k_max: int = 20) -> pd.DataFrame:
    """Impact captured by the top-K predicted zones vs spreading teams evenly.
       optimal = cumulative share of predicted load at the top-K zones;
       even    = K / (#active zones) = expected share without targeting."""
    loads = np.sort(pred["pred_load"].values)[::-1]
    total = loads.sum()
    n = len(loads)
    rows, cum, prev = [], 0.0, 0.0
    for k in range(1, min(k_max, n) + 1):
        cum += float(loads[k - 1])
        opt = cum / total if total else 0.0
        even = k / n if n else 0.0
        rows.append({"teams": k, "optimal": opt, "even": even,
                     "ratio": (opt / even) if even else 0.0,
                     "marginal": opt - prev})
        prev = opt
    return pd.DataFrame(rows)

# --------------------------------------------------------------------------
# Enforcement delay analysis
def delay_by_station(df: pd.DataFrame) -> pd.DataFrame:
    """Average enforcement delay by police station."""
    if "action_delay_mins" not in df.columns:
        return pd.DataFrame(columns=["police_station", "avg_delay", "median_delay", "violations"])
    valid = df[df["action_delay_mins"].notna() & (df["police_station"].str.lower() != "nan")]
    agg = (valid.groupby("police_station")
           .agg(avg_delay=("action_delay_mins", "mean"),
                median_delay=("action_delay_mins", "median"),
                violations=("lat", "size"))
           .reset_index()
           .sort_values("avg_delay", ascending=False))
    return agg

# --------------------------------------------------------------------------
if __name__ == "__main__":
    df = load_clean()
    print("clean:", df.shape)
    print("columns:", list(df.columns))
    zones = add_impact(build_zones(df))
    print("\nzones:", zones.shape)
    print(zones[["gh6", "label", "violations", "avg_severity", "avg_pcu",
                 "impact_score", "top_violation", "place_type"]].head(10).to_string(index=False))

    # test recommendations
    print("\n--- Top zone recommendations ---")
    for zr in zone_recommendations(zones, 3):
        z = zr["zone"]
        print(f"\n{z['label']} (impact={z['impact_score']}):")
        for r in zr["recommendations"]:
            print(f"  [{r['priority']}] {r['action']}: {r['reason'][:80]}...")

    # test impact breakdown
    print("\n--- Impact breakdown (top zone) ---")
    bd = impact_breakdown(zones.iloc[0])
    for k, v in bd.items():
        print(f"  {v['label']:20s}  raw={v['raw_score']:.3f}  weighted={v['weighted']:5.1f}  (×{v['weight']})")

    fc = build_forecaster(df)
    print("\nbacktest:", backtest(df))
    DOW, HRS = 5, range(9, 13)        # Saturday 9am-1pm
    pred = predict_load(fc, DOW, HRS)
    plan = allocate_patrols(zones, pred, k=8)
    print(f"\npatrol plan (Sat 9-13h, 8 teams):")
    disp_cols = ["team", "label", "pred_load", "impact_score", "top_violation"]
    if "recommended_action" in plan.columns:
        disp_cols.append("recommended_action")
    print(plan[disp_cols].to_string(index=False))
