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
def _optimize_clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce resident memory after loading the cleaned deployment dataset."""
    category_cols = [
        "gh6", "gh7", "primary_type", "police_station",
        "junction_name", "vehicle_type", "vehicle_number", "place_type",
    ]
    float_cols = [
        "lat", "lon", "severity", "pcu", "obstruction_weight",
        "action_delay_mins",
    ]
    int_cols = ["hour", "dow"]
    bool_cols = ["has_junction", "is_peak_hour", "is_arterial"]

    for col in category_cols:
        if col in df.columns:
            df[col] = df[col].astype("category")
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="float")
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], downcast="integer")
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def load_clean() -> pd.DataFrame:
    return _optimize_clean_frame(pd.read_pickle(DATA / "clean.pkl"))

def load_junctions() -> pd.DataFrame:
    return pd.read_pickle(DATA / "junctions.pkl")

def vehicle_counts(df: pd.DataFrame) -> pd.Series:
    """Violations per vehicle (descending), excluding unrecorded ('nan') plates.
       Single source of truth for repeat-offender stats across pages."""
    vc = df["vehicle_number"].value_counts()
    return vc[vc.index.str.lower() != "nan"]

# --------------------------------------------------------------------------
def _first_mode(series: pd.Series, default="Unknown"):
    mode = series.dropna().mode()
    return mode.iat[0] if len(mode) else default


def _label_for(group: pd.DataFrame) -> str:
    """Human-readable label for a zone: dominant named junction else police station."""
    j = group.loc[group["has_junction"], "junction_name"]
    if len(j):
        return _first_mode(j, "Unnamed area")
    s = group["police_station"]
    s = s[s.str.lower() != "nan"]
    return _first_mode(s, "Unnamed area") if len(s) else "Unnamed area"

def _safe_col_mean(df, col, default=0.0):
    """Safely compute mean of a column, returning default if column missing."""
    if col in df.columns:
        return df[col].mean()
    return default

def build_zones(df: pd.DataFrame, key: str = "gh6") -> pd.DataFrame:
    """Collapse the point cloud into ~800 neighbourhood zones with rich stats."""
    g = df.groupby(key, observed=True)

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
            lambda s: _first_mode(s, "Street segment")
        ).rename("place_type").reset_index()
        zones = zones.merge(place_agg, on=key)
    else:
        zones["place_type"] = "Street segment"

    # repeat vehicle share
    vc = df["vehicle_number"].map(df["vehicle_number"].value_counts())
    df_temp = df.copy()
    df_temp["_is_repeat"] = (vc >= 2) & (df["vehicle_number"].str.lower() != "nan")
    rep_agg = df_temp.groupby(key, observed=True)["_is_repeat"].mean().rename("repeat_vehicle_share").reset_index()
    zones = zones.merge(rep_agg, on=key)

    # labels and top violation
    labels = g.apply(_label_for, include_groups=False).rename("label").reset_index()
    top_type = g["primary_type"].agg(lambda s: _first_mode(s, "UNKNOWN")).rename("top_violation").reset_index()
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
        })

    # 2. Peak Hour Enforcement — violation clustering during rush hours
    if peak_share >= 0.50 and score >= 40:
        recs.append({
            "action": "Peak Hour Enforcement",
            "priority": "HIGH" if peak_share >= 0.65 else "MEDIUM",
            "reason": f"{peak_share*100:.0f}% of violations during peak hours — targeted enforcement "
                      f"during rush windows will catch maximum offenders.",
            "window": "08:00–11:00 & 17:00–21:00 IST",
        })

    # 3. Camera / ANPR Monitoring — high delay or high impact unreachable by patrol
    if (avg_delay >= 45 and score >= 45) or (score >= 75 and avg_delay >= 30):
        recs.append({
            "action": "CCTV & ANPR Monitoring",
            "priority": "HIGH",
            "reason": f"Average enforcement delay {avg_delay:.0f} mins — manual patrols are too slow. "
                      f"Automated camera enforcement recommended.",
            "window": "Continuous (24/7 automated)",
        })

    # 4. Repeat Offender Escalation — chronic violators
    if repeat_share >= 0.25:
        recs.append({
            "action": "Repeat Offender Escalation",
            "priority": "HIGH" if repeat_share >= 0.40 else "MEDIUM",
            "reason": f"{repeat_share*100:.0f}% of violations from repeat offenders — standard fines "
                      f"aren't deterring. Escalate to towing priority / registration flags.",
            "window": "Target arrival times of repeat plates",
        })

    # 5. Heavy Vehicle Restriction — high PCU weight zone
    if avg_pcu >= 1.5 and score >= 35:
        recs.append({
            "action": "Heavy Vehicle Restriction",
            "priority": "MEDIUM",
            "reason": f"Average PCU {avg_pcu:.1f} — large vehicles (buses, trucks, tempos) dominate "
                      f"violations here. Consider time-based heavy-vehicle entry restrictions.",
            "window": "Peak hours (8–11 AM, 5–9 PM)",
        })

    # 6. Signage & Road Marking Audit — high violations but few repeat offenders
    if repeat_share < 0.15 and score >= 35:
        recs.append({
            "action": "Signage & Marking Audit",
            "priority": "MEDIUM",
            "reason": f"Low repeat ratio ({repeat_share*100:.0f}%) suggests many first-time violators — "
                      f"regulations may be poorly marked. Clear signage could reduce violations 20–30%.",
            "window": "Infrastructure rollout within 14 days",
        })

    # 7. Precinct Escalation — critical risk
    if score >= 80:
        recs.append({
            "action": "Precinct Command Escalation",
            "priority": "CRITICAL",
            "reason": f"Critical impact score ({score:.0f}) — zone requires jurisdictional-level "
                      f"coordination, dedicated towing, and coordinated police sweeps.",
            "window": "Immediate dispatch",
        })

    # 8. Routine Patrol — fallback for lower-risk zones
    if not recs or (score < 35 and peak_share < 0.40):
        recs.append({
            "action": "Routine Patrol",
            "priority": "LOW",
            "reason": "Zone within safe limits. Periodic check-ins maintain compliance and gather data.",
            "window": "Weekly daytime rotation",
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
    n_dow = df.groupby("dow", observed=True)["ymd"].nunique()  # # of each weekday in span
    n_dates = df["ymd"].nunique()
    czdh = df.groupby(["gh6", "dow", "hour"], observed=True).size().rename("c").reset_index()
    # zone-hour backoff rate (avg violations per day at that hour, any weekday)
    r_zh = (df.groupby(["gh6", "hour"], observed=True).size() / n_dates).rename("r_zh").reset_index()
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
    return (sub.groupby("gh6", observed=True)["rate"].sum()
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
    n_dow_te = te.groupby("dow", observed=True)["ymd"].nunique()
    act = te.groupby(["gh6", "dow", "hour"], observed=True).size().rename("c_te").reset_index()
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
    c = df.groupby("hour", observed=True).size()
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
    agg = (valid.groupby("police_station", observed=True)
           .agg(avg_delay=("action_delay_mins", "mean"),
                median_delay=("action_delay_mins", "median"),
                violations=("lat", "size"))
           .reset_index()
           .sort_values("avg_delay", ascending=False))
    return agg

# --------------------------------------------------------------------------
# DBSCAN hotspot clustering (from ParkSight AI)
def dbscan_clusters(df: pd.DataFrame, eps_m: float = 300.0,
                    min_samples: int = 30) -> pd.DataFrame:
    """DBSCAN spatial clustering with haversine metric (ParkSight AI approach).
       Returns a DataFrame with one row per cluster: centroid, counts, severity ratios,
       impact score. eps_m is radius in metres."""
    from sklearn.cluster import DBSCAN
    import json as _json

    clean = df.dropna(subset=["lat", "lon"]).copy()
    coords_rad = np.radians(clean[["lat", "lon"]].values)
    eps_rad = eps_m / 6_371_000.0
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm="ball_tree",
                metric="haversine").fit(coords_rad)
    clean["cluster_id"] = db.labels_

    clustered = clean[clean["cluster_id"] >= 0]
    if clustered.empty:
        return pd.DataFrame()

    HIGH_SEV_TYPES = {"PARKING NEAR ROAD CROSSING", "DOUBLE PARKING",
                      "PARKING ON FOOTPATH",
                      "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS"}
    MED_SEV_TYPES = {"PARKING IN A MAIN ROAD", "NO PARKING"}

    results = []
    for cid, grp in clustered.groupby("cluster_id", observed=True):
        count = len(grp)
        centroid_lat = grp["lat"].mean()
        centroid_lon = grp["lon"].mean()

        # Junction name
        if "junction_name" in grp.columns:
            junc_counts = grp[grp["junction_name"] != "No Junction"]["junction_name"].value_counts()
            dominant_junction = junc_counts.index[0] if len(junc_counts) > 0 else "Unknown Zone"
        else:
            dominant_junction = "Unknown Zone"

        # Station
        if "police_station" in grp.columns:
            dominant_station = _first_mode(grp["police_station"], "Unknown") if len(grp) > 0 else "Unknown"
        else:
            dominant_station = "Unknown"

        # Peak hour ratio
        peak_ratio = grp["is_peak_hour"].mean() if "is_peak_hour" in grp.columns else 0.0

        # Severity ratios from primary_type
        if "primary_type" in grp.columns:
            types = grp["primary_type"]
            high_sev_ratio = types.isin(HIGH_SEV_TYPES).mean()
            med_sev_ratio = types.isin(MED_SEV_TYPES).mean()
        else:
            high_sev_ratio = 0.0
            med_sev_ratio = 0.0

        # Vehicle
        dominant_vehicle = _first_mode(grp["vehicle_type"], "UNKNOWN") if "vehicle_type" in grp.columns and len(grp) > 0 else "UNKNOWN"

        # Top violation
        top_violation = _first_mode(grp["primary_type"], "UNKNOWN") if "primary_type" in grp.columns and len(grp) > 0 else "UNKNOWN"

        # Sunday ratio
        sunday_ratio = (grp["dow"] == 6).mean() if "dow" in grp.columns else 0.0

        # Avg severity
        avg_severity = grp["severity"].mean() if "severity" in grp.columns else 0.5

        results.append({
            "cluster_id": int(cid),
            "violation_count": count,
            "centroid_lat": round(centroid_lat, 6),
            "centroid_lon": round(centroid_lon, 6),
            "dominant_junction": dominant_junction,
            "dominant_station": dominant_station,
            "peak_hour_ratio": round(peak_ratio, 4),
            "high_severity_ratio": round(high_sev_ratio, 4),
            "medium_severity_ratio": round(med_sev_ratio, 4),
            "dominant_vehicle": dominant_vehicle,
            "top_violation": top_violation,
            "sunday_ratio": round(sunday_ratio, 4),
            "avg_severity": round(avg_severity, 4),
        })

    cdf = pd.DataFrame(results)

    # Normalised count and impact score (ParkSight AI formula)
    min_c = cdf["violation_count"].min()
    max_c = cdf["violation_count"].max()
    cdf["norm_count"] = (cdf["violation_count"] - min_c) / max(max_c - min_c, 1)

    # Log-normalised impact (v2 from ParkSight)
    log_counts = np.log1p(cdf["violation_count"])
    log_max = log_counts.max() if log_counts.max() > 0 else 1.0
    cdf["norm_count_log"] = log_counts / log_max

    cdf["impact_score"] = (
        cdf["norm_count_log"]         * 0.40 +
        cdf["high_severity_ratio"]    * 0.35 +
        cdf["peak_hour_ratio"]        * 0.25
    ) * 10
    cdf["impact_score"] = cdf["impact_score"].round(2)

    n_clusters = len(cdf)
    n_noise = int((clean["cluster_id"] == -1).sum())

    cdf = cdf.sort_values("impact_score", ascending=False).reset_index(drop=True)
    cdf.attrs["n_clusters"] = n_clusters
    cdf.attrs["n_noise"] = n_noise
    cdf.attrs["total_points"] = len(clean)
    return cdf


def cluster_quality_metrics(df: pd.DataFrame, eps_m: float = 300.0,
                            min_samples: int = 30,
                            sample_size: int = 10_000) -> dict:
    """Compute DBSCAN cluster quality metrics: Silhouette, Davies-Bouldin, Calinski-Harabasz.
       Uses a sample for efficiency (Silhouette is O(n²))."""
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

    clean = df.dropna(subset=["lat", "lon"]).copy()
    if len(clean) > sample_size:
        clean = clean.sample(n=sample_size, random_state=42)

    coords_rad = np.radians(clean[["lat", "lon"]].values)
    eps_rad = eps_m / 6_371_000.0
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, algorithm="ball_tree",
                metric="haversine").fit(coords_rad)
    labels = db.labels_

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    n_clustered = int((labels >= 0).sum())

    result = {
        "sample_size": len(clean),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_clustered": n_clustered,
        "noise_ratio": round(n_noise / len(labels), 4) if len(labels) else 0,
    }

    mask = labels >= 0
    if mask.sum() > 0 and n_clusters >= 2:
        coords_clustered = clean[["lat", "lon"]].values[mask]
        labels_clustered = labels[mask]

        sil = silhouette_score(coords_clustered, labels_clustered,
                               metric="euclidean",
                               sample_size=min(5000, len(coords_clustered)))
        db_score = davies_bouldin_score(coords_clustered, labels_clustered)
        ch_score = calinski_harabasz_score(coords_clustered, labels_clustered)

        # Interpretations
        if sil > 0.5:
            sil_interp = "GOOD"
        elif sil > 0.25:
            sil_interp = "FAIR"
        elif sil > 0:
            sil_interp = "WEAK"
        else:
            sil_interp = "POOR"

        if db_score < 0.5:
            db_interp = "EXCELLENT"
        elif db_score < 1.0:
            db_interp = "GOOD"
        elif db_score < 2.0:
            db_interp = "MODERATE"
        else:
            db_interp = "WEAK"

        result.update({
            "silhouette_score": round(sil, 4),
            "silhouette_interp": sil_interp,
            "davies_bouldin_score": round(db_score, 4),
            "davies_bouldin_interp": db_interp,
            "calinski_harabasz_score": round(ch_score, 2),
        })
    else:
        result.update({
            "silhouette_score": None,
            "silhouette_interp": "N/A",
            "davies_bouldin_score": None,
            "davies_bouldin_interp": "N/A",
            "calinski_harabasz_score": None,
        })

    return result


# --------------------------------------------------------------------------
# Congestion probability model (from ParkSight AI)
def train_congestion_model(df: pd.DataFrame) -> dict:
    """Train a Random Forest classifier for binary congestion risk.
       Returns model metrics (ROC-AUC, F1, accuracy) and feature importance."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

    # Build daily junction-level features
    if "junction_name" not in df.columns:
        return {"error": "junction_name column missing"}

    junc_df = df[df.get("has_junction", df["junction_name"] != "No Junction")].copy()
    daily = junc_df.groupby(["junction_name", "ymd"], observed=True).agg(
        violations=("lat", "size"),
        avg_severity=("severity", "mean"),
        peak_count=("is_peak_hour", "sum") if "is_peak_hour" in junc_df.columns else ("severity", "size"),
        high_sev_count=("severity", lambda x: (x >= 0.8).sum()),
    ).reset_index()

    daily["peak_ratio"] = daily["peak_count"] / daily["violations"].clip(lower=1)
    daily["high_sev_ratio"] = daily["high_sev_count"] / daily["violations"].clip(lower=1)

    # Add temporal features
    daily["date"] = pd.to_datetime(daily["ymd"])
    daily["dow"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)

    # Target: high congestion = violations above 75th percentile
    threshold = daily["violations"].quantile(0.75)
    daily["is_high_congestion"] = (daily["violations"] >= threshold).astype(int)

    features = ["avg_severity", "peak_ratio", "high_sev_ratio", "dow", "month", "is_weekend"]
    X = daily[features].fillna(0)
    y = daily["is_high_congestion"]

    if len(X) < 20 or y.nunique() < 2:
        return {"error": "Insufficient data for congestion model"}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    model = RandomForestClassifier(n_estimators=100, max_depth=5,
                                   random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    roc_auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    feat_imp = {f: round(float(i), 4)
                for f, i in sorted(zip(features, model.feature_importances_),
                                   key=lambda x: x[1], reverse=True)}

    return {
        "roc_auc": round(roc_auc, 3),
        "f1_score": round(f1, 3),
        "accuracy": round(acc, 3),
        "threshold": int(threshold),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "feature_importance": feat_imp,
    }


# --------------------------------------------------------------------------
# Next-Day Random Forest Prediction (from ParkSight AI)

# Top junctions for per-junction modelling
TOP_JUNCTIONS = [
    "BTP051 - Safina Plaza Junction",
    "BTP082 - KR Market Junction",
    "BTP040 - Elite Junction",
    "BTP044 - Sagar Theatre Junction",
    "BTP211 - Central Street Junction",
    "BTP020 - Hosahalli Metro Station",
]
JUNC_SHORT = {
    "BTP051 - Safina Plaza Junction":   "Safina Plaza",
    "BTP082 - KR Market Junction":      "KR Market",
    "BTP040 - Elite Junction":          "Elite Junction",
    "BTP044 - Sagar Theatre Junction":  "Sagar Theatre",
    "BTP211 - Central Street Junction": "Central Street",
    "BTP020 - Hosahalli Metro Station": "Hosahalli Metro",
}
JUNC_COLORS = {
    "BTP051 - Safina Plaza Junction":   "#EF4444",
    "BTP082 - KR Market Junction":      "#F97316",
    "BTP040 - Elite Junction":          "#F59E0B",
    "BTP044 - Sagar Theatre Junction":  "#8B5CF6",
    "BTP211 - Central Street Junction": "#06B6D4",
    "BTP020 - Hosahalli Metro Station": "#10B981",
}

_NEXTDAY_FEATURES = [
    "lag_1d", "lag_2d", "lag_3d", "lag_7d",
    "roll_3d", "roll_7d", "roll_14d", "roll_7d_std",
    "dow_hist_avg", "month_hist_avg", "trend_7d",
    "dow", "month", "is_weekend", "is_monday", "is_sunday",
    "avg_severity", "high_sev_ratio", "peak_hour_ratio", "unique_vehicles",
]


def _build_daily_features(df: pd.DataFrame, junction: str) -> pd.DataFrame:
    """Build daily feature matrix for a single junction (ParkSight AI approach)."""
    sub = df[df["junction_name"] == junction].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["date"] = pd.to_datetime(sub["ymd"])

    daily = sub.groupby("date", observed=True).agg(
        violations=("severity", "count"),
        high_sev_count=("severity", lambda x: (x >= 0.8).sum()),
        peak_hour_count=("is_peak_hour", "sum") if "is_peak_hour" in sub.columns else ("severity", "count"),
        avg_severity=("severity", "mean"),
        unique_vehicles=("vehicle_type", "nunique"),
        unique_hours=("hour", "nunique"),
    ).reset_index()

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["dow"] = daily["date"].dt.dayofweek
    daily["month"] = daily["date"].dt.month
    daily["is_weekend"] = (daily["dow"] >= 5).astype(int)
    daily["is_monday"] = (daily["dow"] == 0).astype(int)
    daily["is_sunday"] = (daily["dow"] == 6).astype(int)
    daily["high_sev_ratio"] = daily["high_sev_count"] / daily["violations"].clip(lower=1)
    daily["peak_hour_ratio"] = daily["peak_hour_count"] / daily["violations"].clip(lower=1)

    # Lag features
    for lag in [1, 2, 3, 7]:
        daily[f"lag_{lag}d"] = daily["violations"].shift(lag)

    # Rolling averages
    daily["roll_3d"] = daily["violations"].shift(1).rolling(3, min_periods=1).mean()
    daily["roll_7d"] = daily["violations"].shift(1).rolling(7, min_periods=1).mean()
    daily["roll_14d"] = daily["violations"].shift(1).rolling(14, min_periods=1).mean()
    daily["roll_7d_std"] = daily["violations"].shift(1).rolling(7, min_periods=2).std().fillna(0)

    # Historical averages
    dow_avg = daily.groupby("dow", observed=True)["violations"].mean()
    daily["dow_hist_avg"] = daily["dow"].map(dow_avg)
    month_avg = daily.groupby("month", observed=True)["violations"].mean()
    daily["month_hist_avg"] = daily["month"].map(month_avg)

    # Trend: slope of last 7 days
    slopes = []
    for i in range(len(daily)):
        if i < 7:
            slopes.append(0.0)
        else:
            y_vals = daily["violations"].iloc[i - 7:i].values
            x_vals = np.arange(7)
            slope = float(np.polyfit(x_vals, y_vals, 1)[0])
            slopes.append(slope)
    daily["trend_7d"] = slopes

    daily = daily.dropna(subset=["lag_1d", "lag_2d", "lag_3d", "lag_7d"]).reset_index(drop=True)
    return daily


def train_nextday_models(df: pd.DataFrame, junctions: list = None) -> dict:
    """Train per-junction next-day prediction models (RF/GBT with TimeSeriesSplit CV).
       Returns dict with models, results, and city-wide model."""
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit, cross_val_score
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    from datetime import timedelta

    if junctions is None:
        # Auto-detect top junctions present in data
        if "junction_name" not in df.columns:
            return {"error": "junction_name column missing"}
        present = set(df["junction_name"].unique())
        junctions = [j for j in TOP_JUNCTIONS if j in present]
        if not junctions:
            # Fallback: top 6 by frequency
            junctions = (df[df["junction_name"] != "No Junction"]["junction_name"]
                         .value_counts().head(6).index.tolist())

    all_models = {}
    all_results = {}
    tscv = TimeSeriesSplit(n_splits=5)

    for junc in junctions:
        short = JUNC_SHORT.get(junc, junc.split(" - ")[-1][:20] if " - " in junc else junc[:20])
        daily = _build_daily_features(df, junc)

        if len(daily) < 20:
            continue

        X = daily[_NEXTDAY_FEATURES]
        y = daily["violations"]

        # Model selection
        models = {
            "RandomForest": RandomForestRegressor(
                n_estimators=200, max_depth=6, min_samples_leaf=3,
                random_state=42, n_jobs=-1),
            "GradientBoosting": GradientBoostingRegressor(
                n_estimators=150, max_depth=4, learning_rate=0.05,
                random_state=42),
            "Ridge": Ridge(alpha=1.0),
        }

        best_name, best_mae, best_model = None, float("inf"), None

        for mname, model in models.items():
            try:
                if mname == "Ridge":
                    scaler = StandardScaler()
                    X_sc = scaler.fit_transform(X)
                    scores = cross_val_score(model, X_sc, y, cv=tscv,
                                             scoring="neg_mean_absolute_error")
                else:
                    scores = cross_val_score(model, X, y, cv=tscv,
                                             scoring="neg_mean_absolute_error")
                mae_cv = -scores.mean()
                if mae_cv < best_mae:
                    best_mae, best_name, best_model = mae_cv, mname, model
            except Exception:
                continue

        if best_model is None:
            continue

        # Train best model on all data
        scaler = None
        if best_name == "Ridge":
            scaler = StandardScaler()
            X_fit = scaler.fit_transform(X)
        else:
            X_fit = X
        best_model.fit(X_fit, y)

        # Evaluate on last 20%
        split = int(len(daily) * 0.8)
        X_test = daily[_NEXTDAY_FEATURES].iloc[split:]
        y_test = daily["violations"].iloc[split:]
        if scaler:
            y_pred = best_model.predict(scaler.transform(X_test))
        else:
            y_pred = best_model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        r2 = r2_score(y_test, y_pred)
        nz = y_test[y_test > 0]
        nz_pred = y_pred[y_test.values > 0]
        mape = float(np.mean(np.abs((nz.values - nz_pred) / nz.values)) * 100) if len(nz) > 0 else 100.0

        feat_imp = {}
        if hasattr(best_model, "feature_importances_"):
            for fname, fimp in sorted(zip(_NEXTDAY_FEATURES, best_model.feature_importances_),
                                       key=lambda x: x[1], reverse=True)[:8]:
                feat_imp[fname] = round(float(fimp), 4)

        all_models[junc] = {"model": best_model, "scaler": scaler,
                            "daily": daily, "features": _NEXTDAY_FEATURES}
        all_results[junc] = {
            "short": short,
            "best_model": best_name,
            "cv_mae": round(float(best_mae), 2),
            "test_mae": round(float(mae), 2),
            "test_rmse": round(rmse, 2),
            "test_r2": round(float(r2), 3),
            "test_mape": round(mape, 1),
            "accuracy": round(max(0, 100 - mape), 1),
            "n_days": len(daily),
            "feat_imp": feat_imp,
            "avg_daily": round(float(y.mean()), 1),
            "max_daily": int(y.max()),
        }

    # --- City-wide model ---
    city_daily = df.groupby("ymd", observed=True).size().reset_index(name="violations")
    city_daily["date"] = pd.to_datetime(city_daily["ymd"])
    city_daily = city_daily.sort_values("date")
    city_daily["dow"] = city_daily["date"].dt.dayofweek
    city_daily["month"] = city_daily["date"].dt.month
    city_daily["is_weekend"] = (city_daily["dow"] >= 5).astype(int)
    city_daily["is_sunday"] = (city_daily["dow"] == 6).astype(int)

    for lag in [1, 2, 3, 7]:
        city_daily[f"lag_{lag}d"] = city_daily["violations"].shift(lag)
    city_daily["roll_7d"] = city_daily["violations"].shift(1).rolling(7, min_periods=1).mean()
    city_daily["roll_14d"] = city_daily["violations"].shift(1).rolling(14, min_periods=1).mean()
    city_daily["roll_std"] = city_daily["violations"].shift(1).rolling(7, min_periods=2).std().fillna(0)
    dow_avg_city = city_daily.groupby("dow", observed=True)["violations"].mean()
    city_daily["dow_hist"] = city_daily["dow"].map(dow_avg_city)
    city_daily = city_daily.dropna(subset=["lag_1d", "lag_2d", "lag_3d", "lag_7d"])

    CITY_FEATS = ["lag_1d", "lag_2d", "lag_3d", "lag_7d",
                  "roll_7d", "roll_14d", "roll_std", "dow_hist",
                  "dow", "month", "is_weekend", "is_sunday"]
    X_city = city_daily[CITY_FEATS]
    y_city = city_daily["violations"]

    city_model = RandomForestRegressor(n_estimators=200, max_depth=5,
                                       min_samples_leaf=3, random_state=42, n_jobs=-1)
    city_model.fit(X_city, y_city)

    split = int(len(city_daily) * 0.8)
    city_pred = city_model.predict(X_city.iloc[split:])
    city_mae = mean_absolute_error(y_city.iloc[split:], city_pred)
    city_r2 = r2_score(y_city.iloc[split:], city_pred)
    city_mape = float(np.mean(np.abs((y_city.iloc[split:].values - city_pred) /
                                      y_city.iloc[split:].clip(lower=1).values)) * 100)

    city_mean = float(y_city.mean())
    city_std = float(y_city.std())

    return {
        "junction_models": all_models,
        "junction_results": all_results,
        "city_model": city_model,
        "city_daily": city_daily,
        "city_features": CITY_FEATS,
        "city_metrics": {
            "mae": round(city_mae, 2),
            "r2": round(city_r2, 3),
            "accuracy": round(max(0, 100 - city_mape), 1),
            "avg_daily": round(city_mean, 1),
            "high_threshold": round(city_mean + 0.75 * city_std, 1),
        },
        "dow_avg_city": dow_avg_city,
    }


def predict_next_7days(trained: dict, df: pd.DataFrame) -> dict:
    """Generate 7-day ahead forecast using trained models.
       Returns dict with per-junction and city-wide forecasts."""
    from datetime import timedelta

    last_date = pd.to_datetime(df["ymd"].max())
    forecast_dates = [last_date + timedelta(days=i + 1) for i in range(7)]
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    all_models = trained["junction_models"]
    all_results = trained["junction_results"]
    weekly_forecast = {}

    for junc in all_models:
        short = JUNC_SHORT.get(junc, junc.split(" - ")[-1][:20] if " - " in junc else junc[:20])
        model = all_models[junc]["model"]
        scaler = all_models[junc]["scaler"]
        daily_df = all_models[junc]["daily"].copy()
        res = all_results[junc]

        junc_mean = daily_df["violations"].mean()
        junc_std = daily_df["violations"].std()
        high_thresh = junc_mean + 0.75 * junc_std
        med_thresh = junc_mean - 0.25 * junc_std

        day_forecasts = []
        recent = list(daily_df["violations"].tail(14))
        recent_peak = float(daily_df["peak_hour_ratio"].tail(7).mean())
        recent_sev = float(daily_df["high_sev_ratio"].tail(7).mean())
        recent_avg_sev = float(daily_df["avg_severity"].tail(7).mean())
        recent_vehicles = float(daily_df["unique_vehicles"].tail(7).mean())

        for fdate in forecast_dates:
            dow = fdate.dayofweek
            month = fdate.month
            is_weekend = int(dow >= 5)
            is_monday = int(dow == 0)
            is_sunday = int(dow == 6)

            lag_1d = recent[-1]
            lag_2d = recent[-2]
            lag_3d = recent[-3]
            lag_7d = recent[-7] if len(recent) >= 7 else lag_1d
            roll_3d = float(np.mean(recent[-3:]))
            roll_7d = float(np.mean(recent[-7:]))
            roll_14d = float(np.mean(recent[-14:])) if len(recent) >= 14 else roll_7d
            roll_std = float(np.std(recent[-7:]))

            dow_hist = float(daily_df[daily_df["dow"] == dow]["violations"].mean()
                             if len(daily_df[daily_df["dow"] == dow]) > 0 else junc_mean)
            mon_hist = float(daily_df[daily_df["month"] == month]["violations"].mean()
                             if len(daily_df[daily_df["month"] == month]) > 0 else junc_mean)

            last7 = recent[-7:]
            trend = float(np.polyfit(range(len(last7)), last7, 1)[0]) if len(last7) >= 2 else 0.0

            feat_vec = pd.DataFrame([[lag_1d, lag_2d, lag_3d, lag_7d,
                                  roll_3d, roll_7d, roll_14d, roll_std,
                                  dow_hist, mon_hist, trend,
                                  dow, month, is_weekend, is_monday, is_sunday,
                                  recent_avg_sev, recent_sev, recent_peak, recent_vehicles]],
                                  columns=_NEXTDAY_FEATURES)

            if scaler:
                feat_vec = scaler.transform(feat_vec)
            predicted = max(0, round(float(model.predict(feat_vec)[0])))

            ci_half = round(res["test_rmse"] * 1.25)
            ci_low = max(0, predicted - ci_half)
            ci_high = predicted + ci_half

            if predicted >= high_thresh:
                risk = "HIGH"
            elif predicted >= med_thresh:
                risk = "MEDIUM"
            else:
                risk = "LOW"

            if risk == "HIGH":
                rec = f"Pre-position officer before {6 if is_sunday else 7}AM"
            elif risk == "MEDIUM":
                rec = "Standard patrol — monitor 2–6AM window"
            else:
                rec = "Normal operations"

            day_forecasts.append({
                "date": fdate.strftime("%Y-%m-%d"),
                "dayName": day_names[dow],
                "predicted": predicted,
                "ciLow": ci_low,
                "ciHigh": ci_high,
                "risk": risk,
                "recommendation": rec,
                "isWeekend": bool(is_weekend),
            })
            recent.append(predicted)

        weekly_forecast[junc] = {
            "shortName": short,
            "color": JUNC_COLORS.get(junc, "#4C8BF5"),
            "avgDaily": res["avg_daily"],
            "highThreshold": round(high_thresh, 1),
            "mediumThreshold": round(med_thresh, 1),
            "modelUsed": res["best_model"],
            "testMAE": res["test_mae"],
            "testRMSE": res["test_rmse"],
            "testR2": res["test_r2"],
            "accuracy": res["accuracy"],
            "featureImportance": res["feat_imp"],
            "days": day_forecasts,
        }

    # --- City-wide forecast ---
    city_model = trained["city_model"]
    city_daily = trained["city_daily"]
    city_metrics = trained["city_metrics"]
    dow_avg_city = trained["dow_avg_city"]
    city_mean = city_metrics["avg_daily"]
    city_high_thresh = city_metrics["high_threshold"]

    recent_city = list(city_daily["violations"].tail(14))
    city_forecasts = []
    for fdate in forecast_dates:
        dow = fdate.dayofweek
        month = fdate.month
        lag1 = recent_city[-1]
        lag2 = recent_city[-2]
        lag3 = recent_city[-3]
        lag7 = recent_city[-7] if len(recent_city) >= 7 else lag1
        r7 = float(np.mean(recent_city[-7:]))
        r14 = float(np.mean(recent_city[-14:])) if len(recent_city) >= 14 else r7
        rstd = float(np.std(recent_city[-7:]))
        dh = float(dow_avg_city.get(dow, city_mean))

        feat = pd.DataFrame([[lag1, lag2, lag3, lag7, r7, r14, rstd, dh,
                          dow, month, int(dow >= 5), int(dow == 6)]],
                          columns=trained["city_features"])
        pred = max(0, round(float(city_model.predict(feat)[0])))
        risk = ("HIGH" if pred >= city_high_thresh else
                "MEDIUM" if pred >= city_mean else "LOW")

        day_names_full = ["Monday", "Tuesday", "Wednesday", "Thursday",
                          "Friday", "Saturday", "Sunday"]
        city_forecasts.append({
            "date": fdate.strftime("%Y-%m-%d"),
            "dayName": day_names_full[dow],
            "predicted": pred,
            "risk": risk,
        })
        recent_city.append(pred)

    return {
        "data_date_range": f"{df['ymd'].min()} to {df['ymd'].max()}",
        "forecast_from": forecast_dates[0].strftime("%Y-%m-%d"),
        "forecast_to": forecast_dates[-1].strftime("%Y-%m-%d"),
        "junctions": weekly_forecast,
        "junction_results": all_results,
        "city_wide": {**city_metrics, "forecast": city_forecasts},
    }


# --------------------------------------------------------------------------
# Traffic Propagation Analysis (from GRIDLOCK2.0 Prototype script 09)
def traffic_propagation(zones: pd.DataFrame, radius_km: float = 2.0) -> pd.DataFrame:
    """Haversine-based pairwise hotspot proximity analysis.
       Identifies which hotspots affect each other within `radius_km`.
       Returns DataFrame with source→affected links, distance, and propagation risk."""
    results = []
    for i, src in zones.iterrows():
        for j, tgt in zones.iterrows():
            if i == j:
                continue
            dist_m = _haversine(src["lat"], src["lon"], tgt["lat"], tgt["lon"])
            dist_km = dist_m / 1000.0
            if dist_km <= radius_km:
                if dist_km <= 0.5:
                    risk = "Very High"
                elif dist_km <= 1.0:
                    risk = "High"
                elif dist_km <= 1.5:
                    risk = "Medium"
                else:
                    risk = "Low"
                results.append({
                    "source_zone": src.get("label", f"Zone {i}"),
                    "source_lat": src["lat"],
                    "source_lon": src["lon"],
                    "source_impact": src.get("impact_score", 0),
                    "affected_zone": tgt.get("label", f"Zone {j}"),
                    "affected_lat": tgt["lat"],
                    "affected_lon": tgt["lon"],
                    "affected_impact": tgt.get("impact_score", 0),
                    "distance_km": round(dist_km, 3),
                    "propagation_risk": risk,
                })
    if not results:
        return pd.DataFrame()
    pdf = pd.DataFrame(results).sort_values("distance_km")
    return pdf


# --------------------------------------------------------------------------
# Officer Allocation Engine (from GRIDLOCK2.0 Prototype script 10)
def officer_allocation(zones: pd.DataFrame, total_officers: int = 100,
                       top_n: int = 20) -> pd.DataFrame:
    """Priority-proportional allocation of `total_officers` across top N zones.
       Uses impact_score as the priority weight."""
    top = zones.head(top_n).copy()
    total_priority = top["impact_score"].sum()
    if total_priority == 0:
        top["officer_share"] = 0.0
        top["allocated_officers"] = 0
    else:
        top["officer_share"] = top["impact_score"] / total_priority
        top["allocated_officers"] = (top["officer_share"] * total_officers).round().astype(int)
    # Ensure at least 1 officer per zone if possible
    top.loc[top["allocated_officers"] == 0, "allocated_officers"] = 1
    # Adjust to match total
    diff = total_officers - top["allocated_officers"].sum()
    if diff != 0:
        idx = top["impact_score"].idxmax()
        top.loc[idx, "allocated_officers"] += diff
    top["priority_rank"] = range(1, len(top) + 1)
    cols = ["priority_rank", "label", "violations", "impact_score",
            "officer_share", "allocated_officers"]
    available = [c for c in cols if c in top.columns]
    return top[available].sort_values("allocated_officers", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Patrol Route Optimizer (from GRIDLOCK2.0 Prototype script 11)
def optimal_patrol_route(zones: pd.DataFrame, top_n: int = 10) -> dict:
    """Nearest-neighbor greedy TSP route through top `top_n` hotspots.
       Returns dict with ordered route DataFrame and total distance."""
    hotspots = zones.head(top_n).reset_index(drop=True)
    if len(hotspots) < 2:
        return {"route": hotspots, "total_distance_km": 0.0, "stops": len(hotspots)}

    unvisited = list(range(len(hotspots)))
    route_order = [unvisited.pop(0)]
    total_dist = 0.0

    while unvisited:
        current = route_order[-1]
        nearest = min(
            unvisited,
            key=lambda x: _haversine(
                hotspots.iloc[current]["lat"], hotspots.iloc[current]["lon"],
                hotspots.iloc[x]["lat"], hotspots.iloc[x]["lon"]
            )
        )
        d = _haversine(
            hotspots.iloc[current]["lat"], hotspots.iloc[current]["lon"],
            hotspots.iloc[nearest]["lat"], hotspots.iloc[nearest]["lon"]
        )
        total_dist += d / 1000.0
        route_order.append(nearest)
        unvisited.remove(nearest)

    route_df = hotspots.iloc[route_order].reset_index(drop=True)
    route_df.insert(0, "stop", [f"Stop {i+1}" for i in range(len(route_df))])

    # Compute leg distances
    leg_dists = [0.0]
    for i in range(1, len(route_df)):
        d = _haversine(
            route_df.iloc[i-1]["lat"], route_df.iloc[i-1]["lon"],
            route_df.iloc[i]["lat"], route_df.iloc[i]["lon"]
        ) / 1000.0
        leg_dists.append(round(d, 2))
    route_df["leg_distance_km"] = leg_dists

    return {
        "route": route_df,
        "total_distance_km": round(total_dist, 2),
        "stops": len(route_df),
    }


# --------------------------------------------------------------------------
# Parking DNA Profiles (from GRIDLOCK2.0 Prototype script 12)
def parking_dna_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Per-station behavioural DNA: dominant vehicle, dominant violation,
       peak hour, weekend ratio. Min 50 violations per station."""
    profiles = []
    stations = df["police_station"].dropna().unique()
    stations = [s for s in stations if str(s).lower() != "nan"]

    for station in stations:
        temp = df[df["police_station"] == station]
        if len(temp) < 50:
            continue

        dominant_vehicle = (_first_mode(temp["vehicle_type"], "Unknown")
                           if "vehicle_type" in temp.columns
                           else "Unknown")
        dominant_violation = (_first_mode(temp["primary_type"], "Unknown")
                             if "primary_type" in temp.columns
                             else "Unknown")
        peak_hour = (int(_first_mode(temp["hour"], -1))
                     if "hour" in temp.columns
                     else -1)
        weekend_ratio = round(
            temp["dow"].isin([5, 6]).mean() * 100, 1
        ) if "dow" in temp.columns else 0.0

        avg_severity = round(temp["severity"].mean(), 3) if "severity" in temp.columns else 0.5
        junction_frac = round(temp["has_junction"].mean() * 100, 1) if "has_junction" in temp.columns else 0.0
        peak_share = round(
            temp["is_peak_hour"].mean() * 100, 1
        ) if "is_peak_hour" in temp.columns else 0.0

        profiles.append({
            "police_station": station,
            "dominant_vehicle": dominant_vehicle,
            "dominant_violation": dominant_violation,
            "peak_hour": peak_hour,
            "weekend_ratio": weekend_ratio,
            "total_violations": len(temp),
            "avg_severity": avg_severity,
            "junction_frac": junction_frac,
            "peak_hour_share": peak_share,
        })

    dna = pd.DataFrame(profiles)
    dna = dna.sort_values("total_violations", ascending=False).reset_index(drop=True)
    return dna


# --------------------------------------------------------------------------
# Emerging Hotspot Analysis (from GRIDLOCK2.0 Prototype script 06)
def emerging_hotspot_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Early-vs-late period growth analysis per station.
       Auto-detects the midpoint of the date range for comparison."""
    dates = pd.to_datetime(df["ymd"])
    midpoint = dates.min() + (dates.max() - dates.min()) / 2

    df_temp = df.copy()
    df_temp["_date"] = pd.to_datetime(df_temp["ymd"])
    early = df_temp[df_temp["_date"] <= midpoint]
    late = df_temp[df_temp["_date"] > midpoint]

    stations = [s for s in df["police_station"].dropna().unique()
                if str(s).lower() != "nan"]

    early_counts = (early.groupby("police_station", observed=True).size()
                    .reset_index(name="early_count"))
    late_counts = (late.groupby("police_station", observed=True).size()
                   .reset_index(name="late_count"))

    growth = early_counts.merge(late_counts, on="police_station", how="outer")
    growth[["early_count", "late_count"]] = growth[["early_count", "late_count"]].fillna(0)
    growth["early_count"] = growth["early_count"].astype(int)
    growth["late_count"] = growth["late_count"].astype(int)
    growth["change"] = growth["late_count"] - growth["early_count"]
    growth["growth_percent"] = (
        (growth["late_count"] - growth["early_count"])
        / (growth["early_count"] + 1)
    ) * 100
    growth["growth_percent"] = growth["growth_percent"].round(1)
    growth["total"] = growth["early_count"] + growth["late_count"]

    # Classify
    def _classify(row):
        if row["growth_percent"] >= 50:
            return "Rapidly Emerging"
        elif row["growth_percent"] >= 20:
            return "Emerging"
        elif row["growth_percent"] >= -10:
            return "Stable"
        elif row["growth_percent"] >= -30:
            return "Declining"
        else:
            return "Rapidly Declining"

    growth["trend"] = growth.apply(_classify, axis=1)
    growth = growth.sort_values("growth_percent", ascending=False).reset_index(drop=True)

    # Store metadata
    growth.attrs["early_period"] = f"{early['_date'].min().strftime('%Y-%m-%d')} to {midpoint.strftime('%Y-%m-%d')}"
    growth.attrs["late_period"] = f"{midpoint.strftime('%Y-%m-%d')} to {late['_date'].max().strftime('%Y-%m-%d')}"

    return growth


# --------------------------------------------------------------------------
# What-If Simulation (from GRIDLOCK2.0 Prototype script 13)
def what_if_simulation(zones: pd.DataFrame, target_idx: int = 0,
                       additional_officers: int = 5) -> dict:
    """Estimate risk/congestion/propagation reduction from adding officers.
       Uses the prototype's linear estimation formula."""
    if target_idx >= len(zones):
        return {"error": "Zone index out of range"}

    zone = zones.iloc[target_idx]
    current_impact = float(zone.get("impact_score", 0))

    # Risk reduction: each officer reduces impact by ~2 points (from prototype)
    risk_reduction = additional_officers * 2
    new_impact = max(0, current_impact - risk_reduction)
    impact_change = current_impact - new_impact

    # Congestion and propagation reduction estimates (prototype formula)
    congestion_reduction = round(impact_change * 0.8, 2)
    propagation_reduction = round(impact_change * 0.6, 2)
    violation_reduction_pct = round((impact_change / max(current_impact, 1)) * 100, 1)

    return {
        "zone_label": zone.get("label", f"Zone {target_idx}"),
        "zone_idx": target_idx,
        "current_impact": round(current_impact, 1),
        "additional_officers": additional_officers,
        "new_impact": round(new_impact, 1),
        "impact_reduction": round(impact_change, 1),
        "impact_reduction_pct": violation_reduction_pct,
        "congestion_reduction": congestion_reduction,
        "propagation_reduction": propagation_reduction,
        "violations": int(zone.get("violations", 0)),
        "avg_severity": round(float(zone.get("avg_severity", 0)), 2),
        "junction_frac": round(float(zone.get("junction_frac", 0)) * 100, 0),
    }


def what_if_batch(zones: pd.DataFrame, target_indices: list,
                  additional_officers: int = 5) -> list:
    """Run what-if simulation on multiple zones for comparison."""
    return [what_if_simulation(zones, idx, additional_officers)
            for idx in target_indices if idx < len(zones)]


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
