"""
Data foundation for Parking Enforcement Intelligence (Theme 1).
Reads the raw BTP parking-violation CSV, cleans it, derives flow-severity,
encodes geohash zones, resolves timestamps to IST, and writes:
  data/clean.pkl      - one cleaned row per violation (the analytics core)
  data/junctions.pkl  - per-junction centroid + counts (for map / patrol module)
Run once:  python app/prep.py
"""
import pandas as pd, numpy as np, json, ast, time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # round2/
# --- locate the raw CSV (try several common locations) ---
_candidates = [
    ROOT / "dataset theme 1" / "jan to may police violation_anonymized791b166.csv",
    ROOT / "jan to may police violation_anonymized791b166.csv",
    ROOT.parent / "jan to may police violation_anonymized791b166.csv",
]
SRC = next((p for p in _candidates if p.exists()), _candidates[0])
OUT  = ROOT / "data"
OUT.mkdir(exist_ok=True)

t0 = time.time()
df = pd.read_csv(SRC, low_memory=False)
print(f"loaded {df.shape} in {time.time()-t0:.1f}s")

# ---------- 1. parse violation_type (JSON-list string) ----------
def parse_vt(s):
    if isinstance(s, list): return s
    if not isinstance(s, str): return []
    try: return json.loads(s)
    except Exception:
        try: return ast.literal_eval(s)
        except Exception: return []
df["vtypes"] = df["violation_type"].map(parse_vt)

cnt = Counter(t for L in df["vtypes"] for t in L)
print(f"\n# distinct violation types: {len(cnt)}")
for t, c in cnt.most_common():
    print(f"  {c:7d}  {t}")

# flow-impact severity: how much a violation type chokes the carriageway (0..1)
SEV = {
    "PARKING IN A MAIN ROAD":      1.00,
    "OBSTRUCTING TRAFFIC":         1.00,
    "PARKING NEAR ROAD CROSSING":  0.90,
    "PARKING AT BUS STOP":         0.90,
    "PARKING NEAR BUS STOP":       0.90,
    "PARKING NEAR TRAFFIC LIGHT OR ZEBRA CROSS": 0.90,
    "DOUBLE PARKING":              0.85,
    "PARKING OPPOSITE TO ANOTHER PARKED VEHICLE": 0.80,
    "WRONG PARKING":               0.80,
    "PARKING NEAR BUSTOP/SCHOOL/HOSPITAL ETC": 0.75,
    "NO PARKING":                  0.60,
    "PARKING ON FOOTPATH":         0.50,
    "DEFECTIVE NUMBER PLATE":      0.10,
}
DEFAULT_SEV = 0.55
def severity(L):  return max((SEV.get(t, DEFAULT_SEV) for t in L), default=DEFAULT_SEV)
def primary(L):   return max(L, key=lambda t: SEV.get(t, DEFAULT_SEV)) if L else "UNKNOWN"
df["severity"]     = df["vtypes"].map(severity)
df["primary_type"] = df["vtypes"].map(primary)
df["n_viol"]       = df["vtypes"].map(len).clip(lower=1)

# ---------- 1b. arterial / junction flag from violation types ----------
ARTERIAL_TERMS = ("MAIN ROAD", "ROAD CROSSING", "TRAFFIC LIGHT", "ZEBRA",
                  "DOUBLE PARKING", "OPPOSITE")
def is_arterial(L):
    upper = " ".join(L).upper()
    return any(t in upper for t in ARTERIAL_TERMS)
df["is_arterial"] = df["vtypes"].map(is_arterial)

# ---------- 2. PCU vehicle obstruction weights (from ParkSight AI) ----------
PCU = {
    "MOPED":                0.30,
    "SCOOTER":              0.35,
    "MOTOR CYCLE":          0.35,
    "PASSENGER AUTO":       0.75,
    "GOODS AUTO":           0.85,
    "CAR":                  1.00,
    "JEEP":                 1.05,
    "VAN":                  1.15,
    "MAXI-CAB":             1.20,
    "LGV":                  1.50,
    "TEMPO":                1.70,
    "PRIVATE BUS":          2.60,
    "BUS (BMTC/KSRTC)":    2.80,
    "HGV":                  3.00,
    "LORRY/GOODS VEHICLE":  3.00,
    "TANKER":               3.20,
}
DEFAULT_PCU = 1.0
vtype_raw = df["vehicle_type"].astype(str).str.strip().str.upper()
df["pcu"] = vtype_raw.map(PCU).fillna(DEFAULT_PCU)

# Obstruction weight: PCU × severity × junction boost × peak boost (computed after timestamps)

# ---------- 3. timestamps -> IST ----------
df["ts_utc"] = pd.to_datetime(df["created_datetime"], errors="coerce", utc=True)
df = df[df["ts_utc"].notna()].copy()
loc = df["ts_utc"].dt.tz_convert("Asia/Kolkata")
df["ts_ist"]     = loc
df["hour"]       = loc.dt.hour
df["dow"]        = loc.dt.dayofweek            # 0=Mon
df["dow_name"]   = loc.dt.day_name()
df["is_weekend"] = loc.dt.dayofweek >= 5
df["month"]      = loc.dt.month
df["ymd"]        = loc.dt.strftime("%Y-%m-%d")

# ---------- 3b. peak hour flag ----------
PEAK_HOURS = {8, 9, 10, 11, 17, 18, 19, 20, 21}
df["is_peak_hour"] = df["hour"].isin(PEAK_HOURS)

# ---------- 3c. enforcement delay (action time - creation time) ----------
for ts_col in ["action_taken_timestamp", "closed_datetime"]:
    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)

if "action_taken_timestamp" in df.columns:
    delay = (df["action_taken_timestamp"] - df["ts_utc"]).dt.total_seconds() / 60.0
    df["action_delay_mins"] = delay.clip(lower=0)  # negative = data error
else:
    df["action_delay_mins"] = np.nan

# ---------- 3d. compute obstruction weight ----------
junction_boost = np.where(
    (df.get("junction_name", "").astype(str).str.lower() != "nan") &
    (df.get("junction_name", "").astype(str) != "No Junction"),
    1.35, 1.0
)
peak_boost = np.where(df["is_peak_hour"], 1.18, 1.0)
df["obstruction_weight"] = df["pcu"] * df["severity"] * junction_boost * peak_boost

# ---------- 4. place-type tagging (from ParkSight AI) ----------
PLACE_TERMS = {
    "metro":    "Metro spillover",
    "market":   "Commercial market",
    "mall":     "Commercial mall",
    "hospital": "Hospital frontage",
    "school":   "School frontage",
    "college":  "College frontage",
    "bus":      "Bus-stop conflict",
    "station":  "Transit station",
    "theatre":  "Event venue",
    "temple":   "Religious site",
    "mosque":   "Religious site",
    "church":   "Religious site",
}

def detect_place_type(row):
    text = (str(row.get("location", "")) + " " + str(row.get("junction_name", ""))).lower()
    for term, label in PLACE_TERMS.items():
        if term in text:
            return label
    return "Street segment"

df["place_type"] = df.apply(detect_place_type, axis=1)

# ---------- 5. geo clean ----------
df["lat"] = pd.to_numeric(df["latitude"],  errors="coerce")
df["lon"] = pd.to_numeric(df["longitude"], errors="coerce")
box = df["lat"].between(12.7, 13.35) & df["lon"].between(77.3, 77.9)
print(f"\ndropped out-of-box rows: {(~box).sum()}")
df = df[box].copy()

# ---------- 6. geohash encode (precision 7 ~153m, 6 ~1.2km) ----------
_B32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_BITS = [16, 8, 4, 2, 1]
def gh_encode(lat, lon, prec=7):
    la, lo = [-90.0, 90.0], [-180.0, 180.0]
    out, bit, ch, even = [], 0, 0, True
    while len(out) < prec:
        if even:
            mid = (lo[0] + lo[1]) / 2
            if lon >= mid: ch |= _BITS[bit]; lo[0] = mid
            else:          lo[1] = mid
        else:
            mid = (la[0] + la[1]) / 2
            if lat >= mid: ch |= _BITS[bit]; la[0] = mid
            else:          la[1] = mid
        even = not even
        if bit < 4: bit += 1
        else:       out.append(_B32[ch]); bit, ch = 0, 0
    return "".join(out)

df["lat_r"], df["lon_r"] = df["lat"].round(6), df["lon"].round(6)
uc = df[["lat_r", "lon_r"]].drop_duplicates().reset_index(drop=True)
print(f"unique coords to encode: {len(uc)}")
uc["gh7"] = [gh_encode(a, b, 7) for a, b in zip(uc["lat_r"], uc["lon_r"])]
uc["gh6"] = uc["gh7"].str[:6]
df = df.merge(uc, on=["lat_r", "lon_r"], how="left")

# ---------- 7. tidy categoricals ----------
for c in ["police_station", "junction_name", "vehicle_type", "vehicle_number"]:
    df[c] = df[c].astype(str).str.strip()
df["has_junction"] = (df["junction_name"] != "No Junction") & (df["junction_name"].str.lower() != "nan")

# keep only what the app/core actually consume (slim footprint for cloud deploy)
keep = ["lat", "lon", "gh6", "gh7", "primary_type", "severity",
        "hour", "dow", "ymd", "police_station", "junction_name",
        "has_junction", "vehicle_type", "vehicle_number",
        # --- NEW columns ---
        "pcu", "obstruction_weight", "is_peak_hour", "is_arterial",
        "action_delay_mins", "place_type"]
clean = df[keep].reset_index(drop=True)
clean.to_pickle(OUT / "clean.pkl")

# ---------- 8. junction lookup (centroid + load) ----------
jx = (clean[clean["has_junction"]]
      .groupby("junction_name")
      .agg(lat=("lat", "median"), lon=("lon", "median"),
           n=("lat", "size"), severity=("severity", "mean"))
      .reset_index().sort_values("n", ascending=False))
jx.to_pickle(OUT / "junctions.pkl")

# ---------- summary ----------
print(f"\nsaved data/clean.pkl {clean.shape}  |  data/junctions.pkl {jx.shape}")
print(f"total time {time.time()-t0:.1f}s")
print("\nhour-of-day (IST) distribution:")
print(clean["hour"].value_counts().sort_index().to_string())
print("\nseverity stats:"); print(clean["severity"].describe().round(3).to_string())
print(f"\ngh7 zones: {clean['gh7'].nunique()}  |  gh6 zones: {clean['gh6'].nunique()}  |  junctions: {len(jx)}")
print("primary_type mix:"); print(clean["primary_type"].value_counts().head(10).to_string())
print(f"\nPCU stats:"); print(clean["pcu"].describe().round(3).to_string())
print(f"\nplace_type distribution:"); print(clean["place_type"].value_counts().to_string())
print(f"\nis_peak_hour: {clean['is_peak_hour'].mean()*100:.1f}%")
print(f"is_arterial: {clean['is_arterial'].mean()*100:.1f}%")
print(f"action_delay_mins (median): {clean['action_delay_mins'].median():.1f}")
