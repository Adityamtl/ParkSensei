// ParkPulse client-side intelligence — types, data loading, and the patrol allocator
// ported from core.py so the app runs fully client-side (no backend).

export type Zone = {
  gh6: string; lat: number; lon: number; label: string; violations: number;
  impact_score: number; avg_severity: number; junction_frac: number;
  main_road_frac: number; top_violation: string;
};
export type GridCell = { lat: number; lon: number; n: number };
export type RateCell = { gh6: string; dow: number; hour: number; rate: number };
export type Meta = {
  totals: { violations: number; zones: number; days: number; junctions: number };
  coverage: { hour: number; share: number }[];
  coverage_summary: { before_1pm: number; evening_5_9: number };
  repeat: { unique: number; repeat_vehicles: number; share_pct: number; worst: number };
  backtest: { pearson_r: number; mae: number; cells: number; cutoff: string };
  top_hotspots: { label: string; violations: number; impact_score: number }[];
  dow_names: string[];
};

export async function loadJSON<T>(name: string): Promise<T> {
  const r = await fetch(`/data/${name}.json`);
  if (!r.ok) throw new Error(`failed to load ${name}.json`);
  return r.json();
}

export function haversine(lat1: number, lon1: number, lat2: number, lon2: number): number {
  const R = 6371000, rad = Math.PI / 180;
  const dp = (lat2 - lat1) * rad, dl = (lon2 - lon1) * rad;
  const a = Math.sin(dp / 2) ** 2 +
            Math.cos(lat1 * rad) * Math.cos(lat2 * rad) * Math.sin(dl / 2) ** 2;
  return 2 * R * Math.asin(Math.sqrt(a));
}

/** Expected enforcement load per zone for a weekday + hour window (sums shrunk rates). */
export function predictLoad(rates: RateCell[], dow: number, hours: number[]): Map<string, number> {
  const hs = new Set(hours), m = new Map<string, number>();
  for (const r of rates) {
    if (r.dow === dow && hs.has(r.hour)) m.set(r.gh6, (m.get(r.gh6) ?? 0) + r.rate);
  }
  return m;
}

export type Deployment = Zone & { team: string; pred: number };

/** Greedy max-load allocation with a minimum-spacing constraint (mirrors core.allocate_patrols). */
export function allocatePatrols(
  zones: Zone[], load: Map<string, number>, k: number, minSepM = 600,
): Deployment[] {
  const cand = zones
    .map((z) => ({ ...z, pred: load.get(z.gh6) ?? 0 }))
    .filter((z) => z.pred > 0)
    .sort((a, b) => b.pred - a.pred);
  const chosen: (Zone & { pred: number })[] = [];
  for (const z of cand) {
    if (chosen.length >= k) break;
    if (chosen.some((c) => haversine(z.lat, z.lon, c.lat, c.lon) < minSepM)) continue;
    chosen.push(z);
  }
  return chosen.map((z, i) => ({ ...z, team: `Team ${i + 1}` }));
}

export const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

/** blue → amber → red ramp for an impact score (0–100). */
export function impactColor(score: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [
    [0, [46, 134, 222]], [45, [245, 205, 90]], [70, [245, 158, 65]], [100, [226, 53, 43]],
  ];
  const s = Math.max(0, Math.min(100, score));
  for (let i = 0; i < stops.length - 1; i++) {
    const [a, ca] = stops[i], [b, cb] = stops[i + 1];
    if (s <= b) {
      const t = b === a ? 0 : (s - a) / (b - a);
      return [0, 1, 2].map((j) => Math.round(ca[j] + (cb[j] - ca[j]) * t)) as [number, number, number];
    }
  }
  return stops[stops.length - 1][1];
}
