"use client";
import { useEffect, useMemo, useState } from "react";
import dynamic from "next/dynamic";
import { loadJSON, type Meta, type GridCell } from "@/lib/parkpulse";

const HotspotMap = dynamic(() => import("@/components/HotspotMap"), {
  ssr: false,
  loading: () => (
    <div className="absolute inset-0 grid place-items-center text-slate-500">Loading map…</div>
  ),
});

export default function Home() {
  const [meta, setMeta] = useState<Meta | null>(null);
  const [grid, setGrid] = useState<GridCell[] | null>(null);

  useEffect(() => {
    loadJSON<Meta>("meta").then(setMeta).catch(() => {});
    loadJSON<GridCell[]>("grid").then(setGrid).catch(() => {});
  }, []);

  const fmt = (n: number) => n.toLocaleString("en-IN");
  const covMax = useMemo(
    () => (meta ? Math.max(...meta.coverage.map((c) => c.share)) : 1),
    [meta],
  );

  const kpis = meta
    ? [
        { label: "Violations logged", value: fmt(meta.totals.violations) },
        { label: "Hotspot zones", value: fmt(meta.totals.zones) },
        { label: "Named junctions", value: fmt(meta.totals.junctions) },
        { label: "Repeat-offender load", value: `${meta.repeat.share_pct}%` },
      ]
    : [];

  return (
    <main className="min-h-screen">
      {/* ---------------- HERO: map background + overlays ---------------- */}
      <section className="relative h-[80vh] w-full overflow-hidden">
        {grid && <HotspotMap grid={grid} />}
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-b from-[#0b0f17]/85 via-transparent to-[#0b0f17]" />

        <div className="pointer-events-none absolute inset-x-0 top-0 flex items-center gap-3 p-7">
          <span className="text-3xl">🚦</span>
          <div>
            <h1 className="text-2xl font-bold tracking-tight">ParkPulse</h1>
            <p className="text-sm text-[var(--muted)]">Parking Enforcement Intelligence · Bengaluru</p>
          </div>
        </div>

        <div className="pointer-events-none absolute bottom-32 left-7 max-w-2xl">
          <h2 className="text-4xl font-bold leading-[1.15]">
            Turning <span className="text-[var(--accent)]">298,000</span> parking tickets into{" "}
            <span className="text-[var(--accent)]">where to stand tomorrow.</span>
          </h2>
          <p className="mt-3 text-[var(--muted)]">
            Detect → Score → Forecast → Deploy → Target. Drag the map to explore the city&apos;s
            violation density.
          </p>
        </div>

        <div className="absolute inset-x-0 bottom-0 p-6">
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            {kpis.map((k) => (
              <div key={k.label} className="glass rounded-2xl px-5 py-4">
                <div className="text-2xl font-bold">{k.value}</div>
                <div className="text-xs text-[var(--muted)]">{k.label}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* ---------------- INSIGHTS ---------------- */}
      <section className="mx-auto max-w-6xl px-6 py-10">
        <div className="grid gap-6 md:grid-cols-2">
          <div className="card p-6">
            <h3 className="text-lg font-bold">When does enforcement happen?</h3>
            <p className="mt-1 text-sm text-[var(--muted)]">
              {meta
                ? `${meta.coverage_summary.before_1pm}% before 1 PM · ${meta.coverage_summary.evening_5_9}% in the 5–9 PM evening peak`
                : "—"}{" "}
              — the data is enforcement time, not demand. Evenings are a blind spot.
            </p>
            <div className="mt-5 flex h-40 items-end gap-1">
              {meta?.coverage.map((c) => {
                const h = (c.share / covMax) * 100;
                const col = c.hour >= 17 && c.hour <= 21 ? "#E2352B" : c.hour < 13 ? "var(--accent)" : "#3a4254";
                return (
                  <div
                    key={c.hour}
                    title={`${c.hour}:00 — ${(c.share * 100).toFixed(1)}%`}
                    style={{ height: `${h}%`, background: col }}
                    className="flex-1 rounded-t"
                  />
                );
              })}
            </div>
            <div className="mt-1 flex justify-between text-[10px] text-[var(--muted)]">
              <span>12 AM</span><span>12 PM</span><span>11 PM</span>
            </div>
          </div>

          <div className="card p-6">
            <h3 className="text-lg font-bold">Top impact zones</h3>
            <p className="mt-1 text-sm text-[var(--muted)]">Ranked by Congestion Impact Score</p>
            <ul className="mt-4 space-y-2.5">
              {meta?.top_hotspots.slice(0, 8).map((h, i) => (
                <li key={i} className="flex items-center gap-3">
                  <span className="w-5 text-sm text-[var(--muted)]">{i + 1}</span>
                  <span className="flex-1 truncate text-sm">{h.label}</span>
                  <span className="text-xs text-[var(--muted)]">{fmt(h.violations)}</span>
                  <div className="h-2 w-20 overflow-hidden rounded bg-[#222b3d]">
                    <div className="h-full" style={{ width: `${h.impact_score}%`, background: "var(--accent)" }} />
                  </div>
                </li>
              ))}
            </ul>
          </div>
        </div>

        <div className="mt-6 card p-6">
          <h3 className="text-lg font-bold">Honest forecasting</h3>
          <p className="mt-1 text-sm text-[var(--muted)]">
            {meta
              ? `Spatio-temporal forecaster validated on held-out weeks: Pearson r = ${meta.backtest.pearson_r}, MAE ${meta.backtest.mae} across ${fmt(meta.backtest.cells)} cells.`
              : "—"}{" "}
            Trained on the past, tested on the unseen future — no inflated in-sample numbers.
          </p>
        </div>

        <p className="mt-8 text-center text-xs text-[var(--muted)]">
          Built for Bengaluru Traffic Police · Gridlock Hackathon 2.0 · Round 2 prototype
        </p>
      </section>
    </main>
  );
}
