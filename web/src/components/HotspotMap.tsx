"use client";
import { useMemo } from "react";
import { DeckGL, HexagonLayer } from "deck.gl";
import { Map } from "react-map-gl/maplibre";
import "maplibre-gl/dist/maplibre-gl.css";
import type { GridCell } from "@/lib/parkpulse";

const COLOR_RANGE: [number, number, number][] = [
  [46, 134, 222], [92, 160, 180], [245, 205, 90], [245, 158, 65], [238, 110, 55], [226, 53, 43],
];
const BASEMAP = "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json";

export default function HotspotMap({ grid }: { grid: GridCell[] }) {
  const layers = useMemo(
    () => [
      new HexagonLayer<GridCell>({
        id: "hex",
        data: grid,
        getPosition: (d) => [d.lon, d.lat],
        getElevationWeight: (d) => d.n,
        getColorWeight: (d) => d.n,
        elevationAggregation: "SUM",
        colorAggregation: "SUM",
        radius: 160,
        elevationScale: 18,
        elevationRange: [0, 2400],
        extruded: true,
        coverage: 0.92,
        pickable: true,
        colorRange: COLOR_RANGE,
      }),
    ],
    [grid],
  );

  return (
    <DeckGL
      initialViewState={{ longitude: 77.5946, latitude: 12.9716, zoom: 10.4, pitch: 52, bearing: 0 }}
      controller
      layers={layers}
      style={{ position: "absolute", inset: "0" }}
    >
      <Map mapStyle={BASEMAP} />
    </DeckGL>
  );
}
