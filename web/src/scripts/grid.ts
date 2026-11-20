// Grid geometry and discrete colour ramps for the map's thermal layers:
// - 1 km cells (Climb % and Model): 0.01° cells with Mercator-uniform mosaic spacing
// - ~200 m thermal hotspots (Thermals): snapped to their square grid with mosaic spacing
// No DOM dependencies, so the geometry and colour steps can be tested directly in Node.

import type * as maplibregl from "maplibre-gl";

export type CellLayer = "climb_pct" | "predicted_pct";
export type Hotspots = { cols: string[]; step_deg?: number; rows: (number | null)[][] };
export type StepDef = readonly [number | null, string, string]; // [lower bound (null for first bin), fill, border]

export const EMPTY_GEO: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };

export const CELL_HALF = 0.005;   // 0.01° cell, centred on its point (~1 km)
export const CELL_INSET = 0.06;   // 6% inset per side -> 12% gap between adjacent 1 km cells
export const DEFAULT_HOT_STEP = 0.002; // ~220 m of latitude (matches scripts/export_hotspots.py)
export const HOT_INSET = 0.08;    // 8% inset per side -> 16% gap between adjacent ~200 m squares

// Discrete 7-step ramps using the page's terracotta palette, each paired with a darker border tone.
export const CELL_STEPS: readonly StepDef[] = [
  [null, "#fdeee6", "#d9b09a"],
  [12.5, "#fbd3bd", "#d1906e"],
  [25, "#f6a982", "#c76f44"],
  [37.5, "#f08455", "#b84f23"],
  [50, "#eb6834", "#9e3a12"],
  [62.5, "#cf5222", "#7e2a0b"],
  [75, "#a33e16", "#5c1d06"],
];

export const HOT_STEPS: readonly StepDef[] = [
  [null, "#fbd3bd", "#d1906e"],
  [1 / 7, "#f6a982", "#c76f44"],
  [2 / 7, "#f08455", "#b84f23"],
  [3 / 7, "#eb6834", "#9e3a12"],
  [4 / 7, "#cf5222", "#7e2a0b"],
  [5 / 7, "#a33e16", "#5c1d06"],
  [6 / 7, "#7f2f10", "#451504"],
];

/** MapLibre step expression for either a layer's fill or subtle border colour, transparent on null. */
export function stepColorExpr(
  prop: string,
  steps: readonly StepDef[],
  kind: "fill" | "border" = "fill",
): maplibregl.ExpressionSpecification {
  const idx = kind === "fill" ? 1 : 2;
  const step: (string | number | unknown[])[] = ["step", ["to-number", ["get", prop]], steps[0][idx]];
  for (let k = 1; k < steps.length; k++) {
    step.push(steps[k][0] as number, steps[k][idx]);
  }
  return ["case", ["==", ["get", prop], null], "rgba(0,0,0,0)", step] as maplibregl.ExpressionSpecification;
}

/** 1 km grid cells with uniform horizontal and vertical gaps in Web Mercator. */
export function cellsGeo(cols: string[], rows: (number | null)[][]): GeoJSON.FeatureCollection {
  const i = Object.fromEntries(cols.map((c, k) => [c, k]));
  const features: GeoJSON.Feature[] = [];
  let id = 1;
  for (const r of rows) {
    const lat = r[i.lat], lon = r[i.lon];
    if (lat == null || lon == null) continue;
    const padLon = CELL_HALF * CELL_INSET;
    // In Web Mercator, dy_px = dLat / cos(lat), so scaling padLat by cos(lat) keeps gaps equal on screen.
    const padLat = padLon * Math.max(Math.cos((lat * Math.PI) / 180), 0.2);
    const dx = CELL_HALF - padLon;
    const dy = CELL_HALF - padLat;
    features.push({
      type: "Feature",
      id: id++,
      geometry: {
        type: "Polygon",
        coordinates: [[[lon - dx, lat - dy], [lon + dx, lat - dy], [lon + dx, lat + dy], [lon - dx, lat + dy], [lon - dx, lat - dy]]],
      },
      properties: {
        crossings: r[i.crossings],
        climb_pct: r[i.climb_pct],
        predicted_pct: r[i.predicted_pct],
        climb_ms: r[i.climb_ms],
      },
    });
  }
  return { type: "FeatureCollection", features };
}

type CellAcc = {
  iy: number;
  ix: number;
  thermals: number;
  flights: number;
  climbSum: number;
  climbW: number;
  topSum: number;
  topW: number;
  merged: boolean;
  rawClimb: number | null;
  rawTop: number | null;
};

/** ~200 m thermal hotspot cells snapped to a Mercator-square mosaic grid.
 * `w` (0–1) is how many flights climbed in each cell, on a log scale from the site's minimum
 * up to its 95th percentile (floored so tiny sites do not saturate the darkest step). */
export function hotspotsGeo(h: Hotspots | null, siteLat: number): GeoJSON.FeatureCollection {
  if (!h || !h.rows.length) return EMPTY_GEO;
  const i = Object.fromEntries(h.cols.map((c, k) => [c, k]));
  const stepLat = h.step_deg ?? DEFAULT_HOT_STEP;
  const stepLon = stepLat / Math.max(Math.cos((siteLat * Math.PI) / 180), 0.2);

  // Snap each row back to its (iy, ix) bucket and merge rare rounding-boundary collisions.
  const buckets = new Map<string, CellAcc>();
  for (const r of h.rows) {
    const lat = r[i.lat], lon = r[i.lon];
    if (lat == null || lon == null) continue;
    const iy = Math.floor(lat / stepLat);
    const ix = Math.floor(lon / stepLon);
    const key = `${iy},${ix}`;
    const thermals = (r[i.thermals] as number) || 0;
    const flights = (r[i.flights] as number) || 0;
    const climb = r[i.climb_ms];
    const top = r[i.top_agl_m];
    const w = Math.max(1, thermals);
    const cur = buckets.get(key);
    if (!cur) {
      buckets.set(key, {
        iy, ix, thermals, flights,
        climbSum: climb != null ? climb * w : 0, climbW: climb != null ? w : 0,
        topSum: top != null ? top * w : 0, topW: top != null ? w : 0,
        merged: false, rawClimb: climb, rawTop: top,
      });
    } else {
      cur.merged = true;
      cur.thermals += thermals;
      cur.flights = Math.max(cur.flights, flights);
      if (climb != null) { cur.climbSum += climb * w; cur.climbW += w; }
      if (top != null) { cur.topSum += top * w; cur.topW += w; }
    }
  }

  const list = Array.from(buckets.values());
  if (!list.length) return EMPTY_GEO;

  const sorted = list.map((c) => c.flights).sort((a, b) => a - b);
  const minF = Math.max(1, sorted[0] ?? 2);
  const p95F = sorted[Math.floor(0.95 * (sorted.length - 1))] ?? minF;
  const lo = Math.log1p(minF);
  const hi = Math.log1p(Math.max(minF + 4, p95F));
  const span = Math.max(hi - lo, 1e-9);

  const padLat = stepLat * HOT_INSET;
  const padLon = stepLon * HOT_INSET;

  const features: GeoJSON.Feature[] = list.map((c, idx) => {
    const south = c.iy * stepLat + padLat;
    const north = (c.iy + 1) * stepLat - padLat;
    const west = c.ix * stepLon + padLon;
    const east = (c.ix + 1) * stepLon - padLon;
    const climbMs = c.merged
      ? (c.climbW ? Math.round((c.climbSum / c.climbW) * 100) / 100 : null)
      : c.rawClimb;
    const topAgl = c.merged
      ? (c.topW ? Math.round(c.topSum / c.topW) : null)
      : c.rawTop;
    const w = Math.min(1, Math.max(0, (Math.log1p(c.flights) - lo) / span));
    return {
      type: "Feature",
      id: idx + 1,
      geometry: {
        type: "Polygon",
        coordinates: [[[west, south], [east, south], [east, north], [west, north], [west, south]]],
      },
      properties: { thermals: c.thermals, flights: c.flights, climb_ms: climbMs, top: topAgl, w },
    };
  });

  return { type: "FeatureCollection", features };
}
