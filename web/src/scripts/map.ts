import * as maplibregl from "maplibre-gl";
// the map's web worker is its own file: let the bundler copy it and tell MapLibre where it is
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import { attachMap, closePlanner } from "./planner";
import { addRelief } from "./terrain";
import { loadTaskModel, modelSummary, onTaskSelect, renderTasks } from "./tasks";

maplibregl.setWorkerUrl(workerUrl);

type Site = {
  id: string; name: string; country: string; lat: number; lon: number;
  box: [number, number, number, number]; events: number; flights: number;
  first: string; last: string; cells: number;
};
type Summary = {
  sites: number; flights: number; competitions: number; cells: number; min_crossings: number;
  model?: Record<string, { auc?: number; log_loss_skill?: number }>;
  climb_rate?: { r2?: number; mae_ms?: number };
};
type Layer = "thermals" | "climb_pct" | "predicted_pct";
type Hotspots = { cols: string[]; rows: (number | null)[][] };

const STYLE = "https://tiles.openfreemap.org/styles/positron";
const FALLBACK: maplibregl.StyleSpecification = {
  version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#e9ecef" } }],
};
const HALF = 0.005; // a cell is 0.01° on a side, centred on its point
// one hue, light to dark: the share of crossings that found a climb
const RAMP: [number, string][] = [
  [0, "#fdeee6"], [12.5, "#fbd3bd"], [25, "#f6a982"], [37.5, "#f08455"],
  [50, "#eb6834"], [62.5, "#cf5222"], [75, "#a33e16"],
];
const HINT: Record<Layer, string> = {
  thermals: "Where pilots circled in thermals: cells of about 200 m with thermals from at least two flights. Zoom in for each cell's climb rate and height.",
  climb_pct: "Share of pilots who found a climb when they crossed the 1 km cell.",
  predicted_pct: "What the thermal model predicts for those same crossings, scored on sites it never saw.",
};

const $ = (id: string) => document.getElementById(id)!;
const fmt = new Intl.NumberFormat("en");

let sites: Site[] = [];
let layer: Layer = "thermals";     // the layer shown
let chosen: Layer = "thermals";    // the one the user picked (thermals fall back to cells where there are none)
let hasHotspots = false;
let current: Site | null = null;
let allSites = () => {};

async function json<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
}

function colorExpr(prop: Layer): maplibregl.ExpressionSpecification {
  return ["case", ["==", ["get", prop], null], "rgba(0,0,0,0)",
    ["interpolate", ["linear"], ["get", prop], ...RAMP.flat()]] as maplibregl.ExpressionSpecification;
}

function sitesGeo(list: Site[]): GeoJSON.FeatureCollection {
  return {
    type: "FeatureCollection",
    features: list.map((s) => ({
      type: "Feature", id: list.indexOf(s),
      geometry: { type: "Point", coordinates: [s.lon, s.lat] },
      properties: { id: s.id, name: s.name, country: s.country, flights: s.flights, events: s.events },
    })),
  };
}

function cellsGeo(cols: string[], rows: (number | null)[][]): GeoJSON.FeatureCollection {
  const i = Object.fromEntries(cols.map((c, k) => [c, k]));
  return {
    type: "FeatureCollection",
    features: rows.map((r) => {
      const lat = r[i.lat] as number, lon = r[i.lon] as number;
      return {
        type: "Feature",
        geometry: { type: "Polygon", coordinates: [[[lon - HALF, lat - HALF], [lon + HALF, lat - HALF],
          [lon + HALF, lat + HALF], [lon - HALF, lat + HALF], [lon - HALF, lat - HALF]]] },
        properties: { crossings: r[i.crossings], climb_pct: r[i.climb_pct], predicted_pct: r[i.predicted_pct],
          climb_ms: r[i.climb_ms] },
      };
    }),
  };
}

/** Hotspot cells as points; `w` (0–1) is how many flights climbed there, on a log scale, relative to
 * the site's busiest cells (95th percentile), so every site uses the whole colour scale. */
function hotspotsGeo(h: Hotspots | null): GeoJSON.FeatureCollection {
  if (!h) return { type: "FeatureCollection", features: [] };
  const i = Object.fromEntries(h.cols.map((c, k) => [c, k]));
  const flights = h.rows.map((r) => (r[i.flights] as number) || 0).sort((a, b) => a - b);
  const top = Math.log1p(Math.max(3, flights[Math.floor(0.95 * (flights.length - 1))] ?? 3));
  return {
    type: "FeatureCollection",
    features: h.rows.map((r) => ({
      type: "Feature",
      geometry: { type: "Point", coordinates: [r[i.lon] as number, r[i.lat] as number] },
      properties: { thermals: r[i.thermals], flights: r[i.flights], climb_ms: r[i.climb_ms], top: r[i.top_agl_m],
        w: Math.min(1, Math.max(0.25, Math.log1p((r[i.flights] as number) || 0) / top)) },
    })),
  };
}

function renderList(map: maplibregl.Map, filter = "") {
  const q = filter.trim().toLowerCase();
  const ol = $("sites");
  ol.innerHTML = "";
  sites
    .filter((s) => !q || s.name.toLowerCase().includes(q) || (s.country || "").toLowerCase().includes(q))
    .forEach((s) => {
      const li = document.createElement("li");
      const b = document.createElement("button");
      b.innerHTML = `<span class="n">${esc(s.name)}</span><span class="c">${esc(s.country || "")}</span>
        <span class="d">${fmt.format(s.flights)} flights · ${s.events} comps · ${s.first}${s.last !== s.first ? "–" + s.last : ""}</span>`;
      b.onclick = () => selectSite(map, s);
      li.appendChild(b);
      ol.appendChild(li);
    });
}

function esc(s: string) {
  return s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
}

async function selectSite(map: maplibregl.Map, s: Site) {
  current = s;
  history.replaceState(null, "", `#${s.id}`);
  $("site").hidden = false;
  $("list-wrap").hidden = true;
  $("s-name").textContent = s.name;
  $("s-meta").textContent = `${s.country || ""} · ${s.first}${s.last !== s.first ? "–" + s.last : ""}`;
  const facts: [string, string][] = [
    ["Competitions", String(s.events)], ["Flights", fmt.format(s.flights)], ["Cells shown", fmt.format(s.cells)],
  ];
  const [data, hot] = await Promise.all([
    json<{ cols: string[]; rows: (number | null)[][] }>(`/data/cells/${s.id}.json`),
    json<Hotspots>(`/data/hotspots/${s.id}.json`),
  ]);
  if (current !== s) return;
  hasHotspots = !!hot && hot.rows.length > 0;
  (map.getSource("hotspots") as maplibregl.GeoJSONSource).setData(hotspotsGeo(hot));
  setLayer(map, chosen === "thermals" && !hasHotspots ? "climb_pct" : chosen);
  if (hasHotspots) {
    const k = hot!.cols.indexOf("thermals");
    facts.push(["Thermals mapped", fmt.format(hot!.rows.reduce((n, r) => n + ((r[k] as number) || 0), 0))]);
    $("legend").hidden = false;
  }
  const src = map.getSource("cells") as maplibregl.GeoJSONSource;
  if (data && data.rows.length) {
    src.setData(cellsGeo(data.cols, data.rows));
    const i = data.cols.indexOf("climb_pct"), c = data.cols.indexOf("crossings");
    let w = 0, sum = 0;
    for (const r of data.rows) { const n = (r[c] as number) || 0; if (r[i] != null) { sum += (r[i] as number) * n; w += n; } }
    facts.push(["Crossings", fmt.format(w)], ["Found a climb", w ? `${(sum / w).toFixed(0)}%` : "–"]);
    $("legend").hidden = false;
  } else {
    src.setData({ type: "FeatureCollection", features: [] });
  }
  $("s-facts").innerHTML = facts.map(([k, v]) => `<div><dt>${k}</dt><dd>${v}</dd></div>`).join("");
  renderTasks(s, $("tasks-box"), $("planner"));
  map.fitBounds([[s.box[0], s.box[1]], [s.box[2], s.box[3]]], { padding: 40, duration: 900 });
}

/** A task drawn on the map: the line between cylinder centres and each cylinder. */
function routeGeo(route: [number, number, number, string][] | null): GeoJSON.FeatureCollection {
  if (!route) return { type: "FeatureCollection", features: [] };
  const circle = (lat: number, lon: number, r: number): GeoJSON.Position[] => {
    const pts: GeoJSON.Position[] = [];
    const dLat = r / 111320, dLon = r / (111320 * Math.cos((lat * Math.PI) / 180));
    for (let k = 0; k <= 48; k++) {
      const a = (k / 48) * 2 * Math.PI;
      pts.push([lon + dLon * Math.cos(a), lat + dLat * Math.sin(a)]);
    }
    return pts;
  };
  return {
    type: "FeatureCollection",
    features: [
      { type: "Feature", properties: { kind: "line" }, geometry: { type: "LineString", coordinates: route.map(([la, lo]) => [lo, la]) } },
      ...route.filter(([, , r]) => r > 0).map(([la, lo, r, ty]) => ({
        type: "Feature" as const, properties: { kind: "cyl", type: ty },
        geometry: { type: "Polygon" as const, coordinates: [circle(la, lo, r)] },
      })),
    ],
  };
}

function backToAll(map: maplibregl.Map) {
  current = null;
  history.replaceState(null, "", location.pathname);
  $("site").hidden = true;
  $("list-wrap").hidden = false;
  $("legend").hidden = true;
  (map.getSource("cells") as maplibregl.GeoJSONSource).setData({ type: "FeatureCollection", features: [] });
  (map.getSource("hotspots") as maplibregl.GeoJSONSource).setData(hotspotsGeo(null));
  (map.getSource("route") as maplibregl.GeoJSONSource | undefined)?.setData(routeGeo(null));
  closePlanner();
  allSites();
}

const LEGEND: Record<Layer, [string, string[]]> = {
  thermals: ["Thermals found", ["few", "", "", "many"]],
  climb_pct: ["Found a climb", ["0%", "25%", "50%", "75%+"]],
  predicted_pct: ["Model: chance of a climb", ["0%", "25%", "50%", "75%+"]],
};

function setLayer(map: maplibregl.Map, l: Layer) {
  layer = l;
  document.querySelectorAll<HTMLButtonElement>(".seg button").forEach((b) => {
    const on = b.dataset.layer === l;
    b.classList.toggle("on", on);
    b.setAttribute("aria-checked", String(on));
  });
  $("layer-hint").textContent = l === "thermals" && current && !hasHotspots ? "No thermal hotspots exported for this site yet." : HINT[l];
  $("legend-title").textContent = LEGEND[l][0];
  $("legend-scale").innerHTML = LEGEND[l][1].map((t) => `<span>${t}</span>`).join("");
  const vis = (id: string, on: boolean) => map.getLayer(id) && map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  vis("cells-fill", l !== "thermals");
  vis("hot-heat", l === "thermals");
  vis("hot-dot", l === "thermals");
  if (l !== "thermals" && map.getLayer("cells-fill")) map.setPaintProperty("cells-fill", "fill-color", colorExpr(l));
}

function showTip(e: maplibregl.MapMouseEvent, html: string) {
  const tip = $("tip");
  tip.innerHTML = html;
  tip.hidden = false;
  const r = (e.target.getContainer() as HTMLElement).getBoundingClientRect();
  const x = Math.min(e.point.x + 14, r.width - tip.offsetWidth - 8), y = Math.min(e.point.y + 14, r.height - tip.offsetHeight - 8);
  tip.style.transform = `translate(${x}px, ${y}px)`;
}

export async function start() {
  const [summary, list] = await Promise.all([json<Summary>("/data/summary.json"), json<Site[]>("/data/sites.json")]);
  sites = (list || []).sort((a, b) => b.flights - a.flights);
  if (summary) {
    $("t-sites").textContent = fmt.format(summary.sites);
    $("t-comps").textContent = fmt.format(summary.competitions);
    $("t-flights").textContent = fmt.format(summary.flights);
    $("t-cells").textContent = fmt.format(summary.cells);
    const full = summary.model?.["+ weather"];
    $("foot").innerHTML = (full?.auc
      ? `Model tested on sites it never saw: AUC ${full.auc.toFixed(2)}. `
      : "") + `Cells with ${summary.min_crossings}+ crossings. Aggregates only, no pilot data.`;
  }
  if (!sites.length) $("empty").hidden = false;
  loadTaskModel().then((m) => {
    const note = m ? modelSummary(m) : "";
    if (note) { $("model-note").innerHTML = note; $("model-note").hidden = false; }
  });

  // the basemap is optional: without it (offline, blocked) the data still shows on a plain background
  const style = await fetch(STYLE).then((r) => (r.ok ? STYLE : FALLBACK)).catch(() => FALLBACK);
  const map = new maplibregl.Map({ container: "map", style, center: [10, 25], zoom: 1.6, attributionControl: { compact: true } });
  map.addControl(new maplibregl.NavigationControl({ showCompass: true, visualizePitch: true }), "top-right");

  const fitAll = (duration: number) => {
    if (!sites.length) return;
    const lons = sites.map((s) => s.lon), lats = sites.map((s) => s.lat);
    map.fitBounds([[Math.min(...lons), Math.min(...lats)], [Math.max(...lons), Math.max(...lats)]],
      { padding: 60, maxZoom: 6, duration });
  };
  allSites = () => fitAll(900);

  map.on("load", () => {
    if (!location.hash) fitAll(0);
    addRelief(map);
    map.addSource("cells", { type: "geojson", data: { type: "FeatureCollection", features: [] } });
    map.addLayer({ id: "cells-fill", type: "fill", source: "cells",
      paint: { "fill-color": colorExpr("climb_pct"), "fill-opacity": 0.72 } });
    // thermal hotspots: a heat map whose blur follows the cells' spacing (about 220 m), so it reads as
    // a smooth raster at every zoom; invisible dots on top answer the pointer with each cell's numbers
    map.addSource("hotspots", { type: "geojson", data: hotspotsGeo(null) });
    map.addLayer({ id: "hot-heat", type: "heatmap", source: "hotspots",
      paint: {
        "heatmap-weight": ["get", "w"],
        "heatmap-intensity": 0.9,
        "heatmap-radius": ["interpolate", ["exponential", 2], ["zoom"], 8, 1, 16, 256],
        "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"],
          0, "rgba(251,211,189,0)", 0.05, "rgba(251,211,189,0.6)", 0.15, "rgba(246,169,130,0.85)", 0.3, "#f08455",
          0.5, "#eb6834", 0.7, "#cf5222", 0.9, "#a33e16", 1, "#7f2f10"],
        "heatmap-opacity": 0.85,
      } });
    map.addLayer({ id: "hot-dot", type: "circle", source: "hotspots", minzoom: 12,
      paint: {
        "circle-radius": ["interpolate", ["exponential", 2], ["zoom"], 12, 3, 16, 48],
        "circle-color": "#a33e16",
        "circle-opacity": 0.01,
      } });
    setLayer(map, layer);
    map.addSource("route", { type: "geojson", data: routeGeo(null) });
    map.addLayer({ id: "route-cyl", type: "line", source: "route", filter: ["==", ["get", "kind"], "cyl"],
      paint: { "line-color": "#1d4f99", "line-width": 1.5, "line-dasharray": [2, 1] } });
    map.addLayer({ id: "route-line", type: "line", source: "route", filter: ["==", ["get", "kind"], "line"],
      paint: { "line-color": "#1d4f99", "line-width": 2.5 } });
    onTaskSelect((route) => {
      (map.getSource("route") as maplibregl.GeoJSONSource).setData(routeGeo(route ?? null));
      if (route && route.length) {
        const la = route.map((p) => p[0]), lo = route.map((p) => p[1]);
        map.fitBounds([[Math.min(...lo), Math.min(...la)], [Math.max(...lo), Math.max(...la)]], { padding: 60, duration: 800 });
      }
    });
    // the task planner: its own layers, markers and clicks
    attachMap(map);
    map.addSource("sites", { type: "geojson", data: sitesGeo(sites) });
    map.addLayer({ id: "sites-dot", type: "circle", source: "sites",
      paint: {
        "circle-radius": ["interpolate", ["linear"], ["sqrt", ["get", "flights"]], 0, 4, 100, 14],
        "circle-color": "#cf5222", "circle-opacity": 0.9,
        "circle-stroke-color": "#ffffff", "circle-stroke-width": 2,
      },
      maxzoom: 8 });

    map.on("mousemove", "cells-fill", (e) => {
      const p = e.features?.[0]?.properties as Record<string, number | null> | undefined;
      if (!p) return;
      const pct = (v: number | null) => (v == null ? "–" : `${Math.round(v)}%`);
      showTip(e, `<b>${fmt.format(p.crossings as number)}</b> crossings<br>Found a climb <b>${pct(p.climb_pct)}</b>` +
        `<br>Model <b>${pct(p.predicted_pct)}</b>` + (p.climb_ms != null ? `<br>Mean climb <b>${(p.climb_ms as number).toFixed(1)} m/s</b>` : ""));
    });
    map.on("mouseleave", "cells-fill", () => ($("tip").hidden = true));
    map.on("mousemove", "hot-dot", (e) => {
      const p = e.features?.[0]?.properties as Record<string, number | null> | undefined;
      if (!p) return;
      showTip(e, `<b>${fmt.format(p.thermals as number)}</b> thermals from <b>${fmt.format(p.flights as number)}</b> flights` +
        (p.climb_ms != null ? `<br>Mean climb <b>${(p.climb_ms as number).toFixed(1)} m/s</b>` : "") +
        (p.top != null ? `<br>Top <b>${fmt.format(p.top as number)} m</b> above the ground` : ""));
    });
    map.on("mouseleave", "hot-dot", () => ($("tip").hidden = true));
    map.on("mousemove", "sites-dot", (e) => {
      map.getCanvas().style.cursor = "pointer";
      const p = e.features?.[0]?.properties as Record<string, string | number>;
      showTip(e, `<b>${esc(String(p.name))}</b> ${esc(String(p.country || ""))}<br>${fmt.format(p.flights as number)} flights · ${p.events} comps`);
    });
    map.on("mouseleave", "sites-dot", () => { map.getCanvas().style.cursor = ""; $("tip").hidden = true; });
    map.on("click", "sites-dot", (e) => {
      const id = e.features?.[0]?.properties?.id;
      const s = sites.find((x) => x.id === id);
      if (s) selectSite(map, s);
    });
  });

  renderList(map);
  ($("search") as HTMLInputElement).addEventListener("input", (e) => renderList(map, (e.target as HTMLInputElement).value));
  $("back").addEventListener("click", () => backToAll(map));
  document.querySelectorAll<HTMLButtonElement>(".seg button").forEach((b) =>
    b.addEventListener("click", () => { chosen = b.dataset.layer as Layer; setLayer(map, chosen); }));
  const hash = location.hash.slice(1);
  if (hash) map.once("load", () => { const s = sites.find((x) => x.id === hash); if (s) selectSite(map, s); });
}
