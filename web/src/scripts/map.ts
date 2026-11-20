import * as maplibregl from "maplibre-gl";
// the map's web worker is its own file: let the bundler copy it and tell MapLibre where it is
import workerUrl from "maplibre-gl/dist/maplibre-gl-worker.mjs?url";
import {
  CELL_STEPS, EMPTY_GEO, HOT_STEPS, type Hotspots,
  cellsGeo, hotspotsGeo, stepColorExpr,
} from "./grid";
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
type HoverTarget = { source: "cells" | "hotspots"; id: number };

const STYLE = "https://tiles.openfreemap.org/styles/positron";
const FALLBACK: maplibregl.StyleSpecification = {
  version: 8, sources: {}, layers: [{ id: "bg", type: "background", paint: { "background-color": "#e9ecef" } }],
};
const CELL_LAYERS = ["cells-fill", "cells-grid", "cells-hover-casing", "cells-hover"] as const;
const HOT_LAYERS = ["hot-fill", "hot-grid", "hot-hover-casing", "hot-hover"] as const;
const HINT: Record<Layer, string> = {
  thermals: "Where pilots circled in thermals: ~200 m grid cells with thermals from at least two flights. Hover a cell for its climb rate and height.",
  climb_pct: "Share of pilots who found a climb when they crossed each 1 km grid cell.",
  predicted_pct: "What the thermal model predicts for those same 1 km grid cells, scored on sites it never saw.",
};

const $ = (id: string) => document.getElementById(id)!;
const fmt = new Intl.NumberFormat("en");

let sites: Site[] = [];
let layer: Layer = "thermals";     // the layer shown
let chosen: Layer = "thermals";    // the one the user picked (thermals fall back to cells where there are none)
let hasHotspots = false;
let current: Site | null = null;
let hovered: HoverTarget | null = null;
let allSites = () => {};

function setHovered(map: maplibregl.Map, next: HoverTarget | null) {
  if (hovered && (!next || hovered.source !== next.source || hovered.id !== next.id)) {
    if (map.getSource(hovered.source)) map.setFeatureState(hovered, { hover: false });
  }
  if (next && (!hovered || hovered.source !== next.source || hovered.id !== next.id)) {
    if (map.getSource(next.source)) map.setFeatureState(next, { hover: true });
  }
  hovered = next;
}

function clearHover(map: maplibregl.Map, resetSources = false) {
  setHovered(map, null);
  if (resetSources) {
    if (map.getSource("cells")) map.removeFeatureState({ source: "cells" });
    if (map.getSource("hotspots")) map.removeFeatureState({ source: "hotspots" });
  }
}

async function json<T>(url: string): Promise<T | null> {
  try {
    const r = await fetch(url);
    return r.ok ? ((await r.json()) as T) : null;
  } catch {
    return null;
  }
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
  clearHover(map, true);
  $("tip").hidden = true;
  hasHotspots = !!hot && hot.rows.length > 0;
  (map.getSource("hotspots") as maplibregl.GeoJSONSource).setData(hotspotsGeo(hot, s.lat));
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
    src.setData(EMPTY_GEO);
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
  clearHover(map, true);
  $("tip").hidden = true;
  (map.getSource("cells") as maplibregl.GeoJSONSource).setData(EMPTY_GEO);
  (map.getSource("hotspots") as maplibregl.GeoJSONSource).setData(EMPTY_GEO);
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
  clearHover(map);
  $("tip").hidden = true;
  document.querySelectorAll<HTMLButtonElement>(".seg button").forEach((b) => {
    const on = b.dataset.layer === l;
    b.classList.toggle("on", on);
    b.setAttribute("aria-checked", String(on));
  });
  $("layer-hint").textContent = l === "thermals" && current && !hasHotspots ? "No thermal hotspots exported for this site yet." : HINT[l];
  $("legend-title").textContent = LEGEND[l][0];
  const steps = l === "thermals" ? HOT_STEPS : CELL_STEPS;
  $("legend-bar").innerHTML = steps
    .map(([, fill, border]) => `<span style="background:${fill};border-color:${border}"></span>`)
    .join("");
  $("legend-scale").innerHTML = LEGEND[l][1].map((t) => `<span>${t}</span>`).join("");
  const vis = (id: string, on: boolean) => map.getLayer(id) && map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
  for (const id of CELL_LAYERS) vis(id, l !== "thermals");
  for (const id of HOT_LAYERS) vis(id, l === "thermals");
  if (l !== "thermals") {
    const notNull: maplibregl.FilterSpecification = ["!=", ["get", l], null];
    for (const id of CELL_LAYERS) if (map.getLayer(id)) map.setFilter(id, notNull);
    if (map.getLayer("cells-fill")) map.setPaintProperty("cells-fill", "fill-color", stepColorExpr(l, CELL_STEPS, "fill"));
    if (map.getLayer("cells-grid")) map.setPaintProperty("cells-grid", "line-color", stepColorExpr(l, CELL_STEPS, "border"));
  }
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
    const labels = map.getStyle().layers?.find((l) => l.type === "symbol")?.id;
    // 1 km cells (Climb % and Model): mosaic squares with discrete steps, subtle borders and hover frame
    map.addSource("cells", { type: "geojson", data: EMPTY_GEO, tolerance: 0 });
    map.addLayer({ id: "cells-fill", type: "fill", source: "cells",
      paint: {
        "fill-color": stepColorExpr("climb_pct", CELL_STEPS, "fill"),
        "fill-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.96, 0.84],
        "fill-antialias": true,
      } }, labels);
    map.addLayer({ id: "cells-grid", type: "line", source: "cells",
      paint: {
        "line-color": stepColorExpr("climb_pct", CELL_STEPS, "border"),
        "line-width": ["interpolate", ["linear"], ["zoom"], 8, 0.5, 11, 1.0, 14, 1.5],
        "line-opacity": ["interpolate", ["linear"], ["zoom"], 7.5, 0, 8.5, 0.75],
      } }, labels);
    map.addLayer({ id: "cells-hover-casing", type: "line", source: "cells",
      paint: {
        "line-color": "#ffffff",
        "line-width": ["interpolate", ["linear"], ["zoom"], 8, 2.8, 13, 4.2],
        "line-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.92, 0],
      } }, labels);
    map.addLayer({ id: "cells-hover", type: "line", source: "cells",
      paint: {
        "line-color": "#16150f",
        "line-width": ["interpolate", ["linear"], ["zoom"], 8, 1.4, 13, 2.2],
        "line-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 1, 0],
      } }, labels);
    // ~200 m thermal hotspots: snapped square mosaic grid with 7 discrete terracotta steps and hover frame
    map.addSource("hotspots", { type: "geojson", data: EMPTY_GEO, tolerance: 0 });
    map.addLayer({ id: "hot-fill", type: "fill", source: "hotspots",
      paint: {
        "fill-color": stepColorExpr("w", HOT_STEPS, "fill"),
        "fill-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.96, 0.84],
        "fill-antialias": true,
      } }, labels);
    map.addLayer({ id: "hot-grid", type: "line", source: "hotspots",
      paint: {
        "line-color": stepColorExpr("w", HOT_STEPS, "border"),
        "line-width": ["interpolate", ["linear"], ["zoom"], 10.5, 0.4, 12.5, 0.9, 15, 1.4],
        "line-opacity": ["interpolate", ["linear"], ["zoom"], 10, 0, 11.2, 0.72],
      } }, labels);
    map.addLayer({ id: "hot-hover-casing", type: "line", source: "hotspots",
      paint: {
        "line-color": "#ffffff",
        "line-width": ["interpolate", ["linear"], ["zoom"], 10, 2.6, 14, 4.0],
        "line-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 0.92, 0],
      } }, labels);
    map.addLayer({ id: "hot-hover", type: "line", source: "hotspots",
      paint: {
        "line-color": "#16150f",
        "line-width": ["interpolate", ["linear"], ["zoom"], 10, 1.3, 14, 2.1],
        "line-opacity": ["case", ["boolean", ["feature-state", "hover"], false], 1, 0],
      } }, labels);
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
      const f = e.features?.[0];
      const p = f?.properties as Record<string, number | null> | undefined;
      if (!p) return;
      if (typeof f?.id === "number") setHovered(map, { source: "cells", id: f.id });
      const pct = (v: number | null) => (v == null ? "–" : `${Math.round(v)}%`);
      showTip(e, `<b>${fmt.format(p.crossings as number)}</b> crossings<br>Found a climb <b>${pct(p.climb_pct)}</b>` +
        `<br>Model <b>${pct(p.predicted_pct)}</b>` + (p.climb_ms != null ? `<br>Mean climb <b>${(p.climb_ms as number).toFixed(1)} m/s</b>` : ""));
    });
    map.on("mouseleave", "cells-fill", () => { setHovered(map, null); $("tip").hidden = true; });
    map.on("mousemove", "hot-fill", (e) => {
      const f = e.features?.[0];
      const p = f?.properties as Record<string, number | null> | undefined;
      if (!p) return;
      if (typeof f?.id === "number") setHovered(map, { source: "hotspots", id: f.id });
      showTip(e, `<b>${fmt.format(p.thermals as number)}</b> thermals from <b>${fmt.format(p.flights as number)}</b> flights` +
        (p.climb_ms != null ? `<br>Mean climb <b>${(p.climb_ms as number).toFixed(1)} m/s</b>` : "") +
        (p.top != null ? `<br>Top <b>${fmt.format(p.top as number)} m</b> above the ground` : ""));
    });
    map.on("mouseleave", "hot-fill", () => { setHovered(map, null); $("tip").hidden = true; });
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
