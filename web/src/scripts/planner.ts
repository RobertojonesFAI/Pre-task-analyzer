// Task planner: set a competition task on the map (turnpoints, cylinders, start and deadline),
// import it from instruments' files or an XCTrack QR code, export it back to them, and ask the API
// what the task model predicts for it with either a live multi-model forecast or a past competition
// day's ERA5 reanalysis.

import * as maplibregl from "maplibre-gl";
import QRCode from "qrcode";
import * as X from "./xctask";
import {
  API,
  apiAvailable,
  fetchForecastModels,
  predictRoute,
  type ForecastModelInfo,
  type ModelPrediction,
  type Prediction,
} from "./api";

/** A past competition day at the site: its weather (in ERA5 mode) and pilot field (in both modes). */
export type Day = {
  key: string; date: string; competition: string | null; task: string | null; km: number | null;
  pred_kmh: number | null; pred_goal: number | null; route?: [number, number, number, string][];
};
export type PlanContext = { site: { id: string; name: string; lat: number; lon: number }; days: Day[] };
type SiteWaypoints = {
  waypoints: [string, number, number, number | null][];
  tasks: Record<string, { tps: [number, number, string][]; start: string | null; deadline: string | null; utc: number | null }>;
};
type Kind = "takeoff" | "sss" | "tp" | "ess" | "goal";
type WxMode = "forecast" | "era5";
type BlendKind = "median" | "mean";

const COLORS: Record<Kind, string> = { takeoff: "#7a7873", sss: "#2f7d3a", tp: "#1d4f99", ess: "#b26a00", goal: "#cf5222" };
const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };
const MAX_FROM_SITE_DEG = 3;
const SNAP_PX = 12;

const FALLBACK_MODELS: ForecastModelInfo[] = [
  { id: "ecmwf_ifs", name: "ECMWF IFS", region: "global", resolution_km: 9, max_days: 6, blh_source: "native", available: true },
  { id: "ecmwf_aifs", name: "ECMWF AIFS", region: "global", resolution_km: 28, max_days: 6, blh_source: "parcel", available: true },
  { id: "gfs", name: "GFS", region: "global", resolution_km: 25, max_days: 6, blh_source: "native", available: true },
  { id: "icon", name: "ICON", region: "global", resolution_km: 13, max_days: 6, blh_source: "parcel", available: true },
  { id: "gem", name: "GEM", region: "global", resolution_km: 15, max_days: 6, blh_source: "parcel", available: true },
];

let map: maplibregl.Map | null = null;
let markers: maplibregl.Marker[] = [];
let el: HTMLElement | null = null;
let root: HTMLElement | null = null;   // the planner's section; `el` is set while it is open
let ctx: PlanContext | null = null;
let day: Day | null = null;
let task: X.Task = X.emptyTask();
let dist: X.Distances | null = null;
let adding = false;
let dragging = false;
let insertAfter = -1;                  // map clicks insert after this row; -1 appends
let siteWps: X.Waypoint[] = [];
let extraWps: X.Waypoint[] = [];
let utc = 0;                           // local time minus UTC at the site, hours
let tplTask: X.Task | null = null;     // the task as it was set
let tplSS: number | null = null;       // this page's optimized speed section of the task as set
let tplGate: string | null = null;     // its first start gate (UTC)
let raf = 0;
let draggedAt = 0;                     // a drag ends with a click on the marker: ignore it
const cache = new Map<string, Promise<SiteWaypoints | null>>();

// Forecast mode state
let wxMode: WxMode = "forecast";
let fcDate = "";
let fcMinDate = "";
let fcMaxDate = "";
let fcModels: ForecastModelInfo[] = FALLBACK_MODELS;
let fcChoice = "blend";                // "blend" or a model id
let fcBlendKind: BlendKind = "median";
let fcIncluded = new Set<string>();
let lastForecast: Prediction | null = null;

const $ = <T extends HTMLElement = HTMLElement>(sel: string) => el!.querySelector<T>(sel)!;
const esc = (s: string) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);
const pad2 = (n: number) => String(n).padStart(2, "0");
const hm = (h: number | null | undefined) => {
  if (h == null || !Number.isFinite(h) || h <= 0) return "–";
  const m = Math.round(h * 60);
  return `${Math.floor(m / 60)}:${pad2(m % 60)}`;
};
const pct = (v: number | null | undefined) => (v == null || !Number.isFinite(v) ? "–" : `${Math.round(v * 100)}%`);
const km = (v: number | null | undefined) => (v == null ? "–" : `${v.toFixed(1)} km`);

function isoDateOffset(offsetDays: number, offsetHours = 0): string {
  const d = new Date(Date.now() + offsetHours * 3600_000 + offsetDays * 86400_000);
  return `${d.getUTCFullYear()}-${pad2(d.getUTCMonth() + 1)}-${pad2(d.getUTCDate())}`;
}

function initForecastDates() {
  fcMinDate = isoDateOffset(-1);
  fcMaxDate = isoDateOffset(6);
  const localToday = isoDateOffset(0, utc);
  fcDate = localToday < fcMinDate ? fcMinDate : localToday > fcMaxDate ? fcMaxDate : localToday;
}

function relativeDayTag(iso: string): string {
  const localToday = isoDateOffset(0, utc);
  const diff = Math.round((Date.parse(`${iso}T12:00:00Z`) - Date.parse(`${localToday}T12:00:00Z`)) / 86400_000);
  if (diff === 0) return "Today";
  if (diff === 1) return "Tomorrow";
  if (diff === -1) return "Yesterday";
  return diff > 1 ? `+${diff}d` : `${diff}d`;
}

/** Shifts "HH:MM[:SS]" by hours; returns "HH:MM". */
function shift(t: string, h: number): string {
  const [a, b] = t.split(":").map(Number);
  const m = ((((a * 60 + b + Math.round(h * 60)) % 1440) + 1440) % 1440);
  return `${pad2(Math.floor(m / 60))}:${pad2(m % 60)}`;
}
const toLocal = (u: string | null | undefined) => (u ? shift(u, utc) : "");
const toUtc = (local: string) => `${shift(local, -utc)}:00`;
const utcLabel = () => `UTC${utc >= 0 ? "+" : "−"}${Math.abs(utc) % 1 ? Math.abs(utc).toFixed(1) : Math.abs(utc)}`;

function kind(i: number): Kind {
  const t = task.tps[i].type;
  if (i === task.tps.length - 1 && i > 0) return "goal";
  return t === "TAKEOFF" ? "takeoff" : t === "SSS" ? "sss" : t === "ESS" ? "ess" : "tp";
}

function siteWaypoints(id: string): Promise<SiteWaypoints | null> {
  if (!cache.has(id)) {
    cache.set(id, fetch(`/data/waypoints/${id}.json`).then((r) => (r.ok ? r.json() : null)).catch(() => null));
  }
  return cache.get(id)!;
}

/** A day's task as it was set, from the waypoints file (names, radii, times) or the route on the page. */
async function templateTask(site: PlanContext["site"], d: Day): Promise<X.Task | null> {
  const w = await siteWaypoints(site.id);
  const def = w?.tasks[d.key];
  utc = def?.utc ?? Math.round(site.lon / 15);
  if (w && def && def.tps.length >= 2) {
    return {
      ...X.emptyTask(),
      tps: def.tps.map(([i, r, ty]) => {
        const [name, lat, lon, alt] = w.waypoints[i];
        return { name, desc: "", lat, lon, alt: alt ?? 0, radius: r, type: (["TAKEOFF", "SSS", "ESS"].includes(ty) ? ty : "") as X.TpType };
      }),
      gates: def.start ? [toUtc(def.start)] : [],
      deadline: def.deadline ? toUtc(def.deadline) : null,
    };
  }
  const route = d.route || [];
  if (route.length < 2) return null;
  return {
    ...X.emptyTask(),
    tps: route.map(([lat, lon, r, ty], i) => ({ name: `TP${pad2(i + 1)}`, desc: "", lat, lon, alt: 0, radius: r,
      type: (["TAKEOFF", "SSS", "ESS"].includes(ty) ? ty : "") as X.TpType })),
  };
}

// ---------------------------------------------------------------------------------------------
// Map

export function attachMap(m: maplibregl.Map) {
  map = m;
  m.addSource("plan", { type: "geojson", data: EMPTY });
  m.addSource("plan-wps", { type: "geojson", data: EMPTY });
  m.addLayer({ id: "plan-cyl-fill", type: "fill", source: "plan", filter: ["==", ["get", "kind"], "cyl"],
    paint: { "fill-color": ["get", "color"], "fill-opacity": 0.06 } });
  m.addLayer({ id: "plan-cyl-line", type: "line", source: "plan", filter: ["in", ["get", "kind"], ["literal", ["cyl", "goal-line"]]],
    paint: { "line-color": ["get", "color"], "line-width": ["case", ["==", ["get", "kind"], "goal-line"], 4, 1.6] } });
  m.addLayer({ id: "plan-centres", type: "line", source: "plan", filter: ["==", ["get", "kind"], "centres"],
    paint: { "line-color": "#7a7873", "line-width": 1, "line-dasharray": [3, 2] } });
  m.addLayer({ id: "plan-route", type: "line", source: "plan", filter: ["==", ["get", "kind"], "route"],
    layout: { "line-join": "round", "line-cap": "round" }, paint: { "line-color": "#16150f", "line-width": 2.5 } });
  m.addLayer({ id: "plan-wps-dot", type: "circle", source: "plan-wps",
    paint: { "circle-radius": ["interpolate", ["linear"], ["zoom"], 7, 2.5, 11, 4.5],
      "circle-color": ["case", ["get", "extra"], "#7b3fa0", "#3d3c39"], "circle-stroke-color": "#ffffff", "circle-stroke-width": 1.2 } });
  // labels need the basemap's fonts: borrow the first one it uses
  const font = m.getStyle().layers?.map((l) => (l.layout as Record<string, unknown> | undefined)?.["text-font"]).find(Array.isArray);
  if (m.getStyle().glyphs && font) {
    m.addLayer({ id: "plan-wps-label", type: "symbol", source: "plan-wps", minzoom: 9.5,
      layout: { "text-field": ["get", "name"], "text-font": font as string[], "text-size": 11, "text-offset": [0, 0.9], "text-anchor": "top" },
      paint: { "text-color": "#3d3c39", "text-halo-color": "#ffffff", "text-halo-width": 1.4 } });
  }
  m.on("click", (e) => {
    if (!el || !adding) return;
    const wp = nearestWaypoint(e.point);
    const n = task.tps.length;
    add(wp ? { ...wp, radius: n ? 1000 : 400, type: "" }
      : { name: nextName(), desc: "", lat: +e.lngLat.lat.toFixed(5), lon: +e.lngLat.lng.toFixed(5), alt: 0, radius: n ? 1000 : 400, type: "" });
  });
}

function nearestWaypoint(p: maplibregl.Point): X.Waypoint | null {
  let best: X.Waypoint | null = null, bd = SNAP_PX;
  for (const w of [...extraWps, ...siteWps]) {
    const q = map!.project([w.lon, w.lat]);
    const d = Math.hypot(q.x - p.x, q.y - p.y);
    if (d < bd) { bd = d; best = w; }
  }
  return best;
}

function nextName(): string {
  const used = new Set(task.tps.map((t) => t.name));
  for (let i = task.tps.length + 1; ; i++) if (!used.has(`TP${pad2(i)}`)) return `TP${pad2(i)}`;
}

function circle(lat: number, lon: number, r: number): GeoJSON.Position[] {
  const pts: GeoJSON.Position[] = [];
  for (let k = 0; k <= 72; k++) {
    const [la, lo] = X.destination([lat, lon], (k / 72) * 360, r);
    pts.push([lo, la]);
  }
  return pts;
}

function geometry(): GeoJSON.FeatureCollection {
  const tps = task.tps, n = tps.length;
  const f: GeoJSON.Feature[] = [];
  tps.forEach((p, i) => {
    if (p.radius <= 0) return;
    const color = COLORS[kind(i)];
    if (i === n - 1 && n > 1 && task.goalType === "LINE") {
      // the goal line: across the way in, through the goal's centre (S7F 2025: from the optimized route)
      const from = dist?.route[n - 2] ?? [tps[n - 2].lat, tps[n - 2].lon];
      const az = X.bearing(from as [number, number], [p.lat, p.lon]);
      const a = X.destination([p.lat, p.lon], az + 90, p.radius), b = X.destination([p.lat, p.lon], az - 90, p.radius);
      f.push({ type: "Feature", properties: { kind: "goal-line", color }, geometry: { type: "LineString", coordinates: [[a[1], a[0]], [b[1], b[0]]] } });
      return;
    }
    f.push({ type: "Feature", properties: { kind: "cyl", color }, geometry: { type: "Polygon", coordinates: [circle(p.lat, p.lon, p.radius)] } });
  });
  if (n >= 2) {
    f.push({ type: "Feature", properties: { kind: "centres" }, geometry: { type: "LineString", coordinates: tps.map((p) => [p.lon, p.lat]) } });
    if (dist) f.push({ type: "Feature", properties: { kind: "route" }, geometry: { type: "LineString", coordinates: dist.route.map(([la, lo]) => [lo, la]) } });
  }
  return { type: "FeatureCollection", features: f };
}

function drawWaypoints() {
  const src = map?.getSource("plan-wps") as maplibregl.GeoJSONSource | undefined;
  if (!src) return;
  const feats = (list: X.Waypoint[], extra: boolean) => list.map((w) => ({
    type: "Feature" as const, properties: { name: w.name, extra }, geometry: { type: "Point" as const, coordinates: [w.lon, w.lat] },
  }));
  src.setData(el ? { type: "FeatureCollection", features: [...feats(siteWps, false), ...feats(extraWps, true)] } : EMPTY);
}

function drawMarkers() {
  markers.forEach((m) => m.remove());
  markers = [];
  if (!map || !el) return;
  task.tps.forEach((p, i) => {
    const d = document.createElement("button");
    d.type = "button";
    d.className = `tp-marker${i === insertAfter ? " sel" : ""}`;
    d.style.background = COLORS[kind(i)];
    d.textContent = String(i + 1);
    d.title = `${p.name} · ${p.radius >= 1000 ? `${p.radius / 1000} km` : `${p.radius} m`}`;
    const mk = new maplibregl.Marker({ element: d, draggable: true }).setLngLat([p.lon, p.lat]).addTo(map!);
    mk.on("dragstart", () => { dragging = true; });
    mk.on("drag", () => {
      const ll = mk.getLngLat();
      p.lat = +ll.lat.toFixed(5);
      p.lon = +ll.lng.toFixed(5);
      scheduleGeometry();
    });
    mk.on("dragend", () => {
      dragging = false;
      draggedAt = Date.now();
      // dropped on a waypoint: take it; otherwise a point of its own
      const wp = nearestWaypoint(map!.project(mk.getLngLat()));
      if (wp) Object.assign(p, { name: wp.name, desc: wp.desc, lat: wp.lat, lon: wp.lon, alt: wp.alt });
      else if (siteWps.concat(extraWps).some((w) => w.name === p.name)) Object.assign(p, { name: nextName(), desc: "", alt: 0 });
      changed(true);
    });
    d.addEventListener("click", (ev) => { ev.stopPropagation(); if (Date.now() - draggedAt > 400) select(i); });
    markers.push(mk);
  });
}

function scheduleGeometry() {
  if (raf) return;
  raf = requestAnimationFrame(() => {
    raf = 0;
    dist = task.tps.length >= 2 ? X.distances(task) : null;
    (map?.getSource("plan") as maplibregl.GeoJSONSource | undefined)?.setData(geometry());
    showDistances();
  });
}

function fit() {
  if (!map || !task.tps.length) return;
  const b = new maplibregl.LngLatBounds();
  task.tps.forEach((p) => {
    const dLat = p.radius / 111320, dLon = p.radius / (111320 * Math.cos((p.lat * Math.PI) / 180));
    b.extend([p.lon - dLon, p.lat - dLat]).extend([p.lon + dLon, p.lat + dLat]);
  });
  map.fitBounds(b, { padding: 50, duration: 700, maxZoom: 12 });
}

// ---------------------------------------------------------------------------------------------
// Panel

const dayLabel = (d: Day) => `${d.date} · ${d.competition || ""}${d.task ? ` · ${d.task}` : ""}`;

/** Shows the planner for a site, open, with its competition days to predict with (latest first). */
export function mountPlanner(box: HTMLElement, c: PlanContext) {
  closePlanner();
  root = box;
  ctx = { ...c, days: [...c.days].sort((a, b) => (a.date < b.date ? 1 : a.date > b.date ? -1 : 0)) };
  utc = Math.round(c.site.lon / 15);
  wxMode = "forecast";
  fcChoice = "blend";
  fcBlendKind = "median";
  fcModels = FALLBACK_MODELS;
  fcIncluded = new Set(FALLBACK_MODELS.map((m) => m.id));
  lastForecast = null;
  initForecastDates();
  box.hidden = false;
  // the latest day whose task is on record, so "That day's task" has something to show
  setDay(ctx.days.find((d) => (d.route?.length ?? 0) >= 2) ?? ctx.days[0] ?? null);
  expand();
}

/** Opens a past task in the planner, on its own day in ERA5 mode. */
export async function planFrom(key: string) {
  if (!ctx || !root) return;
  const d = ctx.days.find((x) => x.key === key);
  if (!d) return;
  wxMode = "era5";
  if (!el) expand();
  else applyWxModeUI();
  await setDay(d);
  if (el) $<HTMLSelectElement>("#p-day").value = d.key;
  await loadDayTask();
  root.scrollIntoView({ block: "start", behavior: "smooth" });
}

/** The day's task as it was set, measured the same way as new ones, so an unchanged task predicts the same. */
async function setDay(d: Day | null) {
  day = d;
  tplTask = null;
  tplSS = tplGate = null;
  if (!d || !ctx) return;
  const c = ctx;
  const t = await templateTask(c.site, d);
  if (ctx !== c || day !== d) return;
  tplTask = t;
  tplSS = t ? X.distances(t).ssKm : null;
  tplGate = t?.gates[0] ?? null;
}

async function loadDayTask() {
  if (!ctx || !day) return;
  const t = await templateTask(ctx.site, day);
  if (!el) return;
  if (!t) return msg("That day's task has no turnpoints on record.");
  task = t;
  insertAfter = -1;
  setAdding(false);
  changed(true);
  fit();
  msg(`${day.competition || "The task"}${day.task ? `, ${day.task}` : ""}, as it was set: change it, predict it or export it.`);
}

export function closePlanner() {
  hide();
  if (root) { root.innerHTML = ""; root.hidden = true; root.classList.remove("open"); }
  root = null;
  ctx = null;
  day = null;
  task = X.emptyTask();
  dist = null;
  extraWps = [];
  siteWps = [];
  insertAfter = -1;
  lastForecast = null;
}

/** Takes the planner off the map (its task is kept for when it opens again). */
function hide() {
  markers.forEach((m) => m.remove());
  markers = [];
  el = null;
  adding = false;
  if (map) {
    showSetRoute(true);
    (map.getSource("plan") as maplibregl.GeoJSONSource | undefined)?.setData(EMPTY);
    drawWaypoints();
    map.getCanvas().style.cursor = "";
  }
}

function collapse() {
  if (!root) return;
  hide();
  root.classList.remove("open");
  root.innerHTML = `<div class="plan-head"><h3>Create a task</h3><button class="primary-sm" id="p-open">Open the planner</button></div>
    <p class="hint">Set a task on the map, see what the task model predicts, and send it to XCTrack and other instruments.</p>`;
  root.querySelector("#p-open")!.addEventListener("click", () => expand());
}

/** The page's own drawing of an opened task: shown only while the planner has no task of its own. */
function showSetRoute(on: boolean) {
  for (const id of ["route-cyl", "route-line"]) if (map?.getLayer(id)) map.setLayoutProperty(id, "visibility", on ? "visible" : "none");
}

function renderForecastControls() {
  if (!el) return;
  const dateInput = $<HTMLInputElement>("#p-fdate");
  dateInput.min = fcMinDate;
  dateInput.max = fcMaxDate;
  dateInput.value = fcDate;
  $("#p-fdate-tag").textContent = relativeDayTag(fcDate);

  // If the selected single model is out of horizon for this date, fall back to blend.
  if (fcChoice !== "blend" && !fcModels.some((m) => m.id === fcChoice && m.available)) {
    fcChoice = "blend";
  }

  const modelsBox = $("#p-fc-models");
  const resText = (r: number) => `${r >= 10 ? Math.round(r) : r}k`;
  const pills = [
    `<button type="button" data-fc="blend" class="${fcChoice === "blend" ? "on" : ""}" title="Combine every available model for this date">Blend</button>`,
    ...fcModels.map((m) => {
      const on = fcChoice === m.id ? " on" : "";
      const reg = m.region === "regional" ? " reg" : "";
      const dis = m.available ? "" : " disabled";
      const tip = m.available
        ? `${m.name} (${m.resolution_km} km, ${m.region}, BLH: ${m.blh_source})`
        : `${m.name} covers up to ${m.max_days} days ahead`;
      return `<button type="button" data-fc="${esc(m.id)}" class="${(on + reg).trim()}"${dis} title="${esc(tip)}">${esc(m.name)} <span class="res">${resText(m.resolution_km)}</span></button>`;
    }),
  ];
  modelsBox.innerHTML = pills.join("");
  modelsBox.querySelectorAll<HTMLButtonElement>("button[data-fc]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.fc!;
      if (id === fcChoice) return;
      fcChoice = id;
      renderForecastControls();
      if (lastForecast && canRenderCachedChoice(lastForecast)) {
        renderForecastResult();
      } else {
        $("#p-out").innerHTML = lastForecast ? `<p class="hint">Click Predict this task to load ${id === "blend" ? "all forecast models" : esc(modelLabel(id))}.</p>` : "";
      }
    });
  });

  const blendBar = $("#p-blend-bar");
  blendBar.hidden = fcChoice !== "blend";
  blendBar.querySelectorAll<HTMLButtonElement>("button[data-blend]").forEach((b) => {
    const active = b.dataset.blend === fcBlendKind;
    b.classList.toggle("on", active);
    b.setAttribute("aria-checked", String(active));
  });
}

function modelLabel(id: string): string {
  return fcModels.find((m) => m.id === id)?.name ?? id;
}

function canRenderCachedChoice(pred: Prediction): boolean {
  const list = pred.models ?? [];
  if (!list.length) return false;
  if (fcChoice === "blend") {
    const availIds = fcModels.filter((m) => m.available).map((m) => m.id);
    const haveIds = new Set(list.map((m) => m.id));
    // Re-use cached response only if we already fetched more than one model (or all available).
    return list.length >= 2 || availIds.every((id) => haveIds.has(id));
  }
  return list.some((m) => m.id === fcChoice);
}

async function loadForecastModels() {
  if (!ctx || !API) return;
  const c = ctx, dateStr = fcDate;
  try {
    const res = await fetchForecastModels(c.site.id, dateStr);
    if (ctx !== c || fcDate !== dateStr || !el) return;
    fcMinDate = res.min_date;
    fcMaxDate = res.max_date;
    if (fcDate < fcMinDate) fcDate = fcMinDate;
    if (fcDate > fcMaxDate) fcDate = fcMaxDate;
    fcModels = res.models.length ? res.models : FALLBACK_MODELS;
    fcIncluded = new Set(fcModels.filter((m) => m.available).map((m) => m.id));
    renderForecastControls();
  } catch {
    // Keep fallback global list when offline or older API doesn't expose /forecast/models yet.
    if (ctx === c && el) renderForecastControls();
  }
}

function applyWxModeUI() {
  if (!el || !ctx) return;
  el.querySelectorAll<HTMLButtonElement>("#p-wx-mode button[data-mode]").forEach((b) => {
    const on = b.dataset.mode === wxMode;
    b.classList.toggle("on", on);
    b.setAttribute("aria-checked", String(on));
  });
  const isFc = wxMode === "forecast";
  $("#p-fc-box").hidden = !isFc;
  $("#p-day-lbl").textContent = isFc ? "Pilot field from" : "Weather and pilots of";
  $("#p-day-hint").textContent = ctx.days.length
    ? (isFc
        ? "Pilots and ratings come from this competition day; weather comes from the live forecast."
        : "Re-plans the task with that past competition day's ERA5 weather and pilot field.")
    : "No competition day on record at this site: tasks can be set and exported, not predicted.";
  if (isFc) renderForecastControls();
}

function expand() {
  const box = root!, c = ctx!;
  el = box;
  box.classList.add("open");
  const days = c.days.map((d) => `<option value="${esc(d.key)}"${d === day ? " selected" : ""}>${esc(dayLabel(d))}</option>`).join("");
  box.innerHTML = `<div class="plan-head"><h3>Create a task</h3><button class="close" id="p-close" aria-label="Close the planner">✕</button></div>
    <div class="seg seg-2" id="p-wx-mode" role="radiogroup" aria-label="Weather source">
      <button type="button" role="radio" data-mode="forecast" aria-checked="true" class="on">Forecast</button>
      <button type="button" role="radio" data-mode="era5" aria-checked="false">Past day (ERA5)</button>
    </div>
    <div class="fc-box" id="p-fc-box">
      <div class="fc-date-row">
        <label class="fc-date">Forecast date
          <input type="date" id="p-fdate">
        </label>
        <span class="fc-tag" id="p-fdate-tag"></span>
      </div>
      <div class="fc-label">Forecast model</div>
      <div class="btns fc-models" id="p-fc-models"></div>
      <div class="fc-blend-row" id="p-blend-bar">
        <span>Blend</span>
        <div class="seg seg-2 seg-sm" role="radiogroup" aria-label="Blend method">
          <button type="button" role="radio" data-blend="median" aria-checked="true" class="on">Median</button>
          <button type="button" role="radio" data-blend="mean" aria-checked="false">Mean</button>
        </div>
      </div>
    </div>
    <label class="day"><span id="p-day-lbl">Pilot field from</span>
      <select id="p-day"${c.days.length ? "" : " disabled"}>${days || "<option>no competition day on record here</option>"}</select></label>
    <p class="hint" id="p-day-hint"></p>
    <div class="btns">
      <button id="p-tpl"${c.days.length ? "" : " disabled"}>That day's task</button>
      <button id="p-clear">Clear</button>
      <button id="p-add" aria-pressed="false">Add on the map</button>
      <label class="file-btn">Import…<input type="file" id="p-file" accept=".xctsk,.cup,.wpt,.gpx,.txt,.json,image/*"></label>
      <button id="p-paste">Paste QR text</button>
    </div>
    <div class="paste" id="p-paste-box" hidden>
      <textarea id="p-paste-text" rows="3" placeholder="XCTSK:{…} or the contents of a task file"></textarea>
      <div class="btns"><button id="p-paste-go">Read</button></div>
    </div>
    <p class="plan-msg" id="p-msg" role="status"></p>
    <div class="tp-head" aria-hidden="true"><span>#</span><span>Name</span><span>Type</span><span>Radius m</span><span></span></div>
    <ol class="tp-list" id="p-list"></ol>
    <div class="tp-times">
      <label>Start gate<input type="time" id="p-start"></label>
      <label>Deadline<input type="time" id="p-deadline"></label>
      <label class="goal">Goal<select id="p-goal"><option value="CYLINDER">Cylinder</option><option value="LINE">Line</option></select></label>
    </div>
    <p class="hint" id="p-tz"></p>
    <p class="tp-dist" id="p-dist"></p>
    <div class="btns"><button class="primary" id="p-go" disabled>Predict this task</button></div>
    <div id="p-out"></div>
    <h5>Send it to instruments</h5>
    <div class="btns" id="p-export">
      <button data-x="qr">XCTrack QR</button><button data-x="xctsk">XCTrack .xctsk</button><button data-x="cup">SeeYou .cup</button>
      <button data-x="geo">Flymaster .wpt</button><button data-x="compe">CompeGPS .wpt</button><button data-x="gpx">GPX</button>
    </div>
    <div class="qr-box" id="p-qr" hidden>
      <canvas id="p-qr-canvas"></canvas>
      <div class="btns"><a class="btn-like" id="p-qr-png" download="task-qr.png">Download image</a><button id="p-qr-copy">Copy text</button></div>
      <p class="hint">Scan it with XCTrack's task import, or another app that reads XCTrack task codes.</p>
    </div>`;

  $("#p-close").addEventListener("click", collapse);
  box.querySelectorAll<HTMLButtonElement>("#p-wx-mode button[data-mode]").forEach((b) => {
    b.addEventListener("click", () => {
      const m = b.dataset.mode as WxMode;
      if (m === wxMode) return;
      wxMode = m;
      applyWxModeUI();
      staleResult();
      if (wxMode === "forecast") loadForecastModels();
    });
  });
  $<HTMLInputElement>("#p-fdate").addEventListener("change", (e) => {
    const v = (e.target as HTMLInputElement).value;
    if (!v) return;
    fcDate = v;
    staleResult();
    renderForecastControls();
    loadForecastModels();
  });
  box.querySelectorAll<HTMLButtonElement>("#p-blend-bar button[data-blend]").forEach((b) => {
    b.addEventListener("click", () => {
      fcBlendKind = b.dataset.blend as BlendKind;
      renderForecastControls();
      if (lastForecast && fcChoice === "blend") renderForecastResult();
    });
  });
  $<HTMLSelectElement>("#p-day").addEventListener("change", async (e) => {
    const d = c.days.find((x) => x.key === (e.target as HTMLSelectElement).value) ?? null;
    await setDay(d);
    if (el) changed(false);
  });
  $("#p-tpl").addEventListener("click", loadDayTask);
  $("#p-clear").addEventListener("click", () => { task = { ...X.emptyTask(), gates: task.gates, deadline: task.deadline }; insertAfter = -1; setAdding(true); changed(true); startMsg(); });
  $("#p-add").addEventListener("click", () => setAdding(!adding));
  $<HTMLInputElement>("#p-file").addEventListener("change", (e) => {
    const f = (e.target as HTMLInputElement).files?.[0];
    if (f) importFile(f);
    (e.target as HTMLInputElement).value = "";
  });
  $("#p-paste").addEventListener("click", () => { const b = $("#p-paste-box"); b.hidden = !b.hidden; if (!b.hidden) $<HTMLTextAreaElement>("#p-paste-text").focus(); });
  $("#p-paste-go").addEventListener("click", async () => {
    const text = $<HTMLTextAreaElement>("#p-paste-text").value;
    try { apply(await X.parseAny("pasted.txt", text)); $("#p-paste-box").hidden = true; }
    catch (e) { msg(`Could not read it: ${(e as Error).message}`); }
  });
  $<HTMLInputElement>("#p-start").addEventListener("change", (e) => {
    const v = (e.target as HTMLInputElement).value;
    task.gates = v ? [toUtc(v)] : [];
    staleResult();
  });
  $<HTMLInputElement>("#p-deadline").addEventListener("change", (e) => {
    const v = (e.target as HTMLInputElement).value;
    task.deadline = v ? toUtc(v) : null;
    staleResult();
  });
  $<HTMLSelectElement>("#p-goal").addEventListener("change", (e) => { task.goalType = (e.target as HTMLSelectElement).value as X.Task["goalType"]; changed(true); });
  $("#p-go").addEventListener("click", runPredict);
  $("#p-export").addEventListener("click", (e) => {
    const b = (e.target as HTMLElement).closest<HTMLButtonElement>("button[data-x]");
    if (b) exportAs(b.dataset.x!);
  });
  $("#p-qr-copy").addEventListener("click", () => navigator.clipboard?.writeText(X.toQR(named())).then(() => msg("QR text copied.")));

  applyWxModeUI();
  loadForecastModels();
  if (task.tps.length) { setAdding(false); msg(""); }
  else { setAdding(true); startMsg(); }
  changed(true);
  siteWaypoints(c.site.id).then((w) => {
    if (el !== box) return;
    siteWps = (w?.waypoints || []).map(([name, lat, lon, alt]) => ({ name, desc: "", lat, lon, alt: alt ?? 0 }));
    drawWaypoints();
  });
}

function startMsg() {
  msg(`Click the map to place the takeoff, then the start, the turnpoints and the goal${siteWps.length || ctx?.days.length ? "; the dots are waypoints this site's competitions used" : ""}.`);
}

function setAdding(on: boolean) {
  adding = on;
  if (el) {
    const b = $("#p-add");
    b.classList.toggle("on", on);
    b.setAttribute("aria-pressed", String(on));
    b.textContent = on ? "Adding on the map…" : "Add on the map";
  }
  if (map) map.getCanvas().style.cursor = on ? "crosshair" : "";
}

function msg(text: string) {
  if (el) $("#p-msg").textContent = text;
}

function select(i: number) {
  insertAfter = insertAfter === i ? -1 : i;
  renderList();
  drawMarkers();
  if (insertAfter >= 0) { setAdding(true); msg(`New turnpoints go after ${i + 1} (${task.tps[i].name}). Click the number again to add at the end.`); }
}

function add(tp: X.Turnpoint) {
  const at = insertAfter < 0 ? task.tps.length : insertAfter + 1;
  task.tps.splice(at, 0, tp);
  // a task built by clicks: takeoff first, the start next
  if (task.tps.length === 1) task.tps[0].type = "TAKEOFF";
  if (task.tps.length >= 2 && task.tps[0].type === "TAKEOFF" && !task.tps.some((p) => p.type === "SSS") && task.tps[1].type === "") {
    task.tps[1].type = "SSS";
  }
  if (insertAfter >= 0) insertAfter = at;
  changed(true);
}

/** Something changed: redraw the list, the map and the distances. */
function changed(list: boolean) {
  if (!el) return;
  if (list) renderList();
  if (!dragging) drawMarkers();
  showSetRoute(task.tps.length === 0);
  dist = task.tps.length >= 2 ? X.distances(task) : null;
  (map?.getSource("plan") as maplibregl.GeoJSONSource | undefined)?.setData(geometry());
  $<HTMLInputElement>("#p-start").value = toLocal(task.gates[0]);
  $<HTMLInputElement>("#p-deadline").value = toLocal(task.deadline);
  $<HTMLSelectElement>("#p-goal").value = task.goalType;
  $("#p-tz").textContent = `Times are local (${utcLabel()}); instruments get them in UTC.`;
  showDistances();
  staleResult();
}

function staleResult() {
  lastForecast = null;
  if (!el) return;
  $("#p-out").innerHTML = "";
  $("#p-qr").hidden = true;
  updateGo();
}

function showDistances() {
  if (!el) return;
  $("#p-dist").innerHTML = dist
    ? `Task <b>${km(dist.taskKm)}</b> · speed section <b>${km(dist.ssKm)}</b><span>optimized (FAI S7F, ${task.earth === "WGS84" ? "WGS84" : "FAI sphere"}); ${km(dist.centresKm)} through the centres</span>`
    : "";
}

function far(): boolean {
  return !!ctx && task.tps.some((p) => Math.abs(p.lat - ctx!.site.lat) > MAX_FROM_SITE_DEG || Math.abs(p.lon - ctx!.site.lon) > MAX_FROM_SITE_DEG);
}

async function updateGo() {
  if (!el) return;
  const go = $<HTMLButtonElement>("#p-go");
  const ok = API ? await apiAvailable() : false;
  if (!el) return;
  go.disabled = !(ok && day && task.tps.length >= 2 && !far());
  go.title = !day ? "No competition day on record at this site." : !ok ? "The prediction server is not reachable right now."
    : far() ? "The task is too far from this site to predict it with this site's days." : "";
}

function renderList() {
  if (!el) return;
  const ol = $("#p-list");
  const last = task.tps.length - 1;
  ol.innerHTML = task.tps.map((p, i) => `<li class="tp-row${i === insertAfter ? " sel" : ""}" data-i="${i}">
      <button class="tp-n" draggable="true" style="background:${COLORS[kind(i)]}" title="Add the next turnpoints after this one; drag to reorder">${i + 1}</button>
      <input class="tp-name" value="${esc(p.name)}" aria-label="Name of turnpoint ${i + 1}" spellcheck="false">
      <select class="tp-type" aria-label="Type of turnpoint ${i + 1}">
        <option value="TAKEOFF"${p.type === "TAKEOFF" ? " selected" : ""}>Takeoff</option>
        <option value="SSS"${p.type === "SSS" ? " selected" : ""}>Start</option>
        <option value=""${p.type === "" ? " selected" : ""}>${i === last && i > 0 ? "Goal" : "Turnpoint"}</option>
        <option value="ESS"${p.type === "ESS" ? " selected" : ""}>${i === last ? "ESS, goal" : "ESS"}</option>
      </select>
      <input class="tp-r" type="number" min="0" max="200000" step="100" value="${Math.round(p.radius)}" aria-label="Radius of turnpoint ${i + 1}, metres">
      <button class="tp-x" aria-label="Remove turnpoint ${i + 1}">✕</button>
    </li>`).join("");
  ol.querySelectorAll<HTMLLIElement>(".tp-row").forEach((li) => {
    const i = Number(li.dataset.i);
    li.querySelector(".tp-n")!.addEventListener("click", () => select(i));
    li.querySelector<HTMLInputElement>(".tp-name")!.addEventListener("input", (e) => {
      task.tps[i].name = (e.target as HTMLInputElement).value.trim();
      if (markers[i]) markers[i].getElement().title = task.tps[i].name;
      $("#p-qr").hidden = true;
    });
    li.querySelector<HTMLSelectElement>(".tp-type")!.addEventListener("change", (e) => {
      const v = (e.target as HTMLSelectElement).value as X.TpType;
      if (v) task.tps.forEach((p, k) => { if (k !== i && p.type === v) p.type = ""; });   // one takeoff, start and ESS
      task.tps[i].type = v;
      changed(true);
    });
    li.querySelector<HTMLInputElement>(".tp-r")!.addEventListener("input", (e) => {
      const v = Number((e.target as HTMLInputElement).value);
      if (!Number.isFinite(v) || v < 0) return;
      task.tps[i].radius = v;
      scheduleGeometry();
      staleResult();
    });
    li.querySelector(".tp-x")!.addEventListener("click", () => {
      task.tps.splice(i, 1);
      if (insertAfter >= task.tps.length || insertAfter === i) insertAfter = -1;
      changed(true);
    });
    // reorder by dragging the number
    li.querySelector(".tp-n")!.addEventListener("dragstart", (e) => { (e as DragEvent).dataTransfer?.setData("text/plain", String(i)); });
    li.addEventListener("dragover", (e) => { e.preventDefault(); li.classList.add("over"); });
    li.addEventListener("dragleave", () => li.classList.remove("over"));
    li.addEventListener("drop", (e) => {
      e.preventDefault();
      li.classList.remove("over");
      const from = Number((e as DragEvent).dataTransfer?.getData("text/plain"));
      if (!Number.isInteger(from) || from === i) return;
      const [p] = task.tps.splice(from, 1);
      task.tps.splice(i, 0, p);
      insertAfter = -1;
      changed(true);
    });
  });
}

// ---------------------------------------------------------------------------------------------
// Prediction

function sameRoute(a: X.Task, b: X.Task): boolean {
  return a.tps.length === b.tps.length && a.goalType === b.goalType && a.tps.every((p, i) => {
    const q = b.tps[i];
    return Math.abs(p.lat - q.lat) < 1e-5 && Math.abs(p.lon - q.lon) < 1e-5 && p.radius === q.radius && p.type === q.type;
  });
}

function medianOf(nums: number[]): number {
  const s = [...nums].sort((a, b) => a - b);
  const mid = Math.floor(s.length / 2);
  return s.length % 2 ? s[mid] : (s[mid - 1] + s[mid]) / 2;
}

function meanOf(nums: number[]): number {
  return nums.reduce((a, b) => a + b, 0) / nums.length;
}

function rangeBar(minV: number, maxV: number, val: number, scaleMin: number, scaleMax: number): string {
  const span = Math.max(scaleMax - scaleMin, 1e-6);
  const clamp = (x: number) => Math.max(0, Math.min(100, ((x - scaleMin) / span) * 100));
  const lo = clamp(minV), hi = clamp(maxV), pt = clamp(val);
  return `<div class="range-bar" aria-hidden="true"><span class="range-fill" style="left:${lo.toFixed(1)}%;width:${Math.max(2, hi - lo).toFixed(1)}%"></span><span class="range-dot" style="left:${pt.toFixed(1)}%"></span></div>`;
}

function windShort(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "–";
  const sign = v >= 0 ? "+" : "−";
  return `${sign}${Math.abs(v).toFixed(1)}`;
}

function weatherTooltip(m: ModelPrediction): string {
  const w = m.weather || {};
  const parts = [
    `${m.name} (${m.resolution_km} km)`,
    w.blh_mean != null ? `BLH ${Math.round(w.blh_mean)} m (${m.blh_source})` : null,
    w.usable_m != null ? `usable ${Math.round(w.usable_m)} m` : null,
    w.tailwind_700 != null ? `700 hPa ${w.tailwind_700 >= 0 ? "tail" : "head"} ${Math.abs(w.tailwind_700).toFixed(1)} m/s` : null,
    w.wind_700 != null ? `wind700 ${w.wind_700.toFixed(1)} m/s` : null,
    w.cloud_total != null ? `clouds ${Math.round(w.cloud_total * 100)}%` : null,
    w.cape != null ? `CAPE ${Math.round(w.cape)}` : null,
    w.rain_mm != null && w.rain_mm >= 0.1 ? `rain ${w.rain_mm.toFixed(1)} mm` : null,
  ].filter(Boolean);
  return parts.join(" · ");
}

function renderForecastResult() {
  if (!el || !lastForecast || !day) return;
  const out = $("#p-out");
  const all = lastForecast.models ?? [];
  if (!all.length) return;

  const isBlend = fcChoice === "blend";
  let active = isBlend ? all.filter((m) => fcIncluded.has(m.id)) : all.filter((m) => m.id === fcChoice);
  if (!active.length) active = isBlend ? all : [all[0]];

  const agg = fcBlendKind === "mean" ? meanOf : medianOf;
  const hours = active.map((m) => m.winner_h).filter((v): v is number => v != null && Number.isFinite(v));
  const goals = active.map((m) => m.goal_share);
  const headH = hours.length ? agg(hours) : null;
  const headGoal = agg(goals);
  const minH = hours.length ? Math.min(...hours) : null;
  const maxH = hours.length ? Math.max(...hours) : null;
  const minG = Math.min(...goals);
  const maxG = Math.max(...goals);

  // Scale for the time range bar: pad tightly around all fetched models so spread is legible.
  const allHours = all.map((m) => m.winner_h).filter((v): v is number => v != null && Number.isFinite(v));
  const tLo = allHours.length ? Math.max(0, Math.min(...allHours) - 0.25) : 0;
  const tHi = allHours.length ? Math.max(...allHours) + 0.25 : 4;

  const showRange = active.length >= 2 && minH != null && maxH != null && headH != null;
  const timeRangeHtml = showRange
    ? `<span class="range-txt">${hm(minH)} – ${hm(maxH)}</span>${rangeBar(minH!, maxH!, headH!, tLo, tHi)}`
    : "";
  const goalRangeHtml = active.length >= 2
    ? `<span class="range-txt">${pct(minG)} – ${pct(maxG)}</span>${rangeBar(minG, maxG, headGoal, 0, 1)}`
    : "";

  const label = isBlend
    ? `Blend (${fcBlendKind === "median" ? "median" : "mean"} of ${active.length} ${active.length === 1 ? "model" : "models"})`
    : `${active[0].name} (${active[0].resolution_km} km)`;

  const blhVals = active.map((m) => m.weather?.blh_mean).filter((v): v is number => v != null);
  const twVals = active.map((m) => m.weather?.tailwind_700).filter((v): v is number => v != null);
  const cldVals = active.map((m) => m.weather?.cloud_total).filter((v): v is number => v != null);
  const rainVals = active.map((m) => m.weather?.rain_mm).filter((v): v is number => v != null);
  const medBlh = blhVals.length ? Math.round(agg(blhVals)) : null;
  const medTw = twVals.length ? agg(twVals) : null;
  const medCld = cldVals.length ? Math.round(agg(cldVals) * 100) : null;
  const medRain = rainVals.length ? agg(rainVals) : null;
  const chips = active[0].weather?.route_chips;

  const hasParcel = all.some((m) => m.blh_source === "parcel");
  const rowsHtml = all.map((m) => {
    const checked = fcIncluded.has(m.id);
    const focused = !isBlend && m.id === fcChoice;
    const dimmed = isBlend && !checked;
    const w = m.weather || {};
    const blhStr = w.blh_mean == null ? "–" : `${m.blh_source === "parcel" ? "~" : ""}${Math.round(w.blh_mean)}m`;
    const rainTag = w.rain_mm != null && w.rain_mm >= 0.3 ? ` · ${w.rain_mm.toFixed(1)}mm` : "";
    return `<tr class="${focused ? "sel" : ""}${dimmed ? " dim" : ""}" title="${esc(weatherTooltip(m))}">
      <td class="fc-m">
        ${isBlend ? `<input type="checkbox" data-inc="${esc(m.id)}"${checked ? " checked" : ""} aria-label="Include ${esc(m.name)} in blend">` : ""}
        <button type="button" class="fc-pick" data-pick="${esc(m.id)}">${esc(m.name)}</button>
      </td>
      <td class="num">${hm(m.winner_h)}</td>
      <td class="num">${pct(m.goal_share)}</td>
      <td class="num wx-col">${blhStr} · ${windShort(w.tailwind_700)}${rainTag}</td>
    </tr>`;
  }).join("");

  const skippedHtml = lastForecast.skipped?.length
    ? `<p class="hint">Not included: ${esc(lastForecast.skipped.map((s) => `${s.id} (${s.reason})`).join(", "))}.</p>`
    : "";

  out.innerHTML = `<div class="whatif-out">
      <div><span class="lbl">Winner's time</span><b>${hm(headH)}</b>${timeRangeHtml}</div>
      <div><span class="lbl">In goal</span><b>${pct(headGoal)}</b>${goalRangeHtml}</div>
    </div>
    <p class="hint"><b>${esc(label)}</b> for ${esc(lastForecast.date ?? fcDate)} · pilots from ${esc(day.date)}.
      ${lastForecast.ss_km} km speed section · BLH ${medBlh == null ? "–" : `${medBlh} m`} ·
      ${medTw == null ? "–" : `${medTw >= 0 ? "tailwind" : "headwind"} ${Math.abs(medTw).toFixed(1)} m/s`} at 700 hPa ·
      ${medCld == null ? "–" : `clouds ${medCld}%`}${medRain != null && medRain >= 0.2 ? ` · rain ${medRain.toFixed(1)} mm` : ""} ·
      ${chips ?? "–"} weather cells.</p>
    <table class="fc-table">
      <thead><tr><th>Model</th><th class="num">Time</th><th class="num">Goal</th><th class="num">BLH · W700</th></tr></thead>
      <tbody>${rowsHtml}</tbody>
    </table>
    ${hasParcel ? `<p class="hint">~BLH estimated from the model's 925–600 hPa temperature profile. W700 is tailwind (+) / headwind (−) in m/s.</p>` : `<p class="hint">W700 is tailwind (+) / headwind (−) at 700 hPa in m/s.</p>`}
    ${skippedHtml}`;

  out.querySelectorAll<HTMLInputElement>("input[data-inc]").forEach((cb) => {
    cb.addEventListener("change", () => {
      const id = cb.dataset.inc!;
      if (cb.checked) fcIncluded.add(id);
      else if (fcIncluded.size > 1) fcIncluded.delete(id);
      else cb.checked = true;   // keep at least one model in the blend
      renderForecastResult();
    });
  });
  out.querySelectorAll<HTMLButtonElement>("button[data-pick]").forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.dataset.pick!;
      if (all.length < 2) return;
      fcChoice = fcChoice === id ? "blend" : id;
      renderForecastControls();
      renderForecastResult();
    });
  });
}

async function runPredict() {
  if (!ctx || !el || !day || task.tps.length < 2) return;
  const c = ctx, t = day;
  const go = $<HTMLButtonElement>("#p-go"), out = $("#p-out");
  go.disabled = true;

  // the speed section on the same footing as the task's official one: the official distance for the
  // task as set (so it predicts exactly as on the card), otherwise scaled by the ratio between the
  // official distance and this page's measure of the task as set, when the two agree
  let ss = dist?.ssKm ?? undefined;
  if (ss != null && t.km) {
    if (tplTask && sameRoute(task, tplTask)) ss = t.km;
    else if (tplSS && t.km / tplSS > 0.8 && t.km / tplSS < 1.25) ss *= t.km / tplSS;
  }
  // a new start time moves the weather window; solar time = UTC + longitude / 15
  let startSolar: number | undefined;
  if (task.gates[0] && (wxMode === "forecast" || task.gates[0] !== tplGate)) {
    const [h, m] = task.gates[0].split(":").map(Number);
    startSolar = (((h + m / 60 + c.site.lon / 15) % 24) + 24) % 24;
  }
  let windowH: number | undefined;
  if (task.gates[0] && task.deadline) {
    const [sh, sm] = task.gates[0].split(":").map(Number);
    const [eh, em] = task.deadline.split(":").map(Number);
    const diff = (((eh * 60 + em) - (sh * 60 + sm)) + 1440) % 1440;
    if (diff >= 30 && diff <= 600) windowH = Math.round((diff / 60) * 100) / 100;
  }

  const route = task.tps.map((p) => [p.lat, p.lon, p.radius] as [number, number, number]);

  if (wxMode === "forecast") {
    const modelIds = fcChoice === "blend"
      ? fcModels.filter((m) => m.available).map((m) => m.id)
      : [fcChoice];
    out.innerHTML = `<p class="hint">Fetching ${fcChoice === "blend" ? `${modelIds.length} forecast models` : esc(modelLabel(fcChoice))} for ${esc(fcDate)} along your route…</p>`;
    try {
      const j = await predictRoute({
        site: c.site.id,
        template: t.key,
        route,
        ss_km: ss != null ? Math.round(ss * 100) / 100 : undefined,
        start_solar_h: startSolar,
        window_h: windowH,
        weather: { source: "forecast", date: fcDate, models: modelIds },
      });
      if (!el) return;
      lastForecast = j;
      if (fcChoice === "blend") {
        const returnedIds = (j.models ?? []).map((m) => m.id);
        const kept = returnedIds.filter((id) => fcIncluded.has(id));
        fcIncluded = new Set(kept.length ? kept : returnedIds);
      }
      renderForecastResult();
    } catch (e) {
      if (el) out.innerHTML = `<p class="hint">The forecast prediction failed: ${esc(String((e as Error).message || e))}</p>`;
    }
    updateGo();
    return;
  }

  out.innerHTML = `<p class="hint">Reading the weather of ${t.date} along your route…</p>`;
  try {
    const j = await predictRoute({
      site: c.site.id,
      template: t.key,
      route,
      ss_km: ss != null ? Math.round(ss * 100) / 100 : undefined,
      start_solar_h: startSolar,
      window_h: windowH,
      weather: { source: "era5" },
    });
    if (!el) return;
    const w = j.weather || {};
    const asSet = t.pred_kmh && t.km ? t.km / t.pred_kmh : null;
    out.innerHTML = `<div class="whatif-out"><div><span class="lbl">Winner's time</span><b>${hm(j.winner_h)}</b></div>
        <div><span class="lbl">In goal</span><b>${pct(j.goal_share)}</b></div></div>
      <p class="hint">With the weather and pilots of ${esc(t.date)}. For that day's task as it was set the model predicted ${hm(asSet)} and ${pct(t.pred_goal)}.
        ${j.ss_km} km speed section · boundary layer ${w.blh_mean == null ? "–" : Math.round(w.blh_mean)} m ·
        ${w.tailwind_700 == null ? "–" : `${w.tailwind_700 >= 0 ? "tailwind" : "headwind"} ${Math.abs(w.tailwind_700).toFixed(1)} m/s`} at 700 hPa ·
        ${w.route_chips ?? "–"} weather cells along the route.</p>`;
  } catch (e) {
    if (el) out.innerHTML = `<p class="hint">The prediction failed: ${esc(String((e as Error).message || e))}</p>`;
  }
  updateGo();
}

// ---------------------------------------------------------------------------------------------
// Import and export

function apply(p: X.Parsed) {
  if (p.waypoints.length) { extraWps = p.waypoints; drawWaypoints(); }
  if (p.task && p.task.tps.length) {
    task = p.task;
    insertAfter = -1;
    setAdding(false);
    changed(true);
    fit();
    msg(`${task.tps.length} turnpoints from ${p.format}.${far() ? " It is far from this site: it can be exported, not predicted with this site's days." : ""}`);
  } else if (p.waypoints.length) {
    setAdding(true);
    msg(`${p.waypoints.length} waypoints from ${p.format} (purple dots): click them on the map to add turnpoints.`);
    const b = new maplibregl.LngLatBounds();
    p.waypoints.forEach((w) => b.extend([w.lon, w.lat]));
    map?.fitBounds(b, { padding: 50, duration: 700, maxZoom: 12 });
  } else {
    msg(`Nothing to read in that ${p.format} file.`);
  }
}

async function importFile(f: File) {
  try {
    let text: string;
    if (f.type.startsWith("image/")) {
      msg("Looking for a QR code in the image…");
      text = await readQR(f);
    } else {
      const buf = await f.arrayBuffer();
      try { text = new TextDecoder("utf-8", { fatal: true }).decode(buf); }
      catch { text = new TextDecoder("windows-1252").decode(buf); }   // older instruments write Latin-1
    }
    apply(await X.parseAny(f.name, text));
  } catch (e) {
    msg(`Could not read ${f.name}: ${(e as Error).message}`);
  }
}

async function readQR(f: File): Promise<string> {
  const bmp = await createImageBitmap(f);
  const Detector = (window as unknown as { BarcodeDetector?: new (o: object) => { detect: (s: ImageBitmap) => Promise<{ rawValue: string }[]> } }).BarcodeDetector;
  if (Detector) {
    try {
      const codes = await new Detector({ formats: ["qr_code"] }).detect(bmp);
      if (codes[0]?.rawValue) return codes[0].rawValue;
    } catch { /* fall back to jsQR */ }
  }
  const { default: jsQR } = await import("jsqr");
  for (const size of [1600, 1000, 2400, 700]) {
    const s = Math.min(1, size / Math.max(bmp.width, bmp.height));
    const w = Math.round(bmp.width * s), h = Math.round(bmp.height * s);
    const c = document.createElement("canvas");
    c.width = w; c.height = h;
    const g = c.getContext("2d", { willReadFrequently: true })!;
    g.drawImage(bmp, 0, 0, w, h);
    const r = jsQR(g.getImageData(0, 0, w, h).data, w, h, { inversionAttempts: "attemptBoth" });
    if (r) return new TextDecoder().decode(new Uint8Array(r.binaryData));
  }
  throw new Error("no QR code found in the image");
}

/** The task with a name on every turnpoint, as instruments need. */
function named(): X.Task {
  return { ...task, tps: task.tps.map((p, i) => ({ ...p, name: p.name || `TP${pad2(i + 1)}` })) };
}

function fileBase(): string {
  const c = ctx!;
  const dStr = wxMode === "forecast" ? fcDate : day?.date;
  const s = `${c.site.name}${dStr ? `-${dStr}` : ""}`.normalize("NFKD").replace(/[^\w.-]+/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "");
  return `${s || "task"}-task`;
}

function download(name: string, data: string | Uint8Array, type: string) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([data as BlobPart], { type }));
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}

async function exportAs(kind: string) {
  if (task.tps.length < 2) return msg("A task needs at least two turnpoints.");
  const t = named(), base = fileBase();
  switch (kind) {
    case "qr": {
      const text = X.toQR(t);
      const canvas = $<HTMLCanvasElement>("#p-qr-canvas");
      await QRCode.toCanvas(canvas, text, { errorCorrectionLevel: text.length > 900 ? "L" : "M", margin: 2, width: 280 });
      $<HTMLAnchorElement>("#p-qr-png").href = canvas.toDataURL("image/png");
      $<HTMLAnchorElement>("#p-qr-png").download = `${base}-qr.png`;
      $("#p-qr").hidden = false;
      $("#p-qr").scrollIntoView({ block: "nearest", behavior: "smooth" });
      return;
    }
    case "xctsk": return download(`${base}.xctsk`, X.toXctsk(t), "application/xctsk");
    case "cup": return download(`${base}.cup`, X.latin1(X.toCup(t)), "text/plain");
    case "geo": return download(`${base}-flymaster.wpt`, X.latin1(X.toGeoWpt(t)), "text/plain");
    case "compe": return download(`${base}-compegps.wpt`, X.latin1(X.toCompeGpsWpt(t)), "text/plain");
    case "gpx": return download(`${base}.gpx`, X.toGpx(t), "application/gpx+xml");
  }
}
