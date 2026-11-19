// Task planner: set a competition task on the map (turnpoints, cylinders, start and deadline),
// import it from instruments' files or an XCTrack QR code, export it back to them, and ask the API
// what the task model predicts for it with the weather and pilots of a competition day at the site.

import * as maplibregl from "maplibre-gl";
import QRCode from "qrcode";
import * as X from "./xctask";
import { API, apiAvailable, predictRoute } from "./api";

/** A past competition day at the site: its weather and pilots are what a new task is predicted with. */
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

const COLORS: Record<Kind, string> = { takeoff: "#7a7873", sss: "#2f7d3a", tp: "#1d4f99", ess: "#b26a00", goal: "#cf5222" };
const EMPTY: GeoJSON.FeatureCollection = { type: "FeatureCollection", features: [] };
const MAX_FROM_SITE_DEG = 3;
const SNAP_PX = 12;

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
let insertAfter = -1;               // map clicks insert after this row; -1 appends
let siteWps: X.Waypoint[] = [];
let extraWps: X.Waypoint[] = [];
let utc = 0;                        // local time minus UTC at the site, hours
let tplTask: X.Task | null = null; // the task as it was set
let tplSS: number | null = null;    // this page's optimized speed section of the task as set
let tplGate: string | null = null;  // its first start gate (UTC)
let raf = 0;
let draggedAt = 0;                // a drag ends with a click on the marker: ignore it
const cache = new Map<string, Promise<SiteWaypoints | null>>();

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
  box.hidden = false;
  setDay(ctx.days[0] ?? null);
  expand();
}

/** Opens a past task in the planner, on its own day. */
export async function planFrom(key: string) {
  if (!ctx || !root) return;
  const d = ctx.days.find((x) => x.key === key);
  if (!d) return;
  if (!el) expand();
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

function expand() {
  const box = root!, c = ctx!;
  el = box;
  box.classList.add("open");
  const days = c.days.map((d) => `<option value="${esc(d.key)}"${d === day ? " selected" : ""}>${esc(dayLabel(d))}</option>`).join("");
  box.innerHTML = `<div class="plan-head"><h3>Create a task</h3><button class="close" id="p-close" aria-label="Close the planner">✕</button></div>
    <label class="day">Weather and pilots of
      <select id="p-day"${c.days.length ? "" : " disabled"}>${days || "<option>no competition day on record here</option>"}</select></label>
    <p class="hint">${c.days.length
      ? "Until forecasts are in, a task is predicted with the weather and the pilots of a past competition day at this site."
      : "No competition day on record at this site: tasks can be set and exported, not predicted."}</p>
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

async function runPredict() {
  if (!ctx || !el || !day || task.tps.length < 2) return;
  const c = ctx, t = day;
  const go = $<HTMLButtonElement>("#p-go"), out = $("#p-out");
  go.disabled = true;
  out.innerHTML = `<p class="hint">Reading the weather of ${t.date} along your route…</p>`;
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
  if (task.gates[0] && task.gates[0] !== tplGate) {
    const [h, m] = task.gates[0].split(":").map(Number);
    startSolar = (((h + m / 60 + c.site.lon / 15) % 24) + 24) % 24;
  }
  try {
    const j = await predictRoute({ site: c.site.id, template: t.key, route: task.tps.map((p) => [p.lat, p.lon, p.radius]),
      ss_km: ss != null ? Math.round(ss * 100) / 100 : undefined, start_solar_h: startSolar });
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
  const s = `${c.site.name}${day ? `-${day.date}` : ""}`.normalize("NFKD").replace(/[^\w.-]+/g, "-").replace(/-+/g, "-").replace(/^-|-$/g, "");
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
