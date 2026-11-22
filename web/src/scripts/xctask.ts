// Competition tasks: turnpoints with cylinders, read from and written to the files and QR codes that
// flight instruments and apps use, and the optimized distances of FAI Sporting Code S7F §7.2.
//
// Formats: XCTrack task (.xctsk, JSON v1) and its QR code (XCTSK: / XCTSKZ:, v2), SeeYou (.cup),
// waypoint files (.wpt: Flymaster GEO, CompeGPS, OziExplorer) and GPX. No DOM here, so it runs in
// the browser and in tests alike.

export type TpType = "" | "TAKEOFF" | "SSS" | "ESS";
export type Turnpoint = { name: string; desc: string; lat: number; lon: number; alt: number; radius: number; type: TpType };
export type Waypoint = { name: string; desc: string; lat: number; lon: number; alt: number };
export type Task = {
  tps: Turnpoint[];
  sssType: "RACE" | "ELAPSED-TIME";
  gates: string[];            // start gates, UTC "HH:MM:SS"
  deadline: string | null;    // UTC "HH:MM:SS"
  goalType: "CYLINDER" | "LINE";
  earth: "WGS84" | "FAI_SPHERE";
};
export type Parsed = { task: Task | null; waypoints: Waypoint[]; format: string };
type LL = [number, number];   // lat, lon

export const emptyTask = (): Task => ({ tps: [], sssType: "RACE", gates: [], deadline: null, goalType: "CYLINDER", earth: "WGS84" });

// ---------------------------------------------------------------------------------------------
// Geodesy: WGS84 ellipsoid (Vincenty) and the FAI sphere

const D2R = Math.PI / 180;
const WA = 6378137, WF = 1 / 298.257223563, WB = WA * (1 - WF);
const FAI_R = 6371000;

function sphereDist([la1, lo1]: LL, [la2, lo2]: LL): number {
  const p1 = la1 * D2R, p2 = la2 * D2R, dp = p2 - p1, dl = (lo2 - lo1) * D2R;
  const h = Math.sin(dp / 2) ** 2 + Math.cos(p1) * Math.cos(p2) * Math.sin(dl / 2) ** 2;
  return 2 * FAI_R * Math.asin(Math.min(1, Math.sqrt(h)));
}

function sphereAz([la1, lo1]: LL, [la2, lo2]: LL): number {
  const p1 = la1 * D2R, p2 = la2 * D2R, dl = (lo2 - lo1) * D2R;
  return Math.atan2(Math.sin(dl) * Math.cos(p2), Math.cos(p1) * Math.sin(p2) - Math.sin(p1) * Math.cos(p2) * Math.cos(dl)) / D2R;
}

function sphereDest([la, lo]: LL, az: number, d: number): LL {
  const p1 = la * D2R, a = az * D2R, s = d / FAI_R;
  const p2 = Math.asin(Math.sin(p1) * Math.cos(s) + Math.cos(p1) * Math.sin(s) * Math.cos(a));
  const l2 = lo * D2R + Math.atan2(Math.sin(a) * Math.sin(s) * Math.cos(p1), Math.cos(s) - Math.sin(p1) * Math.sin(p2));
  return [p2 / D2R, ((l2 / D2R + 540) % 360) - 180];
}

function vincentyInverse(a: LL, b: LL): number {
  const [la1, lo1] = a, [la2, lo2] = b;
  if (la1 === la2 && lo1 === lo2) return 0;
  const L = (lo2 - lo1) * D2R;
  const tanU1 = (1 - WF) * Math.tan(la1 * D2R), cosU1 = 1 / Math.sqrt(1 + tanU1 * tanU1), sinU1 = tanU1 * cosU1;
  const tanU2 = (1 - WF) * Math.tan(la2 * D2R), cosU2 = 1 / Math.sqrt(1 + tanU2 * tanU2), sinU2 = tanU2 * cosU2;
  let lam = L, lamP = 0, it = 0;
  let sinS = 0, cosS = 0, sig = 0, cos2A = 0, cos2Sm = 0;
  do {
    const sinL = Math.sin(lam), cosL = Math.cos(lam);
    const sinSqS = (cosU2 * sinL) ** 2 + (cosU1 * sinU2 - sinU1 * cosU2 * cosL) ** 2;
    if (sinSqS === 0) return 0;
    sinS = Math.sqrt(sinSqS);
    cosS = sinU1 * sinU2 + cosU1 * cosU2 * cosL;
    sig = Math.atan2(sinS, cosS);
    const sinA = (cosU1 * cosU2 * sinL) / sinS;
    cos2A = 1 - sinA * sinA;
    cos2Sm = cos2A !== 0 ? cosS - (2 * sinU1 * sinU2) / cos2A : 0;
    const C = (WF / 16) * cos2A * (4 + WF * (4 - 3 * cos2A));
    lamP = lam;
    lam = L + (1 - C) * WF * sinA * (sig + C * sinS * (cos2Sm + C * cosS * (-1 + 2 * cos2Sm * cos2Sm)));
  } while (Math.abs(lam - lamP) > 1e-12 && ++it < 200);
  if (it >= 200) return sphereDist(a, b);   // nearly antipodal: never in a task
  const uSq = (cos2A * (WA * WA - WB * WB)) / (WB * WB);
  const A = 1 + (uSq / 16384) * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)));
  const B = (uSq / 1024) * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)));
  const dS = B * sinS * (cos2Sm + (B / 4) * (cosS * (-1 + 2 * cos2Sm * cos2Sm) -
    (B / 6) * cos2Sm * (-3 + 4 * sinS * sinS) * (-3 + 4 * cos2Sm * cos2Sm)));
  return WB * A * (sig - dS);
}

function vincentyDirect([la, lo]: LL, az: number, d: number): LL {
  const a1 = az * D2R, sinA1 = Math.sin(a1), cosA1 = Math.cos(a1);
  const tanU1 = (1 - WF) * Math.tan(la * D2R), cosU1 = 1 / Math.sqrt(1 + tanU1 * tanU1), sinU1 = tanU1 * cosU1;
  const s1 = Math.atan2(tanU1, cosA1);
  const sinA = cosU1 * sinA1, cosSqA = 1 - sinA * sinA;
  const uSq = (cosSqA * (WA * WA - WB * WB)) / (WB * WB);
  const A = 1 + (uSq / 16384) * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)));
  const B = (uSq / 1024) * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)));
  let sig = d / (WB * A), sigP = 0, it = 0, cos2Sm = 0, sinS = 0, cosS = 0;
  do {
    cos2Sm = Math.cos(2 * s1 + sig);
    sinS = Math.sin(sig);
    cosS = Math.cos(sig);
    const dS = B * sinS * (cos2Sm + (B / 4) * (cosS * (-1 + 2 * cos2Sm * cos2Sm) -
      (B / 6) * cos2Sm * (-3 + 4 * sinS * sinS) * (-3 + 4 * cos2Sm * cos2Sm)));
    sigP = sig;
    sig = d / (WB * A) + dS;
  } while (Math.abs(sig - sigP) > 1e-12 && ++it < 100);
  cos2Sm = Math.cos(2 * s1 + sig);
  sinS = Math.sin(sig);
  cosS = Math.cos(sig);
  const x = sinU1 * sinS - cosU1 * cosS * cosA1;
  const la2 = Math.atan2(sinU1 * cosS + cosU1 * sinS * cosA1, (1 - WF) * Math.sqrt(sinA * sinA + x * x));
  const lam = Math.atan2(sinS * sinA1, cosU1 * cosS - sinU1 * sinS * cosA1);
  const C = (WF / 16) * cosSqA * (4 + WF * (4 - 3 * cosSqA));
  const L = lam - (1 - C) * WF * sinA * (sig + C * sinS * (cos2Sm + C * cosS * (-1 + 2 * cos2Sm * cos2Sm)));
  return [la2 / D2R, ((lo + L / D2R + 540) % 360) - 180];
}

type Geo = { dist: (a: LL, b: LL) => number; dest: (c: LL, az: number, d: number) => LL };
const SPHERE: Geo = { dist: sphereDist, dest: sphereDest };
const WGS84: Geo = { dist: vincentyInverse, dest: vincentyDirect };

/** Distance in metres on the task's earth model. */
export const distance = (a: LL, b: LL, earth: Task["earth"] = "WGS84") => (earth === "WGS84" ? WGS84 : SPHERE).dist(a, b);
/** Initial azimuth from one point to another, degrees (sphere: for drawing). */
export const bearing = (a: LL, b: LL) => sphereAz(a, b);
/** A point at a distance and azimuth from another (for drawing cylinders). */
export const destination = (c: LL, az: number, d: number) => vincentyDirect(c, az, d);

// ---------------------------------------------------------------------------------------------
// Optimized route: the shortest path from launch that touches every cylinder's edge in order
// (S7F §7.2). Coordinate descent over each point's azimuth on its cylinder, from three starts
// (centres, chains forward and backward along the edges), on the sphere first and then refined on
// the ellipsoid. Takes about a millisecond per turnpoint.

const GOLD = (Math.sqrt(5) - 1) / 2;

function bestAz(geo: Geo, c: LL, r: number, a: LL, b: LL | null, from: number, to: number, step: number): number {
  const f = (az: number) => { const p = geo.dest(c, az, r); return geo.dist(a, p) + (b ? geo.dist(p, b) : 0); };
  let best = from, fb = Infinity;
  for (let az = from; az <= to + 1e-9; az += step) {
    const v = f(az);
    if (v < fb) { fb = v; best = az; }
  }
  let lo = best - step, hi = best + step;
  let x1 = hi - GOLD * (hi - lo), x2 = lo + GOLD * (hi - lo), f1 = f(x1), f2 = f(x2);
  const tol = Math.max(1e-7, 0.002 / Math.max(r, 1) / D2R);   // about 2 mm on the edge
  while (hi - lo > tol) {
    if (f1 < f2) { hi = x2; x2 = x1; f2 = f1; x1 = hi - GOLD * (hi - lo); f1 = f(x1); }
    else { lo = x1; x1 = x2; f1 = f2; x2 = lo + GOLD * (hi - lo); f2 = f(x2); }
  }
  const az = (lo + hi) / 2;
  return f(az) <= fb ? az : best;
}

type Zone = { c: LL; r: number; fixed: boolean };   // fixed: the point is the centre (launch, line goal, radius 0)

function descend(geo: Geo, zones: Zone[], az: number[], window: number | null, maxPass: number): LL[] {
  const pts: LL[] = zones.map((z, i) => (z.fixed ? z.c : geo.dest(z.c, az[i], z.r)));
  let prev = Infinity;
  for (let pass = 0; pass < maxPass; pass++) {
    for (let i = 1; i < zones.length; i++) {
      const z = zones[i];
      if (z.fixed) continue;
      const b = i + 1 < zones.length ? pts[i + 1] : null;
      const w = window ?? (pass === 0 ? 180 : 20);
      const step = w >= 180 ? 10 : Math.max(w / 6, 0.05);
      az[i] = bestAz(geo, z.c, z.r, pts[i - 1], b, az[i] - w, az[i] + w, step);
      pts[i] = geo.dest(z.c, az[i], z.r);
    }
    const len = pathLength(geo, pts);
    if (prev - len < 0.001) break;
    prev = len;
  }
  return pts;
}

function pathLength(geo: Geo, pts: LL[]): number {
  let s = 0;
  for (let i = 1; i < pts.length; i++) s += geo.dist(pts[i - 1], pts[i]);
  return s;
}

function shortest(zones: Zone[], earth: Task["earth"]): LL[] {
  if (zones.length < 2) return zones.map((z) => z.c);
  const n = zones.length;
  const toward = (i: number, p: LL) => sphereAz(zones[i].c, p);
  const starts: number[][] = [
    zones.map((z, i) => (i + 1 < n ? sphereAz(z.c, zones[i + 1].c) : i > 0 ? sphereAz(z.c, zones[i - 1].c) : 0)),
  ];
  // forward chain: each point on its edge nearest the previous one
  const fwd: number[] = new Array(n).fill(0);
  let p: LL = zones[0].c;
  for (let i = 1; i < n; i++) { fwd[i] = toward(i, p); p = zones[i].fixed ? zones[i].c : sphereDest(zones[i].c, fwd[i], zones[i].r); }
  starts.push(fwd);
  // backward chain, from the goal
  const bwd: number[] = new Array(n).fill(0);
  p = zones[n - 1].c;
  for (let i = n - 2; i >= 1; i--) { bwd[i] = toward(i, p); p = zones[i].fixed ? zones[i].c : sphereDest(zones[i].c, bwd[i], zones[i].r); }
  bwd[n - 1] = toward(n - 1, p);
  starts.push(bwd);

  let best: { pts: LL[]; az: number[]; len: number } | null = null;
  for (const s of starts) {
    const az = s.slice();
    const pts = descend(SPHERE, zones, az, null, 60);
    const len = pathLength(SPHERE, pts);
    if (!best || len < best.len) best = { pts, az, len };
  }
  if (earth === "FAI_SPHERE") return best!.pts;
  // refine on the ellipsoid near the sphere's answer
  return descend(WGS84, zones, best!.az, 2, 6);
}

export type Distances = {
  taskKm: number | null;          // launch to goal, optimized
  ssKm: number | null;            // start of speed section to ESS, optimized (S7F §7.2)
  centresKm: number | null;       // through every centre, launch to goal
  route: LL[];                    // the optimized route's points, for drawing
};

function zonesOf(task: Task, upTo: number): Zone[] {
  return task.tps.slice(0, upTo + 1).map((t, i) => ({
    c: [t.lat, t.lon] as LL, r: t.radius,
    fixed: i === 0 || t.radius <= 0 || (i === task.tps.length - 1 && i === upTo && task.goalType === "LINE"),
  }));
}

/** Optimized task and speed-section distances. */
export function distances(task: Task): Distances {
  const tps = task.tps;
  if (tps.length < 2) return { taskKm: null, ssKm: null, centresKm: null, route: tps.map((t) => [t.lat, t.lon]) };
  const geo = task.earth === "WGS84" ? WGS84 : SPHERE;
  const route = shortest(zonesOf(task, tps.length - 1), task.earth);
  const taskKm = pathLength(geo, route) / 1000;
  let centres = 0;
  for (let i = 1; i < tps.length; i++) centres += geo.dist([tps[i - 1].lat, tps[i - 1].lon], [tps[i].lat, tps[i].lon]);
  // speed section: the optimized path from launch to ESS, less the part before the start
  const sss = tps.findIndex((t) => t.type === "SSS");
  let ess = tps.findIndex((t) => t.type === "ESS");
  if (ess < 0) ess = tps.length - 1;
  let ssKm: number | null = null;
  if (ess > Math.max(sss, 0)) {
    const toEss = ess === tps.length - 1 ? route : shortest(zonesOf(task, ess), task.earth);
    ssKm = pathLength(geo, toEss.slice(Math.max(sss, 0))) / 1000;
  }
  return { taskKm, ssKm, centresKm: centres / 1000, route };
}

// ---------------------------------------------------------------------------------------------
// XCTrack: .xctsk (JSON v1) and QR (XCTSK: JSON v2 with polyline-encoded turnpoints)

function encodeNum(num: number): string {
  let p = num < 0 ? ~(num << 1) : num << 1;
  let out = "";
  while (p >= 0x20) { out += String.fromCharCode((0x20 | (p & 0x1f)) + 63); p >>= 5; }
  return out + String.fromCharCode(p + 63);
}

function decodeNums(s: string): number[] {
  const out: number[] = [];
  let cur = 0, shift = 0;
  for (let i = 0; i < s.length; i++) {
    const c = s.charCodeAt(i) - 63;
    if (c < 0 || c > 63) throw new Error(`not a polyline character at ${i}`);
    cur |= (c & 0x1f) << shift;
    shift += 5;
    if (c < 0x20) { out.push(cur & 1 ? ~(cur >> 1) : cur >> 1); cur = 0; shift = 0; }
  }
  if (shift) throw new Error("unterminated polyline number");
  return out;
}

const roundHalfUp = (v: number) => Math.floor(v + 0.5);
const hms = (s: string | null | undefined) => {
  const m = /^(\d{1,2}):(\d{2})(?::(\d{2}))?/.exec(s || "");
  return m ? `${m[1].padStart(2, "0")}:${m[2]}:${m[3] || "00"}` : null;
};

/** Fills in what instruments expect: a start (SSS) and an ESS (the goal when none is set). Without
 * a start, the first point is the launch and the second the start, as in most tasks. */
export function normalized(task: Task): Task {
  const tps = task.tps.map((t) => ({ ...t }));
  if (tps.length >= 2 && !tps.some((t) => t.type === "ESS")) tps[tps.length - 1].type = "ESS";
  if (tps.length >= 2 && !tps.some((t) => t.type === "SSS")) {
    if (tps.length > 2 && (tps[0].type === "TAKEOFF" || tps[0].type === "")) {
      tps[0].type = "TAKEOFF";
      if (tps[1].type === "") tps[1].type = "SSS";
    } else if (tps[0].type === "") tps[0].type = "SSS";
  }
  return { ...task, tps };
}

export function toXctsk(task: Task): string {
  const t = normalized(task);
  const out: Record<string, unknown> = {
    taskType: "CLASSIC", version: 1, earthModel: t.earth,
    turnpoints: t.tps.map((p) => ({
      ...(p.type ? { type: p.type } : {}),
      radius: Math.round(p.radius),
      waypoint: { name: p.name, ...(p.desc ? { description: p.desc } : {}), lat: +p.lat.toFixed(6), lon: +p.lon.toFixed(6), altSmoothed: Math.round(p.alt) },
    })),
    sss: { type: t.sssType, direction: "EXIT", timeGates: t.gates.map((g) => `${g}Z`) },
    goal: { type: t.goalType, ...(t.deadline ? { deadline: `${t.deadline}Z` } : {}) },
  };
  return JSON.stringify(out, null, 2);
}

function fromXctskObject(o: any): Task {
  if (!o || !Array.isArray(o.turnpoints)) throw new Error("not an XCTrack task");
  const tps: Turnpoint[] = o.turnpoints.map((p: any) => {
    const w = p.waypoint || {};
    if (typeof w.lat !== "number" || typeof w.lon !== "number") throw new Error("a turnpoint has no coordinates");
    return { name: String(w.name ?? ""), desc: String(w.description ?? ""), lat: w.lat, lon: w.lon, alt: Number(w.altSmoothed) || 0,
      radius: Number(p.radius) || 0, type: (["TAKEOFF", "SSS", "ESS"].includes(p.type) ? p.type : "") as TpType };
  });
  return {
    tps,
    sssType: o.sss?.type === "ELAPSED-TIME" ? "ELAPSED-TIME" : "RACE",
    gates: (o.sss?.timeGates || []).map(hms).filter(Boolean) as string[],
    deadline: hms(o.goal?.deadline),
    goalType: o.goal?.type === "LINE" ? "LINE" : "CYLINDER",
    earth: o.earthModel === "FAI_SPHERE" ? "FAI_SPHERE" : "WGS84",
  };
}

export function parseXctsk(text: string): Task {
  const o = JSON.parse(text);
  return Array.isArray(o.turnpoints) ? fromXctskObject(o) : fromV2(o);
}

/** The QR code's text, as XCTrack writes it. */
export function toQR(task: Task): string {
  const t = normalized(task);
  const tt: Record<string, number> = { SSS: 2, ESS: 3 };
  const o: Record<string, unknown> = {
    taskType: "CLASSIC", version: 2,
    t: t.tps.map((p) => ({
      z: encodeNum(roundHalfUp(p.lon * 1e5)) + encodeNum(roundHalfUp(p.lat * 1e5)) + encodeNum(Math.round(p.alt)) + encodeNum(Math.round(p.radius)),
      n: p.name, ...(p.desc ? { d: p.desc } : {}), ...(tt[p.type] ? { t: tt[p.type] } : {}),
    })),
    s: { d: 2, g: t.gates.map((g) => `${g}Z`), t: t.sssType === "RACE" ? 1 : 2 },
    g: { ...(t.deadline ? { d: `${t.deadline}Z` } : {}), t: t.goalType === "LINE" ? 1 : 2 },
    ...(t.earth === "FAI_SPHERE" ? { e: 1 } : {}),
  };
  return "XCTSK:" + JSON.stringify(o);
}

async function inflate(b64: string): Promise<string> {
  let clean = b64.replace(/\s+/g, "").replace(/-/g, "+").replace(/_/g, "/");
  while (clean.length % 4) clean += "=";
  const bytes = Uint8Array.from(atob(clean), (c) => c.charCodeAt(0));
  let lastErr: unknown;
  for (const fmt of ["deflate", "deflate-raw", "gzip"] as CompressionFormat[]) {
    try {
      const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream(fmt));
      return await new Response(stream).text();
    } catch (err) {
      lastErr = err;
    }
  }
  throw lastErr instanceof Error ? lastErr : new Error("could not decompress XCTSKZ payload");
}

/** The v2 task (QR codes): a competition task, or a plain route of waypoints ("T": "W"). */
function fromV2(o: any): Task {
  if (!o || !Array.isArray(o.t)) throw new Error("not an XCTrack task");
  const route = o.T === "W";
  const types: Record<number, TpType> = { 1: "TAKEOFF", 2: "SSS", 3: "ESS" };
  const tps: Turnpoint[] = o.t.map((p: any) => {
    const n = decodeNums(String(p.z || ""));
    if (n.length < 3) throw new Error("a turnpoint has no coordinates");
    return { name: String(p.n ?? ""), desc: String(p.d ?? ""), lon: n[0] / 1e5, lat: n[1] / 1e5, alt: n[2],
      radius: route ? 400 : n[3] ?? 0, type: types[p.t] ?? "" };
  });
  // XCTrack does not mark the takeoff in a QR code: the first point before the start is the launch
  const sss = tps.findIndex((p) => p.type === "SSS");
  if ((sss > 0 || route) && tps.length && !tps.some((p) => p.type === "TAKEOFF")) tps[0].type = "TAKEOFF";
  return {
    tps,
    sssType: o.s?.t === 2 ? "ELAPSED-TIME" : "RACE",
    gates: (o.s?.g || []).map(hms).filter(Boolean) as string[],
    deadline: hms(o.g?.d),
    goalType: o.g?.t === 1 ? "LINE" : "CYLINDER",
    earth: o.e === 1 ? "FAI_SPHERE" : "WGS84",
  };
}

export async function parseQR(text: string): Promise<Task> {
  const s = text.replace(/^\uFEFF/, "").trim();
  let json: string;
  if (/^XCTSKZ:/i.test(s)) json = await inflate(s.slice(7));
  else if (/^XCTSK:/i.test(s)) {
    const body = s.slice(6).trim();
    json = body.startsWith("%7B") || body.startsWith("%7b") ? decodeURIComponent(body) : body;
  } else throw new Error("not an XCTrack QR code");
  const o = JSON.parse(json);
  return Array.isArray(o.turnpoints) ? fromXctskObject(o) : fromV2(o);   // some apps put a v1 task in the QR
}

// ---------------------------------------------------------------------------------------------
// Waypoint names and coordinates in the text formats

const csvQuote = (s: string) => `"${s.replace(/"/g, '""')}"`;
const shortName = (s: string, i: number) => (s.trim().replace(/\s+/g, "_") || `TP${String(i + 1).padStart(2, "0")}`);

function dm(v: number, degDigits: number, pos: string, neg: string): string {
  const a = Math.abs(v), d = Math.floor(a);
  let m = +((a - d) * 60).toFixed(3), dd = d;
  if (m >= 60) { m = 0; dd += 1; }
  return `${String(dd).padStart(degDigits, "0")}${m.toFixed(3).padStart(6, "0")}${v < 0 ? neg : pos}`;
}

function dms(v: number, degDigits: number): [string, string, string] {
  const a = Math.abs(v);
  let d = Math.floor(a), m = Math.floor((a - d) * 60), s = +((((a - d) * 60) - m) * 60).toFixed(2);
  if (s >= 60) { s = 0; m += 1; }
  if (m >= 60) { m = 0; d += 1; }
  return [String(d).padStart(degDigits, "0"), String(m).padStart(2, "0"), s.toFixed(2).padStart(5, "0")];
}

/** Waypoints of a task, one per name (a task can pass the same waypoint twice). */
function taskWaypoints(task: Task): Waypoint[] {
  const seen = new Map<string, Waypoint>();
  task.tps.forEach((p, i) => {
    const name = shortName(p.name, i);
    if (!seen.has(name)) seen.set(name, { name, desc: p.desc || p.name, lat: p.lat, lon: p.lon, alt: p.alt });
  });
  return [...seen.values()];
}

// SeeYou .cup: waypoints, then the task with an observation zone per point after the takeoff
export function toCup(task: Task): string {
  const t = normalized(task);
  const names = t.tps.map((p, i) => shortName(p.name, i));
  const lines = ["name,code,country,lat,lon,elev,style,rwdir,rwlen,freq,desc"];
  for (const w of taskWaypoints(t)) {
    lines.push([csvQuote(w.desc || w.name), csvQuote(w.name), "", dm(w.lat, 2, "N", "S"), dm(w.lon, 3, "E", "W"),
      `${Math.round(w.alt)}.0m`, "1", "", "", "", csvQuote(w.desc)].join(","));
  }
  const hasTakeoff = t.tps[0]?.type === "TAKEOFF";
  const route = hasTakeoff ? t.tps.slice(1) : t.tps;
  const rnames = hasTakeoff ? names.slice(1) : names;
  lines.push("-----Related Tasks-----");
  lines.push([csvQuote("Task"), csvQuote(hasTakeoff ? names[0] : "???"), ...rnames.map(csvQuote), csvQuote(rnames[rnames.length - 1] || "???")].join(","));
  const opts = ["Options"];
  if (t.gates[0]) opts.push(`NoStart=${t.gates[0]}`);
  lines.push(opts.join(","));
  route.forEach((p, k) => {
    const last = k === route.length - 1;
    const line = last && t.goalType === "LINE";
    lines.push(`ObsZone=${k},Style=${k === 0 ? 2 : last ? 3 : 1},R1=${Math.round(p.radius)}m,A1=${line ? 90 : 180}${line ? ",Line=1" : ""}`);
  });
  return lines.join("\r\n") + "\r\n";
}

function splitCsv(line: string): string[] {
  const out: string[] = [];
  let cur = "", q = false;
  for (let i = 0; i < line.length; i++) {
    const c = line[i];
    if (q) {
      if (c === '"' && line[i + 1] === '"') { cur += '"'; i++; }
      else if (c === '"') q = false;
      else cur += c;
    } else if (c === '"') q = true;
    else if (c === ",") { out.push(cur); cur = ""; }
    else cur += c;
  }
  out.push(cur);
  return out.map((s) => s.trim());
}

function cupCoord(s: string): number | null {
  const m = /^(\d+)(\d\d\.\d+)([NSEW])$/i.exec(s.trim());
  if (!m) return null;
  const v = Number(m[1]) + Number(m[2]) / 60;
  return /[SW]/i.test(m[3]) ? -v : v;
}

function cupElev(s: string): number {
  const m = /^(-?[\d.]+)\s*(m|ft)?$/i.exec(s.trim());
  if (!m) return 0;
  return Math.round(Number(m[1]) * (m[2]?.toLowerCase() === "ft" ? 0.3048 : 1));
}

export function parseCup(text: string): Parsed {
  const lines = text.split(/\r?\n/);
  const waypoints: Waypoint[] = [];
  let header: string[] | null = null, i = 0;
  for (; i < lines.length; i++) {
    const l = lines[i].trim();
    if (!l) continue;
    if (/^-+\s*Related Tasks/i.test(l)) { i++; break; }
    const f = splitCsv(l);
    if (!header) { header = f.map((h) => h.toLowerCase()); if (header.includes("lat")) continue; header = ["name", "code", "country", "lat", "lon", "elev"]; }
    const at = (k: string) => f[header!.indexOf(k)] ?? "";
    const lat = cupCoord(at("lat")), lon = cupCoord(at("lon"));
    if (lat == null || lon == null) continue;
    waypoints.push({ name: at("code") || at("name"), desc: at("name") !== at("code") ? at("name") : at("desc") || "", lat, lon, alt: cupElev(at("elev")) });
  }
  // the first task, if any
  let task: Task | null = null;
  const byName = new Map(waypoints.map((w) => [w.name.toLowerCase(), w] as const));
  const byDesc = new Map(waypoints.map((w) => [w.desc.toLowerCase(), w] as const));
  for (; i < lines.length; i++) {
    const l = lines[i].trim();
    if (!l || /^(Options|ObsZone|Point|STARTS)/i.test(l)) continue;
    const f = splitCsv(l);
    const pts = f.slice(1).map((n) => (n === "???" ? null : byName.get(n.toLowerCase()) ?? byDesc.get(n.toLowerCase()) ?? null));
    const takeoff = pts[0], landing = pts[pts.length - 1];
    let route = pts.slice(1, -1).filter(Boolean) as Waypoint[];
    if (!route.length) break;
    if (landing && landing !== route[route.length - 1]) route = [...route, landing];
    const zones: Record<number, { r: number; line: boolean }> = {};
    let opts = "";
    for (let j = i + 1; j < lines.length; j++) {
      const z = lines[j].trim();
      if (/^Options/i.test(z)) { opts = z; continue; }
      const m = /^ObsZone=(\d+)(.*)$/i.exec(z);
      if (!m) { if (z && !/^Point/i.test(z)) break; continue; }
      const r = /R1=([\d.]+)\s*(m|km|nm|ml)?/i.exec(m[2]);
      const mult: Record<string, number> = { m: 1, km: 1000, nm: 1852, ml: 1609.344 };
      zones[Number(m[1])] = { r: r ? Number(r[1]) * mult[(r[2] || "m").toLowerCase()] : 400, line: /Line=1/i.test(m[2]) };
    }
    const tps: Turnpoint[] = [];
    if (takeoff) tps.push({ name: takeoff.name, desc: takeoff.desc, lat: takeoff.lat, lon: takeoff.lon, alt: takeoff.alt, radius: 400, type: "TAKEOFF" });
    route.forEach((w, k) => tps.push({ name: w.name, desc: w.desc, lat: w.lat, lon: w.lon, alt: w.alt, radius: Math.round(zones[k]?.r ?? 400),
      type: k === 0 ? "SSS" : "" }));
    if (tps.length >= 2) {
      const start = /NoStart=(\d{1,2}:\d{2}(?::\d{2})?)/i.exec(opts);
      task = { ...emptyTask(), tps, goalType: zones[route.length - 1]?.line ? "LINE" : "CYLINDER",
        gates: start ? [hms(start[1])!] : [] };
    }
    break;
  }
  return { task, waypoints, format: "SeeYou (.cup)" };
}

// .wpt: Flymaster's GEO format, CompeGPS and OziExplorer
export function toGeoWpt(task: Task): string {
  const lines = ["$FormatGEO"];
  for (const w of taskWaypoints(task)) {
    const [ad, am, as] = dms(w.lat, 2), [od, om, os] = dms(w.lon, 3);
    lines.push(`${w.name.padEnd(10)}${w.lat < 0 ? "S" : "N"} ${ad} ${am} ${as}    ${w.lon < 0 ? "W" : "E"} ${od} ${om} ${os}  ${String(Math.round(w.alt)).padStart(4)}  ${w.desc}`);
  }
  return lines.join("\r\n") + "\r\n";
}

export function toCompeGpsWpt(task: Task): string {
  const lines = ["G  WGS 84", "U  1"];
  for (const w of taskWaypoints(task)) {
    lines.push(`W  ${w.name} A ${Math.abs(w.lat).toFixed(10)}º${w.lat < 0 ? "S" : "N"} ${Math.abs(w.lon).toFixed(10)}º${w.lon < 0 ? "W" : "E"} 27-MAR-62 00:00:00 ${w.alt.toFixed(6)} ${w.desc}`);
    lines.push("w Waypoint,0,-1.0,16777215,255,0,0,7,,0.0,");
  }
  return lines.join("\r\n") + "\r\n";
}

export function parseWpt(text: string): Parsed {
  const lines = text.split(/\r?\n/);
  const waypoints: Waypoint[] = [];
  const first = lines.find((l) => l.trim()) || "";
  if (/^\$FormatGEO/i.test(first.trim())) {
    for (const l of lines) {
      const m = /^\s*(\S+)\s+([NS])\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([EW])\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(-?[\d.]+)\s*(.*)$/i.exec(l);
      if (!m) continue;
      const lat = (Number(m[3]) + Number(m[4]) / 60 + Number(m[5]) / 3600) * (/S/i.test(m[2]) ? -1 : 1);
      const lon = (Number(m[7]) + Number(m[8]) / 60 + Number(m[9]) / 3600) * (/W/i.test(m[6]) ? -1 : 1);
      waypoints.push({ name: m[1], desc: m[11].trim(), lat, lon, alt: Math.round(Number(m[10])) });
    }
    return { task: null, waypoints, format: "Flymaster (.wpt)" };
  }
  if (/^OziExplorer/i.test(first.trim())) {
    for (const l of lines) {
      const f = l.split(",").map((s) => s.trim());
      if (f.length < 4 || !/^\d+$/.test(f[0])) continue;
      const lat = Number(f[2]), lon = Number(f[3]), ft = Number(f[14]);
      if (!Number.isFinite(lat) || !Number.isFinite(lon)) continue;
      waypoints.push({ name: f[1], desc: f[10] || "", lat, lon, alt: Number.isFinite(ft) && ft !== -777 ? Math.round(ft * 0.3048) : 0 });
    }
    return { task: null, waypoints, format: "OziExplorer (.wpt)" };
  }
  // CompeGPS: "W  NAME A 46.123ºN 8.123ºE date time alt description"
  for (const l of lines) {
    const m = /^W\s+(\S+)\s+A\s+(-?[\d.]+)\D*?([NS])\s+(-?[\d.]+)\D*?([EW])\s+\S+\s+\S+\s+(-?[\d.]+)\s*(.*)$/i.exec(l);
    if (!m) continue;
    const lat = Number(m[2]) * (/S/i.test(m[3]) ? -1 : 1), lon = Number(m[4]) * (/W/i.test(m[5]) ? -1 : 1);
    waypoints.push({ name: m[1], desc: m[7].trim(), lat, lon, alt: Math.round(Number(m[6])) });
  }
  return { task: null, waypoints, format: "CompeGPS (.wpt)" };
}

// GPX: waypoints and the task as a route
const xmlEsc = (s: string) => s.replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&apos;" })[c]!);
const xmlUnesc = (s: string) => s.replace(/<!\[CDATA\[([\s\S]*?)\]\]>/g, "$1")
  .replace(/&(lt|gt|quot|apos|amp);/g, (_, e) => ({ lt: "<", gt: ">", quot: '"', apos: "'", amp: "&" })[e as "lt"]!);

export function toGpx(task: Task): string {
  const pt = (tag: string, p: { name: string; desc: string; lat: number; lon: number; alt: number }) =>
    `  <${tag} lat="${p.lat.toFixed(6)}" lon="${p.lon.toFixed(6)}"><ele>${Math.round(p.alt)}</ele><name>${xmlEsc(p.name)}</name>${p.desc ? `<desc>${xmlEsc(p.desc)}</desc>` : ""}</${tag}>`;
  return `<?xml version="1.0" encoding="UTF-8"?>
<gpx version="1.1" creator="Pre-task Analyzer" xmlns="http://www.topografix.com/GPX/1/1">
${taskWaypoints(task).map((w) => pt("wpt", w)).join("\n")}
  <rte><name>Task</name>
${task.tps.map((p, i) => "  " + pt("rtept", { ...p, name: shortName(p.name, i) })).join("\n")}
  </rte>
</gpx>
`;
}

export function parseGpx(text: string): Parsed {
  const read = (tag: string) => [...text.matchAll(new RegExp(`<${tag}\\b([^>]*)>([\\s\\S]*?)</${tag}>|<${tag}\\b([^>]*)/>`, "g"))].map((m) => {
    const attrs = m[1] ?? m[3] ?? "", body = m[2] ?? "";
    const a = (k: string) => new RegExp(`${k}\\s*=\\s*["']([^"']+)["']`).exec(attrs)?.[1];
    const e = (k: string) => { const r = new RegExp(`<${k}>([\\s\\S]*?)</${k}>`).exec(body); return r ? xmlUnesc(r[1]).trim() : ""; };
    return { name: e("name"), desc: e("desc") || e("cmt"), lat: Number(a("lat")), lon: Number(a("lon")), alt: Math.round(Number(e("ele")) || 0) };
  }).filter((w) => Number.isFinite(w.lat) && Number.isFinite(w.lon));
  const waypoints = read("wpt");
  const rte = read("rtept");
  let task: Task | null = null;
  if (rte.length >= 2) task = { ...emptyTask(), tps: rte.map((w, i) => ({ ...w, radius: 400, type: i === 0 ? "TAKEOFF" : "" })) };
  return { task, waypoints: waypoints.length ? waypoints : rte, format: "GPX" };
}

/** Reads any supported file or QR text. */
export async function parseAny(name: string, text: string): Promise<Parsed> {
  const ext = (/\.([a-z0-9]+)$/i.exec(name)?.[1] || "").toLowerCase();
  const s = text.replace(/^﻿/, "").trim();
  if (/^XCTSKZ?:/i.test(s)) return { task: await parseQR(s), waypoints: [], format: "XCTrack QR" };
  if (ext === "xctsk" || (s.startsWith("{") && /"(turnpoints|t)"\s*:/.test(s))) return { task: parseXctsk(s), waypoints: [], format: "XCTrack (.xctsk)" };
  if (ext === "cup" || /^name,code,country/i.test(s)) return parseCup(s);
  if (ext === "gpx" || /<gpx[\s>]/i.test(s)) return parseGpx(s);
  if (ext === "wpt" || /^(\$FormatGEO|OziExplorer|G\s+WGS)/i.test(s)) return parseWpt(s);
  throw new Error("this file is not a task or waypoint file this page reads");
}

/** Text as an instrument expects it: Latin-1 for the .wpt formats, UTF-8 otherwise. */
export function latin1(text: string): Uint8Array {
  return Uint8Array.from(text, (c) => (c.charCodeAt(0) < 256 ? c.charCodeAt(0) : 63));
}
