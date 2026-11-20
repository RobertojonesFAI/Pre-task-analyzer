// The API (the OptiPlex, behind a Cloudflare Tunnel) re-plans a task with a new route. Its address
// is ?api= in the page address, else PUBLIC_API_URL at build time, else the project's own API.
// PUBLIC_API_URL="" builds a page that works alone, without the API.
const DEFAULT_API = "https://api.xcmeteo.com";
export const API = (new URLSearchParams(location.search).get("api") ?? import.meta.env.PUBLIC_API_URL ?? DEFAULT_API).replace(/\/$/, "");

let ok: Promise<boolean> | null = null;
export function apiAvailable(): Promise<boolean> {
  if (!API) return Promise.resolve(false);
  ok ??= fetch(`${API}/health`).then((r) => r.ok).catch(() => false);
  return ok;
}

export type Prediction = {
  ss_km: number; route_km: number; winner_kmh: number; winner_h: number | null; goal_share: number;
  window_utc: [string, string]; weather: Record<string, number | null>;
};

export async function predictRoute(body: {
  site: string; template: string; route: [number, number, number][]; ss_km?: number; start_solar_h?: number;
}): Promise<Prediction> {
  const r = await fetch(`${API}/predict`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : r.statusText);
  return j as Prediction;
}
