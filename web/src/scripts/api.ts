// The API (the OptiPlex, behind a Cloudflare Tunnel) evaluates a task with a new route using
// either a past competition day's ERA5 reanalysis or live Open-Meteo weather forecasts. Its
// address is ?api= in the page address, else PUBLIC_API_URL at build time, else the project's own API.
// PUBLIC_API_URL="" builds a page that works alone, without the API.
const DEFAULT_API = "https://api.xcmeteo.com";
export const API = (new URLSearchParams(location.search).get("api") ?? import.meta.env.PUBLIC_API_URL ?? DEFAULT_API).replace(/\/$/, "");

let ok: Promise<boolean> | null = null;
export function apiAvailable(): Promise<boolean> {
  if (!API) return Promise.resolve(false);
  ok ??= fetch(`${API}/health`).then((r) => r.ok).catch(() => false);
  return ok;
}

export type ForecastModelInfo = {
  id: string;
  name: string;
  region: "global" | "regional";
  resolution_km: number;
  max_days: number;
  blh_source: "native" | "parcel";
  available: boolean;
};

export type ForecastModelsResponse = {
  site: string | null;
  today_utc: string;
  min_date: string;
  max_date: string;
  models: ForecastModelInfo[];
};

export type ModelPrediction = {
  id: string;
  name: string;
  region: "global" | "regional";
  resolution_km: number;
  blh_source: "native" | "parcel";
  winner_kmh: number;
  winner_h: number | null;
  goal_share: number;
  weather: Record<string, number | null>;
};

export type BlendSummary = {
  n_models: number;
  median: { winner_kmh: number; winner_h: number | null; goal_share: number };
  mean: { winner_kmh: number; winner_h: number | null; goal_share: number };
  range: {
    winner_kmh: [number, number];
    winner_h: [number, number] | null;
    goal_share: [number, number];
  };
};

export type Prediction = {
  source?: "era5" | "forecast";
  date?: string;
  ss_km: number;
  route_km: number;
  winner_kmh: number;
  winner_h: number | null;
  goal_share: number;
  window_utc: [string, string];
  weather: Record<string, number | null>;
  blend?: BlendSummary;
  models?: ModelPrediction[];
  skipped?: { id: string; reason: string }[];
};

export type WeatherRequest = {
  source: "era5" | "forecast";
  date?: string;
  models?: string[];
  weights?: Record<string, number>;
};

const modelCache = new Map<string, Promise<ForecastModelsResponse>>();

export function fetchForecastModels(site: string, date?: string): Promise<ForecastModelsResponse> {
  const key = `${site}|${date ?? ""}`;
  if (!modelCache.has(key)) {
    const qs = new URLSearchParams({ site });
    if (date) qs.set("date", date);
    const p = fetch(`${API}/forecast/models?${qs}`)
      .then(async (r) => {
        const j = await r.json().catch(() => ({}));
        if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : r.statusText);
        return j as ForecastModelsResponse;
      })
      .catch((err) => {
        modelCache.delete(key);
        throw err;
      });
    modelCache.set(key, p);
  }
  return modelCache.get(key)!;
}

export async function predictRoute(body: {
  site: string;
  template: string;
  route: [number, number, number][];
  ss_km?: number;
  start_solar_h?: number;
  window_h?: number;
  weather?: WeatherRequest;
}): Promise<Prediction> {
  const r = await fetch(`${API}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  const j = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(typeof j.detail === "string" ? j.detail : r.statusText);
  return j as Prediction;
}
