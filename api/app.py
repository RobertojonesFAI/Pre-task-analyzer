"""Pre-task Analyzer API: predictions for a task a committee draws on the map.

Runs on the OptiPlex behind a Cloudflare Tunnel (api/README.md). The web page calls it to evaluate
a route drawn on the map using either:
  - a past competition day's ERA5 reanalysis (same day, same pilots, new route/start), or
  - live Open-Meteo forecasts (one or more NWP/AI models plus their median/mean blend) with a past
    competition day's pilot field as template.

    PTA_DB=C:/pta/database  python -m uvicorn api.app:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health           what is loaded
    GET  /forecast/models  forecast models available for a site (and optional date)
    POST /predict          predict winner's time and goal share for a route (ERA5 or live forecast)
"""

import json
import math
import os
import sys
import time
from datetime import date as date_cls
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import task_model as TM  # noqa: E402
from api import forecast as FC  # noqa: E402

DB = Path(os.environ.get("PTA_DB", ROOT / "database"))
WEB = Path(os.environ.get("PTA_WEB", DB / "web_data"))
ORIGINS = [o.strip() for o in os.environ.get("PTA_ORIGINS", "*").split(",") if o.strip()]
MAX_POINTS, MAX_KM, MAX_FROM_SITE_DEG = 25, 400.0, 3.0

WEATHER_KEEP = [
    "blh_mean", "blh_max", "usable_m", "cloud_base_above_m", "sensible_heat_wm2", "sunshine_wm2",
    "cloud_total", "cape", "wind100", "tailwind_100", "wind_700", "tailwind_700", "crosswind_700",
    "lapse_850_700", "rain_mm", "route_chips",
]

app = FastAPI(title="Pre-task Analyzer API", version="1")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["GET", "POST"], allow_headers=["*"])
STATE = {}


def evaluate(trees, x):
    s = 0.0
    for node in trees:
        while isinstance(node, list):
            f, thr, nan_left, left, right = node
            v = x[f]
            node = (left if nan_left else right) if v is None or (isinstance(v, float) and math.isnan(v)) else (
                left if v <= thr else right)
        s += node
    return s


@app.on_event("startup")
def load():  # noqa: D401  (FastAPI startup hook)
    t0 = time.time()
    STATE["model"] = json.loads((WEB / "task_model.json").read_text(encoding="utf-8"))
    STATE["sites"] = {s["id"]: s for s in json.loads((WEB / "sites.json").read_text(encoding="utf-8"))}
    STATE["tables"] = TM.load_tables(DB)
    STATE["loaded_s"] = round(time.time() - t0, 1)


@lru_cache(maxsize=64)
def site_tasks(site_id):
    p = WEB / "tasks" / f"{site_id}.json"
    if not p.exists():
        raise HTTPException(404, f"no tasks for site {site_id}")
    return json.loads(p.read_text(encoding="utf-8"))


@app.get("/health")
def health():
    t = STATE.get("tables", {})
    return {
        "ok": "model" in STATE,
        "sites": len(STATE.get("sites", {})),
        "loaded_in_s": STATE.get("loaded_s"),
        "weather_rows": {k: (0 if v is None else len(v)) for k, v in t.items()},
        "forecast_models": len(FC.MODELS),
    }


def _parse_date(raw: str | None) -> date_cls | None:
    if raw is None:
        return None
    try:
        return date_cls.fromisoformat(raw[:10])
    except ValueError as exc:
        raise HTTPException(422, f"invalid date {raw!r}, expected YYYY-MM-DD") from exc


@app.get("/forecast/models")
def forecast_models(
    site: str | None = None,
    lat: float | None = None,
    lon: float | None = None,
    date: str | None = None,
):
    if site is not None:
        s = STATE.get("sites", {}).get(site)
        if s is None:
            raise HTTPException(404, "unknown site")
        lat, lon = float(s["lat"]), float(s["lon"])
    elif lat is None or lon is None:
        raise HTTPException(422, "provide either site or both lat and lon")
    target = _parse_date(date)
    d_min, d_max = FC.date_bounds()
    return {
        "site": site,
        "today_utc": str(FC.utc_today()),
        "min_date": str(d_min),
        "max_date": str(d_max),
        "models": FC.available_models(lat, lon, target),
    }


class WeatherSpec(BaseModel):
    source: Literal["era5", "forecast"] = "era5"
    date: str | None = Field(None, description="YYYY-MM-DD forecast date (required when source='forecast')")
    models: list[str] | None = Field(None, description="forecast model ids; omit for all models available at this site/date")
    weights: dict[str, float] | None = Field(None, description="optional per-model blend weights")


class Request(BaseModel):
    site: str
    template: str = Field(description="task_key of the day to re-plan (or pilot-field template in forecast mode)")
    route: list[list[float]] = Field(description="[[lat, lon, radius_m], ...] in order, takeoff first")
    start_solar_h: float | None = None
    window_h: float | None = Field(None, description="task window from start gate to deadline, hours")
    ss_km: float | None = Field(None, description="the route's optimized speed-section distance (S7F), when the page "
                                                  "computed it from the cylinders")
    weather: WeatherSpec | None = None


def route_km(pts):
    return sum(TM.km_between(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]) for i in range(len(pts) - 1))


def _resolve_window(task_x, at, site, date_str, start_solar_h, window_h_given):
    start_solar = start_solar_h if start_solar_h is not None else task_x[at["start_solar_h"]] if "start_solar_h" in at else None
    window = window_h_given if (window_h_given is not None and window_h_given > 0) else (
        task_x[at["window_h"]] if "window_h" in at and task_x[at["window_h"]] is not None else None
    )
    if start_solar is None:
        start_solar = TM.DEFAULT_WINDOW[0]
        if window is None:
            window = TM.DEFAULT_WINDOW[1] - TM.DEFAULT_WINDOW[0]
    window = min(max(float(window) if window and window > 0 else 3.0, 0.5), float(TM.MAX_WINDOW_H))
    start = pd.Timestamp(date_str) + pd.Timedelta(hours=float(start_solar) - site["lon"] / 15.0)
    end = start + pd.Timedelta(hours=window)
    return float(start_solar), window, start, end


def _resolve_ss_km(task, pts, tpl, ss_given):
    km = route_km(pts)
    if ss_given is not None:
        return ss_given, km
    if task.get("km") and len(tpl) >= 2 and route_km(tpl) > 0:
        return task["km"] * km / route_km(tpl), km
    return km, km


def _format_weather(wx):
    return {k: (None if pd.isna(wx.get(k)) else round(float(wx[k]), 2)) for k in WEATHER_KEEP}


def features(task, feats, site, pts, start_solar_h, tpl, ss_given=None, window_h_given=None):
    """The task's feature vector with a new route (and start): ERA5 weather along the route in the
    task's window, distance, turnpoints. Returns (x, ss_km, weather, window)."""
    at = {f: i for i, f in enumerate(feats)}
    x = list(task["x"])
    _, window, start, end = _resolve_window(x, at, site, task["date"], start_solar_h, window_h_given)
    chips, legs = TM.route_from_points([p[0] for p in pts], [p[1] for p in pts])
    need = pd.DataFrame(TM.need_rows("new", chips, start, end), columns=["task_key", "chip_id", "time"])
    wx = TM.features_from(need, STATE["tables"], {"new": legs}).loc["new"]
    for f in TM.WEATHER:
        if f in at:
            x[at[f]] = None if pd.isna(wx[f]) else float(wx[f])
    ss_km, _ = _resolve_ss_km(task, pts, tpl, ss_given)
    if "ss_km" in at:
        x[at["ss_km"]] = ss_km
    if "n_turnpoints" in at and x[at["n_turnpoints"]] is not None:   # only where the model knew it for this task
        x[at["n_turnpoints"]] = float(len(pts))
    if "start_solar_h" in at and start_solar_h is not None:
        x[at["start_solar_h"]] = float(start_solar_h)
    if "window_h" in at and window_h_given is not None and window_h_given > 0:
        x[at["window_h"]] = window
    return x, ss_km, wx, [str(start), str(end)]


def _forecast_base_x(task, feats, site, all_tasks, pts, target_date, start_solar_h, window_h_given, tpl, ss_given):
    """Build the non-weather feature vector for a forecast day: pilot field from `task`,
    solar geometry for `target_date`, and latest site history across `all_tasks`."""
    at = {f: i for i, f in enumerate(feats)}
    x = list(task["x"])
    _, window, start, end = _resolve_window(x, at, site, str(target_date), start_solar_h, window_h_given)
    ss_km, km = _resolve_ss_km(task, pts, tpl, ss_given)

    # Solar geometry for the forecast date (same formulas as scripts/task_baseline.py).
    doy = target_date.timetuple().tm_yday
    decl = 23.44 * math.sin(math.radians(360.0 / 365.0 * (doy - 81)))
    lat_rad = math.radians(float(site["lat"]))
    cos_h = max(-1.0, min(1.0, -math.tan(lat_rad) * math.tan(math.radians(decl))))
    if "day_length_h" in at:
        x[at["day_length_h"]] = 2.0 * math.degrees(math.acos(cos_h)) / 15.0
    if "sun_decl_local" in at:
        x[at["sun_decl_local"]] = decl * (1.0 if float(site["lat"]) >= 0 else -1.0)

    # Latest site history across all stored tasks at this site (instead of an older template date).
    speeds = [t["km"] / t["winner_h"] for t in all_tasks if t.get("km") and t.get("winner_h") and t["winner_h"] > 0]
    goals = [t["goal"] for t in all_tasks if t.get("goal") is not None]
    if "site_prev_winner_kmh" in at and speeds:
        x[at["site_prev_winner_kmh"]] = float(np.median(speeds))
    if "site_prev_goal_share" in at and goals:
        x[at["site_prev_goal_share"]] = float(np.median(goals))
    if "site_prev_tasks" in at and all_tasks:
        x[at["site_prev_tasks"]] = float(len(all_tasks))

    if "ss_km" in at:
        x[at["ss_km"]] = ss_km
    if "n_turnpoints" in at and x[at["n_turnpoints"]] is not None:
        x[at["n_turnpoints"]] = float(len(pts))
    if "start_solar_h" in at and start_solar_h is not None:
        x[at["start_solar_h"]] = float(start_solar_h)
    if "window_h" in at and window_h_given is not None and window_h_given > 0:
        x[at["window_h"]] = window

    return x, at, ss_km, km, start, end


def _weighted_median(values: list[float], weights: list[float]) -> float:
    order = sorted(range(len(values)), key=lambda i: values[i])
    vs = [values[i] for i in order]
    ws = [weights[i] for i in order]
    total = sum(ws)
    if total <= 0:
        return float(np.median(vs))
    half = total / 2.0
    acc = 0.0
    for idx, (v, w) in enumerate(zip(vs, ws)):
        acc += w
        if math.isclose(acc, half, rel_tol=1e-9) and idx + 1 < len(vs):
            return 0.5 * (v + vs[idx + 1])
        if acc > half:
            return v
    return vs[-1]


def _blend_predictions(results: list[dict], weights_map: dict[str, float] | None) -> tuple[dict, dict]:
    """Compute median, mean and min/max uncertainty range across per-model predictions."""
    ws = [max(0.0, float(weights_map.get(r["id"], 1.0))) if weights_map else 1.0 for r in results]
    if sum(ws) <= 0:
        ws = [1.0] * len(results)
    w_sum = sum(ws)

    kmhs = [r["winner_kmh"] for r in results]
    hours = [r["winner_h"] for r in results if r["winner_h"] is not None]
    goals = [r["goal_share"] for r in results]

    med_kmh = _weighted_median(kmhs, ws) if weights_map else float(np.median(kmhs))
    mean_kmh = sum(v * w for v, w in zip(kmhs, ws)) / w_sum
    med_h = (_weighted_median(hours, ws) if weights_map else float(np.median(hours))) if len(hours) == len(results) else None
    mean_h = (sum(v * w for v, w in zip(hours, ws)) / w_sum) if len(hours) == len(results) else None
    med_goal = _weighted_median(goals, ws) if weights_map else float(np.median(goals))
    mean_goal = sum(v * w for v, w in zip(goals, ws)) / w_sum

    blend = {
        "n_models": len(results),
        "median": {
            "winner_kmh": round(med_kmh, 2),
            "winner_h": round(med_h, 3) if med_h is not None else None,
            "goal_share": round(med_goal, 3),
        },
        "mean": {
            "winner_kmh": round(mean_kmh, 2),
            "winner_h": round(mean_h, 3) if mean_h is not None else None,
            "goal_share": round(mean_goal, 3),
        },
        "range": {
            "winner_kmh": [round(min(kmhs), 2), round(max(kmhs), 2)],
            "winner_h": [round(min(hours), 3), round(max(hours), 3)] if hours else None,
            "goal_share": [round(min(goals), 3), round(max(goals), 3)],
        },
    }
    wx_med = {}
    for k in WEATHER_KEEP:
        vals = [r["weather"][k] for r in results if r["weather"].get(k) is not None]
        wx_med[k] = round(float(np.median(vals)), 2) if vals else None
    return blend, wx_med


def _predict_forecast(req: Request, site: dict, data: dict, task: dict, pts: list[list[float]]):
    wx_spec = req.weather
    assert wx_spec is not None
    target_date = _parse_date(wx_spec.date)
    if target_date is None:
        raise HTTPException(422, "weather.date (YYYY-MM-DD) is required in forecast mode")
    d_min, d_max = FC.date_bounds()
    if not (d_min <= target_date <= d_max):
        raise HTTPException(422, f"forecast date {target_date} is outside {d_min}..{d_max}")

    avail = FC.available_models(float(site["lat"]), float(site["lon"]), target_date)
    if wx_spec.models:
        requested = list(dict.fromkeys(wx_spec.models))
    else:
        requested = [m["id"] for m in avail if m["available"]]
    if not requested:
        raise HTTPException(422, "no forecast models selected")

    tpl = [p[:3] for p in (task.get("route") or [])]
    x_base, at, ss_km, km, start, end = _forecast_base_x(
        task, data["features"], site, data["tasks"], pts, target_date,
        req.start_solar_h, req.window_h, tpl, req.ss_km,
    )
    chips, legs = TM.route_from_points([p[0] for p in pts], [p[1] for p in pts])
    need = pd.DataFrame(TM.need_rows("new", chips, start, end), columns=["task_key", "chip_id", "time"])

    tables_by_model, skipped = FC.fetch_models(chips, requested, target_date, STATE["tables"])
    m = STATE["model"]["targets"]

    model_results = []
    for mid in requested:
        tbls = tables_by_model.get(mid)
        if tbls is None:
            continue
        wx = TM.features_from(need, tbls, {"new": legs}).loc["new"]
        if pd.isna(wx.get("t2m_c")) and pd.isna(wx.get("wind_700")) and pd.isna(wx.get("wind100")):
            skipped.append({"id": mid, "reason": "forecast does not cover task window"})
            continue
        x_m = list(x_base)
        for f in TM.WEATHER:
            if f in at:
                x_m[at[f]] = None if pd.isna(wx[f]) else float(wx[f])
        # Direct tree evaluation for unseen future days (no anchoring to a past template day's residual).
        kmh = max(5.0, float(evaluate(m["winner_kmh"]["trees"], x_m)))
        goal = min(1.0, max(0.0, float(evaluate(m["goal_share"]["trees"], x_m))))
        spec = FC.MODEL_BY_ID[mid]
        model_results.append({
            "id": mid,
            "name": spec["name"],
            "region": spec["region"],
            "resolution_km": spec["resolution_km"],
            "blh_source": spec["blh_source"],
            "winner_kmh": round(kmh, 2),
            "winner_h": round(ss_km / kmh, 3) if kmh > 0 else None,
            "goal_share": round(goal, 3),
            "weather": _format_weather(wx),
        })

    if not model_results:
        reasons = "; ".join(f"{s['id']}: {s['reason']}" for s in skipped) or "no data"
        raise HTTPException(502, f"no forecast models returned data for {target_date}: {reasons}")

    blend, wx_med = _blend_predictions(model_results, wx_spec.weights)
    return {
        "source": "forecast",
        "date": str(target_date),
        "ss_km": round(ss_km, 1),
        "route_km": round(km, 1),
        "window_utc": [str(start), str(end)],
        "winner_kmh": blend["median"]["winner_kmh"],
        "winner_h": blend["median"]["winner_h"],
        "goal_share": blend["median"]["goal_share"],
        "weather": wx_med,
        "blend": blend,
        "models": model_results,
        "skipped": skipped,
    }


@app.post("/predict")
def predict(req: Request):
    site = STATE["sites"].get(req.site)
    if site is None:
        raise HTTPException(404, "unknown site")
    data = site_tasks(req.site)
    task = next((t for t in data["tasks"] if t["key"] == req.template), None)
    if task is None:
        raise HTTPException(404, "unknown task for this site")
    pts = [p[:3] + [0.0] * (3 - len(p[:3])) for p in req.route]
    if not 2 <= len(pts) <= MAX_POINTS:
        raise HTTPException(422, f"a route needs 2 to {MAX_POINTS} points")
    if any(abs(p[0] - site["lat"]) > MAX_FROM_SITE_DEG or abs(p[1] - site["lon"]) > MAX_FROM_SITE_DEG for p in pts):
        raise HTTPException(422, "the route must stay near the site")
    km = route_km(pts)
    if not 1.0 <= km <= MAX_KM:
        raise HTTPException(422, f"route length {km:.0f} km is out of range")
    if req.ss_km is not None and not 1.0 <= req.ss_km <= MAX_KM:
        raise HTTPException(422, f"speed section {req.ss_km:.0f} km is out of range")

    if req.weather is not None and req.weather.source == "forecast":
        return _predict_forecast(req, site, data, task, pts)

    tpl = [p[:3] for p in (task.get("route") or [])]
    x_new, ss_km, wx, window_utc = features(
        task, data["features"], site, pts, req.start_solar_h, tpl, req.ss_km, req.window_h
    )
    # the task as it was set, through the same steps, so rounding in the stored data cancels out
    x_tpl = features(task, data["features"], site, tpl, None, tpl)[0] if len(tpl) >= 2 else task["x"]
    m = STATE["model"]["targets"]
    kmh_raw, goal_raw = evaluate(m["winner_kmh"]["trees"], x_new), evaluate(m["goal_share"]["trees"], x_new)
    kmh_tpl, goal_tpl = evaluate(m["winner_kmh"]["trees"], x_tpl), evaluate(m["goal_share"]["trees"], x_tpl)
    # start from the prediction made without this competition and add the model's change
    kmh = (task.get("pred_kmh") or kmh_tpl) * kmh_raw / kmh_tpl
    goal = min(1.0, max(0.0, (task.get("pred_goal") if task.get("pred_goal") is not None else goal_tpl) + goal_raw - goal_tpl))
    return {
        "source": "era5",
        "ss_km": round(ss_km, 1),
        "route_km": round(km, 1),
        "winner_kmh": round(kmh, 2),
        "winner_h": round(ss_km / kmh, 3) if kmh > 0 else None,
        "goal_share": round(goal, 3),
        "window_utc": window_utc,
        "weather": _format_weather(wx),
    }
