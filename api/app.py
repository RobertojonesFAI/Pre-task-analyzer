"""Pre-task Analyzer API: predictions for a task a committee draws on the map.

Runs on the OptiPlex behind a Cloudflare Tunnel (api/README.md). The web page calls it to re-plan
a past task: same day, same pilots, a new route (and start time). The weather along the new route
is read from ERA5 for that day, exactly as for training (scripts/task_model.py), and the task model
predicts the winner's time and the share in goal.

    PTA_DB=C:/pta/database  python -m uvicorn api.app:app --host 127.0.0.1 --port 8000

Endpoints:
    GET  /health    what is loaded
    POST /predict   {"site": site_id, "template": task_key, "route": [[lat, lon, radius_m], ...],
                     "start_solar_h": optional}
"""

import json
import math
import os
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import task_model as TM  # noqa: E402

DB = Path(os.environ.get("PTA_DB", ROOT / "database"))
WEB = Path(os.environ.get("PTA_WEB", DB / "web_data"))
ORIGINS = [o.strip() for o in os.environ.get("PTA_ORIGINS", "*").split(",") if o.strip()]
MAX_POINTS, MAX_KM, MAX_FROM_SITE_DEG = 15, 400.0, 3.0

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
    STATE["model"] = json.loads((WEB / "task_model.json").read_text())
    STATE["sites"] = {s["id"]: s for s in json.loads((WEB / "sites.json").read_text())}
    STATE["tables"] = TM.load_tables(DB)
    STATE["loaded_s"] = round(time.time() - t0, 1)


@lru_cache(maxsize=64)
def site_tasks(site_id):
    p = WEB / "tasks" / f"{site_id}.json"
    if not p.exists():
        raise HTTPException(404, f"no tasks for site {site_id}")
    return json.loads(p.read_text())


@app.get("/health")
def health():
    t = STATE.get("tables", {})
    return {"ok": "model" in STATE, "sites": len(STATE.get("sites", {})), "loaded_in_s": STATE.get("loaded_s"),
            "weather_rows": {k: (0 if v is None else len(v)) for k, v in t.items()}}


class Request(BaseModel):
    site: str
    template: str = Field(description="task_key of the day to re-plan")
    route: list[list[float]] = Field(description="[[lat, lon, radius_m], ...] in order, takeoff first")
    start_solar_h: float | None = None


def route_km(pts):
    return sum(TM.km_between(pts[i][0], pts[i][1], pts[i + 1][0], pts[i + 1][1]) for i in range(len(pts) - 1))


def features(task, feats, site, pts, start_solar_h, tpl):
    """The task's feature vector with a new route (and start): weather along the route in the
    task's window, distance, turnpoints. Returns (x, ss_km, weather, window)."""
    at = {f: i for i, f in enumerate(feats)}
    x = list(task["x"])
    km = route_km(pts)
    # the day's window, as in training: from the start (solar time) for the task's window, at most 5 h
    start_solar = start_solar_h if start_solar_h is not None else x[at["start_solar_h"]] if "start_solar_h" in at else None
    window = x[at["window_h"]] if "window_h" in at and x[at["window_h"]] is not None else None
    if start_solar is None:
        start_solar, window = TM.DEFAULT_WINDOW[0], TM.DEFAULT_WINDOW[1] - TM.DEFAULT_WINDOW[0]
    window = min(window if window and window > 0 else 3.0, TM.MAX_WINDOW_H)
    start = pd.Timestamp(task["date"]) + pd.Timedelta(hours=start_solar - site["lon"] / 15)
    end = start + pd.Timedelta(hours=window)
    chips, legs = TM.route_from_points([p[0] for p in pts], [p[1] for p in pts])
    need = pd.DataFrame(TM.need_rows("new", chips, start, end), columns=["task_key", "chip_id", "time"])
    wx = TM.features_from(need, STATE["tables"], {"new": legs}).loc["new"]
    for f in TM.WEATHER:
        if f in at:
            x[at[f]] = None if pd.isna(wx[f]) else float(wx[f])
    # distance on the same footing as the task's own: its speed-section distance scaled by route length
    ss_km = task["km"] * km / route_km(tpl) if task.get("km") and len(tpl) >= 2 and route_km(tpl) > 0 else km
    if "ss_km" in at:
        x[at["ss_km"]] = ss_km
    if "n_turnpoints" in at and x[at["n_turnpoints"]] is not None:   # only where the model knew it for this task
        x[at["n_turnpoints"]] = float(len(pts))
    if "start_solar_h" in at and start_solar_h is not None:
        x[at["start_solar_h"]] = float(start_solar_h)
    return x, ss_km, wx, [str(start), str(end)]


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

    tpl = [p[:3] for p in (task.get("route") or [])]
    x_new, ss_km, wx, window_utc = features(task, data["features"], site, pts, req.start_solar_h, tpl)
    # the task as it was set, through the same steps, so rounding in the stored data cancels out
    x_tpl = features(task, data["features"], site, tpl, None, tpl)[0] if len(tpl) >= 2 else task["x"]
    m = STATE["model"]["targets"]
    kmh_raw, goal_raw = evaluate(m["winner_kmh"]["trees"], x_new), evaluate(m["goal_share"]["trees"], x_new)
    kmh_tpl, goal_tpl = evaluate(m["winner_kmh"]["trees"], x_tpl), evaluate(m["goal_share"]["trees"], x_tpl)
    # start from the prediction made without this competition and add the model's change
    kmh = (task.get("pred_kmh") or kmh_tpl) * kmh_raw / kmh_tpl
    goal = min(1.0, max(0.0, (task.get("pred_goal") if task.get("pred_goal") is not None else goal_tpl) + goal_raw - goal_tpl))
    keep = ["blh_mean", "usable_m", "cloud_base_above_m", "sensible_heat_wm2", "sunshine_wm2", "cloud_total", "cape",
            "wind_700", "tailwind_700", "crosswind_700", "lapse_850_700", "rain_mm", "route_chips"]
    return {"ss_km": round(ss_km, 1), "route_km": round(km, 1), "winner_kmh": round(kmh, 2),
            "winner_h": round(ss_km / kmh, 3) if kmh > 0 else None, "goal_share": round(goal, 3),
            "window_utc": window_utc,
            "weather": {k: (None if pd.isna(wx.get(k)) else round(float(wx[k]), 2)) for k in keep}}
