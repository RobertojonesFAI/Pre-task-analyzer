#!/usr/bin/env python3
"""Verify Open-Meteo historical forecasts against pilot-measured winds from IGC tracks and derive
empirical blend weights for the forecast models.

Reads the pooled pilot wind measurements built by notebook 11 (models/wind_check/groups.parquet:
median u, v per site, chip, hour and height band from at least 3 thermals and 2 flights), fetches
matching archived forecasts from Open-Meteo's Historical Forecast API, interpolates each model's
wind profile (10 m, 100 m and 925/850/700/600/500 hPa) to the exact altitude of each group, and:

  1. Scores every global and regional model per height band alongside ERA5 on identical rows.
  2. Compares the core global models head-to-head on their common date intersection with
     event-clustered bootstrap 95% confidence intervals, and checks each regional model inside
     its domain.
  3. Fits global, regional and site-shrunk model weights on the middle + free-air bands (above
     400 m AGL or above local crests) and validates them against equal-weight mean and median
     blends using 5-fold grouped cross-validation by competition (event_key, zero leakage).
  4. Exports forecast_weights.json (and copies it to DB/web_data/) for weighted blending in the API.

Downloads are resumable and cached per model in DB/weather/hist_forecast/. Safe to stop and rerun.

    python3 scripts/forecast_wind_weights.py /content/drive/MyDrive/IGCS/database
    python3 scripts/forecast_wind_weights.py /content/drive/MyDrive/IGCS/database --skip-download
"""

from __future__ import annotations

import argparse
import gzip
import http.client
import json
import math
import socket
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pta import grid  # noqa: E402

HIST_FORECAST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
LEVELS = [925, 850, 700, 600, 500]
CALM_MS = 3.0
MERGE_GAP_DAYS = 14        # merge dates within 14 days into one API request window
MAX_WINDOW_DAYS = 45       # cap window length so responses stay compact
BATCH_CHIPS = 8            # chips per multi-coordinate request sharing a window
CHECKPOINT_EVERY = 20      # flush parquet cache to Drive every N batches
HTTP_TIMEOUT_S = 35
SHRINKAGE_K = 20.0         # Empirical Bayes shrinkage toward parent (region/global) skill
BOOTSTRAP_ITERS = 1000
WEIGHT_BANDS = ("free air", "middle")

# Archived Open-Meteo models matching api/forecast.py model ids.
# Note: historical ECMWF IFS upper-air winds live in ecmwf_ifs025 (ecmwf_ifs 9 km has no pressure levels).
MODELS: list[dict[str, Any]] = [
    {"id": "ecmwf_ifs",  "name": "ECMWF IFS",  "slug": "ecmwf_ifs025",              "region": "global",   "bbox": None,                         "wind100": "100m"},
    {"id": "gfs",        "name": "GFS",        "slug": "gfs_global",                "region": "global",   "bbox": None,                         "wind100": "100m"},
    {"id": "icon",       "name": "ICON",       "slug": "icon_global",               "region": "global",   "bbox": None,                         "wind100": "100m"},
    {"id": "gem",        "name": "GEM",        "slug": "gem_global",                "region": "global",   "bbox": None,                         "wind100": "80_120"},
    {"id": "ecmwf_aifs", "name": "ECMWF AIFS", "slug": "ecmwf_aifs025_single",      "region": "global",   "bbox": None,                         "wind100": "100m"},
    {"id": "icon_eu",    "name": "ICON-EU",    "slug": "icon_eu",                   "region": "regional", "bbox": (30.0, 70.0, -23.0, 44.5),    "wind100": "100m"},
    {"id": "icon_d2",    "name": "ICON-D2",    "slug": "icon_d2",                   "region": "regional", "bbox": (43.5, 57.5, -3.5, 20.0),     "wind100": "100m"},
    {"id": "arome",      "name": "AROME",      "slug": "meteofrance_arome_france",  "region": "regional", "bbox": (38.0, 55.0, -11.5, 15.5),    "wind100": "100m"},
    {"id": "hrdps",      "name": "HRDPS",      "slug": "gem_hrdps_continental",     "region": "regional", "bbox": (36.0, 68.0, -148.0, -48.0),  "wind100": "80_120"},
    {"id": "rrfs",       "name": "RRFS",       "slug": "ncep_rrfs_conus",           "region": "regional", "bbox": (22.0, 52.0, -133.0, -62.0),  "wind100": "100m"},
]
MODEL_BY_ID = {m["id"]: m for m in MODELS}
CORE_GLOBALS = ["ecmwf_ifs", "gfs", "icon", "gem"]

CACHE_COLS = (
    ["chip_id", "hour", "u10", "v10", "u100", "v100"]
    + [f"{v}_{p}" for p in LEVELS for v in ("u", "v", "z")]
)


# ── HTTP client (IPv4-preferred, gzip, 429 backoff) ──────────────────────────

_SSL_CTX = ssl.create_default_context()


class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        last_err: Exception | None = None
        infos = socket.getaddrinfo(self.host, self.port, 0, socket.SOCK_STREAM)
        v4 = [r for r in infos if r[0] == socket.AF_INET]
        ordered = v4 + [r for r in infos if r[0] != socket.AF_INET]
        for af, socktype, proto, _, sa in ordered:
            sock = None
            try:
                sock = socket.socket(af, socktype, proto)
                if self.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                    sock.settimeout(self.timeout)
                sock.connect(sa)
                self.sock = self._context.wrap_socket(sock, server_hostname=self.host)
                return
            except OSError as exc:
                last_err = exc
                if sock is not None:
                    sock.close()
        if last_err is not None:
            raise last_err


class _IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_IPv4HTTPSConnection, req, context=_SSL_CTX)


_OPENER = urllib.request.build_opener(_IPv4HTTPSHandler)


class DailyQuotaExceeded(RuntimeError):
    """Raised when Open-Meteo reports a daily API quota limit so progress is saved cleanly."""


def _http_get_json(params: dict[str, Any]) -> Any:
    url = f"{HIST_FORECAST_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "pre-task-analyzer-hist/1"})
    last_err: Exception | None = None
    for attempt in range(5):
        try:
            with _OPENER.open(req, timeout=HTTP_TIMEOUT_S) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            if exc.code == 429:
                if "daily" in body.lower():
                    raise DailyQuotaExceeded(body[:200]) from exc
                retry_hdr = exc.headers.get("Retry-After") if exc.headers else None
                try:
                    delay = min(30.0, max(2.0, float(retry_hdr))) if retry_hdr else min(30.0, 2.5 * (2 ** attempt))
                except ValueError:
                    delay = min(30.0, 2.5 * (2 ** attempt))
                print(f"    429 rate-limited, waiting {delay:.1f}s...", flush=True)
                time.sleep(delay)
                last_err = exc
                continue
            if exc.code == 400:
                raise RuntimeError(f"HTTP 400: {body[:180]}") from exc
            last_err = exc
            time.sleep(min(20.0, 2.0 * (attempt + 1)))
        except Exception as exc:
            last_err = exc
            time.sleep(min(15.0, 1.5 * (attempt + 1)))
    raise RuntimeError(f"Open-Meteo request failed after retries: {last_err}")


# ── Helpers: regions, bboxes, wind math ──────────────────────────────────────

def in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float] | None) -> bool:
    if bbox is None:
        return True
    s, n, w, e = bbox
    return s <= lat <= n and w <= lon <= e


def region_of(lat: float, lon: float) -> str:
    if 34.0 <= lat <= 72.0 and -15.0 <= lon <= 45.0:
        return "europe"
    if 15.0 <= lat <= 72.0 and -168.0 <= lon <= -50.0:
        return "north_america"
    if -56.0 <= lat < 15.0 and -90.0 <= lon <= -32.0:
        return "south_america"
    return "rest_of_world"


def _wind_uv(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    rad = np.radians(direction_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def _to_arr(values: list[Any] | None, n: int) -> np.ndarray:
    if not values or len(values) != n:
        return np.full(n, np.nan, dtype=np.float64)
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _vars_for_model(spec: dict[str, Any]) -> list[str]:
    v = ["wind_speed_10m", "wind_direction_10m"]
    if spec["wind100"] == "100m":
        v += ["wind_speed_100m", "wind_direction_100m"]
    else:
        v += ["wind_speed_80m", "wind_direction_80m", "wind_speed_120m", "wind_direction_120m"]
    for p in LEVELS:
        v += [f"wind_speed_{p}hPa", f"wind_direction_{p}hPa", f"geopotential_height_{p}hPa"]
    return v


# ── Step 1: Load groups.parquet + attach event_key, region, ground_m ─────────

def load_groups(db: Path, from_year: int, max_sites: int | None = None) -> pd.DataFrame:
    gp = db / "models" / "wind_check" / "groups.parquet"
    if not gp.exists():
        raise SystemExit(f"Missing {gp}. Run notebook 11 (wind_check.py) first.")
    g = pd.read_parquet(gp)
    g["hour"] = pd.to_datetime(g["hour"]).astype("datetime64[ns]")
    g["date"] = g["hour"].dt.strftime("%Y-%m-%d")
    g = g[g["hour"].dt.year >= from_year].copy()

    sites = pd.read_parquet(db / "sites.parquet", columns=["site_id", "name", "country", "lat", "lon"])
    g = g.merge(sites, on="site_id", how="left")
    g["region"] = [region_of(la, lo) for la, lo in zip(g["lat"].fillna(0.0), g["lon"].fillna(0.0))]

    # Link each group to its competition (event_key) for strict grouped CV.
    # Shift UTC hour by approximate solar offset (lon / 15 h) so western/eastern tasks that cross
    # 00:00 UTC still land on the task's local flying date instead of splitting into two event keys.
    g["local_date"] = (g["hour"] + pd.to_timedelta(g["lon"].fillna(0.0) / 15.0, unit="h")).dt.strftime("%Y-%m-%d")
    tasks = pd.read_parquet(db / "tasks.parquet", columns=["event_key", "date"])
    ev_sites = pd.read_parquet(db / "event_sites.parquet", columns=["event_key", "site_id"])
    te = tasks.merge(ev_sites, on="event_key", how="inner").dropna(subset=["date", "site_id"])
    te["date"] = pd.to_datetime(te["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    te = te.dropna(subset=["date"]).drop_duplicates(["site_id", "date"])
    g = g.merge(
        te[["site_id", "date", "event_key"]].rename(columns={"date": "local_date"}),
        on=["site_id", "local_date"],
        how="left",
    )
    if g["event_key"].isna().any():
        utc_ev = g[["site_id", "date"]].merge(te[["site_id", "date", "event_key"]], on=["site_id", "date"], how="left")["event_key"]
        g["event_key"] = g["event_key"].fillna(utc_ev)
    fallback_ev = g["site_id"].astype(str) + ":" + g["local_date"].str.slice(0, 7)
    g["event_key"] = g["event_key"].fillna(fallback_ev)

    static_p = db / "weather" / "era5_static.parquet"
    if static_p.exists():
        st = pd.read_parquet(static_p, columns=["chip_id", "ground_m"]).drop_duplicates("chip_id")
        g = g.merge(st, on="chip_id", how="left")
    else:
        g["ground_m"] = np.nan
    if g["ground_m"].isna().any() and (db / "cells.parquet").exists():
        cells = pd.read_parquet(db / "cells.parquet", columns=["chip_id", "elev_mean"])
        cell_ground = cells.groupby("chip_id")["elev_mean"].mean()
        g["ground_m"] = g["ground_m"].fillna(g["chip_id"].map(cell_ground))
    if g["ground_m"].isna().any():
        g["ground_m"] = g["ground_m"].fillna(g["h_asl"] - g["agl"])

    if max_sites is not None and max_sites > 0:
        top_sites = g.groupby("site_id").size().sort_values(ascending=False).head(max_sites).index
        g = g[g["site_id"].isin(top_sites)].copy()

    return g.reset_index(drop=True)


# ── Step 2: Resumable Historical Forecast downloader ─────────────────────────

def _merge_date_windows(dates: list[date]) -> list[tuple[date, date]]:
    """Collapse a sorted list of dates into contiguous [start, end] windows."""
    if not dates:
        return []
    ds = sorted(set(dates))
    windows: list[tuple[date, date]] = []
    w_start = w_end = ds[0]
    for d in ds[1:]:
        if (d - w_end).days <= MERGE_GAP_DAYS and (d - w_start).days <= MAX_WINDOW_DAYS:
            w_end = d
        else:
            windows.append((w_start, w_end))
            w_start = w_end = d
    windows.append((w_start, w_end))
    return windows


def _plan_batches(g_sub: pd.DataFrame, done_keys: set[str]) -> list[tuple[str, str, list[int]]]:
    """Build (start_iso, end_iso, [chip_ids]) download batches not already recorded in done_keys."""
    # Group by site first so nearby chips from the same competition share one date window.
    by_window: dict[tuple[str, str], set[int]] = {}
    for _, grp in g_sub.groupby("site_id"):
        chips = sorted(grp["chip_id"].astype(int).unique())
        dates = [date.fromisoformat(s) for s in grp["date"].unique()]
        for w0, w1 in _merge_date_windows(dates):
            s0, s1 = str(w0), str(w1)
            need_chips = [c for c in chips if f"{c}|{s0}|{s1}" not in done_keys]
            if need_chips:
                by_window.setdefault((s0, s1), set()).update(need_chips)

    batches: list[tuple[str, str, list[int]]] = []
    for (s0, s1), chip_set in sorted(by_window.items()):
        c_list = sorted(chip_set)
        for i in range(0, len(c_list), BATCH_CHIPS):
            batches.append((s0, s1, c_list[i : i + BATCH_CHIPS]))
    return batches


def _parse_batch_response(
    items: list[dict[str, Any]],
    chips: list[int],
    spec: dict[str, Any],
    needed_hours: dict[int, set[pd.Timestamp]],
) -> pd.DataFrame:
    frames = []
    for cid, item in zip(chips, items):
        hourly = item.get("hourly") or {}
        times_str = hourly.get("time") or []
        if not times_str:
            continue
        n = len(times_str)
        times = pd.to_datetime(times_str).astype("datetime64[ns]")
        want = needed_hours.get(cid)
        if want:
            keep_mask = times.isin(want)
            if not keep_mask.any():
                continue
        else:
            keep_mask = np.ones(n, dtype=bool)

        u10, v10 = _wind_uv(_to_arr(hourly.get("wind_speed_10m"), n), _to_arr(hourly.get("wind_direction_10m"), n))
        if spec["wind100"] == "100m":
            u100, v100 = _wind_uv(_to_arr(hourly.get("wind_speed_100m"), n), _to_arr(hourly.get("wind_direction_100m"), n))
        else:
            u80, v80 = _wind_uv(_to_arr(hourly.get("wind_speed_80m"), n), _to_arr(hourly.get("wind_direction_80m"), n))
            u120, v120 = _wind_uv(_to_arr(hourly.get("wind_speed_120m"), n), _to_arr(hourly.get("wind_direction_120m"), n))
            u100 = 0.5 * (u80 + u120)
            v100 = 0.5 * (v80 + v120)

        cols: dict[str, Any] = {
            "chip_id": np.full(int(keep_mask.sum()), cid, dtype=np.int64),
            "hour": times[keep_mask],
            "u10": u10[keep_mask].astype(np.float32),
            "v10": v10[keep_mask].astype(np.float32),
            "u100": u100[keep_mask].astype(np.float32),
            "v100": v100[keep_mask].astype(np.float32),
        }
        for p in LEVELS:
            up, vp = _wind_uv(_to_arr(hourly.get(f"wind_speed_{p}hPa"), n), _to_arr(hourly.get(f"wind_direction_{p}hPa"), n))
            zp = _to_arr(hourly.get(f"geopotential_height_{p}hPa"), n)
            cols[f"u_{p}"] = up[keep_mask].astype(np.float32)
            cols[f"v_{p}"] = vp[keep_mask].astype(np.float32)
            cols[f"z_{p}"] = zp[keep_mask].astype(np.float32)

        df = pd.DataFrame(cols)
        # Drop rows where every wind column is NaN (before a model's archive start date).
        wind_cols = ["u10", "u100"] + [f"u_{p}" for p in LEVELS]
        df = df[df[wind_cols].notna().any(axis=1)]
        if len(df):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=CACHE_COLS)


def _save_cache(cache_path: Path, done_path: Path, existing: pd.DataFrame, new_frames: list[pd.DataFrame], done_keys: set[str]) -> pd.DataFrame:
    if new_frames:
        parts = [existing] + new_frames if len(existing) else new_frames
        combined = pd.concat(parts, ignore_index=True).drop_duplicates(["chip_id", "hour"], keep="last")
        tmp = cache_path.with_suffix(".parquet.tmp")
        combined.to_parquet(tmp, index=False)
        tmp.replace(cache_path)
    else:
        combined = existing
    done_tmp = done_path.with_suffix(".json.tmp")
    done_tmp.write_text(json.dumps(sorted(done_keys)), encoding="utf-8")
    done_tmp.replace(done_path)
    return combined


def download_model_history(g: pd.DataFrame, spec: dict[str, Any], cache_dir: Path, pause: float) -> pd.DataFrame:
    """Download and cache historical forecast wind profiles for one model over all eligible groups."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    mid = spec["id"]
    cache_path = cache_dir / f"{mid}.parquet"
    done_path = cache_dir / f"{mid}_fetched.json"

    existing = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=CACHE_COLS)
    if len(existing):
        existing["hour"] = pd.to_datetime(existing["hour"]).astype("datetime64[ns]")
    done_keys: set[str] = set(json.loads(done_path.read_text(encoding="utf-8"))) if done_path.exists() else set()

    # Filter groups to chips inside this model's domain.
    chip_centres = {int(c): grid.chip_centre(int(c)) for c in g["chip_id"].unique()}
    valid_chips = {c for c, (la, lo) in chip_centres.items() if in_bbox(la, lo, spec["bbox"])}
    g_sub = g[g["chip_id"].isin(valid_chips)]
    if not len(g_sub):
        print(f"  {mid:12s}: 0 groups inside domain", flush=True)
        return existing

    needed_hours: dict[int, set[pd.Timestamp]] = {
        int(cid): set(grp["hour"]) for cid, grp in g_sub.groupby("chip_id")
    }
    batches = _plan_batches(g_sub, done_keys)
    if not batches:
        print(f"  {mid:12s}: cache complete ({len(existing):,} chip-hours)", flush=True)
        return existing

    print(f"  {mid:12s} ({spec['slug']}): fetching {len(batches)} batches ({g_sub['chip_id'].nunique()} chips, {len(g_sub):,} groups)...", flush=True)
    req_vars = ",".join(_vars_for_model(spec))
    new_frames: list[pd.DataFrame] = []
    quota_hit = False
    n_err = 0

    for idx, (s0, s1, chips) in enumerate(batches, 1):
        lats = ",".join(f"{chip_centres[c][0]:.4f}" for c in chips)
        lons = ",".join(f"{chip_centres[c][1]:.4f}" for c in chips)
        try:
            data = _http_get_json({
                "latitude": lats,
                "longitude": lons,
                "start_date": s0,
                "end_date": s1,
                "hourly": req_vars,
                "models": spec["slug"],
                "wind_speed_unit": "ms",
                "timezone": "UTC",
            })
            items = data if isinstance(data, list) else [data]
            if len(items) == len(chips):
                df = _parse_batch_response(items, chips, spec, needed_hours)
                if len(df):
                    new_frames.append(df)
                for c in chips:
                    done_keys.add(f"{c}|{s0}|{s1}")
        except DailyQuotaExceeded as exc:
            print(f"    Open-Meteo daily quota reached at batch {idx}/{len(batches)}: {exc}", flush=True)
            quota_hit = True
            break
        except Exception as exc:
            if "HTTP 400" in str(exc) and len(chips) > 1:
                # A single edge chip outside a regional grid can cause HTTP 400 for the whole batch;
                # retry chip-by-chip so valid chips in this batch are kept.
                for c in chips:
                    try:
                        la, lo = chip_centres[c]
                        d1 = _http_get_json({
                            "latitude": f"{la:.4f}",
                            "longitude": f"{lo:.4f}",
                            "start_date": s0,
                            "end_date": s1,
                            "hourly": req_vars,
                            "models": spec["slug"],
                            "wind_speed_unit": "ms",
                            "timezone": "UTC",
                        })
                        df1 = _parse_batch_response([d1] if isinstance(d1, dict) else d1, [c], spec, needed_hours)
                        if len(df1):
                            new_frames.append(df1)
                        done_keys.add(f"{c}|{s0}|{s1}")
                    except DailyQuotaExceeded as qexc:
                        print(f"    Open-Meteo daily quota reached during single-chip retry: {qexc}", flush=True)
                        quota_hit = True
                        break
                    except Exception as exc1:
                        n_err += 1
                        if "HTTP 400" in str(exc1):
                            done_keys.add(f"{c}|{s0}|{s1}")
                        if n_err <= 5:
                            print(f"    chip {c} ({s0}..{s1}) skipped: {exc1}", flush=True)
                    if pause > 0:
                        time.sleep(pause)
                if quota_hit:
                    break
            else:
                n_err += 1
                if "HTTP 400" in str(exc):
                    # Mark single out-of-domain chip as attempted so reruns do not stall on it.
                    for c in chips:
                        done_keys.add(f"{c}|{s0}|{s1}")
                if n_err <= 5:
                    print(f"    batch {idx} ({s0}..{s1}, {len(chips)} chips) skipped: {exc}", flush=True)

        if idx % CHECKPOINT_EVERY == 0 or idx == len(batches):
            existing = _save_cache(cache_path, done_path, existing, new_frames, done_keys)
            new_frames = []
            print(f"    {mid:12s}: {idx}/{len(batches)} batches saved ({len(existing):,} chip-hours)", flush=True)
        if pause > 0:
            time.sleep(pause)

    if new_frames or quota_hit:
        existing = _save_cache(cache_path, done_path, existing, new_frames, done_keys)
    if quota_hit:
        raise DailyQuotaExceeded(f"Saved progress for {mid}. Re-run later to finish remaining batches.")
    return existing


# ── Step 3: Vertical interpolation to each group's exact h_asl ───────────────

def interpolate_model_wind(g: pd.DataFrame, wx: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate a model's (u, v) wind profile to each group's altitude h_asl (m ASL),
    matching scripts/wind_check.py (10 m, 100 m, and pressure levels > ground + 150 m)."""
    if not len(wx):
        return np.full(len(g), np.nan), np.full(len(g), np.nan)
    wx = wx.drop_duplicates(["chip_id", "hour"])
    m = g[["chip_id", "hour", "ground_m", "h_asl"]].merge(wx, on=["chip_id", "hour"], how="left")

    ground = m["ground_m"].to_numpy(float)
    H = [ground + 10.0, ground + 100.0]
    U = [m["u10"].to_numpy(float), m["u100"].to_numpy(float)]
    V = [m["v10"].to_numpy(float), m["v100"].to_numpy(float)]
    for p in LEVELS:
        z = m[f"z_{p}"].to_numpy(float)
        z = np.where(z > ground + 150.0, z, np.nan)
        H.append(z)
        U.append(m[f"u_{p}"].to_numpy(float))
        V.append(m[f"v_{p}"].to_numpy(float))

    H_mat, U_mat, V_mat = np.column_stack(H), np.column_stack(U), np.column_stack(V)
    h_asl = m["h_asl"].to_numpy(float)
    out_u = np.full(len(g), np.nan, dtype=np.float64)
    out_v = np.full(len(g), np.nan, dtype=np.float64)
    for i in range(len(g)):
        if math.isnan(h_asl[i]):
            continue
        ok = ~(np.isnan(H_mat[i]) | np.isnan(U_mat[i]) | np.isnan(V_mat[i]))
        if ok.sum() < 2:
            continue
        hh = H_mat[i][ok]
        order = np.argsort(hh)
        hh = hh[order]
        out_u[i] = float(np.interp(h_asl[i], hh, U_mat[i][ok][order]))
        out_v[i] = float(np.interp(h_asl[i], hh, V_mat[i][ok][order]))
    return out_u, out_v


# ── Step 4: Metrics & event-clustered bootstrap CIs ──────────────────────────

def wind_metrics(pu: np.ndarray, pv: np.ndarray, mu: np.ndarray, mv: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(pu) & np.isfinite(pv) & np.isfinite(mu) & np.isfinite(mv)
    if not ok.any():
        return {"groups": 0}
    pu, pv, mu, mv = pu[ok], pv[ok], mu[ok], mv[ok]
    err_sq = (pu - mu) ** 2 + (pv - mv) ** 2
    vec_err = np.sqrt(err_sq)
    pspd = np.hypot(pu, pv)
    mspd = np.hypot(mu, mv)
    pdir = (np.degrees(np.arctan2(-pu, -pv)) + 360.0) % 360.0
    mdir = (np.degrees(np.arctan2(-mu, -mv)) + 360.0) % 360.0
    derr = np.abs((pdir - mdir + 180.0) % 360.0 - 180.0)
    windy = mspd >= CALM_MS
    corr = float(np.corrcoef(pspd, mspd)[0, 1]) if len(pspd) > 2 and np.std(pspd) > 0 and np.std(mspd) > 0 else None
    return {
        "groups": int(len(vec_err)),
        "vector_rmse_ms": float(np.sqrt(np.mean(err_sq))),
        "vector_mae_ms": float(np.mean(vec_err)),
        "vector_median_ms": float(np.median(vec_err)),
        "speed_bias_ms": float(np.median(pspd - mspd)),
        "speed_corr": corr,
        "dir_error_deg": float(np.median(derr[windy])) if windy.any() else None,
        "dir_within_30": float(np.mean(derr[windy] <= 30.0)) if windy.any() else None,
    }


def cluster_bootstrap_ci(
    events: np.ndarray,
    values: np.ndarray,
    stat_fn,
    n_iter: int = BOOTSTRAP_ITERS,
    seed: int = 42,
) -> tuple[float, float]:
    """95% CI by resampling whole competitions (event_key clusters) with replacement."""
    rng = np.random.default_rng(seed)
    uniq, inv = np.unique(events, return_inverse=True)
    if len(uniq) < 3:
        v = float(stat_fn(values))
        return v, v
    by_ev = [values[inv == i] for i in range(len(uniq))]
    stats = np.empty(n_iter, dtype=np.float64)
    for b in range(n_iter):
        picks = rng.integers(0, len(uniq), size=len(uniq))
        sample = np.concatenate([by_ev[p] for p in picks])
        stats[b] = stat_fn(sample)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ── Step 5: Relative skill weights & leakage-free grouped CV ─────────────────

def _fit_relative_skill(df: pd.DataFrame, model_ids: list[str]) -> dict[str, tuple[float, int]]:
    """Return {model_id: (rel_rmse_vs_era5, n_groups)} on df.
    Dividing by ERA5 RMSE on the exact same rows normalizes differences in archive dates/sites."""
    pu, pv = df["pilot_u"].to_numpy(float), df["pilot_v"].to_numpy(float)
    eu, ev = df["model_u"].to_numpy(float), df["model_v"].to_numpy(float)
    era5_sq = (pu - eu) ** 2 + (pv - ev) ** 2
    out: dict[str, tuple[float, int]] = {}
    for mid in model_ids:
        u_col, v_col = f"{mid}_u", f"{mid}_v"
        if u_col not in df:
            continue
        mu, mv = df[u_col].to_numpy(float), df[v_col].to_numpy(float)
        ok = np.isfinite(pu) & np.isfinite(pv) & np.isfinite(mu) & np.isfinite(mv) & np.isfinite(era5_sq)
        n = int(ok.sum())
        if n < 5:
            continue
        m_rmse = float(np.sqrt(np.mean((pu[ok] - mu[ok]) ** 2 + (pv[ok] - mv[ok]) ** 2)))
        e_rmse = float(np.sqrt(np.mean(era5_sq[ok])))
        rel = m_rmse / max(e_rmse, 0.25)
        out[mid] = (rel, n)
    return out


def _skill_to_weights(rel_skills: dict[str, float], beta: float) -> dict[str, float]:
    """Convert relative RMSE map {model: rel_rmse} to normalized positive weights via softmax(-beta * rel)."""
    if not rel_skills:
        return {}
    vals = np.array(list(rel_skills.values()), dtype=np.float64)
    shifted = -beta * (vals - vals.min())
    raw = np.exp(np.clip(shifted, -20.0, 0.0))
    raw /= raw.sum()
    return {m: float(w) for m, w in zip(rel_skills.keys(), raw)}


def fit_hierarchy(train_df: pd.DataFrame, model_ids: list[str], beta: float, shrink_k: float = SHRINKAGE_K) -> dict[str, Any]:
    """Fit global, regional, and site-shrunk weights strictly on train_df."""
    g_raw = _fit_relative_skill(train_df, model_ids)
    # Neutral fallback (median relative RMSE) for models with too few historical observations.
    default_rel = float(np.median([v[0] for v in g_raw.values()])) if g_raw else 1.2
    global_rel = {mid: g_raw[mid][0] if mid in g_raw else default_rel for mid in model_ids}
    global_w = _skill_to_weights(global_rel, beta)

    region_rel: dict[str, dict[str, float]] = {}
    region_w: dict[str, dict[str, float]] = {}
    for reg, grp in train_df.groupby("region"):
        r_raw = _fit_relative_skill(grp, model_ids)
        shrunk = {}
        for mid in model_ids:
            if mid in r_raw:
                rel_r, n_r = r_raw[mid]
                shrunk[mid] = (n_r * rel_r + shrink_k * global_rel[mid]) / (n_r + shrink_k)
            else:
                shrunk[mid] = global_rel[mid]
        region_rel[str(reg)] = shrunk
        region_w[str(reg)] = _skill_to_weights(shrunk, beta)

    site_w: dict[str, dict[str, Any]] = {}
    for sid, grp in train_df.groupby("site_id"):
        n_ev = int(grp["event_key"].nunique())
        if len(grp) < 10 or n_ev < 2:
            continue
        reg = str(grp["region"].iloc[0])
        parent_rel = region_rel.get(reg, global_rel)
        s_raw = _fit_relative_skill(grp, model_ids)
        if not s_raw:
            continue
        shrunk = {}
        for mid in model_ids:
            if mid in s_raw:
                rel_s, n_s = s_raw[mid]
                shrunk[mid] = (n_s * rel_s + shrink_k * parent_rel[mid]) / (n_s + shrink_k)
            else:
                shrunk[mid] = parent_rel[mid]
        site_w[str(sid)] = {
            "weights": {k: round(v, 4) for k, v in _skill_to_weights(shrunk, beta).items()},
            "n_groups": int(len(grp)),
            "n_events": n_ev,
            "region": reg,
        }

    return {
        "beta": beta,
        "shrinkage_k": shrink_k,
        "global_rel_rmse": {k: round(v, 4) for k, v in global_rel.items()},
        "global": {k: round(v, 4) for k, v in global_w.items()},
        "by_region": {r: {k: round(v, 4) for k, v in wm.items()} for r, wm in region_w.items()},
        "by_site": site_w,
    }


def predict_blend_uv(df: pd.DataFrame, model_ids: list[str], mode: str, weights_spec: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Combine available model wind vectors row by row under the chosen blend strategy."""
    u_cols = [f"{m}_u" for m in model_ids if f"{m}_u" in df]
    v_cols = [f"{m}_v" for m in model_ids if f"{m}_v" in df]
    active_ids = [m for m in model_ids if f"{m}_u" in df]
    U = df[u_cols].to_numpy(float)
    V = df[v_cols].to_numpy(float)
    valid = np.isfinite(U) & np.isfinite(V)

    if mode == "equal_mean":
        cnt = valid.sum(axis=1)
        out_u = np.where(cnt > 0, np.where(valid, U, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
        out_v = np.where(cnt > 0, np.where(valid, V, 0.0).sum(axis=1) / np.maximum(cnt, 1), np.nan)
        return out_u, out_v

    if mode == "median":
        with np.errstate(all="ignore"):
            return np.nanmedian(U, axis=1), np.nanmedian(V, axis=1)

    assert weights_spec is not None
    n = len(df)
    W = np.zeros_like(U)
    g_w = weights_spec["global"]
    r_w = weights_spec["by_region"]
    s_w = weights_spec["by_site"]
    regs = df["region"].astype(str).to_numpy()
    sids = df["site_id"].astype(str).to_numpy()

    for i in range(n):
        if mode == "site_weighted" and sids[i] in s_w:
            wm = s_w[sids[i]]["weights"]
        elif mode in ("region_weighted", "site_weighted") and regs[i] in r_w:
            wm = r_w[regs[i]]
        else:
            wm = g_w
        for j, mid in enumerate(active_ids):
            if valid[i, j]:
                W[i, j] = wm.get(mid, 0.0)

    w_sum = W.sum(axis=1)
    out_u = np.where(w_sum > 0, (np.where(valid, U, 0.0) * W).sum(axis=1) / np.maximum(w_sum, 1e-9), np.nan)
    out_v = np.where(w_sum > 0, (np.where(valid, V, 0.0) * W).sum(axis=1) / np.maximum(w_sum, 1e-9), np.nan)
    return out_u, out_v


def run_grouped_cv(df: pd.DataFrame, model_ids: list[str], k_folds: int = 5) -> dict[str, Any]:
    """5-fold grouped CV by competition (event_key) comparing unweighted vs weighted blends."""
    # Keep rows with at least 2 forecast models available so blending is meaningful.
    u_cols = [f"{m}_u" for m in model_ids if f"{m}_u" in df]
    d = df[df["band"].isin(WEIGHT_BANDS)].copy()
    d = d[d[u_cols].notna().sum(axis=1) >= 2].reset_index(drop=True)
    n_events = int(d["event_key"].nunique())
    if n_events < 3 or len(d) < 20:
        return {"error": f"too few matched events ({n_events}) or groups ({len(d)}) for grouped CV"}

    n_splits = min(k_folds, n_events)
    gkf = GroupKFold(n_splits=n_splits)
    betas = [1.0, 2.0, 4.0, 6.0, 8.0]

    # Select best global temperature beta strictly inside each training fold.
    oof: dict[str, tuple[np.ndarray, np.ndarray]] = {
        name: (np.full(len(d), np.nan), np.full(len(d), np.nan))
        for name in ("best_single", "equal_mean", "median", "global_weighted", "region_weighted", "site_weighted")
    }
    chosen_betas: list[float] = []

    for tr_idx, te_idx in gkf.split(d, groups=d["event_key"]):
        tr, te = d.iloc[tr_idx], d.iloc[te_idx]
        # Pick best single full-coverage model on train fold (lowest relative RMSE with >= 80% coverage,
        # falling back to >= 20% coverage if archives are sparse).
        tr_rel = _fit_relative_skill(tr, model_ids)
        eligible = {m: v[0] for m, v in tr_rel.items() if v[1] >= 0.8 * len(tr)}
        if not eligible:
            eligible = {m: v[0] for m, v in tr_rel.items() if v[1] >= 0.2 * len(tr)}
        best_mid = min(eligible, key=eligible.get) if eligible else model_ids[0]
        oof["best_single"][0][te_idx] = te[f"{best_mid}_u"].to_numpy(float)
        oof["best_single"][1][te_idx] = te[f"{best_mid}_v"].to_numpy(float)

        # Tune beta on inner 3-fold split of train fold (zero test-fold leakage).
        inner_events = int(tr["event_key"].nunique())
        best_beta = 4.0
        if inner_events >= 3:
            inner_gkf = GroupKFold(n_splits=min(3, inner_events))
            best_inner = float("inf")
            for b in betas:
                errs = []
                for itr, iva in inner_gkf.split(tr, groups=tr["event_key"]):
                    spec_in = fit_hierarchy(tr.iloc[itr], model_ids, beta=b)
                    va = tr.iloc[iva]
                    bu, bv = predict_blend_uv(va, model_ids, "global_weighted", spec_in)
                    sq = (va["pilot_u"].to_numpy(float) - bu) ** 2 + (va["pilot_v"].to_numpy(float) - bv) ** 2
                    errs.append(sq[np.isfinite(sq)])
                if errs:
                    rmse = float(np.sqrt(np.mean(np.concatenate(errs))))
                    if rmse < best_inner:
                        best_inner, best_beta = rmse, b
        chosen_betas.append(best_beta)

        spec = fit_hierarchy(tr, model_ids, beta=best_beta)
        for mode in ("equal_mean", "median", "global_weighted", "region_weighted", "site_weighted"):
            pu_hat, pv_hat = predict_blend_uv(te, model_ids, mode, spec)
            oof[mode][0][te_idx] = pu_hat
            oof[mode][1][te_idx] = pv_hat

    pu, pv = d["pilot_u"].to_numpy(float), d["pilot_v"].to_numpy(float)
    evs = d["event_key"].astype(str).to_numpy()
    results: dict[str, Any] = {
        "n_groups": int(len(d)),
        "n_events": n_events,
        "n_sites": int(d["site_id"].nunique()),
        "n_folds": n_splits,
        "chosen_betas": chosen_betas,
        "median_beta": float(np.median(chosen_betas)),
        "strategies": {},
    }

    eq_sq = (pu - oof["equal_mean"][0]) ** 2 + (pv - oof["equal_mean"][1]) ** 2
    for name, (mu, mv) in oof.items():
        m = wind_metrics(pu, pv, mu, mv)
        sq = (pu - mu) ** 2 + (pv - mv) ** 2
        ok = np.isfinite(sq)
        if not ok.any():
            results["strategies"][name] = m
            continue
        lo, hi = cluster_bootstrap_ci(evs[ok], sq[ok], lambda x: float(np.sqrt(np.mean(x))))
        m["vector_rmse_ci95"] = [round(lo, 3), round(hi, 3)]
        if name != "equal_mean":
            both = ok & np.isfinite(eq_sq)
            if both.any():
                diff_lo, diff_hi = cluster_bootstrap_ci(
                    evs[both],
                    np.column_stack([sq[both], eq_sq[both]]),
                    lambda x: float(np.sqrt(np.mean(x[:, 0])) - np.sqrt(np.mean(x[:, 1]))),
                )
                delta = float(np.sqrt(np.mean(sq[both]))) - float(np.sqrt(np.mean(eq_sq[both])))
                m["delta_rmse_vs_equal_ms"] = round(delta, 4)
                m["delta_rmse_ci95"] = [round(diff_lo, 4), round(diff_hi, 4)]
        results["strategies"][name] = m

    # ERA5 reanalysis reference on the exact same CV groups
    eu, ev = d["model_u"].to_numpy(float), d["model_v"].to_numpy(float)
    era_m = wind_metrics(pu, pv, eu, ev)
    era_sq = (pu - eu) ** 2 + (pv - ev) ** 2
    era_ok = np.isfinite(era_sq)
    if era_ok.any():
        lo, hi = cluster_bootstrap_ci(evs[era_ok], era_sq[era_ok], lambda x: float(np.sqrt(np.mean(x))))
        era_m["vector_rmse_ci95"] = [round(lo, 3), round(hi, 3)]
    results["strategies"]["era5_reanalysis"] = era_m
    return results


# ── Main CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder (e.g. /content/drive/MyDrive/IGCS/database)")
    parser.add_argument("--out", type=Path, help="output folder (default DB/models/forecast_wind_weights)")
    parser.add_argument("--from-year", type=int, default=2022, help="earliest year to query (default 2022)")
    parser.add_argument("--models", type=str, default="all", help="comma-separated model ids or 'all'")
    parser.add_argument("--pause", type=float, default=0.4, help="seconds between Open-Meteo requests (default 0.4)")
    parser.add_argument("--max-sites", type=int, default=None, help="limit to top N sites by group count (smoke test)")
    parser.add_argument("--skip-download", action="store_true", help="score using already cached parquets only")
    args = parser.parse_args()

    out = args.out or (args.db / "models" / "forecast_wind_weights")
    cache_dir = args.db / "weather" / "hist_forecast"
    out.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    selected = MODELS if args.models == "all" else [MODEL_BY_ID[m.strip()] for m in args.models.split(",") if m.strip() in MODEL_BY_ID]
    if not selected:
        raise SystemExit(f"No valid models in --models={args.models!r}. Valid: {list(MODEL_BY_ID)}")

    g = load_groups(args.db, from_year=args.from_year, max_sites=args.max_sites)
    print(
        f"Loaded {len(g):,} wind groups since {args.from_year} across {g['site_id'].nunique()} sites "
        f"and {g['event_key'].nunique()} competitions "
        f"(free air: {(g['band'] == 'free air').sum():,}, middle: {(g['band'] == 'middle').sum():,}, "
        f"near ground: {(g['band'] == 'near the ground').sum():,})",
        flush=True,
    )

    # 1. Download (resumable) and interpolate each model to each group's exact altitude.
    model_tables: dict[str, pd.DataFrame] = {}
    quota_stopped = False
    for spec in selected:
        mid = spec["id"]
        cache_p = cache_dir / f"{mid}.parquet"
        if args.skip_download or quota_stopped:
            wx = pd.read_parquet(cache_p) if cache_p.exists() else pd.DataFrame(columns=CACHE_COLS)
            if len(wx):
                wx["hour"] = pd.to_datetime(wx["hour"]).astype("datetime64[ns]")
        else:
            try:
                wx = download_model_history(g, spec, cache_dir, pause=args.pause)
            except DailyQuotaExceeded as exc:
                print(
                    f"\n[PAUSED] {exc}\n"
                    f"         All downloaded batches are saved in {cache_dir}.\n"
                    "         Scoring models cached so far; re-run later to finish remaining batches.\n",
                    flush=True,
                )
                quota_stopped = True
                wx = pd.read_parquet(cache_p) if cache_p.exists() else pd.DataFrame(columns=CACHE_COLS)
                if len(wx):
                    wx["hour"] = pd.to_datetime(wx["hour"]).astype("datetime64[ns]")
        model_tables[mid] = wx
        u, v = interpolate_model_wind(g, wx)
        g[f"{mid}_u"] = u.astype(np.float32)
        g[f"{mid}_v"] = v.astype(np.float32)
        g[f"{mid}_err"] = np.hypot(g["pilot_u"] - u, g["pilot_v"] - v).astype(np.float32)

    g.to_parquet(out / "matched_groups.parquet", index=False)

    # 2. Per-model scores by height band (paired with ERA5 on the exact same rows).
    score_rows = []
    pu_all, pv_all = g["pilot_u"].to_numpy(float), g["pilot_v"].to_numpy(float)
    eu_all, ev_all = g["model_u"].to_numpy(float), g["model_v"].to_numpy(float)
    band_defs = [("near the ground", ["near the ground"]), ("middle", ["middle"]), ("free air", ["free air"]), ("free+middle", list(WEIGHT_BANDS))]
    active_ids = [s["id"] for s in selected if g[f"{s['id']}_u"].notna().any()]
    if not active_ids:
        print("\nNo cached historical forecast rows matched any wind groups yet. Re-run when API quota resets.")
        return

    for spec in selected:
        mid = spec["id"]
        mu_all, mv_all = g[f"{mid}_u"].to_numpy(float), g[f"{mid}_v"].to_numpy(float)
        for bname, blist in band_defs:
            mask = g["band"].isin(blist).to_numpy() & np.isfinite(mu_all) & np.isfinite(mv_all)
            if not mask.any():
                continue
            sub = g[mask]
            m = wind_metrics(pu_all[mask], pv_all[mask], mu_all[mask], mv_all[mask])
            e = wind_metrics(pu_all[mask], pv_all[mask], eu_all[mask], ev_all[mask])
            score_rows.append({
                "model": mid,
                "name": spec["name"],
                "region": spec["region"],
                "band": bname,
                "events": int(sub["event_key"].nunique()),
                "sites": int(sub["site_id"].nunique()),
                "first_date": str(sub["date"].min()),
                "last_date": str(sub["date"].max()),
                **m,
                "era5_rmse_ms": e.get("vector_rmse_ms"),
                "era5_median_ms": e.get("vector_median_ms"),
                "rel_rmse_vs_era5": m["vector_rmse_ms"] / max(e["vector_rmse_ms"], 0.25) if e.get("vector_rmse_ms") else None,
            })

    scores_df = pd.DataFrame(score_rows)
    scores_df.to_csv(out / "model_scores.csv", index=False)

    print("\n=== Model accuracy vs IGC pilot winds (free air + middle, paired with ERA5 on same rows) ===")
    fm = scores_df[scores_df["band"] == "free+middle"].sort_values("rel_rmse_vs_era5")
    if len(fm):
        cols = ["model", "groups", "events", "first_date", "vector_rmse_ms", "vector_median_ms", "era5_rmse_ms", "rel_rmse_vs_era5", "dir_error_deg", "dir_within_30"]
        print(fm[cols].round(3).to_string(index=False))

    # 3. Head-to-head on common global intersection with event-cluster bootstrap 95% CIs.
    common_summary: dict[str, Any] = {}
    core_present = [m for m in CORE_GLOBALS if m in active_ids]
    if len(core_present) >= 2:
        d_fm = g[g["band"].isin(WEIGHT_BANDS)].copy()
        common_mask = np.ones(len(d_fm), dtype=bool)
        for m in core_present:
            common_mask &= d_fm[f"{m}_u"].notna().to_numpy()
        dc = d_fm[common_mask]
        if len(dc) >= 20:
            pu_c, pv_c = dc["pilot_u"].to_numpy(float), dc["pilot_v"].to_numpy(float)
            evs_c = dc["event_key"].astype(str).to_numpy()
            print(f"\n=== Common intersection of {core_present} (free+middle: {len(dc):,} groups, {dc['event_key'].nunique()} events) ===")
            c_rows = []
            for m in [*core_present, "era5"]:
                mu = dc["model_u"].to_numpy(float) if m == "era5" else dc[f"{m}_u"].to_numpy(float)
                mv = dc["model_v"].to_numpy(float) if m == "era5" else dc[f"{m}_v"].to_numpy(float)
                met = wind_metrics(pu_c, pv_c, mu, mv)
                sq = (pu_c - mu) ** 2 + (pv_c - mv) ** 2
                lo, hi = cluster_bootstrap_ci(evs_c, sq, lambda x: float(np.sqrt(np.mean(x))))
                met["rmse_ci95"] = [round(lo, 3), round(hi, 3)]
                common_summary[m] = met
                c_rows.append({"model": m, "rmse": round(met["vector_rmse_ms"], 3), "ci95": f"[{lo:.2f}, {hi:.2f}]",
                               "median": round(met["vector_median_ms"], 3), "dir_deg": round(met["dir_error_deg"], 1) if met["dir_error_deg"] else None,
                               "within_30": f"{met['dir_within_30']:.1%}" if met["dir_within_30"] else "-"})
            print(pd.DataFrame(c_rows).to_string(index=False))

    # 4. Per-site breakdown (free+middle)
    site_rows = []
    d_fm = g[g["band"].isin(WEIGHT_BANDS)]
    for (sid, sname, country, reg), grp in d_fm.groupby(["site_id", "name", "country", "region"], dropna=False):
        for mid in active_ids:
            ok = grp[f"{mid}_u"].notna().to_numpy()
            if ok.sum() < 10:
                continue
            sub = grp[ok]
            m = wind_metrics(sub["pilot_u"].to_numpy(float), sub["pilot_v"].to_numpy(float),
                             sub[f"{mid}_u"].to_numpy(float), sub[f"{mid}_v"].to_numpy(float))
            site_rows.append({"site_id": sid, "name": sname, "country": country, "region": reg,
                              "model": mid, "events": int(sub["event_key"].nunique()), **m})
    by_site_cols = [
        "site_id", "name", "country", "region", "model", "events", "groups",
        "vector_rmse_ms", "vector_mae_ms", "vector_median_ms", "speed_bias_ms",
        "speed_corr", "dir_error_deg", "dir_within_30",
    ]
    by_site_df = pd.DataFrame(site_rows) if site_rows else pd.DataFrame(columns=by_site_cols)
    by_site_df.to_csv(out / "by_site.csv", index=False)

    # 5. Grouped 5-fold CV by competition (event_key) for blend strategies
    cv_res = run_grouped_cv(g, active_ids, k_folds=5)
    cv_res["common_global"] = common_summary
    cv_res["incomplete_download"] = quota_stopped
    (out / "cv_comparison.json").write_text(json.dumps(cv_res, indent=2), encoding="utf-8")

    print("\n=== Grouped 5-fold CV (competitions held out, free+middle) ===")
    if "strategies" in cv_res:
        srows = []
        for name, met in cv_res["strategies"].items():
            if not met.get("groups"):
                continue
            srows.append({
                "strategy": name,
                "rmse_ms": round(met["vector_rmse_ms"], 3),
                "ci95": met.get("vector_rmse_ci95"),
                "delta_vs_equal": met.get("delta_rmse_vs_equal_ms"),
                "median_ms": round(met["vector_median_ms"], 3),
                "dir_deg": round(met["dir_error_deg"], 1) if met.get("dir_error_deg") else None,
                "within_30": f"{met['dir_within_30']:.1%}" if met.get("dir_within_30") else "-",
            })
        print(pd.DataFrame(srows).to_string(index=False))
    else:
        print(" ", cv_res)

    # 6. Final fit of empirical weights on all events using CV-selected beta, export JSON
    best_beta = float(cv_res.get("median_beta", 4.0))
    final_weights = fit_hierarchy(d_fm, active_ids, beta=best_beta, shrink_k=SHRINKAGE_K)
    payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from_year": args.from_year,
        "bands_used": list(WEIGHT_BANDS),
        "incomplete_download": quota_stopped,
        "n_groups": int(len(d_fm)),
        "n_events": int(d_fm["event_key"].nunique()),
        "n_sites": int(d_fm["site_id"].nunique()),
        **final_weights,
        "cv_summary": {
            k: {m: v for m, v in met.items() if m in ("vector_rmse_ms", "vector_median_ms", "dir_error_deg", "dir_within_30", "delta_rmse_vs_equal_ms", "delta_rmse_ci95")}
            for k, met in cv_res.get("strategies", {}).items()
        },
    }
    weights_txt = json.dumps(payload, indent=2)
    (out / "forecast_weights.json").write_text(weights_txt, encoding="utf-8")
    web_dir = args.db / "web_data"
    published_web = False
    if web_dir.exists() and not quota_stopped:
        (web_dir / "forecast_weights.json").write_text(weights_txt, encoding="utf-8")
        published_web = True

    print("\nFinal global weights (relative wind skill vs IGC thermals):")
    for mid, w in sorted(payload["global"].items(), key=lambda kv: -kv[1]):
        rel = payload["global_rel_rmse"].get(mid)
        print(f"  {mid:12s}: weight={w:.3f}  (rel RMSE vs ERA5={rel:.3f})")
    msg = f"\nSaved outputs to {out}"
    if published_web:
        msg += f" and {web_dir / 'forecast_weights.json'}"
    elif quota_stopped:
        msg += " (did not overwrite web_data/forecast_weights.json because download paused at daily quota)"
    print(msg)


if __name__ == "__main__":
    main()
