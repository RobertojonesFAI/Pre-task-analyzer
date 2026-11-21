"""Live weather forecasts from Open-Meteo, converted to the ERA5 table layout used by
scripts/task_model.py so features_from() runs unchanged.

Each forecast model returns the same five-table dictionary as task_model.load_tables():
    single   hourly surface fields (blh_m, cloud_base_m, cape, cin, sunshine_wm2, cloud_total,
             cloud_low, rain_mmh, t2m_c, dewpoint_c, u100, v100, u10, v10, gust10)
    levels   pressure-level fields at 850 and 700 hPa (t_*, z_*, u_*, v_*, rh_*) keyed on time3
    extra    ground and air-mass fields (skin_temp_c, soil_moisture, freezing_level_m)
    static   chip ground height (ground_m)
    elev     chip mean route elevation (route_elev)

Models that do not publish a native boundary-layer height (blh_m) use a dry-adiabatic parcel
intersection against the model's 925/850/700/600 hPa temperature profile; each model reports
blh_source ("native" or "parcel") so callers and future bias checks can tell them apart.

Run as a script to probe live model coverage at sample sites:
    python -m api.forecast --probe
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
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pta import grid  # noqa: E402

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
CACHE_TTL_S = 3600       # 1 h cache per (chip_id, open-meteo model slug)
CACHE_MAX_ITEMS = 800
HTTP_TIMEOUT_S = 25
MAX_FORECAST_DAYS = 7    # today through +6 days (UTC); request 8 days so western tasks crossing 00 UTC stay covered
FETCH_FORECAST_DAYS = 8

# Supported forecast models: global models work at every site; regional models are offered only
# where the site (and route) falls inside their bounding box (south, north, west, east) and the
# forecast date is within their horizon.
MODELS: list[dict[str, Any]] = [
    # Global models
    {
        "id": "ecmwf_ifs",
        "name": "ECMWF IFS",
        "region": "global",
        "resolution_km": 9,
        "max_days": 6,
        "blh_source": "native",
        # Open-Meteo splits ECMWF IFS: 9 km surface (with BLH & CIN) + 0.25 deg upper-air levels.
        "slugs": ["ecmwf_ifs", "ecmwf_ifs025"],
        "bbox": None,
    },
    {
        "id": "ecmwf_aifs",
        "name": "ECMWF AIFS",
        "region": "global",
        "resolution_km": 28,
        "max_days": 6,
        "blh_source": "parcel",
        "slugs": ["ecmwf_aifs025_single"],
        "bbox": None,
    },
    {
        "id": "gfs",
        "name": "GFS",
        "region": "global",
        "resolution_km": 25,
        "max_days": 6,
        "blh_source": "native",
        "slugs": ["gfs_global"],
        "bbox": None,
    },
    {
        "id": "icon",
        "name": "ICON",
        "region": "global",
        "resolution_km": 13,
        "max_days": 6,
        "blh_source": "parcel",
        "slugs": ["icon_global"],
        "bbox": None,
    },
    {
        "id": "gem",
        "name": "GEM",
        "region": "global",
        "resolution_km": 15,
        "max_days": 6,
        "blh_source": "parcel",
        "slugs": ["gem_global"],
        "bbox": None,
    },
    # Regional high-resolution models
    {
        "id": "rrfs",
        "name": "RRFS",
        "region": "regional",
        "resolution_km": 3,
        "max_days": 3,
        "blh_source": "native",
        "slugs": ["ncep_rrfs_conus"],
        "bbox": (22.0, 52.0, -133.0, -62.0),
    },
    {
        "id": "hrdps",
        "name": "HRDPS",
        "region": "regional",
        "resolution_km": 2.5,
        "max_days": 2,
        "blh_source": "parcel",
        "slugs": ["gem_hrdps_continental"],
        "bbox": (36.0, 68.0, -148.0, -48.0),
    },
    {
        "id": "icon_d2",
        "name": "ICON-D2",
        "region": "regional",
        "resolution_km": 2,
        "max_days": 2,
        "blh_source": "parcel",
        "slugs": ["icon_d2"],
        "bbox": (43.5, 57.5, -3.5, 20.0),
    },
    {
        "id": "icon_eu",
        "name": "ICON-EU",
        "region": "regional",
        "resolution_km": 7,
        "max_days": 5,
        "blh_source": "parcel",
        "slugs": ["icon_eu"],
        "bbox": (30.0, 70.0, -23.0, 44.5),
    },
    {
        "id": "arome",
        "name": "AROME",
        "region": "regional",
        "resolution_km": 1.3,
        "max_days": 2,
        "blh_source": "parcel",
        "slugs": ["meteofrance_arome_france"],
        "bbox": (38.0, 55.0, -11.5, 15.5),
    },
]

MODEL_BY_ID: dict[str, dict[str, Any]] = {m["id"]: m for m in MODELS}
GLOBAL_SLUGS: set[str] = {s for m in MODELS if m["region"] == "global" for s in m["slugs"]}

HOURLY_VARS: list[str] = [
    "boundary_layer_height",
    "cape",
    "convective_inhibition",
    "shortwave_radiation",
    "cloud_cover",
    "cloud_cover_low",
    "precipitation",
    "temperature_2m",
    "dew_point_2m",
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_80m",
    "wind_direction_80m",
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_speed_120m",
    "wind_direction_120m",
    "wind_gusts_10m",
    "soil_temperature_0cm",
    "soil_temperature_0_to_7cm",
    "soil_temperature_0_to_10cm",
    "soil_moisture_0_to_1cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_0_to_10cm",
    "freezing_level_height",
    "temperature_925hPa",
    "geopotential_height_925hPa",
    "temperature_850hPa",
    "geopotential_height_850hPa",
    "wind_speed_850hPa",
    "wind_direction_850hPa",
    "relative_humidity_850hPa",
    "temperature_700hPa",
    "geopotential_height_700hPa",
    "wind_speed_700hPa",
    "wind_direction_700hPa",
    "relative_humidity_700hPa",
    "temperature_600hPa",
    "geopotential_height_600hPa",
]

# Cache: (chip_id, slug) -> (fetched_at_monotonic, elevation_m, times_str_list, {var: np.ndarray})
_CACHE: dict[tuple[int, str], tuple[float, float, list[str], dict[str, np.ndarray]]] = {}
_CACHE_LOCK = threading.Lock()


# ── HTTP client (IPv4-preferred + gzip, zero extra dependencies) ─────────────

class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection preferring IPv4 to avoid IPv6 TLS handshake stalls."""

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
        return self.do_open(_IPv4HTTPSConnection, req, context=ssl.create_default_context())


_OPENER = urllib.request.build_opener(_IPv4HTTPSHandler)


def _http_get_json(params: dict[str, Any]) -> Any:
    url = f"{OPEN_METEO_URL}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"Accept-Encoding": "gzip", "User-Agent": "pre-task-analyzer/1"})
    last_err: Exception | None = None
    for attempt in range(2):
        try:
            with _OPENER.open(req, timeout=HTTP_TIMEOUT_S) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding", "").lower() == "gzip":
                    raw = gzip.decompress(raw)
                return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError:
            raise
        except Exception as exc:
            last_err = exc
            if attempt == 0:
                time.sleep(0.5)
    raise RuntimeError(f"Open-Meteo request failed: {last_err}")


# ── Model discovery ──────────────────────────────────────────────────────────

def in_bbox(lat: float, lon: float, bbox: tuple[float, float, float, float] | None) -> bool:
    if bbox is None:
        return True
    s, n, w, e = bbox
    return s <= lat <= n and w <= lon <= e


def utc_today() -> date:
    return datetime.now(timezone.utc).date()


def date_bounds() -> tuple[date, date]:
    """Allowed forecast dates (UTC): yesterday (covers Americas local today) through +6 days."""
    today = utc_today()
    return today - timedelta(days=1), today + timedelta(days=MAX_FORECAST_DAYS - 1)


def available_models(lat: float, lon: float, target_date: date | None = None) -> list[dict[str, Any]]:
    """Models covering (lat, lon), annotated with horizon availability for target_date."""
    today = utc_today()
    delta = (target_date - today).days if target_date is not None else 0
    out = []
    for m in MODELS:
        if not in_bbox(lat, lon, m["bbox"]):
            continue
        in_horizon = target_date is None or (-1 <= delta <= m["max_days"])
        out.append({
            "id": m["id"],
            "name": m["name"],
            "region": m["region"],
            "resolution_km": m["resolution_km"],
            "max_days": m["max_days"],
            "blh_source": m["blh_source"],
            "available": in_horizon,
        })
    return out


# ── Fetch and cache per (chip_id, open-meteo slug) ───────────────────────────

def _to_float_array(values: list[Any] | None, n: int) -> np.ndarray:
    if not values or len(values) != n:
        return np.full(n, np.nan, dtype=np.float64)
    return np.array([np.nan if v is None else float(v) for v in values], dtype=np.float64)


def _prune_cache(now: float) -> None:
    expired = [k for k, v in _CACHE.items() if now - v[0] > CACHE_TTL_S]
    for k in expired:
        _CACHE.pop(k, None)
    if len(_CACHE) > CACHE_MAX_ITEMS:
        oldest = sorted(_CACHE.items(), key=lambda kv: kv[1][0])[: len(_CACHE) - CACHE_MAX_ITEMS]
        for k, _ in oldest:
            _CACHE.pop(k, None)


def _fetch_batch(chips: list[int], slugs: list[str]) -> dict[tuple[int, str], tuple[float, list[str], dict[str, np.ndarray]]]:
    """Fetch one Open-Meteo multi-coordinate, multi-model batch and parse per (chip_id, slug)."""
    lats, lons = [], []
    for cid in chips:
        la, lo = grid.chip_centre(int(cid))
        lats.append(f"{la:.4f}")
        lons.append(f"{lo:.4f}")
    data = _http_get_json({
        "latitude": ",".join(lats),
        "longitude": ",".join(lons),
        "hourly": ",".join(HOURLY_VARS),
        "models": ",".join(slugs),
        "wind_speed_unit": "ms",
        "timezone": "UTC",
        "past_days": 1,
        "forecast_days": FETCH_FORECAST_DAYS,
    })
    items = data if isinstance(data, list) else [data]
    if len(items) != len(chips):
        raise RuntimeError(f"Open-Meteo returned {len(items)} locations for {len(chips)} chips")
    multi = len(slugs) > 1
    parsed: dict[tuple[int, str], tuple[float, list[str], dict[str, np.ndarray]]] = {}
    for cid, item in zip(chips, items):
        elev = float(item.get("elevation") or 0.0)
        hourly = item.get("hourly") or {}
        times = list(hourly.get("time") or [])
        n = len(times)
        for slug in slugs:
            suffix = f"_{slug}" if multi else ""
            cols = {v: _to_float_array(hourly.get(f"{v}{suffix}"), n) for v in HOURLY_VARS}
            parsed[(int(cid), slug)] = (elev, times, cols)
    return parsed


def _ensure_cached(chips: list[int], slugs: list[str]) -> dict[str, str]:
    """Ensure every (chip, slug) is present in _CACHE. Returns {slug: error} for failed slugs."""
    now = time.monotonic()
    missing_by_slug: dict[str, list[int]] = {}
    with _CACHE_LOCK:
        _prune_cache(now)
        for slug in slugs:
            miss = [c for c in chips if (c, slug) not in _CACHE or now - _CACHE[(c, slug)][0] > CACHE_TTL_S]
            if miss:
                missing_by_slug[slug] = miss

    errors: dict[str, str] = {}
    if not missing_by_slug:
        return errors

    # Batch global slugs that share the same missing chip set into a single HTTP request.
    global_groups: dict[tuple[int, ...], list[str]] = {}
    regional_slugs: list[tuple[str, list[int]]] = []
    for slug, miss in missing_by_slug.items():
        if slug in GLOBAL_SLUGS:
            global_groups.setdefault(tuple(sorted(miss)), []).append(slug)
        else:
            regional_slugs.append((slug, sorted(miss)))

    for chip_tuple, gslugs in global_groups.items():
        try:
            fetched = _fetch_batch(list(chip_tuple), gslugs)
            ts = time.monotonic()
            with _CACHE_LOCK:
                for key, val in fetched.items():
                    _CACHE[key] = (ts, val[0], val[1], val[2])
        except Exception as exc:
            # Fall back to per-slug fetch if a combined request fails.
            for slug in gslugs:
                try:
                    fetched = _fetch_batch(list(chip_tuple), [slug])
                    ts = time.monotonic()
                    with _CACHE_LOCK:
                        for key, val in fetched.items():
                            _CACHE[key] = (ts, val[0], val[1], val[2])
                except Exception as inner:
                    errors[slug] = str(inner)

    for slug, miss in regional_slugs:
        try:
            fetched = _fetch_batch(miss, [slug])
            ts = time.monotonic()
            with _CACHE_LOCK:
                for key, val in fetched.items():
                    _CACHE[key] = (ts, val[0], val[1], val[2])
        except Exception as exc:
            errors[slug] = str(exc)

    return errors


# ── Meteorological conversions (Open-Meteo -> ERA5 column semantics) ─────────

def _wind_uv(speed: np.ndarray, direction_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert meteorological wind (speed m/s, direction blowing FROM in deg) to (u_east, v_north)."""
    rad = np.radians(direction_deg)
    return -speed * np.sin(rad), -speed * np.cos(rad)


def _coalesce(*arrays: np.ndarray) -> np.ndarray:
    out = np.full_like(arrays[0], np.nan, dtype=np.float64)
    for a in arrays:
        mask = np.isnan(out) & ~np.isnan(a)
        out[mask] = a[mask]
    return out


def _parcel_blh_asl(
    t2m: np.ndarray,
    rad_wm2: np.ndarray,
    z_sfc: float,
    prof_z: list[np.ndarray],
    prof_t: list[np.ndarray],
) -> np.ndarray:
    """Estimate convective mixing-layer top (m ASL) via dry-adiabatic parcel intersection
    against the available pressure-level temperature profile (925, 850, 700, 600 hPa).
    Uses a +1.0 C convective trigger excess when solar heating is active (>= 20 W/m2).
    """
    n = len(t2m)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        ts = t2m[i]
        if math.isnan(ts):
            continue
        pts = [(z_sfc, ts)]
        for za, ta in zip(prof_z, prof_t):
            z, t = za[i], ta[i]
            if not (math.isnan(z) or math.isnan(t)) and z > z_sfc + 30.0:
                pts.append((z, t))
        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p[0])
        excess = 1.0 if (not math.isnan(rad_wm2[i]) and rad_wm2[i] >= 20.0) else 0.0
        tp0 = ts + excess
        prev_z, prev_diff = pts[0][0], excess
        z_top = pts[-1][0]
        for z, te in pts[1:]:
            tp = tp0 - 0.0098 * (z - z_sfc)
            diff = tp - te
            if diff <= 0.0:
                if prev_diff <= 0.0:
                    z_top = prev_z
                else:
                    frac = prev_diff / (prev_diff - diff)
                    z_top = prev_z + frac * (z - prev_z)
                break
            prev_z, prev_diff = z, diff
        out[i] = min(max(z_top, z_sfc), z_sfc + 6000.0)
    return out


def _derived_freezing_asl(
    t2m: np.ndarray,
    z_sfc: float,
    prof_z: list[np.ndarray],
    prof_t: list[np.ndarray],
) -> np.ndarray:
    """Linear 0 C level (m ASL) from surface and upper-air temperatures when freezing_level_height
    is absent."""
    n = len(t2m)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        ts = t2m[i]
        if math.isnan(ts):
            continue
        if ts <= 0.0:
            out[i] = z_sfc
            continue
        pts = [(z_sfc, ts)]
        for za, ta in zip(prof_z, prof_t):
            z, t = za[i], ta[i]
            if not (math.isnan(z) or math.isnan(t)) and z > z_sfc + 30.0:
                pts.append((z, t))
        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p[0])
        prev_z, prev_t = pts[0]
        found = False
        for z, t in pts[1:]:
            if t <= 0.0:
                if prev_t <= 0.0:
                    out[i] = prev_z
                else:
                    out[i] = prev_z + (prev_t / (prev_t - t)) * (z - prev_z)
                found = True
                break
            prev_z, prev_t = z, t
        if not found:
            # Extrapolate above highest level with the last segment lapse rate (min 5 C/km).
            z0, t0 = pts[-2]
            z1, t1 = pts[-1]
            lapse = max(0.005, (t0 - t1) / max(z1 - z0, 1.0))
            out[i] = z1 + t1 / lapse
    return out


def _build_model_tables(
    chips: list[int],
    spec: dict[str, Any],
    base_tables: dict[str, pd.DataFrame | None],
) -> dict[str, pd.DataFrame | None] | None:
    """Assemble task_model-compatible weather tables for one forecast model across `chips`."""
    static_df = base_tables.get("static")
    elev_df = base_tables.get("elev")
    ground_lookup = (
        dict(zip(static_df["chip_id"].astype(int), static_df["ground_m"].astype(float)))
        if static_df is not None and len(static_df)
        else {}
    )
    route_elev_lookup = (
        dict(zip(elev_df["chip_id"].astype(int), elev_df["route_elev"].astype(float)))
        if elev_df is not None and len(elev_df)
        else {}
    )

    single_frames, level_frames, extra_frames = [], [], []
    static_rows, elev_rows = [], []

    with _CACHE_LOCK:
        cached_per_chip = {}
        for cid in chips:
            entries = []
            for slug in spec["slugs"]:
                val = _CACHE.get((cid, slug))
                if val is not None:
                    entries.append(val)
            if not entries:
                return None
            cached_per_chip[cid] = entries

    for cid in chips:
        entries = cached_per_chip[cid]
        # Primary slug supplies timestamps and DEM elevation; secondary fills missing variables
        # (e.g. ecmwf_ifs surface + ecmwf_ifs025 pressure levels).
        _, om_elev, times_str, primary_cols = entries[0]
        if not times_str:
            return None
        n = len(times_str)
        times = pd.to_datetime(times_str).astype("datetime64[ns]")

        merged: dict[str, np.ndarray] = {}
        time_index = {t: idx for idx, t in enumerate(times_str)}
        for v in HOURLY_VARS:
            arrs = []
            for e in entries:
                e_times, e_vals = e[2], e[3][v]
                if e_times == times_str:
                    arrs.append(e_vals)
                else:
                    aligned = np.full(n, np.nan, dtype=np.float64)
                    for src_i, ts in enumerate(e_times):
                        dst_i = time_index.get(ts)
                        if dst_i is not None and src_i < len(e_vals):
                            aligned[dst_i] = e_vals[src_i]
                    arrs.append(aligned)
            merged[v] = _coalesce(*arrs) if len(arrs) > 1 else arrs[0].copy()

        ground_m = ground_lookup.get(cid, om_elev)
        route_elev = route_elev_lookup.get(cid, om_elev)
        static_rows.append({"chip_id": cid, "ground_m": ground_m})
        elev_rows.append({"chip_id": cid, "route_elev": route_elev})

        t2m = merged["temperature_2m"]
        td2m = merged["dew_point_2m"]
        rad = merged["shortwave_radiation"]
        cloud_total = np.clip(merged["cloud_cover"] / 100.0, 0.0, 1.0)
        cloud_low = np.clip(merged["cloud_cover_low"] / 100.0, 0.0, 1.0)

        prof_z = [
            merged["geopotential_height_925hPa"],
            merged["geopotential_height_850hPa"],
            merged["geopotential_height_700hPa"],
            merged["geopotential_height_600hPa"],
        ]
        prof_t = [
            merged["temperature_925hPa"],
            merged["temperature_850hPa"],
            merged["temperature_700hPa"],
            merged["temperature_600hPa"],
        ]

        # Boundary-layer height (m above ground_m): native where published, else parcel estimate.
        blh_native = merged["boundary_layer_height"]
        if np.isfinite(blh_native).all():
            blh_m = blh_native
        else:
            blh_asl = _parcel_blh_asl(t2m, rad, om_elev, prof_z, prof_t)
            blh_parcel = np.maximum(0.0, blh_asl - ground_m)
            blh_m = _coalesce(blh_native, blh_parcel)

        # Cloud base (m above ground_m): LCL when clouds are present, NaN on clear blue sky
        # (matches ERA5 cloud_base_height semantics).
        lcl_asl = om_elev + np.maximum(0.0, 125.0 * (t2m - td2m))
        cloud_base_m = np.where(cloud_total >= 0.05, np.maximum(0.0, lcl_asl - ground_m), np.nan)

        # Wind components (m/s, east/north). Prefer 100 m; fall back to vector average of 80 & 120 m
        # (needed for GEM / HRDPS), then 80 m, then 10 m.
        u10, v10 = _wind_uv(merged["wind_speed_10m"], merged["wind_direction_10m"])
        u100_raw, v100_raw = _wind_uv(merged["wind_speed_100m"], merged["wind_direction_100m"])
        u80, v80 = _wind_uv(merged["wind_speed_80m"], merged["wind_direction_80m"])
        u120, v120 = _wind_uv(merged["wind_speed_120m"], merged["wind_direction_120m"])
        u100_mid = 0.5 * (u80 + u120)
        v100_mid = 0.5 * (v80 + v120)
        u100 = _coalesce(u100_raw, u100_mid, u80, u10)
        v100 = _coalesce(v100_raw, v100_mid, v80, v10)

        u850, v850 = _wind_uv(merged["wind_speed_850hPa"], merged["wind_direction_850hPa"])
        u700, v700 = _wind_uv(merged["wind_speed_700hPa"], merged["wind_direction_700hPa"])

        cin = np.abs(merged["convective_inhibition"])  # ERA5 stores CIN as positive J/kg

        skin_temp = _coalesce(
            merged["soil_temperature_0cm"],
            merged["soil_temperature_0_to_7cm"],
            merged["soil_temperature_0_to_10cm"],
        )
        soil_moist = _coalesce(
            merged["soil_moisture_0_to_7cm"],
            merged["soil_moisture_0_to_10cm"],
            merged["soil_moisture_0_to_1cm"],
        )

        flh_native = merged["freezing_level_height"]
        if np.isfinite(flh_native).all():
            flh_asl = flh_native
        else:
            flh_asl = _coalesce(flh_native, _derived_freezing_asl(t2m, om_elev, prof_z, prof_t))
        freezing_level_m = np.maximum(0.0, flh_asl - ground_m)

        single_frames.append(pd.DataFrame({
            "chip_id": cid,
            "time": times,
            "blh_m": blh_m,
            "cloud_base_m": cloud_base_m,
            "cape": merged["cape"],
            "cin": cin,
            "sensible_heat_wm2": np.nan,
            "sunshine_wm2": rad,
            "cloud_total": cloud_total,
            "cloud_low": cloud_low,
            "rain_mmh": merged["precipitation"],
            "t2m_c": t2m,
            "dewpoint_c": td2m,
            "u100": u100,
            "v100": v100,
            "u10": u10,
            "v10": v10,
            "gust10": merged["wind_gusts_10m"],
        }))
        level_frames.append(pd.DataFrame({
            "chip_id": cid,
            "time3": times,
            "t_850": merged["temperature_850hPa"],
            "t_700": merged["temperature_700hPa"],
            "z_850": merged["geopotential_height_850hPa"],
            "z_700": merged["geopotential_height_700hPa"],
            "u_850": u850,
            "v_850": v850,
            "u_700": u700,
            "v_700": v700,
            "rh_700": merged["relative_humidity_700hPa"],
            "rh_850": merged["relative_humidity_850hPa"],
        }))
        extra_frames.append(pd.DataFrame({
            "chip_id": cid,
            "time": times,
            "skin_temp_c": skin_temp,
            "soil_moisture": soil_moist,
            "freezing_level_m": freezing_level_m,
        }))

    return {
        "single": pd.concat(single_frames, ignore_index=True),
        "levels": pd.concat(level_frames, ignore_index=True),
        "extra": pd.concat(extra_frames, ignore_index=True),
        "static": pd.DataFrame(static_rows),
        "elev": pd.DataFrame(elev_rows),
    }


def fetch_models(
    chips: list[int] | np.ndarray,
    model_ids: list[str],
    target_date: date,
    base_tables: dict[str, pd.DataFrame | None],
) -> tuple[dict[str, dict[str, pd.DataFrame | None]], list[dict[str, str]]]:
    """Download (or serve from cache) forecast weather tables for each requested model.

    Returns:
        tables_by_model: {model_id: tables_dict} for models that cover the chips and date
        skipped: [{"id": model_id, "reason": ...}] for out-of-domain, out-of-horizon, or failed models
    """
    chip_list = sorted({int(c) for c in chips})
    centres = [grid.chip_centre(c) for c in chip_list]
    today = utc_today()
    delta = (target_date - today).days

    eligible: list[dict[str, Any]] = []
    skipped: list[dict[str, str]] = []
    for mid in model_ids:
        spec = MODEL_BY_ID.get(mid)
        if spec is None:
            skipped.append({"id": mid, "reason": "unknown model"})
            continue
        if delta > spec["max_days"]:
            skipped.append({"id": mid, "reason": f"horizon is {spec['max_days']} days"})
            continue
        if spec["bbox"] is not None and not all(in_bbox(la, lo, spec["bbox"]) for la, lo in centres):
            skipped.append({"id": mid, "reason": "route outside regional domain"})
            continue
        eligible.append(spec)

    slugs_needed: list[str] = []
    for spec in eligible:
        for slug in spec["slugs"]:
            if slug not in slugs_needed:
                slugs_needed.append(slug)

    slug_errors = _ensure_cached(chip_list, slugs_needed) if slugs_needed else {}

    tables_by_model: dict[str, dict[str, pd.DataFrame | None]] = {}
    for spec in eligible:
        failed = [s for s in spec["slugs"] if s in slug_errors]
        if failed:
            skipped.append({"id": spec["id"], "reason": f"fetch failed ({slug_errors[failed[0]]})"})
            continue
        tbls = _build_model_tables(chip_list, spec, base_tables)
        if tbls is None:
            skipped.append({"id": spec["id"], "reason": "no data returned"})
            continue
        tables_by_model[spec["id"]] = tbls

    return tables_by_model, skipped


# ── CLI coverage probe ───────────────────────────────────────────────────────

def _run_probe() -> None:
    """Query all configured models at three reference sites and print coverage."""
    probe_sites = [
        ("Cervidae (US)", 43.62, -115.99),
        ("St-Andre (EU)", 43.97, 6.51),
        ("Roldanillo (SA)", 4.41, -76.15),
    ]
    tomorrow = utc_today() + timedelta(days=1)
    for name, lat, lon in probe_sites:
        cid = int(grid.chip_id(lat, lon))
        avail = available_models(lat, lon, tomorrow)
        ids = [m["id"] for m in avail if m["available"]]
        print(f"\n=== {name} ({lat}, {lon}, chip {cid}) for {tomorrow} ===", flush=True)
        tables_map, skipped = fetch_models([cid], ids, tomorrow, {"static": None, "elev": None})
        for mid in ids:
            if mid not in tables_map:
                reason = next((s["reason"] for s in skipped if s["id"] == mid), "skipped")
                print(f"  {mid:12s} SKIPPED: {reason}")
                continue
            t = tables_map[mid]
            s, lv, ex = t["single"], t["levels"], t["extra"]
            miss = [c for df in (s, lv, ex) for c in df.columns if c not in ("chip_id", "time", "time3") and df[c].isna().all()]
            blh_max = float(s["blh_m"].max()) if s["blh_m"].notna().any() else math.nan
            w700_max = float(np.hypot(lv["u_700"], lv["v_700"]).max()) if lv["u_700"].notna().any() else math.nan
            spec = MODEL_BY_ID[mid]
            print(
                f"  {mid:12s} ({spec['name']:10s}, blh={spec['blh_source']:6s}) "
                f"blh_max={blh_max:5.0f} m  w700_max={w700_max:4.1f} m/s  missing={miss or 'none'}",
                flush=True,
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Open-Meteo forecast fetcher and coverage probe.")
    parser.add_argument("--probe", action="store_true", help="Probe configured models at reference sites")
    args = parser.parse_args()
    if args.probe:
        _run_probe()
    else:
        parser.print_help()
