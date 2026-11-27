#!/usr/bin/env python3
"""Multi-variable verification of archived Open-Meteo forecasts against high-precision IGC
atmospheric truth (`database/truth/` from notebook 15) and ERA5 reanalysis.

Evaluates all 9 archived forecast models across four physical pillars without re-downloading wind:

  Pillar 1  Clean thermal & glide wind (`truth_wind_groups`, `truth_glide_wind`) using the
            existing wind cache (`DB/weather/hist_forecast/` from notebook 14; 0 API calls).
  Pillar 2  Calibrated multi-pilot ASL top-of-lift (`truth_ceiling`, excluding airspace caps),
            scoring native BLH (GFS, ECMWF IFS 9 km) vs dry-adiabatic parcel BLH and calibrating
            the parcel convective trigger excess (+0.0..+2.5 C) via 5-fold competition-grouped CV.
  Pillar 3  Core 60 s thermal climb strength (`truth_climb_groups.core60_p75_ms`) predicted from
            each model's thermodynamic profile in 5-fold competition-grouped CV alongside ERA5.
  Pillar 4  End-to-end competition task speed & goal share (`truth_tasks`, 2022+) by running
            leakage-free `task_v1` folds on each model's historical route weather: raw vs
            monotone quantile-mapped (thermo-only QM and full-weather QM to ERA5), comparing
            equal mean, median, clean-wind weights, and multi-variable composite weights.
            Every gain is tested with a paired bootstrap on identical tasks (whole competitions
            resampled), and quantile mapping is only recommended when its gain is significant.
            Composite weights shrink each pillar's skill toward the median by sample size.

Thermodynamic historical forecasts are cached per model in `DB/weather/hist_forecast_thermo/`
with 14-day window batching and clean hourly/daily Open-Meteo quota pausing.

    python3 scripts/forecast_multivar_eval.py /content/drive/MyDrive/IGCS/database --wind-only
    python3 scripts/forecast_multivar_eval.py /content/drive/MyDrive/IGCS/database --models ecmwf_ifs --download-only
    python3 scripts/forecast_multivar_eval.py /content/drive/MyDrive/IGCS/database --skip-download
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from api import forecast as FC  # noqa: E402
import forecast_wind_weights as FWW  # noqa: E402
from pta import grid  # noqa: E402
import task_baseline as TB  # noqa: E402
import task_model as TM  # noqa: E402

MERGE_GAP_DAYS = FWW.MERGE_GAP_DAYS
MAX_WINDOW_DAYS = FWW.MAX_WINDOW_DAYS
BATCH_CHIPS = FWW.BATCH_CHIPS
CHECKPOINT_EVERY = 20
SHRINKAGE_K = 20.0
# Composite weights: each pillar's rel_rmse_vs_era5 is shrunk toward the median across models
# by n / (n + k), where n is that model's sample in the pillar (groups, or tasks for "task").
# A model with k samples in a pillar keeps half of its own deviation from the median.
COMPOSITE_SHRINK_N = {"wind": 500.0, "ceiling": 500.0, "climb": 500.0, "task": 100.0}
PAIRED_BOOTSTRAP_ITERS = 2000
PARCEL_EXCESS_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)
QM_PROBS = np.array(
    [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99],
    dtype=np.float64,
)

# Features eligible for Forecast -> ERA5 quantile mapping before task_v1 evaluation.
THERMO_QM_FEATURES = [
    "blh_mean", "blh_max", "usable_m", "cloud_base_m", "cloud_base_above_m",
    "cape", "cin", "sunshine_wm2", "cloud_total", "cloud_low", "spread_c", "t2m_c",
    "lapse_850_700", "rh_700", "rh_850", "skin_minus_air_c", "soil_moisture", "freezing_level_m",
]
WIND_QM_FEATURES = [
    "wind100", "gust10", "wind_700", "wind_850",
    "tailwind_100", "crosswind_100", "tailwind_700", "crosswind_700",
]
ALL_QM_FEATURES = THERMO_QM_FEATURES + WIND_QM_FEATURES

# Verified historical Open-Meteo thermodynamic variable sets per slug (probed live against
# historical-forecast-api.open-meteo.com so unsupported variables are never requested).
_CORE_THERMO = [
    "shortwave_radiation", "cloud_cover", "cloud_cover_low", "precipitation",
    "temperature_2m", "dew_point_2m",
]
_PRESSURE_THERMO = [
    "temperature_925hPa", "geopotential_height_925hPa",
    "temperature_850hPa", "geopotential_height_850hPa", "relative_humidity_850hPa",
    "temperature_700hPa", "geopotential_height_700hPa", "relative_humidity_700hPa",
    "temperature_600hPa", "geopotential_height_600hPa",
]

# Archive start dates (min_date / blh_min_date) verified live against Open-Meteo Historical Forecast API
# with a 1-3 day safety margin so pre-archive dates are skipped before batching.
THERMO_MODELS: list[dict[str, Any]] = [
    {
        "id": "ecmwf_ifs", "name": "ECMWF IFS", "slug": "ecmwf_ifs025", "blh_slug": "ecmwf_ifs",
        "min_date": "2024-02-01", "blh_min_date": "2024-09-01",
        "region": "global", "bbox": None,
        "vars": [*_CORE_THERMO, "cape", "wind_gusts_10m", "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm", *_PRESSURE_THERMO],
    },
    {
        "id": "gfs", "name": "GFS", "slug": "gfs_global", "blh_slug": None,
        "min_date": "2022-01-01",
        "region": "global", "bbox": None,
        "vars": [*_CORE_THERMO, "boundary_layer_height", "cape", "convective_inhibition", "wind_gusts_10m",
                 "soil_temperature_0_to_10cm", "soil_moisture_0_to_10cm", "freezing_level_height", *_PRESSURE_THERMO],
    },
    {
        "id": "icon", "name": "ICON", "slug": "icon_global", "blh_slug": None,
        "min_date": "2022-11-15",
        "region": "global", "bbox": None,
        "vars": [*_CORE_THERMO, "cape", "wind_gusts_10m", "soil_temperature_0cm", "soil_moisture_0_to_1cm",
                 "freezing_level_height", *_PRESSURE_THERMO],
    },
    {
        "id": "gem", "name": "GEM", "slug": "gem_global", "blh_slug": None,
        "min_date": "2022-11-20",
        "region": "global", "bbox": None,
        "vars": [*_CORE_THERMO, "cape", "wind_gusts_10m", "soil_temperature_0_to_10cm", "soil_moisture_0_to_10cm",
                 *_PRESSURE_THERMO],
    },
    {
        "id": "ecmwf_aifs", "name": "ECMWF AIFS", "slug": "ecmwf_aifs025_single", "blh_slug": None,
        "min_date": "2025-02-15",
        "region": "global", "bbox": None,
        "vars": [*_CORE_THERMO, "soil_temperature_0_to_7cm", "soil_moisture_0_to_7cm", *_PRESSURE_THERMO],
    },
    {
        "id": "icon_eu", "name": "ICON-EU", "slug": "icon_eu", "blh_slug": None,
        "min_date": "2022-11-20",
        "region": "regional", "bbox": (30.0, 70.0, -23.0, 44.5),
        "vars": [*_CORE_THERMO, "cape", "wind_gusts_10m", "soil_temperature_0cm", "soil_moisture_0_to_1cm",
                 "freezing_level_height", *_PRESSURE_THERMO],
    },
    {
        "id": "icon_d2", "name": "ICON-D2", "slug": "icon_d2", "blh_slug": None,
        "min_date": "2022-11-10",
        "region": "regional", "bbox": (43.5, 57.5, -3.5, 20.0),
        "vars": [*_CORE_THERMO, "cape", "wind_gusts_10m", "soil_temperature_0cm", "soil_moisture_0_to_1cm",
                 "freezing_level_height", *_PRESSURE_THERMO],
    },
    {
        "id": "arome", "name": "AROME", "slug": "meteofrance_arome_france", "blh_slug": None,
        "min_date": "2023-12-25",
        "region": "regional", "bbox": (38.0, 55.0, -11.5, 15.5),
        "vars": [*_CORE_THERMO, "cape", "wind_gusts_10m", *_PRESSURE_THERMO],
    },
    {
        "id": "hrdps", "name": "HRDPS", "slug": "gem_hrdps_continental", "blh_slug": None,
        "min_date": "2023-03-01",
        "region": "regional", "bbox": (36.0, 68.0, -148.0, -48.0),
        "vars": [*_CORE_THERMO, "cape", "wind_gusts_10m", "soil_temperature_0_to_10cm", "soil_moisture_0_to_10cm",
                 *_PRESSURE_THERMO],
    },
]
MODEL_BY_ID = {m["id"]: m for m in THERMO_MODELS}
CORE_GLOBALS = ["ecmwf_ifs", "gfs", "icon", "gem"]

THERMO_CACHE_COLS = [
    "chip_id", "hour", "om_elev",
    "blh_native_m", "t2m_c", "dewpoint_c", "sunshine_wm2", "cloud_total", "cloud_low",
    "rain_mmh", "cape", "cin", "gust10", "skin_temp_c", "soil_moisture", "freezing_level_m",
    "t_925", "z_925", "t_850", "z_850", "rh_850", "t_700", "z_700", "rh_700", "t_600", "z_600",
]


# ── Pure mathematical helpers (parcel BLH, quantile mapping, task regressor) ─

def parcel_blh_asl_with_excess(
    t2m: np.ndarray,
    rad_wm2: np.ndarray,
    z_sfc: np.ndarray,
    prof_z: list[np.ndarray],
    prof_t: list[np.ndarray],
    excess_c: float = 1.0,
) -> np.ndarray:
    """Dry-adiabatic parcel mixing-layer top (m ASL) matching api/forecast.py:_parcel_blh_asl,
    generalized for per-row `z_sfc` and tunable convective trigger `excess_c` (deg C)."""
    n = len(t2m)
    out = np.full(n, np.nan, dtype=np.float64)
    for i in range(n):
        ts = float(t2m[i])
        zs = float(z_sfc[i])
        if math.isnan(ts) or math.isnan(zs):
            continue
        pts = [(zs, ts)]
        for za, ta in zip(prof_z, prof_t):
            z, t = float(za[i]), float(ta[i])
            if not (math.isnan(z) or math.isnan(t)) and z > zs + 30.0:
                pts.append((z, t))
        if len(pts) < 2:
            continue
        pts.sort(key=lambda p: p[0])
        rad_i = float(rad_wm2[i])
        exc = float(excess_c) if (not math.isnan(rad_i) and rad_i >= 20.0) else 0.0
        tp0 = ts + exc
        prev_z, prev_diff = pts[0][0], exc
        z_top = pts[-1][0]
        for z, te in pts[1:]:
            tp = tp0 - 0.0098 * (z - zs)
            diff = tp - te
            if diff <= 0.0:
                if prev_diff <= 0.0:
                    z_top = prev_z
                else:
                    frac = prev_diff / (prev_diff - diff)
                    z_top = prev_z + frac * (z - prev_z)
                break
            prev_z, prev_diff = z, diff
        out[i] = min(max(z_top, zs), zs + 6000.0)
    return out


def fit_quantile_map(
    src: np.ndarray,
    ref: np.ndarray,
    min_pairs: int = 30,
    probs: np.ndarray = QM_PROBS,
) -> dict[str, Any] | None:
    """Fit a monotone piecewise-linear quantile map `src -> ref` strictly on paired finite values."""
    s = np.asarray(src, dtype=np.float64)
    r = np.asarray(ref, dtype=np.float64)
    ok = np.isfinite(s) & np.isfinite(r)
    n = int(ok.sum())
    if n < min_pairs:
        return None
    s_ok, r_ok = s[ok], r[ok]
    if float(np.std(s_ok)) < 1e-6 or float(np.std(r_ok)) < 1e-6:
        return None
    qs = np.quantile(s_ok, probs)
    qr = np.quantile(r_ok, probs)
    # Collapse duplicate x-knots by averaging corresponding reference quantiles so np.interp stays strict.
    ux, inv = np.unique(qs, return_inverse=True)
    if len(ux) < 3:
        return None
    uy = np.array([float(np.mean(qr[inv == k])) for k in range(len(ux))], dtype=np.float64)
    uy = np.maximum.accumulate(uy)
    return {
        "n": n,
        "src_q": [round(float(x), 4) for x in ux],
        "ref_q": [round(float(y), 4) for y in uy],
        "shift_lo": round(float(uy[0] - ux[0]), 4),
        "shift_hi": round(float(uy[-1] - ux[-1]), 4),
    }


def apply_quantile_map(values: np.ndarray, qmap: dict[str, Any] | None) -> np.ndarray:
    """Apply a fitted monotone quantile map with linear tail shift outside the training knots."""
    v = np.asarray(values, dtype=np.float64)
    if qmap is None:
        return v.copy()
    ux = np.asarray(qmap["src_q"], dtype=np.float64)
    uy = np.asarray(qmap["ref_q"], dtype=np.float64)
    out = np.full_like(v, np.nan, dtype=np.float64)
    ok = np.isfinite(v)
    if not ok.any():
        return out
    x = v[ok]
    mapped = np.interp(x, ux, uy)
    # Preserve tail sensitivity outside [ux[0], ux[-1]] via constant boundary offset.
    lo = x < ux[0]
    hi = x > ux[-1]
    if lo.any():
        mapped[lo] = x[lo] + float(qmap["shift_lo"])
    if hi.any():
        mapped[hi] = x[hi] + float(qmap["shift_hi"])
    out[ok] = mapped
    return out


def _make_task_regressor():
    try:
        return TM.model()
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15, min_samples_leaf=20, random_state=7)


def _coalesce_arr(*arrays: np.ndarray) -> np.ndarray:
    out = np.full_like(arrays[0], np.nan, dtype=np.float64)
    for a in arrays:
        mask = np.isnan(out) & np.isfinite(a)
        out[mask] = a[mask]
    return out


# ── Step 1: Compact target planning & resumable thermo downloader ────────────

def build_thermo_targets(
    db: Path,
    from_year: int,
    max_sites: int | None = None,
) -> pd.DataFrame:
    """Union of (site_id, chip_id, hour, date) needed for ceiling, climb, and task route verification."""
    truth_dir = db / "truth"
    frames = []
    for fname in ("truth_ceiling.parquet", "truth_climb_groups.parquet"):
        p = truth_dir / fname
        if p.exists():
            d = pd.read_parquet(p, columns=["site_id", "chip_id", "hour"])
            d["hour"] = pd.to_datetime(d["hour"]).astype("datetime64[ns]")
            d = d[d["hour"].dt.year >= from_year]
            frames.append(d)

    # Also include every route chip-hour for 2022+ tasks so Pillar 4 (end-to-end task_v1) has full coverage.
    if (db / "tasks.parquet").exists() and (db / "results.parquet").exists():
        t = TB.tasks_table(db)
        t = t[t["date"].dt.year >= from_year].copy()
        if len(t):
            rt = TM.routes(db, t)
            sc = pd.read_parquet(db / "site_chips.parquet", columns=["site_id", "chip_id", "in_area"])
            area = {s_: g.loc[g["in_area"].fillna(False).astype(bool), "chip_id"].to_numpy() for s_, g in sc.groupby("site_id")}
            start, end = TM.windows(t)
            t_rows = []
            for i, r in enumerate(t.itertuples(index=False)):
                chips = rt[r.task_key][0] if r.task_key in rt else area.get(r.site_id, np.array([], int))
                if not len(chips):
                    continue
                hrs = pd.date_range(pd.Timestamp(start.iloc[i]).floor("h"), pd.Timestamp(end.iloc[i]).floor("h"), freq="h")
                for h in hrs:
                    for c in chips:
                        t_rows.append((r.site_id, int(c), h))
            if t_rows:
                td = pd.DataFrame(t_rows, columns=["site_id", "chip_id", "hour"])
                td["hour"] = pd.to_datetime(td["hour"]).astype("datetime64[ns]")
                frames.append(td)

    if not frames:
        return pd.DataFrame(columns=["site_id", "chip_id", "hour", "date"])

    all_t = pd.concat(frames, ignore_index=True).drop_duplicates(["chip_id", "hour"])
    all_t["date"] = all_t["hour"].dt.strftime("%Y-%m-%d")
    if max_sites is not None and max_sites > 0:
        top = all_t.groupby("site_id").size().sort_values(ascending=False).head(max_sites).index
        all_t = all_t[all_t["site_id"].isin(top)].copy()
    return all_t.reset_index(drop=True)


def _parse_thermo_batch(
    items: list[dict[str, Any]],
    chips: list[int],
    needed_hours: dict[int, set[pd.Timestamp]],
) -> pd.DataFrame:
    frames = []
    for cid, item in zip(chips, items):
        hourly = item.get("hourly") or {}
        times_str = hourly.get("time") or []
        if not times_str:
            continue
        n = len(times_str)
        om_elev = float(item.get("elevation") or 0.0)
        times = pd.to_datetime(times_str).astype("datetime64[ns]")
        want = needed_hours.get(cid)
        if want:
            # Keep target hours and also -24h hours inside the same response window for rain_day_before_mm.
            want_prev = {h - pd.Timedelta(days=1) for h in want}
            keep_mask = times.isin(want | want_prev)
            if not keep_mask.any():
                continue
        else:
            keep_mask = np.ones(n, dtype=bool)

        arr = lambda k: FWW._to_arr(hourly.get(k), n)  # noqa: E731
        t2m = arr("temperature_2m")
        td2m = arr("dew_point_2m")
        rad = arr("shortwave_radiation")
        cld_tot = np.clip(arr("cloud_cover") / 100.0, 0.0, 1.0)
        cld_low = np.clip(arr("cloud_cover_low") / 100.0, 0.0, 1.0)
        rain = arr("precipitation")
        cape = arr("cape")
        cin = np.abs(arr("convective_inhibition"))
        gust = arr("wind_gusts_10m")
        blh = arr("boundary_layer_height")
        skin = _coalesce_arr(
            arr("soil_temperature_0cm"),
            arr("soil_temperature_0_to_7cm"),
            arr("soil_temperature_0_to_10cm"),
        )
        soil_m = _coalesce_arr(
            arr("soil_moisture_0_to_7cm"),
            arr("soil_moisture_0_to_10cm"),
            arr("soil_moisture_0_to_1cm"),
        )
        flh_asl = arr("freezing_level_height")
        flh_agl = np.where(np.isfinite(flh_asl), np.maximum(0.0, flh_asl - om_elev), np.nan)

        m_cnt = int(keep_mask.sum())
        df = pd.DataFrame({
            "chip_id": np.full(m_cnt, cid, dtype=np.int64),
            "hour": times[keep_mask],
            "om_elev": np.full(m_cnt, om_elev, dtype=np.float32),
            "blh_native_m": blh[keep_mask].astype(np.float32),
            "t2m_c": t2m[keep_mask].astype(np.float32),
            "dewpoint_c": td2m[keep_mask].astype(np.float32),
            "sunshine_wm2": rad[keep_mask].astype(np.float32),
            "cloud_total": cld_tot[keep_mask].astype(np.float32),
            "cloud_low": cld_low[keep_mask].astype(np.float32),
            "rain_mmh": rain[keep_mask].astype(np.float32),
            "cape": cape[keep_mask].astype(np.float32),
            "cin": cin[keep_mask].astype(np.float32),
            "gust10": gust[keep_mask].astype(np.float32),
            "skin_temp_c": skin[keep_mask].astype(np.float32),
            "soil_moisture": soil_m[keep_mask].astype(np.float32),
            "freezing_level_m": flh_agl[keep_mask].astype(np.float32),
            "t_925": arr("temperature_925hPa")[keep_mask].astype(np.float32),
            "z_925": arr("geopotential_height_925hPa")[keep_mask].astype(np.float32),
            "t_850": arr("temperature_850hPa")[keep_mask].astype(np.float32),
            "z_850": arr("geopotential_height_850hPa")[keep_mask].astype(np.float32),
            "rh_850": arr("relative_humidity_850hPa")[keep_mask].astype(np.float32),
            "t_700": arr("temperature_700hPa")[keep_mask].astype(np.float32),
            "z_700": arr("geopotential_height_700hPa")[keep_mask].astype(np.float32),
            "rh_700": arr("relative_humidity_700hPa")[keep_mask].astype(np.float32),
            "t_600": arr("temperature_600hPa")[keep_mask].astype(np.float32),
            "z_600": arr("geopotential_height_600hPa")[keep_mask].astype(np.float32),
        })
        core = ["t2m_c", "t_850", "t_700", "blh_native_m"]
        df = df[df[core].notna().any(axis=1)]
        if len(df):
            frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=THERMO_CACHE_COLS)


def _download_slug_batches(
    targets: pd.DataFrame,
    slug: str,
    req_vars: list[str],
    bbox: tuple[float, float, float, float] | None,
    cache_path: Path,
    done_path: Path,
    label: str,
    pause: float,
    blh_only: bool = False,
    min_date: str | None = None,
    api_url: str = FWW.HIST_FORECAST_URL,
    timeout: float | None = None,
    batch_chips: int = BATCH_CHIPS,
    workers: int = 1,
) -> pd.DataFrame:
    cols = ["chip_id", "hour", "blh_native_m"] if blh_only else THERMO_CACHE_COLS
    existing = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=cols)
    if len(existing):
        existing["hour"] = pd.to_datetime(existing["hour"]).astype("datetime64[ns]")
    done_keys: set[str] = set(json.loads(done_path.read_text(encoding="utf-8"))) if done_path.exists() else set()

    local_api = FWW.is_local_api_url(api_url)
    req_timeout = float(timeout) if timeout is not None else float(FWW.LOCAL_HTTP_TIMEOUT_S if local_api else FWW.HTTP_TIMEOUT_S)
    eff_workers = max(1, int(workers))
    if eff_workers > 1 and not local_api:
        print(f"  {label:16s}: clamping --workers {eff_workers} -> 1 for public Open-Meteo endpoint", flush=True)
        eff_workers = 1

    chip_centres = {int(c): grid.chip_centre(int(c)) for c in targets["chip_id"].unique()}
    valid_chips = {c for c, (la, lo) in chip_centres.items() if FWW.in_bbox(la, lo, bbox)}
    sub = targets[targets["chip_id"].isin(valid_chips)]
    if min_date:
        sub = sub[sub["date"] >= min_date]
    if not len(sub):
        print(f"  {label:16s}: 0 target chip-hours inside domain (min_date={min_date or 'none'})", flush=True)
        return existing

    needed_hours = {int(cid): set(grp["hour"]) for cid, grp in sub.groupby("chip_id")}
    batches = FWW._plan_batches(sub, done_keys, batch_chips=batch_chips)
    if not batches:
        print(f"  {label:16s}: thermo cache complete ({len(existing):,} chip-hours)", flush=True)
        return existing

    since_tag = f", >={min_date}" if min_date else ""
    mode_tag = f", local w={eff_workers}" if local_api else ""
    print(
        f"  {label:16s} ({slug}): fetching {len(batches)} batches "
        f"({sub['chip_id'].nunique()} chips, {len(sub):,} chip-hours{since_tag}{mode_tag})...",
        flush=True,
    )
    vars_csv = ",".join(req_vars)
    new_frames: list[pd.DataFrame] = []
    quota_hit = False
    quota_msg = ""
    consecutive_429 = 0
    n_err = 0
    t_start = time.monotonic()

    def _parse(items: list[dict[str, Any]], chips: list[int]) -> pd.DataFrame:
        if not blh_only:
            return _parse_thermo_batch(items, chips, needed_hours)
        parts = []
        for cid, it in zip(chips, items):
            h = it.get("hourly") or {}
            ts = h.get("time") or []
            if not ts:
                continue
            times = pd.to_datetime(ts).astype("datetime64[ns]")
            want = needed_hours.get(cid)
            mask = times.isin(want) if want else np.ones(len(ts), dtype=bool)
            if not mask.any():
                continue
            blh = FWW._to_arr(h.get("boundary_layer_height"), len(ts))[mask]
            ok = np.isfinite(blh)
            if ok.any():
                parts.append(pd.DataFrame({
                    "chip_id": np.full(int(ok.sum()), cid, dtype=np.int64),
                    "hour": times[mask][ok],
                    "blh_native_m": blh[ok].astype(np.float32),
                }))
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=cols)

    def _fetch_one_batch(batch_item: tuple[int, str, str, list[int]]) -> dict[str, Any]:
        idx_b, s0, s1, chips = batch_item
        lats = ",".join(f"{chip_centres[c][0]:.4f}" for c in chips)
        lons = ",".join(f"{chip_centres[c][1]:.4f}" for c in chips)
        frames_out: list[pd.DataFrame] = []
        keys_out: list[str] = []
        errs_out: list[str] = []
        saw_429 = False
        try:
            data = FWW._http_get_json({
                "latitude": lats, "longitude": lons,
                "start_date": s0, "end_date": s1,
                "hourly": vars_csv, "models": slug,
                "wind_speed_unit": "ms", "timezone": "UTC",
            }, base_url=api_url, timeout=req_timeout)
            items = data if isinstance(data, list) else [data]
            if len(items) == len(chips):
                df = _parse(items, chips)
                if len(df):
                    frames_out.append(df)
                keys_out.extend(f"{c}|{s0}|{s1}" for c in chips)
            return {"idx": idx_b, "frames": frames_out, "keys": keys_out, "errs": errs_out, "quota": None, "saw_429": False}
        except FWW.QuotaExceeded as exc:
            return {"idx": idx_b, "frames": frames_out, "keys": keys_out, "errs": errs_out, "quota": str(exc), "saw_429": True}
        except Exception as exc:
            if "429" in str(exc):
                saw_429 = True
            if "HTTP 400" in str(exc) and len(chips) > 1:
                for c in chips:
                    try:
                        la, lo = chip_centres[c]
                        d1 = FWW._http_get_json({
                            "latitude": f"{la:.4f}", "longitude": f"{lo:.4f}",
                            "start_date": s0, "end_date": s1,
                            "hourly": vars_csv, "models": slug,
                            "wind_speed_unit": "ms", "timezone": "UTC",
                        }, base_url=api_url, timeout=req_timeout)
                        df1 = _parse([d1] if isinstance(d1, dict) else d1, [c])
                        if len(df1):
                            frames_out.append(df1)
                        keys_out.append(f"{c}|{s0}|{s1}")
                    except FWW.QuotaExceeded as qexc:
                        return {"idx": idx_b, "frames": frames_out, "keys": keys_out, "errs": errs_out, "quota": str(qexc), "saw_429": True}
                    except Exception as exc1:
                        if "HTTP 400" in str(exc1):
                            keys_out.append(f"{c}|{s0}|{s1}")
                        errs_out.append(f"chip {c} ({s0}..{s1}) skipped: {exc1}")
                    if pause > 0:
                        time.sleep(pause)
                return {"idx": idx_b, "frames": frames_out, "keys": keys_out, "errs": errs_out, "quota": None, "saw_429": saw_429}
            if "HTTP 400" in str(exc):
                keys_out.extend(f"{c}|{s0}|{s1}" for c in chips)
            errs_out.append(f"batch {idx_b} ({s0}..{s1}) skipped: {exc}")
            return {"idx": idx_b, "frames": frames_out, "keys": keys_out, "errs": errs_out, "quota": None, "saw_429": saw_429}

    def _consume_result(res: dict[str, Any], done_count: int) -> bool:
        nonlocal existing, new_frames, quota_hit, quota_msg, consecutive_429, n_err
        if res["frames"]:
            new_frames.extend(res["frames"])
        for k in res["keys"]:
            done_keys.add(k)
        for em in res["errs"]:
            n_err += 1
            if n_err <= 5:
                print(f"    {em}", flush=True)
        if res["quota"]:
            print(f"    Open-Meteo quota reached at batch {res['idx']}/{len(batches)}: {res['quota']}", flush=True)
            quota_hit = True
            quota_msg = res["quota"]
            return True
        if res["saw_429"]:
            consecutive_429 += 1
            if consecutive_429 >= 2:
                quota_hit = True
                quota_msg = f"repeated HTTP 429 at batch {res['idx']}/{len(batches)}"
                return True
        else:
            consecutive_429 = 0

        if done_count % CHECKPOINT_EVERY == 0 or done_count == len(batches):
            existing = FWW._save_cache(cache_path, done_path, existing, new_frames, done_keys)
            new_frames = []
            elapsed = max(time.monotonic() - t_start, 1e-6)
            rate_bpm = done_count / elapsed * 60.0
            eta_min = (len(batches) - done_count) / max(done_count / elapsed, 1e-9) / 60.0
            print(
                f"    {label:16s}: {done_count}/{len(batches)} batches saved "
                f"({len(existing):,} chip-hours | {rate_bpm:.1f} b/min, ~{eta_min:.1f} min left)",
                flush=True,
            )
        return False

    indexed = [(idx, s0, s1, chips) for idx, (s0, s1, chips) in enumerate(batches, 1)]
    if eff_workers == 1:
        for done_count, item in enumerate(indexed, 1):
            res = _fetch_one_batch(item)
            if _consume_result(res, done_count):
                break
            if pause > 0:
                time.sleep(pause)
    else:
        # Bounded chronological rolling window preserves S3 time-chunk cache locality across workers.
        max_in_flight = eff_workers * 2
        it = iter(indexed)
        done_count = 0
        with ThreadPoolExecutor(max_workers=eff_workers) as pool:
            pending = set()
            for _ in range(min(max_in_flight, len(indexed))):
                pending.add(pool.submit(_fetch_one_batch, next(it)))
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                stop_now = False
                for fut in finished:
                    done_count += 1
                    res = fut.result()
                    if _consume_result(res, done_count):
                        stop_now = True
                        break
                if stop_now:
                    for fut in pending:
                        fut.cancel()
                    break
                while len(pending) < max_in_flight:
                    try:
                        pending.add(pool.submit(_fetch_one_batch, next(it)))
                    except StopIteration:
                        break

    if new_frames or quota_hit:
        existing = FWW._save_cache(cache_path, done_path, existing, new_frames, done_keys)
    if quota_hit:
        raise FWW.QuotaExceeded(f"Saved progress for {label} ({quota_msg}). Re-run once quota resets.")
    return existing


def download_model_thermo(
    targets: pd.DataFrame,
    spec: dict[str, Any],
    cache_dir: Path,
    pause: float,
    api_url: str = FWW.HIST_FORECAST_URL,
    timeout: float | None = None,
    batch_chips: int = BATCH_CHIPS,
    workers: int = 1,
) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    mid = spec["id"]
    df = _download_slug_batches(
        targets, spec["slug"], spec["vars"], spec["bbox"],
        cache_dir / f"{mid}.parquet", cache_dir / f"{mid}_fetched.json",
        mid, pause=pause, blh_only=False, min_date=spec.get("min_date"),
        api_url=api_url, timeout=timeout, batch_chips=batch_chips, workers=workers,
    )
    if spec.get("blh_slug"):
        blh_df = _download_slug_batches(
            targets, spec["blh_slug"], ["boundary_layer_height"], spec["bbox"],
            cache_dir / f"{mid}_blh9k.parquet", cache_dir / f"{mid}_blh9k_fetched.json",
            f"{mid}_blh9k", pause=pause, blh_only=True, min_date=spec.get("blh_min_date") or spec.get("min_date"),
            api_url=api_url, timeout=timeout, batch_chips=batch_chips, workers=workers,
        )
        df = _merge_blh_sidecar(df, blh_df)
    return df


def _merge_blh_sidecar(df: pd.DataFrame, blh_df: pd.DataFrame) -> pd.DataFrame:
    if not len(df) or not len(blh_df):
        return df
    m = df.merge(
        blh_df.drop_duplicates(["chip_id", "hour"]).rename(columns={"blh_native_m": "_blh9k"}),
        on=["chip_id", "hour"],
        how="left",
    )
    m["blh_native_m"] = m["blh_native_m"].fillna(m["_blh9k"]).astype(np.float32)
    return m.drop(columns=["_blh9k"])


def load_cached_thermo(cache_dir: Path, spec: dict[str, Any]) -> pd.DataFrame:
    mid = spec["id"]
    p = cache_dir / f"{mid}.parquet"
    if not p.exists():
        return pd.DataFrame(columns=THERMO_CACHE_COLS)
    df = pd.read_parquet(p)
    if len(df):
        df["hour"] = pd.to_datetime(df["hour"]).astype("datetime64[ns]")
    blh_p = cache_dir / f"{mid}_blh9k.parquet"
    if spec.get("blh_slug") and blh_p.exists() and len(df):
        bdf = pd.read_parquet(blh_p)
        if len(bdf):
            bdf["hour"] = pd.to_datetime(bdf["hour"]).astype("datetime64[ns]")
            df = _merge_blh_sidecar(df, bdf)
    return df


# ── Pillar 1: Offline re-scoring against clean IGC wind & glide wind ─────────

def evaluate_clean_wind(
    db: Path,
    selected: list[dict[str, Any]],
    from_year: int,
    out_dir: Path,
    max_sites: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Re-score already-cached wind parquets (NB 14) against clean truth_wind_groups + truth_glide_wind."""
    truth_dir = db / "truth"
    wg_p = truth_dir / "truth_wind_groups.parquet"
    if not wg_p.exists():
        raise SystemExit(f"Missing {wg_p}. Run notebook 15 (build_igc_truth.py) first.")

    g = pd.read_parquet(wg_p)
    g["hour"] = pd.to_datetime(g["hour"]).astype("datetime64[ns]")
    g = g[g["hour"].dt.year >= from_year].copy()
    g["date"] = g["hour"].dt.strftime("%Y-%m-%d")

    sites = pd.read_parquet(db / "sites.parquet", columns=["site_id", "name", "country", "lat", "lon"])
    g = g.merge(sites, on="site_id", how="left")
    g["region"] = [FWW.region_of(la, lo) for la, lo in zip(g["lat"].fillna(0.0), g["lon"].fillna(0.0))]
    if "ground_m" not in g:
        g["ground_m"] = g.get("chip_ground_m", g["h_asl"] - g["agl"])
    g["ground_m"] = g["ground_m"].fillna(g["h_asl"] - g["agl"])
    if max_sites is not None and max_sites > 0:
        top = g.groupby("site_id").size().sort_values(ascending=False).head(max_sites).index
        g = g[g["site_id"].isin(top)].copy()
    g = g.reset_index(drop=True)

    # Alias columns so forecast_wind_weights CV & hierarchy helpers work seamlessly on clean wind.
    g["pilot_u"] = g["clean_u"].astype(float)
    g["pilot_v"] = g["clean_v"].astype(float)
    g["model_u"] = g["era5_u"].astype(float) if "era5_u" in g else np.nan
    g["model_v"] = g["era5_v"].astype(float) if "era5_v" in g else np.nan

    wind_cache_dir = db / "weather" / "hist_forecast"
    active_ids = []
    for spec in selected:
        mid = spec["id"]
        cp = wind_cache_dir / f"{mid}.parquet"
        if not cp.exists():
            continue
        wx = pd.read_parquet(cp)
        if not len(wx):
            continue
        wx["hour"] = pd.to_datetime(wx["hour"]).astype("datetime64[ns]")
        u, v = FWW.interpolate_model_wind(g, wx)
        g[f"{mid}_u"] = u.astype(np.float32)
        g[f"{mid}_v"] = v.astype(np.float32)
        if np.isfinite(u).any():
            active_ids.append(mid)

    score_rows = []
    cu, cv_arr = g["clean_u"].to_numpy(float), g["clean_v"].to_numpy(float)
    hu, hv = g["hodo_u"].to_numpy(float), g["hodo_v"].to_numpy(float)
    ou, ov = g["old_u"].to_numpy(float), g["old_v"].to_numpy(float)
    eu, ev = g["model_u"].to_numpy(float), g["model_v"].to_numpy(float)

    band_defs = [("near the ground", ["near the ground"]), ("middle", ["middle"]), ("free air", ["free air"]), ("free+middle", ["free air", "middle"])]
    for spec in selected:
        mid = spec["id"]
        if mid not in active_ids:
            continue
        mu, mv = g[f"{mid}_u"].to_numpy(float), g[f"{mid}_v"].to_numpy(float)
        for bname, blist in band_defs:
            mask = g["band"].isin(blist).to_numpy() & np.isfinite(mu) & np.isfinite(mv)
            if not mask.any():
                continue
            sub = g[mask]
            m_clean = FWW.wind_metrics(cu[mask], cv_arr[mask], mu[mask], mv[mask])
            m_hodo = FWW.wind_metrics(hu[mask], hv[mask], mu[mask], mv[mask])
            m_old = FWW.wind_metrics(ou[mask], ov[mask], mu[mask], mv[mask])
            m_era = FWW.wind_metrics(cu[mask], cv_arr[mask], eu[mask], ev[mask])
            score_rows.append({
                "model": mid, "name": spec["name"], "region": spec["region"], "band": bname,
                "events": int(sub["event_key"].nunique()), "sites": int(sub["site_id"].nunique()),
                "first_date": str(sub["date"].min()), "last_date": str(sub["date"].max()),
                **m_clean,
                "hodo_speed_bias_ms": m_hodo.get("speed_bias_ms"),
                "old_vector_rmse_ms": m_old.get("vector_rmse_ms"),
                "old_speed_bias_ms": m_old.get("speed_bias_ms"),
                "era5_rmse_ms": m_era.get("vector_rmse_ms"),
                "era5_speed_bias_ms": m_era.get("speed_bias_ms"),
                "rel_rmse_vs_era5": m_clean["vector_rmse_ms"] / max(m_era["vector_rmse_ms"], 0.25) if m_era.get("vector_rmse_ms") else None,
            })

    scores_df = pd.DataFrame(score_rows)
    scores_df.to_csv(out_dir / "wind_clean_scores.csv", index=False)

    # Cross-check against independent multi-heading glide wind.
    glide_scores: dict[str, Any] = {}
    gw_p = truth_dir / "truth_glide_wind.parquet"
    if gw_p.exists() and active_ids:
        gw = pd.read_parquet(gw_p)
        gw["hour"] = pd.to_datetime(gw["hour"]).astype("datetime64[ns]")
        mg = g.merge(gw[["site_id", "chip_id", "hour", "band", "glide_u", "glide_v"]], on=["site_id", "chip_id", "hour", "band"], how="inner")
        if len(mg):
            gu, gv = mg["glide_u"].to_numpy(float), mg["glide_v"].to_numpy(float)
            for mid in [*active_ids, "era5"]:
                mu = mg["model_u"].to_numpy(float) if mid == "era5" else mg[f"{mid}_u"].to_numpy(float)
                mv = mg["model_v"].to_numpy(float) if mid == "era5" else mg[f"{mid}_v"].to_numpy(float)
                glide_scores[mid] = FWW.wind_metrics(gu, gv, mu, mv)

    cv_res = FWW.run_grouped_cv(g, active_ids, k_folds=5) if len(active_ids) >= 2 else {"error": "fewer than 2 cached wind models"}
    cv_res["vs_glide_wind"] = glide_scores
    # Sample size per model on free air + middle, used to shrink the composite wind pillar.
    fm_rows = scores_df[scores_df["band"] == "free+middle"] if len(scores_df) else scores_df
    cv_res["free_middle_groups"] = {str(r.model): int(r.groups) for r in fm_rows.itertuples(index=False)} if len(fm_rows) else {}
    best_beta = float(cv_res.get("median_beta", 4.0))
    d_fm = g[g["band"].isin(["free air", "middle"])]
    cv_res["clean_wind_hierarchy"] = FWW.fit_hierarchy(d_fm, active_ids, beta=best_beta, shrink_k=SHRINKAGE_K) if active_ids else {}
    (out_dir / "wind_clean_cv.json").write_text(json.dumps(cv_res, indent=2), encoding="utf-8")
    return scores_df, cv_res


# ── Pillar 2: Top of Lift ASL & Parcel BLH trigger calibration ───────────────

def _within_site_day_corr(df: pd.DataFrame, obs_col: str, pred_col: str) -> tuple[float | None, int]:
    sub = df.dropna(subset=[obs_col, pred_col]).copy()
    if not len(sub):
        return None, 0
    sub["local_date"] = (sub["hour"] + pd.to_timedelta(sub["centroid_lon"].fillna(0.0) / 15.0, unit="h")).dt.strftime("%Y-%m-%d")
    daily = sub.groupby(["site_id", "local_date"]).agg(obs=(obs_col, "max"), pred=(pred_col, "max")).reset_index()
    within = daily.groupby("site_id").filter(lambda x: len(x) >= 5)
    if not len(within):
        return None, 0
    corrs = []
    for _, grp in within.groupby("site_id"):
        if grp["obs"].std() > 0 and grp["pred"].std() > 0:
            corrs.append(float(grp["obs"].corr(grp["pred"])))
    if not corrs:
        return None, 0
    return float(np.median(corrs)), len(corrs)


def _ceiling_metrics(obs: np.ndarray, pred: np.ndarray, df_sub: pd.DataFrame, obs_col: str, pred_col: str) -> dict[str, Any]:
    ok = np.isfinite(obs) & np.isfinite(pred)
    if not ok.any():
        return {"groups": 0}
    o, p = obs[ok], pred[ok]
    diff = p - o
    corr = float(np.corrcoef(o, p)[0, 1]) if len(o) > 2 and np.std(o) > 0 and np.std(p) > 0 else None
    tmp = df_sub.loc[ok, ["site_id", "hour", "centroid_lon"]].copy()
    tmp[obs_col] = o
    tmp[pred_col] = p
    day_r, n_sites5 = _within_site_day_corr(tmp, obs_col, pred_col)
    return {
        "groups": int(len(o)),
        "rmse_m": round(float(np.sqrt(np.mean(diff * diff))), 1),
        "mae_m": round(float(np.mean(np.abs(diff))), 1),
        "bias_pred_minus_obs_m": round(float(np.median(diff)), 1),
        "corr_all": round(corr, 4) if corr is not None else None,
        "within_site_day_corr": round(day_r, 4) if day_r is not None else None,
        "sites_5d": int(n_sites5),
    }


def evaluate_ceiling(
    db: Path,
    selected: list[dict[str, Any]],
    thermo_tables: dict[str, pd.DataFrame],
    from_year: int,
    out_dir: Path,
    max_sites: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score native vs parcel BLH and calibrate optimal parcel excess_c via 5-fold competition CV."""
    cp = db / "truth" / "truth_ceiling.parquet"
    if not cp.exists():
        return pd.DataFrame(), {"error": "missing truth_ceiling.parquet"}

    c = pd.read_parquet(cp)
    c["hour"] = pd.to_datetime(c["hour"]).astype("datetime64[ns]")
    c = c[(c["hour"].dt.year >= from_year) & (~c["airspace_cap_flag"].fillna(False).astype(bool))].copy()
    if max_sites is not None and max_sites > 0:
        top = c.groupby("site_id").size().sort_values(ascending=False).head(max_sites).index
        c = c[c["site_id"].isin(top)].copy()
    c = c.reset_index(drop=True)

    rows = []
    trigger_cv: dict[str, Any] = {}

    for spec in selected:
        mid = spec["id"]
        tx = thermo_tables.get(mid)
        if tx is None or not len(tx):
            continue
        # truth_ceiling already carries ERA5 columns (e.g. cloud_total); the forecast model's keep the bare name.
        m = c.merge(tx.drop_duplicates(["chip_id", "hour"]), on=["chip_id", "hour"], how="inner", suffixes=("_era5", ""))
        if len(m) < 20:
            continue

        obs = m["ceiling_p90_asl"].to_numpy(float)
        era_blh = m["era5_blh_asl"].to_numpy(float) if "era5_blh_asl" in m else np.full(len(m), np.nan)
        era_eff = m["era5_min_blh_cb_asl"].to_numpy(float) if "era5_min_blh_cb_asl" in m else era_blh

        om_elev = m["om_elev"].to_numpy(float)
        t2m = m["t2m_c"].to_numpy(float)
        td2m = m["dewpoint_c"].to_numpy(float)
        rad = m["sunshine_wm2"].to_numpy(float)
        cld = m["cloud_total"].to_numpy(float)
        prof_z = [m[f"z_{p}"].to_numpy(float) for p in (925, 850, 700, 600)]
        prof_t = [m[f"t_{p}"].to_numpy(float) for p in (925, 850, 700, 600)]
        lcl_asl = np.where(cld >= 0.05, om_elev + np.maximum(0.0, 125.0 * (t2m - td2m)), np.nan)

        # Evaluate parcel BLH across all candidate convective triggers.
        cand_blh: dict[float, np.ndarray] = {}
        cand_eff: dict[float, np.ndarray] = {}
        for exc in PARCEL_EXCESS_GRID:
            bp = parcel_blh_asl_with_excess(t2m, rad, om_elev, prof_z, prof_t, excess_c=exc)
            cand_blh[exc] = bp
            cand_eff[exc] = np.where(np.isfinite(lcl_asl), np.minimum(bp, lcl_asl), bp)

        # Nested 5-fold CV by event_key to select optimal excess_c out-of-fold.
        evs = m["event_key"].astype(str).to_numpy()
        n_ev = int(len(np.unique(evs)))
        oof_parcel = np.full(len(m), np.nan)
        oof_eff = np.full(len(m), np.nan)
        chosen: list[float] = []
        if n_ev >= 3:
            gkf = GroupKFold(n_splits=min(5, n_ev))
            for tr_i, te_i in gkf.split(m, groups=evs):
                best_exc, best_r = 1.0, float("inf")
                for exc in PARCEL_EXCESS_GRID:
                    p_tr = cand_eff[exc][tr_i]
                    ok_tr = np.isfinite(p_tr) & np.isfinite(obs[tr_i])
                    if ok_tr.sum() < 10:
                        continue
                    r_tr = float(np.sqrt(np.mean((p_tr[ok_tr] - obs[tr_i][ok_tr]) ** 2)))
                    if r_tr < best_r:
                        best_r, best_exc = r_tr, exc
                chosen.append(best_exc)
                oof_parcel[te_i] = cand_blh[best_exc][te_i]
                oof_eff[te_i] = cand_eff[best_exc][te_i]
        else:
            chosen = [1.0]
            oof_parcel = cand_blh[1.0]
            oof_eff = cand_eff[1.0]

        best_global_exc = float(np.median(chosen))
        # Snap median to nearest grid point.
        best_global_exc = min(PARCEL_EXCESS_GRID, key=lambda x: abs(x - best_global_exc))

        grid_summary = {}
        for exc in PARCEL_EXCESS_GRID:
            ok_e = np.isfinite(cand_eff[exc]) & np.isfinite(obs)
            if ok_e.any():
                grid_summary[f"{exc:.1f}"] = {
                    "rmse_m": round(float(np.sqrt(np.mean((cand_eff[exc][ok_e] - obs[ok_e]) ** 2))), 1),
                    "bias_m": round(float(np.median(cand_eff[exc][ok_e] - obs[ok_e])), 1),
                }
        trigger_cv[mid] = {
            "chosen_per_fold": chosen,
            "recommended_excess_c": best_global_exc,
            "grid_eff_top": grid_summary,
        }

        era_m = _ceiling_metrics(obs, era_eff, m, "obs", "era5")
        variants: list[tuple[str, np.ndarray]] = [
            ("parcel_default_1.0C", cand_eff[1.0]),
            ("parcel_cv_tuned", oof_eff),
            ("parcel_dry_only", oof_parcel),
        ]
        blh_nat = m["blh_native_m"].to_numpy(float)
        if np.isfinite(blh_nat).sum() >= 20:
            nat_asl = om_elev + blh_nat
            nat_eff = np.where(np.isfinite(lcl_asl), np.minimum(nat_asl, lcl_asl), nat_asl)
            variants.insert(0, ("native_min_blh_lcl", nat_eff))
            variants.insert(1, ("native_blh_only", nat_asl))

        for vname, pred in variants:
            met = _ceiling_metrics(obs, pred, m, "obs", "pred")
            if not met.get("groups"):
                continue
            ok_pair = np.isfinite(obs) & np.isfinite(pred) & np.isfinite(era_eff)
            era_pair_rmse = float(np.sqrt(np.mean((era_eff[ok_pair] - obs[ok_pair]) ** 2))) if ok_pair.any() else None
            mod_pair_rmse = float(np.sqrt(np.mean((pred[ok_pair] - obs[ok_pair]) ** 2))) if ok_pair.any() else None
            rows.append({
                "model": mid, "name": spec["name"], "region": spec["region"], "variant": vname,
                "events": int(m["event_key"].nunique()), "sites": int(m["site_id"].nunique()),
                **met,
                "era5_rmse_m": round(era_pair_rmse, 1) if era_pair_rmse is not None else era_m.get("rmse_m"),
                "rel_rmse_vs_era5": round(mod_pair_rmse / max(era_pair_rmse, 50.0), 4) if (era_pair_rmse and mod_pair_rmse) else None,
            })

    ceil_df = pd.DataFrame(rows)
    ceil_df.to_csv(out_dir / "ceiling_scores.csv", index=False)
    (out_dir / "parcel_trigger_cv.json").write_text(json.dumps(trigger_cv, indent=2), encoding="utf-8")
    return ceil_df, trigger_cv


# ── Pillar 3: Core60 climbing speed verification ─────────────────────────────

def _climb_thermo_features(df: pd.DataFrame, excess_c: float) -> pd.DataFrame:
    om_elev = df["om_elev"].to_numpy(float)
    t2m = df["t2m_c"].to_numpy(float)
    td2m = df["dewpoint_c"].to_numpy(float)
    rad = df["sunshine_wm2"].to_numpy(float)
    cld = df["cloud_total"].to_numpy(float)
    prof_z = [df[f"z_{p}"].to_numpy(float) for p in (925, 850, 700, 600)]
    prof_t = [df[f"t_{p}"].to_numpy(float) for p in (925, 850, 700, 600)]
    blh_nat = df["blh_native_m"].to_numpy(float)
    blh_parcel = np.maximum(0.0, parcel_blh_asl_with_excess(t2m, rad, om_elev, prof_z, prof_t, excess_c=excess_c) - om_elev)
    blh_m = _coalesce_arr(blh_nat, blh_parcel)

    z_diff_km = (df["z_700"].to_numpy(float) - df["z_850"].to_numpy(float)) / 1000.0
    lapse = np.where(z_diff_km > 0.2, (df["t_850"].to_numpy(float) - df["t_700"].to_numpy(float)) / z_diff_km, np.nan)
    wstar = np.cbrt(np.maximum(blh_m, 0.0) * np.maximum(rad, 0.0) * np.clip(1.0 - 0.5 * np.nan_to_num(cld, nan=0.0), 0.1, 1.0)) / 100.0
    return pd.DataFrame({
        "blh_m": blh_m,
        "wstar_proxy": wstar,
        "lapse_850_700": lapse,
        "sunshine_wm2": rad,
        "cloud_total": cld,
        "cloud_low": df["cloud_low"].to_numpy(float),
        "spread_c": t2m - td2m,
        "t2m_c": t2m,
        "cape": df["cape"].to_numpy(float),
        "rh_850": df["rh_850"].to_numpy(float),
        "rh_700": df["rh_700"].to_numpy(float),
        "solar_hour": df["solar_hour"].to_numpy(float),
        "elev_m": om_elev,
    }, index=df.index)


def _cv_ridge_score(X: pd.DataFrame, y: np.ndarray, events: np.ndarray) -> dict[str, Any]:
    keep_cols = [c for c in X.columns if np.isfinite(X[c].to_numpy(float)).sum() >= 10]
    if not keep_cols or len(y) < 25 or len(np.unique(events)) < 3:
        return {"groups": 0}
    Xv = X[keep_cols].to_numpy(float)
    gkf = GroupKFold(n_splits=min(5, len(np.unique(events))))
    oof = np.full(len(y), np.nan)
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc", StandardScaler()),
        ("reg", Ridge(alpha=10.0)),
    ])
    for tr, te in gkf.split(Xv, y, groups=events):
        pipe.fit(Xv[tr], y[tr])
        oof[te] = pipe.predict(Xv[te])
    ok = np.isfinite(oof) & np.isfinite(y)
    if not ok.any():
        return {"groups": 0}
    yo, po = y[ok], oof[ok]
    err = po - yo
    rmse = float(np.sqrt(np.mean(err * err)))
    ss_res = float(np.sum(err * err))
    ss_tot = float(np.sum((yo - np.mean(yo)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-9)
    corr = float(np.corrcoef(yo, po)[0, 1]) if np.std(po) > 0 and np.std(yo) > 0 else None
    return {
        "groups": int(len(yo)),
        "rmse_ms": round(rmse, 4),
        "mae_ms": round(float(np.mean(np.abs(err))), 4),
        "r2": round(r2, 4),
        "corr": round(corr, 4) if corr is not None else None,
    }


def evaluate_climb_strength(
    db: Path,
    selected: list[dict[str, Any]],
    thermo_tables: dict[str, pd.DataFrame],
    trigger_cv: dict[str, Any],
    from_year: int,
    out_dir: Path,
    max_sites: int | None = None,
) -> pd.DataFrame:
    """Evaluate which forecast model's thermodynamics best predict IGC core60_p75_ms in 5-fold CV."""
    cp = db / "truth" / "truth_climb_groups.parquet"
    if not cp.exists():
        return pd.DataFrame()

    cl = pd.read_parquet(cp)
    cl["hour"] = pd.to_datetime(cl["hour"]).astype("datetime64[ns]")
    cl = cl[(cl["hour"].dt.year >= from_year) & cl["core60_p75_ms"].notna()].copy()
    if max_sites is not None and max_sites > 0:
        top = cl.groupby("site_id").size().sort_values(ascending=False).head(max_sites).index
        cl = cl[cl["site_id"].isin(top)].copy()
    cl = cl.reset_index(drop=True)

    # Load matching ERA5 thermo once for paired comparison on identical rows.
    era_tbl = None
    w_dir = db / "weather"
    if (w_dir / "era5_single.parquet").exists():
        chips = set(cl["chip_id"].unique())
        s = pd.read_parquet(w_dir / "era5_single.parquet", columns=["chip_id", "time", "blh_m", "sunshine_wm2", "cloud_total", "cloud_low", "t2m_c", "dewpoint_c", "cape"])
        s = s[s["chip_id"].isin(chips)].rename(columns={"time": "hour", "blh_m": "blh_native_m"})
        s["hour"] = pd.to_datetime(s["hour"]).astype("datetime64[ns]")
        era_tbl = s.drop_duplicates(["chip_id", "hour"])
        if (w_dir / "era5_static.parquet").exists():
            st = pd.read_parquet(w_dir / "era5_static.parquet", columns=["chip_id", "ground_m"]).drop_duplicates("chip_id")
            era_tbl = era_tbl.merge(st.rename(columns={"ground_m": "om_elev"}), on="chip_id", how="left")
        else:
            era_tbl["om_elev"] = 0.0
        if (w_dir / "era5_levels.parquet").exists():
            lv = pd.read_parquet(w_dir / "era5_levels.parquet", columns=["chip_id", "time", "t_850", "z_850", "rh_850", "t_700", "z_700", "rh_700"])
            lv = lv[lv["chip_id"].isin(chips)].rename(columns={"time": "hour3"})
            lv["hour3"] = pd.to_datetime(lv["hour3"]).astype("datetime64[ns]")
            era_tbl["hour3"] = era_tbl["hour"].dt.round("3h")
            era_tbl = era_tbl.merge(lv.drop_duplicates(["chip_id", "hour3"]), on=["chip_id", "hour3"], how="left").drop(columns=["hour3"])
        for col in ("t_925", "z_925", "t_600", "z_600", "t_850", "z_850", "rh_850", "t_700", "z_700", "rh_700"):
            if col not in era_tbl:
                era_tbl[col] = np.nan

    rows = []
    for spec in selected:
        mid = spec["id"]
        tx = thermo_tables.get(mid)
        if tx is None or not len(tx):
            continue
        m = cl.merge(tx.drop_duplicates(["chip_id", "hour"]), on=["chip_id", "hour"], how="inner")
        if len(m) < 30:
            continue
        exc = float((trigger_cv.get(mid) or {}).get("recommended_excess_c", 1.0))
        Xm = _climb_thermo_features(m, excess_c=exc)
        y = m["core60_p75_ms"].to_numpy(float)
        evs = m["event_key"].astype(str).to_numpy()
        met = _cv_ridge_score(Xm, y, evs)
        if not met.get("groups"):
            continue
        wstar_corr = float(pd.Series(Xm["wstar_proxy"]).corr(pd.Series(y))) if Xm["wstar_proxy"].notna().sum() > 10 else None

        era_rmse = None
        if era_tbl is not None:
            me = m[["site_id", "chip_id", "hour", "event_key", "solar_hour", "core60_p75_ms"]].merge(era_tbl, on=["chip_id", "hour"], how="inner")
            if len(me) >= 30:
                Xe = _climb_thermo_features(me, excess_c=1.0)
                em = _cv_ridge_score(Xe, me["core60_p75_ms"].to_numpy(float), me["event_key"].astype(str).to_numpy())
                era_rmse = em.get("rmse_ms")

        rows.append({
            "model": mid, "name": spec["name"], "region": spec["region"],
            "events": int(m["event_key"].nunique()), "sites": int(m["site_id"].nunique()),
            **met,
            "wstar_corr": round(wstar_corr, 4) if wstar_corr is not None else None,
            "era5_rmse_ms": era_rmse,
            "rel_rmse_vs_era5": round(met["rmse_ms"] / max(era_rmse, 0.1), 4) if era_rmse else None,
        })

    climb_df = pd.DataFrame(rows)
    climb_df.to_csv(out_dir / "climb_scores.csv", index=False)
    return climb_df


# ── Pillar 4: End-to-end task_v1 + leakage-free Quantile Mapping to ERA5 ─────

def _assemble_task_tables(
    wind_df: pd.DataFrame,
    thermo_df: pd.DataFrame,
    base_tables: dict[str, pd.DataFrame | None],
    excess_c: float,
) -> dict[str, pd.DataFrame | None] | None:
    """Combine cached historical wind + thermo into the 5-table dict expected by task_model.features_from."""
    if not len(thermo_df):
        return None
    d = thermo_df.drop_duplicates(["chip_id", "hour"]).copy()
    if len(wind_df):
        w_cols = [c for c in ["chip_id", "hour", "u10", "v10", "u100", "v100", "u_850", "v_850", "u_700", "v_700"] if c in wind_df]
        d = d.merge(wind_df[w_cols].drop_duplicates(["chip_id", "hour"]), on=["chip_id", "hour"], how="left")
    for c in ("u10", "v10", "u100", "v100", "u_850", "v_850", "u_700", "v_700"):
        if c not in d:
            d[c] = np.nan

    static_df = base_tables.get("static")
    g_map = dict(zip(static_df["chip_id"].astype(int), static_df["ground_m"].astype(float))) if static_df is not None and len(static_df) else {}
    ground_m = np.array([g_map.get(int(cid), float(oe)) for cid, oe in zip(d["chip_id"], d["om_elev"])], dtype=np.float64)
    om_elev = d["om_elev"].to_numpy(float)

    t2m = d["t2m_c"].to_numpy(float)
    td2m = d["dewpoint_c"].to_numpy(float)
    rad = d["sunshine_wm2"].to_numpy(float)
    cld = d["cloud_total"].to_numpy(float)
    prof_z = [d[f"z_{p}"].to_numpy(float) for p in (925, 850, 700, 600)]
    prof_t = [d[f"t_{p}"].to_numpy(float) for p in (925, 850, 700, 600)]

    blh_nat = d["blh_native_m"].to_numpy(float)
    blh_asl_p = parcel_blh_asl_with_excess(t2m, rad, om_elev, prof_z, prof_t, excess_c=excess_c)
    blh_parcel = np.maximum(0.0, blh_asl_p - ground_m)
    blh_m = _coalesce_arr(blh_nat, blh_parcel)

    lcl_asl = om_elev + np.maximum(0.0, 125.0 * (t2m - td2m))
    cloud_base_m = np.where(cld >= 0.05, np.maximum(0.0, lcl_asl - ground_m), np.nan)

    flh_m = d["freezing_level_m"].to_numpy(float)
    if not np.isfinite(flh_m).all():
        # Derive 0 C level row-wise where missing.
        flh_der = np.full(len(d), np.nan, dtype=np.float64)
        for cid_val, idxs in d.groupby("chip_id").groups.items():
            pos = d.index.get_indexer(idxs)
            oe0 = float(om_elev[pos[0]])
            flh_asl = FC._derived_freezing_asl(t2m[pos], oe0, [pz[pos] for pz in prof_z], [pt[pos] for pt in prof_t])
            flh_der[pos] = np.maximum(0.0, flh_asl - ground_m[pos])
        flh_m = _coalesce_arr(flh_m, flh_der)

    single = pd.DataFrame({
        "chip_id": d["chip_id"].to_numpy(np.int64),
        "time": d["hour"],
        "blh_m": blh_m,
        "cloud_base_m": cloud_base_m,
        "cape": d["cape"].to_numpy(float),
        "cin": d["cin"].to_numpy(float),
        "sensible_heat_wm2": np.nan,
        "sunshine_wm2": rad,
        "cloud_total": cld,
        "cloud_low": d["cloud_low"].to_numpy(float),
        "rain_mmh": d["rain_mmh"].to_numpy(float),
        "t2m_c": t2m,
        "dewpoint_c": td2m,
        "u100": d["u100"].to_numpy(float),
        "v100": d["v100"].to_numpy(float),
        "u10": d["u10"].to_numpy(float),
        "v10": d["v10"].to_numpy(float),
        "gust10": d["gust10"].to_numpy(float),
    })
    levels = pd.DataFrame({
        "chip_id": d["chip_id"].to_numpy(np.int64),
        "time3": d["hour"],
        "t_850": d["t_850"].to_numpy(float),
        "t_700": d["t_700"].to_numpy(float),
        "z_850": d["z_850"].to_numpy(float),
        "z_700": d["z_700"].to_numpy(float),
        "u_850": d["u_850"].to_numpy(float),
        "v_850": d["v_850"].to_numpy(float),
        "u_700": d["u_700"].to_numpy(float),
        "v_700": d["v_700"].to_numpy(float),
        "rh_700": d["rh_700"].to_numpy(float),
        "rh_850": d["rh_850"].to_numpy(float),
    })
    extra = pd.DataFrame({
        "chip_id": d["chip_id"].to_numpy(np.int64),
        "time": d["hour"],
        "skin_temp_c": d["skin_temp_c"].to_numpy(float),
        "soil_moisture": d["soil_moisture"].to_numpy(float),
        "freezing_level_m": flh_m,
    })
    return {
        "single": single,
        "levels": levels,
        "extra": extra,
        "static": base_tables.get("static"),
        "elev": base_tables.get("elev"),
    }


def _winner_time_err(df_eval: pd.DataFrame, pred_kmh: np.ndarray) -> np.ndarray:
    """Absolute winner time error in minutes per task (NaN where there is no prediction or no winner)."""
    err = np.full(len(df_eval), np.nan)
    ok = np.isfinite(pred_kmh) & df_eval["winner_kmh"].notna().to_numpy()
    if ok.any():
        pk = np.maximum(5.0, pred_kmh[ok])
        yh = df_eval["winner_h"].to_numpy(float)[ok]
        km = df_eval["ss_km"].to_numpy(float)[ok]
        err[ok] = np.abs(yh - km / pk) * 60.0
    return err


def _goal_err(df_eval: pd.DataFrame, pred_goal: np.ndarray) -> np.ndarray:
    """Absolute goal share error in points per task (NaN where missing)."""
    err = np.full(len(df_eval), np.nan)
    ok = np.isfinite(pred_goal) & df_eval["goal_share"].notna().to_numpy()
    if ok.any():
        err[ok] = np.abs(df_eval["goal_share"].to_numpy(float)[ok] - np.clip(pred_goal[ok], 0.0, 1.0)) * 100.0
    return err


def paired_bootstrap_diff(
    events: np.ndarray,
    err_a: np.ndarray,
    err_b: np.ndarray,
    n_iter: int = PAIRED_BOOTSTRAP_ITERS,
    seed: int = 42,
) -> dict[str, Any]:
    """Mean of (err_a - err_b) over the tasks both cover, with a 95% CI from resampling whole
    competitions (event_key clusters). Negative means A is better. `p_a_not_better` is the share
    of resamples where A's mean error is not below B's (a one-sided bootstrap p-value)."""
    ok = np.isfinite(err_a) & np.isfinite(err_b)
    if not ok.any():
        return {"tasks": 0}
    d = err_a[ok] - err_b[ok]
    uniq, inv = np.unique(np.asarray(events)[ok].astype(str), return_inverse=True)
    out: dict[str, Any] = {
        "tasks": int(ok.sum()),
        "events": int(len(uniq)),
        "mean_a": round(float(err_a[ok].mean()), 3),
        "mean_b": round(float(err_b[ok].mean()), 3),
        "mean_diff": round(float(d.mean()), 3),
    }
    if len(uniq) < 3:
        out.update({"ci95": [out["mean_diff"], out["mean_diff"]], "p_a_not_better": None, "significant": False})
        return out
    # Resampling events with replacement = multinomial multiplicities per event; the statistic is
    # the pooled task mean (sum of differences / number of tasks) of each resample.
    sum_d = np.bincount(inv, weights=d)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    mult = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n_iter).astype(float)
    stats = (mult @ sum_d) / np.maximum(mult @ cnt, 1.0)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    out.update({
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "p_a_not_better": round(float(np.mean(stats >= 0.0)), 4),
        "significant": bool(hi < 0.0 or lo > 0.0),
    })
    return out


def shrink_pillar(rel: dict[str, float], n: dict[str, float], k: float) -> dict[str, float]:
    """Shrink each model's relative RMSE toward the median across models by n / (n + k)."""
    if not rel:
        return {}
    prior = float(np.median(list(rel.values())))
    out = {}
    for mid, r in rel.items():
        n_m = max(float(n.get(mid) or 0.0), 0.0)
        out[mid] = (n_m * r + k * prior) / (n_m + k)
    return out


def paired_task_comparisons(
    t_eval: pd.DataFrame,
    era_pred: tuple[np.ndarray, np.ndarray],
    model_preds: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    blend_preds: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    reference_blend: tuple[str, str] | None,
    n_iter: int = PAIRED_BOOTSTRAP_ITERS,
) -> list[dict[str, Any]]:
    """Paired, competition-clustered bootstrap of winner time and goal share errors.

    `model_preds` and `blend_preds` map (name, mode) to full-length out-of-fold (kmh, goal)
    predictions aligned with `t_eval` (NaN where not covered). Each row compares A against B on
    exactly the tasks both cover:
      blend_mode_vs_raw   every blend strategy under thermo_qm / full_qm vs the same strategy raw
      blend_vs_equal_raw  every blend vs the simplest baseline (raw equal_mean)
      blend_vs_era5       every blend vs task_v1 on ERA5
      model_mode_vs_raw   each model's thermo_qm / full_qm vs its own raw
      model_vs_era5       each model (every mode) vs ERA5
      model_vs_blend      each model (every mode) vs `reference_blend` (the recommended blend)
    """
    evs = t_eval["event_key"].astype(str).to_numpy()
    cache: dict[tuple[str, str, str], tuple[np.ndarray, np.ndarray]] = {}

    def errs(kind: str, name: str, mode: str) -> tuple[np.ndarray, np.ndarray]:
        key = (kind, name, mode)
        if key not in cache:
            src = era_pred if kind == "era5" else (model_preds if kind == "model" else blend_preds)[(name, mode)]
            cache[key] = (_winner_time_err(t_eval, src[0]), _goal_err(t_eval, src[1]))
        return cache[key]

    rows: list[dict[str, Any]] = []

    def add(comparison: str, a: tuple[str, str, str], b: tuple[str, str, str]) -> None:
        ta, ga = errs(*a)
        tb, gb = errs(*b)
        tm = paired_bootstrap_diff(evs, ta, tb, n_iter=n_iter)
        if not tm.get("tasks"):
            return
        gm = paired_bootstrap_diff(evs, ga, gb, n_iter=n_iter)
        rows.append({
            "comparison": comparison,
            "a": f"{a[1]}/{a[2]}", "b": f"{b[1]}/{b[2]}",
            "tasks": tm["tasks"], "events": tm["events"],
            "time_mae_a_min": tm["mean_a"], "time_mae_b_min": tm["mean_b"],
            "time_diff_min": tm["mean_diff"], "time_diff_ci95": tm.get("ci95"),
            "time_p_a_not_better": tm.get("p_a_not_better"), "time_significant": tm.get("significant"),
            "goal_diff_pts": gm.get("mean_diff"), "goal_diff_ci95": gm.get("ci95"),
            "goal_significant": gm.get("significant"),
        })

    era = ("era5", "era5_reanalysis", "reanalysis")
    blend_modes = sorted({m for _, m in blend_preds})
    blend_names = sorted({s for s, _ in blend_preds})
    for sname in blend_names:
        for mode in blend_modes:
            if (sname, mode) not in blend_preds:
                continue
            if mode != "raw" and (sname, "raw") in blend_preds:
                add("blend_mode_vs_raw", ("blend", sname, mode), ("blend", sname, "raw"))
            # equal_mean under QM vs equal_mean raw is already a blend_mode_vs_raw row.
            if sname != "equal_mean" and ("equal_mean", "raw") in blend_preds:
                add("blend_vs_equal_raw", ("blend", sname, mode), ("blend", "equal_mean", "raw"))
            add("blend_vs_era5", ("blend", sname, mode), era)

    for mid, mode in sorted(model_preds):
        if mode != "raw" and (mid, "raw") in model_preds:
            add("model_mode_vs_raw", ("model", mid, mode), ("model", mid, "raw"))
        add("model_vs_era5", ("model", mid, mode), era)
        if reference_blend is not None and reference_blend in blend_preds:
            add("model_vs_blend", ("model", mid, mode), ("blend", *reference_blend))
    return rows


def _task_eval_metrics(df_eval: pd.DataFrame, pred_kmh: np.ndarray, pred_goal: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ok_k = np.isfinite(pred_kmh) & df_eval["winner_kmh"].notna().to_numpy()
    if ok_k.any():
        yk = df_eval["winner_kmh"].to_numpy(float)[ok_k]
        pk = np.maximum(5.0, pred_kmh[ok_k])
        err_min = _winner_time_err(df_eval, pred_kmh)[ok_k]
        evs = df_eval["event_key"].astype(str).to_numpy()[ok_k]
        lo, hi = FWW.cluster_bootstrap_ci(evs, err_min, lambda x: float(np.nanmean(x)), n_iter=500)
        out.update({
            "tasks_time": int(ok_k.sum()),
            "winner_kmh_mae": round(float(np.mean(np.abs(yk - pk))), 3),
            "winner_time_mae_min": round(float(np.nanmean(err_min)), 2),
            "winner_time_ci95": [round(lo, 2), round(hi, 2)],
            "within_15_min": round(float(np.nanmean(err_min <= 15.0)), 4),
            "within_30_min": round(float(np.nanmean(err_min <= 30.0)), 4),
        })
    ok_g = np.isfinite(pred_goal) & df_eval["goal_share"].notna().to_numpy()
    if ok_g.any():
        yg = df_eval["goal_share"].to_numpy(float)[ok_g]
        pg = np.clip(pred_goal[ok_g], 0.0, 1.0)
        out.update({
            "tasks_goal": int(ok_g.sum()),
            "goal_share_mae_pts": round(float(np.mean(np.abs(yg - pg)) * 100.0), 2),
            "within_10_pts": round(float(np.mean(np.abs(yg - pg) <= 0.10)), 4),
        })
    return out


def evaluate_tasks_and_calibration(
    db: Path,
    selected: list[dict[str, Any]],
    thermo_tables: dict[str, pd.DataFrame],
    trigger_cv: dict[str, Any],
    wind_cv: dict[str, Any],
    ceil_scores: pd.DataFrame,
    climb_scores: pd.DataFrame,
    from_year: int,
    out_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Leakage-free 5-fold CV of task_v1 run on historical forecasts: raw vs thermo-QM vs full-QM,
    and equal vs wind-weighted vs multi-variable composite blends."""
    t_all = TB.tasks_table(db)
    if not len(t_all):
        return pd.DataFrame(), {"error": "empty tasks_table"}, {}, {}

    rt = TM.routes(db, t_all)
    sc = pd.read_parquet(db / "site_chips.parquet", columns=["site_id", "chip_id", "in_area"])
    area = {s_: g.loc[g["in_area"].fillna(False).astype(bool), "chip_id"].to_numpy() for s_, g in sc.groupby("site_id")}
    start, end = TM.windows(t_all)
    need_list = []
    for i, r in enumerate(t_all.itertuples(index=False)):
        chips = rt[r.task_key][0] if r.task_key in rt else area.get(r.site_id, np.array([], int))
        need_list += TM.need_rows(r.task_key, chips, start.iloc[i], end.iloc[i])
    need_all = pd.DataFrame(need_list, columns=["task_key", "chip_id", "time"])
    legs = {k: v[1] for k, v in rt.items()}

    era_tables = TM.load_tables(db, set(need_all["chip_id"]))
    era_wx = TM.features_from(need_all, era_tables, legs)
    t_all = t_all.merge(era_wx, left_on="task_key", right_index=True, how="left")

    base_feats = [f for f in TB.FEATURES if f in t_all and t_all[f].notna().any()]
    wx_feats = [f for f in TM.WEATHER if f in t_all and t_all[f].notna().any()]
    full_feats = base_feats + wx_feats

    t_eval = t_all[t_all["date"].dt.year >= from_year].reset_index(drop=True)
    t_pre = t_all[t_all["date"].dt.year < from_year].reset_index(drop=True)
    eval_keys = set(t_eval["task_key"])
    need_eval = need_all[need_all["task_key"].isin(eval_keys)].copy()

    # Build per-model forecast weather feature tables for all 2022+ tasks.
    wind_cache_dir = db / "weather" / "hist_forecast"
    model_wx: dict[str, pd.DataFrame] = {}
    for spec in selected:
        mid = spec["id"]
        tx = thermo_tables.get(mid)
        if tx is None or not len(tx):
            continue
        wp = wind_cache_dir / f"{mid}.parquet"
        wdf = pd.read_parquet(wp) if wp.exists() else pd.DataFrame()
        if len(wdf):
            wdf["hour"] = pd.to_datetime(wdf["hour"]).astype("datetime64[ns]")
        exc = float((trigger_cv.get(mid) or {}).get("recommended_excess_c", 1.0))
        tbls = _assemble_task_tables(wdf, tx, era_tables, excess_c=exc)
        if tbls is None:
            continue
        wx_m = TM.features_from(need_eval, tbls, legs)
        # Keep only tasks where this model actually has temperature/wind data.
        has_data = wx_m["t2m_c"].notna() | wx_m["blh_mean"].notna()
        wx_m = wx_m[has_data]
        if len(wx_m) >= 15:
            model_wx[mid] = wx_m

    active_mids = sorted(model_wx.keys())
    n_ev = int(t_eval["event_key"].nunique())
    if not active_mids or n_ev < 3:
        return pd.DataFrame(), {"error": f"insufficient forecast task overlap (models={len(active_mids)}, events={n_ev})"}, {}, {}

    # 5-fold CV strictly grouped by event_key on 2022+ tasks; pre-2022 tasks always stay in train.
    gkf = GroupKFold(n_splits=min(5, n_ev))
    era_oof_k = np.full(len(t_eval), np.nan)
    era_oof_g = np.full(len(t_eval), np.nan)

    # Store OOF predictions per (model, mode) where mode in ("raw", "thermo_qm", "full_qm")
    oof_preds: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {
        (mid, mode): (np.full(len(t_eval), np.nan), np.full(len(t_eval), np.nan))
        for mid in active_mids
        for mode in ("raw", "thermo_qm", "full_qm")
    }

    for tr_i, te_i in gkf.split(t_eval, groups=t_eval["event_key"]):
        tr_df = pd.concat([t_pre, t_eval.iloc[tr_i]], ignore_index=True)
        te_df = t_eval.iloc[te_i]
        tr_keys = set(t_eval.iloc[tr_i]["task_key"])
        te_keys = list(te_df["task_key"])

        # Train leakage-free task_v1 models on ERA5 training fold.
        mk = _make_task_regressor()
        tr_k = tr_df.dropna(subset=["winner_kmh"])
        mk.fit(tr_k[full_feats], tr_k["winner_kmh"])

        mg = _make_task_regressor()
        tr_g = tr_df.dropna(subset=["goal_share"])
        mg.fit(tr_g[full_feats], tr_g["goal_share"])

        era_oof_k[te_i] = mk.predict(te_df[full_feats])
        era_oof_g[te_i] = mg.predict(te_df[full_feats])

        era_tr_wx = t_eval.iloc[tr_i].set_index("task_key")

        for mid in active_mids:
            wx_m = model_wx[mid]
            te_present = [idx for idx, tk in zip(te_i, te_keys) if tk in wx_m.index]
            if not te_present:
                continue
            te_sub = t_eval.loc[te_present].copy()
            te_wx = wx_m.loc[te_sub["task_key"]].copy()

            # Fit quantile maps strictly on tr_keys matched with ERA5.
            tr_matched_keys = [tk for tk in tr_keys if tk in wx_m.index]
            qmaps: dict[str, dict[str, Any] | None] = {}
            if len(tr_matched_keys) >= 25:
                src_tr = wx_m.loc[tr_matched_keys]
                ref_tr = era_tr_wx.loc[tr_matched_keys]
                for col in ALL_QM_FEATURES:
                    if col in src_tr and col in ref_tr:
                        qmaps[col] = fit_quantile_map(src_tr[col].to_numpy(float), ref_tr[col].to_numpy(float), min_pairs=25)

            for mode, q_cols in (("raw", []), ("thermo_qm", THERMO_QM_FEATURES), ("full_qm", ALL_QM_FEATURES)):
                X_te = te_sub[full_feats].copy()
                for col in wx_feats:
                    if col in te_wx:
                        vals = te_wx[col].to_numpy(float)
                        if col in q_cols and qmaps.get(col) is not None:
                            vals = apply_quantile_map(vals, qmaps[col])
                        X_te[col] = vals
                oof_preds[(mid, mode)][0][te_present] = mk.predict(X_te)
                oof_preds[(mid, mode)][1][te_present] = mg.predict(X_te)

    # Score individual models across raw / thermo_qm / full_qm.
    e2e_rows = []
    era_met = _task_eval_metrics(t_eval, era_oof_k, era_oof_g)
    e2e_rows.append({"model": "era5_reanalysis", "mode": "reanalysis", **era_met})

    best_mode_by_model: dict[str, str] = {}
    task_rel_by_model: dict[str, float] = {}
    task_n_by_model: dict[str, float] = {}
    for mid in active_mids:
        mode_scores = {}
        for mode in ("raw", "thermo_qm", "full_qm"):
            pk, pg = oof_preds[(mid, mode)]
            met = _task_eval_metrics(t_eval, pk, pg)
            # Paired ERA5 error on the exact same tasks covered by this model.
            ok_m = np.isfinite(pk) & t_eval["winner_kmh"].notna().to_numpy()
            era_pair = _task_eval_metrics(t_eval.loc[ok_m], era_oof_k[ok_m], era_oof_g[ok_m]) if ok_m.any() else {}
            rel = (
                round(met["winner_time_mae_min"] / max(era_pair["winner_time_mae_min"], 1.0), 4)
                if met.get("winner_time_mae_min") and era_pair.get("winner_time_mae_min")
                else None
            )
            e2e_rows.append({
                "model": mid, "mode": mode, **met,
                "era5_pair_time_mae_min": era_pair.get("winner_time_mae_min"),
                "rel_time_mae_vs_era5": rel,
            })
            mode_scores[mode] = (met.get("winner_time_mae_min", float("inf")), rel, met.get("tasks_time", 0))
        best_m = min(mode_scores, key=lambda k: mode_scores[k][0])
        best_mode_by_model[mid] = best_m
        if mode_scores[best_m][1] is not None:
            task_rel_by_model[mid] = float(mode_scores[best_m][1])
            task_n_by_model[mid] = float(mode_scores[best_m][2])

    # Derive multi-pillar composite global/regional/site weights and evaluate blends out-of-fold.
    wind_hier = wind_cv.get("clean_wind_hierarchy") or {}
    wind_rel = wind_hier.get("global_rel_rmse") or {}
    wind_n = {k: float(v) for k, v in (wind_cv.get("free_middle_groups") or {}).items()}
    ceil_rel: dict[str, float] = {}
    ceil_n: dict[str, float] = {}
    if len(ceil_scores):
        for mid, grp in ceil_scores.groupby("model"):
            pref = grp[grp["variant"].isin(["native_min_blh_lcl", "parcel_cv_tuned"])]
            use = pref if len(pref) else grp
            best_row = use.dropna(subset=["rel_rmse_vs_era5"]).sort_values("rel_rmse_vs_era5").head(1)
            if len(best_row):
                ceil_rel[str(mid)] = float(best_row["rel_rmse_vs_era5"].iloc[0])
                ceil_n[str(mid)] = float(best_row["groups"].iloc[0])
    climb_rel: dict[str, float] = {}
    climb_n: dict[str, float] = {}
    if len(climb_scores):
        for r in climb_scores.dropna(subset=["rel_rmse_vs_era5"]).itertuples(index=False):
            climb_rel[str(r.model)] = float(r.rel_rmse_vs_era5)
            climb_n[str(r.model)] = float(r.groups)

    # Shrink each pillar toward its median across models by sample size, so a model scored on a
    # few hundred groups or a few dozen tasks (e.g. HRDPS) cannot earn a large weight by chance.
    pillar_rel = {"wind": wind_rel, "ceiling": ceil_rel, "climb": climb_rel, "task": task_rel_by_model}
    pillar_n = {"wind": wind_n, "ceiling": ceil_n, "climb": climb_n, "task": task_n_by_model}
    # Without a stored sample size (an older wind_clean_cv.json), a pillar is left unshrunk.
    pillar_shrunk = {
        p: shrink_pillar(rel, pillar_n[p], COMPOSITE_SHRINK_N[p]) if pillar_n[p] else dict(rel)
        for p, rel in pillar_rel.items()
    }

    # Composite relative skill: weighted mean of available (shrunk) pillar rel_rmse_vs_era5
    # (0.35 clean wind + 0.25 top-of-lift + 0.25 core60 climb + 0.15 end-to-end task speed).
    pillar_weights = {"wind": 0.35, "ceiling": 0.25, "climb": 0.25, "task": 0.15}
    composite_rel: dict[str, float] = {}
    pillar_breakdown: dict[str, dict[str, float | None]] = {}
    pillar_breakdown_shrunk: dict[str, dict[str, float | None]] = {}
    pillar_samples: dict[str, dict[str, int | None]] = {}
    for spec in selected:
        mid = spec["id"]
        raw_vals = {p: pillar_rel[p].get(mid) for p in pillar_weights}
        vals = {p: pillar_shrunk[p].get(mid) for p in pillar_weights}
        pillar_breakdown[mid] = {k: (round(v, 4) if v is not None else None) for k, v in raw_vals.items()}
        pillar_breakdown_shrunk[mid] = {k: (round(v, 4) if v is not None else None) for k, v in vals.items()}
        pillar_samples[mid] = {p: (int(pillar_n[p][mid]) if mid in pillar_n[p] else None) for p in pillar_weights}
        avail = [(pillar_weights[k], v) for k, v in vals.items() if v is not None]
        if avail:
            composite_rel[mid] = sum(w * v for w, v in avail) / sum(w for w, _ in avail)

    beta = float(wind_hier.get("beta", 4.0))
    comp_global_w = FWW._skill_to_weights(composite_rel, beta) if composite_rel else {}
    wind_global_w = wind_hier.get("global") or {}

    # Blend comparison on 2022+ tasks where >= 2 models are available.
    blend_cv: dict[str, Any] = {}
    blend_preds: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    for mode in ("raw", "thermo_qm", "full_qm"):
        K_mat = np.column_stack([oof_preds[(m, mode)][0] for m in active_mids])
        G_mat = np.column_stack([oof_preds[(m, mode)][1] for m in active_mids])
        valid_row = np.isfinite(K_mat).sum(axis=1) >= 2
        if not valid_row.any():
            continue
        sub_t = t_eval.loc[valid_row].reset_index(drop=True)
        Km, Gm = K_mat[valid_row], G_mat[valid_row]
        strategies: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        with np.errstate(all="ignore"):
            strategies["equal_mean"] = (np.nanmean(Km, axis=1), np.nanmean(Gm, axis=1))
            strategies["median"] = (np.nanmedian(Km, axis=1), np.nanmedian(Gm, axis=1))
        for sname, wdict in (("wind_weighted", wind_global_w), ("multivar_weighted", comp_global_w)):
            if not wdict:
                continue
            wv = np.array([wdict.get(m, 0.0) for m in active_mids], dtype=float)
            ok_k = np.isfinite(Km)
            ok_g = np.isfinite(Gm)
            wk = np.where(ok_k, wv[None, :], 0.0)
            wg = np.where(ok_g, wv[None, :], 0.0)
            pk = np.where(wk.sum(axis=1) > 0, (np.where(ok_k, Km, 0.0) * wk).sum(axis=1) / np.maximum(wk.sum(axis=1), 1e-9), np.nan)
            pg = np.where(wg.sum(axis=1) > 0, (np.where(ok_g, Gm, 0.0) * wg).sum(axis=1) / np.maximum(wg.sum(axis=1), 1e-9), np.nan)
            strategies[sname] = (pk, pg)
        blend_cv[mode] = {
            sname: _task_eval_metrics(sub_t, pk, pg)
            for sname, (pk, pg) in strategies.items()
        }
        blend_cv[mode]["era5_reanalysis"] = _task_eval_metrics(sub_t, era_oof_k[valid_row], era_oof_g[valid_row])
        # Full-length copies aligned with t_eval for the paired tests below.
        for sname, (pk, pg) in strategies.items():
            full_k = np.full(len(t_eval), np.nan)
            full_g = np.full(len(t_eval), np.nan)
            full_k[valid_row] = pk
            full_g[valid_row] = pg
            blend_preds[(sname, mode)] = (full_k, full_g)

    # Pick the QM mode whose preferred blend has the lowest out-of-fold winner time MAE ...
    def _best_blend(mode_name: str) -> tuple[str | None, float]:
        bm = blend_cv.get(mode_name) or {}
        for key in ("multivar_weighted", "wind_weighted", "equal_mean"):
            if key in bm and bm[key].get("winner_time_mae_min") is not None:
                return key, float(bm[key]["winner_time_mae_min"])
        return None, float("inf")

    mode_maes = {m: _best_blend(m)[1] for m in ("raw", "thermo_qm", "full_qm") if m in blend_cv}
    recommended_qm_mode = min(mode_maes, key=mode_maes.get) if mode_maes else "raw"
    recommended_strategy = _best_blend(recommended_qm_mode)[0]
    reference_blend = (recommended_strategy, recommended_qm_mode) if recommended_strategy else None

    # ... and test every gain with a paired, competition-clustered bootstrap on identical tasks.
    paired_rows = paired_task_comparisons(
        t_eval, (era_oof_k, era_oof_g), oof_preds, blend_preds, reference_blend,
    )
    # Apply QM only when its gain over the same blend without QM is significant (95% CI below 0).
    apply_test = next(
        (r for r in paired_rows
         if r["comparison"] == "blend_mode_vs_raw" and r["a"] == f"{recommended_strategy}/{recommended_qm_mode}"),
        None,
    )
    recommended_apply = bool(recommended_qm_mode != "raw" and apply_test is not None and apply_test["time_significant"] and apply_test["time_diff_min"] < 0)

    # Fit final full-sample quantile maps per model for export in forecast_calibration.json.
    era_all_eval = t_eval.set_index("task_key")
    cal_models: dict[str, Any] = {}
    for mid in active_mids:
        wx_m = model_wx[mid]
        matched = [tk for tk in t_eval["task_key"] if tk in wx_m.index]
        qmaps_full: dict[str, Any] = {}
        if len(matched) >= 25:
            src = wx_m.loc[matched]
            ref = era_all_eval.loc[matched]
            for col in ALL_QM_FEATURES:
                if col in src and col in ref:
                    qm = fit_quantile_map(src[col].to_numpy(float), ref[col].to_numpy(float), min_pairs=25)
                    if qm is not None:
                        qmaps_full[col] = qm
        cal_models[mid] = {
            "parcel_excess_c": float((trigger_cv.get(mid) or {}).get("recommended_excess_c", 1.0)),
            "best_cv_mode": best_mode_by_model.get(mid, "raw"),
            "n_tasks_matched": int(len(matched)),
            "quantile_maps": qmaps_full,
        }

    cal_payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from_year": from_year,
        "recommended_qm_mode": recommended_qm_mode,
        "recommended_strategy": recommended_strategy,
        "recommended_apply": recommended_apply,
        "recommended_apply_rule": "QM mode with the lowest blend MAE, applied only if its paired 95% CI vs the same blend raw is below 0",
        "recommended_apply_test": apply_test,
        "thermo_qm_features": THERMO_QM_FEATURES,
        "all_qm_features": ALL_QM_FEATURES,
        "blend_cv_by_mode": blend_cv,
        "paired_bootstrap": paired_rows,
        "models": cal_models,
    }

    # Combine composite global skill with site-level wind shrinkage from clean wind hierarchy.
    multivar_weights_payload = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "from_year": from_year,
        "beta": beta,
        "pillar_weights": pillar_weights,
        "pillar_rel_rmse": pillar_breakdown,
        "pillar_samples": pillar_samples,
        "pillar_shrink_n": COMPOSITE_SHRINK_N,
        "pillar_rel_rmse_shrunk": pillar_breakdown_shrunk,
        "composite_rel_rmse": {k: round(v, 4) for k, v in composite_rel.items()},
        "global": {k: round(v, 4) for k, v in comp_global_w.items()},
        "by_region": wind_hier.get("by_region") or {},
        "by_site": wind_hier.get("by_site") or {},
        "blend_cv_by_mode": blend_cv,
    }

    e2e_df = pd.DataFrame(e2e_rows)
    e2e_df.to_csv(out_dir / "task_e2e_scores.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out_dir / "paired_bootstrap.csv", index=False)
    (out_dir / "bias_correction_cv.json").write_text(json.dumps(blend_cv, indent=2), encoding="utf-8")
    (out_dir / "forecast_calibration.json").write_text(json.dumps(cal_payload, indent=2), encoding="utf-8")
    (out_dir / "forecast_multivar_weights.json").write_text(json.dumps(multivar_weights_payload, indent=2), encoding="utf-8")
    return e2e_df, blend_cv, cal_payload, multivar_weights_payload


# ── Optional experiment: regional high-res models at thermal centroid vs chip centre ─

def run_regional_centroid_check(
    db: Path,
    selected: list[dict[str, Any]],
    from_year: int,
    out_dir: Path,
    n_sample: int,
    pause: float,
    seed: int = 42,
    api_url: str = FWW.HIST_FORECAST_URL,
    timeout: float | None = None,
) -> dict[str, Any]:
    """Compare regional high-res models (ICON-D2, AROME, ICON-EU, HRDPS) sampled at the actual
    thermal centroid vs fixed 0.25 deg chip centre on identical wind observations."""
    if n_sample <= 0:
        return {"skipped": True}
    wg_p = db / "truth" / "truth_wind_groups.parquet"
    if not wg_p.exists():
        return {"skipped": True, "reason": "missing truth_wind_groups.parquet"}
    req_timeout = float(timeout) if timeout is not None else float(
        FWW.LOCAL_HTTP_TIMEOUT_S if FWW.is_local_api_url(api_url) else FWW.HTTP_TIMEOUT_S
    )

    g = pd.read_parquet(wg_p)
    g["hour"] = pd.to_datetime(g["hour"]).astype("datetime64[ns]")
    g = g[(g["hour"].dt.year >= from_year) & g["band"].isin(["free air", "middle"])].copy()
    g["date"] = g["hour"].dt.strftime("%Y-%m-%d")
    c_lat, c_lon = grid.chip_centre(g["chip_id"].to_numpy())
    dist_km = np.hypot((g["centroid_lat"] - c_lat) * 111.32, (g["centroid_lon"] - c_lon) * 111.32 * np.cos(np.radians(c_lat)))
    g = g[dist_km >= 3.0].copy()

    wind_cache_dir = db / "weather" / "hist_forecast"
    rng = np.random.default_rng(seed)
    results: dict[str, Any] = {}

    for spec in selected:
        if spec["region"] != "regional":
            continue
        mid = spec["id"]
        w_spec = FWW.MODEL_BY_ID.get(mid)
        cp = wind_cache_dir / f"{mid}.parquet"
        if w_spec is None or not cp.exists():
            continue
        sub = g[[FWW.in_bbox(la, lo, spec["bbox"]) for la, lo in zip(g["centroid_lat"], g["centroid_lon"])]].copy()
        if len(sub) < 15:
            continue
        wx_chip = pd.read_parquet(cp)
        wx_chip["hour"] = pd.to_datetime(wx_chip["hour"]).astype("datetime64[ns]")
        sub["ground_m"] = sub.get("ground_m", sub.get("chip_ground_m", sub["h_asl"] - sub["agl"]))
        cu_chip, cv_chip = FWW.interpolate_model_wind(sub, wx_chip)
        sub["chip_u"], sub["chip_v"] = cu_chip, cv_chip
        sub = sub[sub["chip_u"].notna()].copy()
        if len(sub) < 15:
            continue

        pick = sub.iloc[rng.choice(len(sub), size=min(n_sample, len(sub)), replace=False)].reset_index(drop=True)
        req_vars = ",".join(FWW._vars_for_model(w_spec))
        cent_u = np.full(len(pick), np.nan)
        cent_v = np.full(len(pick), np.nan)

        for idx, r in enumerate(pick.itertuples(index=False)):
            try:
                data = FWW._http_get_json({
                    "latitude": f"{r.centroid_lat:.4f}", "longitude": f"{r.centroid_lon:.4f}",
                    "start_date": r.date, "end_date": r.date,
                    "hourly": req_vars, "models": w_spec["slug"],
                    "wind_speed_unit": "ms", "timezone": "UTC",
                }, base_url=api_url, timeout=req_timeout)
                df1 = FWW._parse_batch_response([data] if isinstance(data, dict) else data, [int(r.chip_id)], w_spec, {int(r.chip_id): {r.hour}})
                if len(df1):
                    row_df = pick.iloc[[idx]]
                    uu, vv = FWW.interpolate_model_wind(row_df, df1)
                    cent_u[idx], cent_v[idx] = uu[0], vv[0]
            except FWW.QuotaExceeded as exc:
                results[mid] = {"paused_at_quota": str(exc)}
                break
            except Exception:
                continue
            if pause > 0:
                time.sleep(pause)

        ok = np.isfinite(cent_u) & np.isfinite(cent_v)
        if ok.sum() >= 10:
            pu = pick.loc[ok, "clean_u"].to_numpy(float)
            pv = pick.loc[ok, "clean_v"].to_numpy(float)
            m_cent = FWW.wind_metrics(pu, pv, cent_u[ok], cent_v[ok])
            m_chip = FWW.wind_metrics(pu, pv, pick.loc[ok, "chip_u"].to_numpy(float), pick.loc[ok, "chip_v"].to_numpy(float))
            results[mid] = {
                "n_matched": int(ok.sum()),
                "centroid": m_cent,
                "chip_centre": m_chip,
                "delta_rmse_ms": round(m_cent["vector_rmse_ms"] - m_chip["vector_rmse_ms"], 4),
            }

    (out_dir / "regional_centroid_check.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


# ── Main CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder (e.g. /content/drive/MyDrive/IGCS/database)")
    parser.add_argument("--out", type=Path, help="output folder (default DB/models/forecast_multivar_eval)")
    parser.add_argument("--from-year", type=int, default=2022, help="earliest year to evaluate (default 2022)")
    parser.add_argument("--models", type=str, default="all", help="comma-separated model ids or 'all'")
    parser.add_argument("--api-url", type=str, default=FWW.HIST_FORECAST_URL, help=f"forecast API endpoint (default {FWW.HIST_FORECAST_URL})")
    parser.add_argument("--timeout", type=float, default=None, help="HTTP timeout in seconds (default 35 remote, 120 local)")
    parser.add_argument("--batch-chips", type=int, default=BATCH_CHIPS, help=f"chips per multi-coordinate request (default {BATCH_CHIPS})")
    parser.add_argument("--workers", type=int, default=1, help="concurrent request workers (local server only; clamped to 1 on public API)")
    parser.add_argument("--pause", type=float, default=None, help="seconds between requests (default 0.4 remote, 0.0 local)")
    parser.add_argument("--max-sites", type=int, default=None, help="limit to top N sites (smoke test)")
    parser.add_argument("--wind-only", action="store_true", help="only run Pillar 1 offline re-scoring on cached wind parquets (0 API calls)")
    parser.add_argument("--download-only", action="store_true", help="only download/update thermodynamic caches for --models")
    parser.add_argument("--skip-download", action="store_true", help="run all 4 pillars offline using cached wind + thermo parquets")
    parser.add_argument("--regional-centroid-check", type=int, default=0, help="sample N regional wind groups to compare centroid vs chip centre (default 0)")
    args = parser.parse_args()
    local_api = FWW.is_local_api_url(args.api_url)
    pause_s = float(args.pause) if args.pause is not None else (0.0 if local_api else 0.4)

    out = args.out or (args.db / "models" / "forecast_multivar_eval")
    thermo_dir = args.db / "weather" / "hist_forecast_thermo"
    out.mkdir(parents=True, exist_ok=True)
    thermo_dir.mkdir(parents=True, exist_ok=True)

    selected = THERMO_MODELS if args.models == "all" else [MODEL_BY_ID[m.strip()] for m in args.models.split(",") if m.strip() in MODEL_BY_ID]
    if not selected:
        raise SystemExit(f"No valid models in --models={args.models!r}. Valid: {list(MODEL_BY_ID)}")

    if args.wind_only:
        print("=== Pillar 1: Offline re-scoring cached winds vs IGC truth_wind_groups ===", flush=True)
        w_scores, w_cv = evaluate_clean_wind(args.db, selected, args.from_year, out, max_sites=args.max_sites)
        fm = w_scores[w_scores["band"] == "free+middle"].sort_values("rel_rmse_vs_era5") if len(w_scores) else pd.DataFrame()
        if len(fm):
            cols = ["model", "groups", "vector_rmse_ms", "speed_bias_ms", "old_speed_bias_ms", "era5_rmse_ms", "rel_rmse_vs_era5", "dir_error_deg", "dir_within_30"]
            print(fm[cols].round(3).to_string(index=False), flush=True)
        if "strategies" in w_cv:
            print("\nGrouped 5-fold CV on clean IGC wind (free+middle):", flush=True)
            for sname, met in w_cv["strategies"].items():
                if met.get("groups"):
                    ddeg = f"{met['dir_error_deg']:.1f} deg" if met.get("dir_error_deg") is not None else "-"
                    w30 = f"{met['dir_within_30']:.1%}" if met.get("dir_within_30") is not None else "-"
                    print(f"  {sname:18s}: RMSE={met['vector_rmse_ms']:.3f} m/s  bias={met['speed_bias_ms']:+.3f} m/s  dir={ddeg}  <30={w30}", flush=True)
        return

    targets = build_thermo_targets(args.db, from_year=args.from_year, max_sites=args.max_sites)
    print(f"Thermodynamic targets since {args.from_year}: {len(targets):,} chip-hours across {targets['site_id'].nunique()} sites and {targets['chip_id'].nunique()} chips", flush=True)

    thermo_tables: dict[str, pd.DataFrame] = {}
    quota_stopped = False
    for spec in selected:
        mid = spec["id"]
        if args.skip_download or quota_stopped:
            thermo_tables[mid] = load_cached_thermo(thermo_dir, spec)
        else:
            try:
                thermo_tables[mid] = download_model_thermo(
                    targets, spec, thermo_dir, pause=pause_s,
                    api_url=args.api_url, timeout=args.timeout,
                    batch_chips=args.batch_chips, workers=args.workers,
                )
            except FWW.QuotaExceeded as exc:
                print(f"\n[PAUSED] {exc}\n         Re-run this cell after Open-Meteo's quota resets.\n", flush=True)
                quota_stopped = True
                thermo_tables[mid] = load_cached_thermo(thermo_dir, spec)

    if args.download_only:
        print("\nThermo cache summary:", flush=True)
        for spec in selected:
            mid = spec["id"]
            df = thermo_tables[mid]
            print(f"  {mid:12s}: {len(df):,} cached chip-hours" + (" (paused at quota)" if quota_stopped else " (done)"), flush=True)
        return

    print("\n=== Pillar 1: Clean wind verification (cached wind parquets) ===", flush=True)
    w_scores, w_cv = evaluate_clean_wind(args.db, selected, args.from_year, out, max_sites=args.max_sites)
    if len(w_scores):
        fm = w_scores[w_scores["band"] == "free+middle"].sort_values("rel_rmse_vs_era5")
        if len(fm):
            print(fm[["model", "groups", "vector_rmse_ms", "speed_bias_ms", "rel_rmse_vs_era5", "dir_within_30"]].round(3).to_string(index=False), flush=True)

    print("\n=== Pillar 2: Top of Lift ASL & Parcel BLH trigger CV ===", flush=True)
    ceil_scores, trigger_cv = evaluate_ceiling(args.db, selected, thermo_tables, args.from_year, out, max_sites=args.max_sites)
    if len(ceil_scores):
        print(ceil_scores[["model", "variant", "groups", "rmse_m", "bias_pred_minus_obs_m", "corr_all", "within_site_day_corr", "rel_rmse_vs_era5"]].round(3).to_string(index=False), flush=True)

    print("\n=== Pillar 3: Core60 climbing speed verification (5-fold CV) ===", flush=True)
    climb_scores = evaluate_climb_strength(args.db, selected, thermo_tables, trigger_cv, args.from_year, out, max_sites=args.max_sites)
    if len(climb_scores):
        print(climb_scores[["model", "groups", "rmse_ms", "mae_ms", "r2", "wstar_corr", "era5_rmse_ms", "rel_rmse_vs_era5"]].round(4).to_string(index=False), flush=True)

    print("\n=== Pillar 4: End-to-end task_v1 & Forecast -> ERA5 Quantile Mapping (5-fold CV) ===", flush=True)
    e2e_df, blend_cv, cal_payload, mw_payload = evaluate_tasks_and_calibration(
        args.db, selected, thermo_tables, trigger_cv, w_cv, ceil_scores, climb_scores, args.from_year, out,
    )
    if len(e2e_df):
        cols = [c for c in ["model", "mode", "tasks_time", "winner_time_mae_min", "within_15_min", "goal_share_mae_pts", "rel_time_mae_vs_era5"] if c in e2e_df]
        print(e2e_df[cols].to_string(index=False), flush=True)
    if blend_cv:
        print(f"\nRecommended QM mode: {cal_payload.get('recommended_qm_mode')} (apply={cal_payload.get('recommended_apply')})", flush=True)
        for mode, strats in blend_cv.items():
            print(f"  [{mode}]", flush=True)
            for sname, met in strats.items():
                if met.get("tasks_time"):
                    print(f"    {sname:18s}: time_MAE={met['winner_time_mae_min']:.2f} min  <=15m={met['within_15_min']:.1%}  goal_MAE={met.get('goal_share_mae_pts', float('nan')):.2f} pts", flush=True)
        paired = cal_payload.get("paired_bootstrap") or []
        if paired:
            ref = f"{cal_payload.get('recommended_strategy')}/{cal_payload.get('recommended_qm_mode')}"
            print("\nPaired bootstrap (A − B winner time MAE on identical tasks, competitions resampled; all rows in paired_bootstrap.csv):", flush=True)
            shown = {
                "blend_mode_vs_raw": lambda r: True,
                "blend_vs_equal_raw": lambda r: r["a"] == ref,
                "blend_vs_era5": lambda r: r["a"] == ref,
                "model_vs_blend": lambda r: r["a"].endswith("/raw"),
            }
            for comp, keep in shown.items():
                rows = [r for r in paired if r["comparison"] == comp and keep(r)]
                if not rows:
                    continue
                print(f"  [{comp}]", flush=True)
                for r in rows:
                    lo, hi = r["time_diff_ci95"]
                    flag = " *" if r["time_significant"] else ""
                    print(f"    {r['a']:30s} vs {r['b']:30s}: {r['time_diff_min']:+.2f} min [{lo:+.2f}, {hi:+.2f}]  n={r['tasks']}{flag}", flush=True)
            print("  (* = 95% CI excludes 0)", flush=True)

    if args.regional_centroid_check > 0:
        print(f"\n=== Optional regional centroid vs chip-centre check (n={args.regional_centroid_check}) ===", flush=True)
        rc = run_regional_centroid_check(
            args.db, selected, args.from_year, out, n_sample=args.regional_centroid_check,
            pause=pause_s, api_url=args.api_url, timeout=args.timeout,
        )
        print(json.dumps(rc, indent=2), flush=True)

    print(f"\nSaved multi-variable verification outputs to {out}", flush=True)


if __name__ == "__main__":
    main()
