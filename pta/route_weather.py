"""Shared hourly space-time route weather, vertical flight-wind interpolation,
MacCready route kinematics, and chip-hour Quantile Mapping for Task Model v2.

Designed as the single source of truth shared by:
  - scripts/task_model_v2.py (ERA5 training, ablation ladder, and CV)
  - scripts/forecast_multivar_eval.py (2022+ Open-Meteo hourly forecast verification)
  - api/app.py & api/forecast.py (future live inference integration)

All features are strictly pre-task observables (zero target or post-task leakage).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pta import grid, terrain

STEP_KM = 5.0
MAX_WINDOW_H = 5
DEFAULT_WINDOW = (13, 16)   # local solar hours when start/deadline are unpublished
LEADER_KMH = 28.0           # fixed pre-task nominal leader pace for space-time ETA sampling
PACK_KMH = 18.0             # fixed pre-task nominal pack/cutoff pace
MAX_LEVEL_INTERP_GAP_H = 3.5

PRESSURE_LEVELS = (925, 850, 700, 600, 500)
SLOT_HOURS = (0, 1, 2, 3, 4)

SINGLE_COLS = [
    "blh_m", "cloud_base_m", "cape", "cin", "sensible_heat_wm2", "latent_heat_wm2",
    "sunshine_wm2", "cloud_total", "cloud_low", "rain_mmh", "t2m_c", "dewpoint_c",
    "u100", "v100", "u10", "v10", "gust10",
]
LEVEL_COLS = (
    [f"{v}_{p}" for p in PRESSURE_LEVELS for v in ("u", "v", "z", "t")]
    + ["rh_850", "rh_700"]
)
EXTRA_COLS = [
    "skin_temp_c", "soil_moisture", "freezing_level_m", "albedo", "snow_depth_m", "conv_rain_mmh",
]
LAND_COLS = [
    "sensible_heat_wm2", "sunshine_wm2", "skin_temp_c", "soil_moisture", "albedo", "snow_cover",
]
TERRAIN_CELL_COLS = [
    "elev_mean", "elev_min", "elev_max", "elev_std",
    "slope_mean", "slope_p90", "steep_share",
    "facing_east", "facing_north", "curvature",
    "ridge_share", "valley_share", "tpi_5km", "tpi_20km", "relief_5km", "flat_share",
]

# Realistic competition polar approximations: (v_trim_kmh, sink_trim_ms, v_bar_kmh, sink_bar_ms).
GLIDER_POLARS: dict[str, tuple[float, float, float, float]] = {
    "CCC":  (42.0, 1.05, 63.0, 2.55),
    "EN-D": (40.0, 1.05, 58.0, 2.45),
    "EN-C": (38.0, 1.08, 53.0, 2.35),
    "EN-B": (36.0, 1.12, 48.0, 2.30),
    "EN-A": (34.0, 1.18, 44.0, 2.30),
}

# Feature group registries (kept in strict sync with extract_hourly_route_features).
SLOT_VARS = ("usable", "blh", "sun", "cloud", "rain", "wind_flt", "tail_flt", "wstar_rad")
HOURLY_SLOT_FEATURES = [f"{v}_h{k}" for k in SLOT_HOURS for v in SLOT_VARS]

HOURLY_DYNAMICS_FEATURES = [
    "blh_mean", "blh_max", "blh_min", "blh_growth_0_2", "blh_decay_late",
    "usable_mean", "usable_min_h", "usable_p10_h", "hours_usable_gt_500", "hours_usable_gt_800",
    "cloud_base_above_mean", "cloud_base_above_min",
    "sunshine_mean", "sunshine_min_h", "cloud_total_mean", "cloud_low_mean", "cloud_buildup",
    "rain_total_mm", "rain_max_mmh", "rain_hours", "rain_onset_rel_h", "rain_day_before_mm",
    "cape_mean", "cape_max_h", "cin_mean", "cin_h0", "spread_mean", "t2m_mean",
    "lapse_850_700_mean", "lapse_850_700_max_h", "lapse_700_600_mean",
    "rh_850_mean", "rh_700_mean", "rh_700_max_h",
    "skin_minus_air_mean", "soil_moisture_mean", "freezing_level_mean",
    "wstar_rad_mean", "wstar_rad_min_h", "wstar_rad_max_h", "wstar_rad_decay",
    "wind100_mean", "gust10_max_h", "wind_flt_mean", "wind_flt_max_h", "wind_flt_pickup",
    "shear_sfc_flt_mean",
]

HOURLY_ETA_FEATURES = [
    "tail_lead_mean", "tail_lead_min_leg", "tail_lead_last_leg", "headwind_share_lead", "cross_lead_mean",
    "tail_pack_mean", "tail_pack_min_leg", "tail_pack_last_leg", "headwind_share_pack",
    "usable_lead_mean", "usable_lead_min", "usable_pack_mean", "usable_pack_min", "pack_late_collapse_m",
    "rain_lead_max", "rain_pack_max", "cloud_pack_last_third",
    "wstar_lead_mean", "wstar_pack_mean", "wstar_lead_min",
    "mc_rad_speed_lead_kmh", "mc_rad_margin_lead_h", "mc_rad_margin_pack_h", "mc_rad_unflyable_share",
    "route_chips", "route_n_legs",
]

# All M2 features are 100% available in both hourly ERA5 and hourly Open-Meteo forecasts.
HOURLY_CORE_FEATURES = HOURLY_SLOT_FEATURES + HOURLY_DYNAMICS_FEATURES + HOURLY_ETA_FEATURES

# M3a: 1 km terrain & hourly sun-on-slope along the route (also 100% forecast-compatible).
TERRAIN_ROUTE_FEATURES = [
    "route_elev_mean", "route_elev_max", "route_elev_p90", "route_elev_std",
    "route_relief_5km_mean", "route_relief_5km_max", "route_tpi_5km_mean", "route_tpi_20km_mean",
    "route_slope_mean", "route_ridge_share", "route_valley_share",
    "route_sun_direct_lead", "route_sun_energy_lead", "route_sun_direct_pack",
    "route_upslope_flt_lead", "route_lee_share_lead",
    "cell_usable_min_lead", "cell_usable_p10_lead", "cell_usable_min_pack",
]

# M3b: ERA5-specific surface flux & 0.1-deg ERA5-Land features (ablation group; absent in Open-Meteo).
ERA5_LAND_FEATURES = [
    "sensible_heat_mean", "sensible_heat_min_h", "wstar_shf_mean",
    "land_sensible_heat_lead", "land_sensible_heat_min_h", "land_sensible_heat_decay",
    "land_sunshine_lead", "land_skin_minus_air_lead", "land_soil_moisture_lead",
    "land_albedo_mean", "land_snow_max",
]

# Hourly variables eligible for chip-hour Forecast -> ERA5 Quantile Mapping.
HOURLY_QM_THERMO_SINGLE = [
    "blh_m", "cloud_base_m", "t2m_c", "dewpoint_c", "sunshine_wm2",
    "cloud_total", "cloud_low", "rain_mmh", "cape", "cin", "gust10",
]
HOURLY_QM_THERMO_EXTRA = ["skin_temp_c", "soil_moisture", "freezing_level_m"]
HOURLY_QM_THERMO_LEVELS = ["t_925", "t_850", "t_700", "t_600", "rh_850", "rh_700"]
HOURLY_QM_WIND_SINGLE = [("u10", "v10", "wind10"), ("u100", "v100", "wind100")]
HOURLY_QM_WIND_LEVELS = [
    ("u_925", "v_925", "wind_925"),
    ("u_850", "v_850", "wind_850"),
    ("u_700", "v_700", "wind_700"),
    ("u_600", "v_600", "wind_600"),
    ("u_500", "v_500", "wind_500"),
]

QM_PROBS = np.array(
    [0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
     0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.97, 0.99],
    dtype=np.float64,
)


# ── Geometry & Route Sampling ────────────────────────────────────────────────

def km_between(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2.0) ** 2
    return 6371.0 * 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def bearing_rad(lat1: Any, lon1: Any, lat2: Any, lon2: Any) -> Any:
    """Direction of travel, radians clockwise from north."""
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


def sample_route_segments(lat: Any, lon: Any, step_km: float = STEP_KM) -> pd.DataFrame:
    """Split an ordered list of turnpoints into midpoint-sampled segments of up to `step_km` km.

    Returns DataFrame with columns:
        seg_idx, leg_idx, n_legs, lat, lon, chip_id, cell_id, seg_km, cum_km, frac_route, leg_bearing_rad
    """
    la = np.asarray(lat, dtype=np.float64)
    lo = np.asarray(lon, dtype=np.float64)
    cols = [
        "seg_idx", "leg_idx", "n_legs", "lat", "lon",
        "chip_id", "cell_id", "seg_km", "cum_km", "frac_route", "leg_bearing_rad",
    ]
    if len(la) < 2:
        return pd.DataFrame(columns=cols)

    rows = []
    cum = 0.0
    leg_idx = 0
    for i in range(len(la) - 1):
        d = float(km_between(la[i], lo[i], la[i + 1], lo[i + 1]))
        if not np.isfinite(d) or d < 0.5:
            continue
        brg = float(bearing_rad(la[i], lo[i], la[i + 1], lo[i + 1]))
        n = max(int(math.ceil(d / step_km)), 1)
        seg_len = d / n
        fracs = (np.arange(n, dtype=np.float64) + 0.5) / n
        s_la = la[i] + fracs * (la[i + 1] - la[i])
        s_lo = lo[i] + fracs * (lo[i + 1] - lo[i])
        for k in range(n):
            rows.append((leg_idx, float(s_la[k]), float(s_lo[k]), seg_len, cum + (k + 0.5) * seg_len, brg))
        cum += d
        leg_idx += 1

    if not rows:
        return pd.DataFrame(columns=cols)

    df = pd.DataFrame(rows, columns=["leg_idx", "lat", "lon", "seg_km", "cum_km", "leg_bearing_rad"])
    df.insert(0, "seg_idx", np.arange(len(df), dtype=np.int32))
    df["n_legs"] = np.int32(leg_idx)
    df["chip_id"] = grid.chip_id(df["lat"].to_numpy(), df["lon"].to_numpy()).astype(np.int64)
    df["cell_id"] = grid.cell_id(df["lat"].to_numpy(), df["lon"].to_numpy()).astype(np.int64)
    total_km = max(cum, 1e-6)
    df["frac_route"] = (df["cum_km"] / total_km).astype(np.float64)
    return df[cols]


def routes_segments(db: Path, tasks_df: pd.DataFrame, step_km: float = STEP_KM) -> pd.DataFrame:
    """Segment table for every task in `tasks_df`.

    Tasks with turnpoint coordinates get exact leg-ordered 5 km segments (`has_route=True`).
    Tasks without turnpoint coordinates fall back to the site's `in_area` chips (`has_route=False`).
    """
    tp_path = db / "turnpoints.parquet"
    rt_frames: list[pd.DataFrame] = []
    have_route: set[str] = set()
    if tp_path.exists():
        tp = pd.read_parquet(tp_path, columns=["task_key", "seq", "lat", "lon"])
        tp = tp[tp["task_key"].isin(tasks_df["task_key"])].dropna(subset=["lat", "lon"])
        for key, g in tp.sort_values(["task_key", "seq"]).groupby("task_key", sort=False):
            if len(g) >= 2:
                seg = sample_route_segments(g["lat"].to_numpy(), g["lon"].to_numpy(), step_km=step_km)
                if len(seg):
                    seg.insert(0, "task_key", key)
                    seg["has_route"] = True
                    rt_frames.append(seg)
                    have_route.add(str(key))

    missing_tasks = tasks_df[~tasks_df["task_key"].isin(have_route)]
    if len(missing_tasks) and (db / "site_chips.parquet").exists():
        sc = pd.read_parquet(db / "site_chips.parquet", columns=["site_id", "chip_id", "in_area"])
        area = {
            s_: g.loc[g["in_area"].fillna(False).astype(bool), "chip_id"].to_numpy(np.int64)
            for s_, g in sc.groupby("site_id")
        }
        fb_rows = []
        for r in missing_tasks.itertuples(index=False):
            chips = area.get(r.site_id, np.array([], dtype=np.int64))
            if not len(chips):
                continue
            la, lo = grid.chip_centre(chips)
            cids = grid.cell_id(la, lo)
            ss_km = float(r.ss_km) if pd.notna(getattr(r, "ss_km", np.nan)) and float(r.ss_km) > 0 else 50.0
            seg_km = ss_km / len(chips)
            for idx_c, cid in enumerate(chips):
                fb_rows.append({
                    "task_key": r.task_key,
                    "seg_idx": idx_c,
                    "leg_idx": 0,
                    "n_legs": 0,
                    "lat": float(la[idx_c]),
                    "lon": float(lo[idx_c]),
                    "chip_id": int(cid),
                    "cell_id": int(cids[idx_c]),
                    "seg_km": seg_km,
                    "cum_km": 0.5 * ss_km,
                    "frac_route": 0.5,
                    "leg_bearing_rad": np.nan,
                    "has_route": False,
                })
        if fb_rows:
            rt_frames.append(pd.DataFrame(fb_rows))

    if not rt_frames:
        return pd.DataFrame(columns=[
            "task_key", "seg_idx", "leg_idx", "n_legs", "lat", "lon",
            "chip_id", "cell_id", "seg_km", "cum_km", "frac_route", "leg_bearing_rad", "has_route",
        ])
    return pd.concat(rt_frames, ignore_index=True)


def _hhmm(s: Any) -> float:
    try:
        parts = str(s).split(":")
        h, m = int(parts[0]), int(parts[1])
        sec = int(parts[2]) if len(parts) > 2 else 0
        return h + m / 60.0 + sec / 3600.0
    except (ValueError, TypeError, IndexError):
        return float("nan")


def task_windows(t: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """UTC start and end timestamps of each task's flying window (matching task_model.windows)."""
    off = t["utc_offset_h"].astype(float).where(t["utc_offset_h"].notna(), (t["lon"] / 15.0).round())
    start_l = t["start_time"].map(_hhmm)
    end_l = t["deadline"].map(_hhmm)
    solar = t["lon"] / 15.0 - off
    start_l = start_l.where(start_l.notna(), DEFAULT_WINDOW[0] + solar)
    end_l = end_l.where(end_l.notna() & (end_l > start_l), DEFAULT_WINDOW[1] + solar)
    end_l = np.minimum(end_l, start_l + MAX_WINDOW_H)
    day = pd.to_datetime(t["date"])
    start = day + pd.to_timedelta(start_l - off, unit="h")
    end = day + pd.to_timedelta(end_l - off, unit="h")
    return start, end


# ── Loading Weather & Terrain Tables ─────────────────────────────────────────

def _read_parquet_filtered(
    path: Path,
    cols: list[str],
    chips: set[int] | None = None,
    cells: set[int] | None = None,
    min_time: pd.Timestamp | None = None,
    max_time: pd.Timestamp | None = None,
) -> pd.DataFrame | None:
    if not path.exists():
        return None
    import pyarrow.parquet as pq

    have = set(pq.ParquetFile(path).schema.names)
    use = [c for c in cols if c in have]
    if not use:
        return None
    d = pd.read_parquet(path, columns=use)
    if chips is not None and "chip_id" in d:
        d = d[d["chip_id"].isin(chips)]
    if cells is not None and "cell_id" in d:
        d = d[d["cell_id"].isin(cells)]
    if "time" in d:
        d["time"] = pd.to_datetime(d["time"]).astype("datetime64[ns]")
        if min_time is not None:
            d = d[d["time"] >= min_time]
        if max_time is not None:
            d = d[d["time"] <= max_time]
    return d.reset_index(drop=True)


def load_era5_hourly_tables(
    db: Path,
    chips: set[int] | None = None,
    cells_needed: set[int] | None = None,
    include_land: bool = True,
    land_chips: set[int] | None = None,
    min_time: pd.Timestamp | None = None,
    max_time: pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame | None]:
    """Load ERA5 hourly tables (single, levels, extra, static, optional land) + 1 km terrain cells.

    `land_chips` allows filtering the large 0.1-deg `era5_land` table strictly to route chips while
    keeping `single/levels/extra/static` broader (e.g. for thermal sub-models across all site chips).
    """
    import pyarrow.parquet as pq

    w = db / "weather"
    single = _read_parquet_filtered(w / "era5_single.parquet", ["chip_id", "time", *SINGLE_COLS], chips, min_time=min_time, max_time=max_time)
    # Pad levels time bounds by +/- 3h so boundary interpolation never clips.
    lv_min = (min_time - pd.Timedelta(hours=3)) if min_time is not None else None
    lv_max = (max_time + pd.Timedelta(hours=3)) if max_time is not None else None
    levels = _read_parquet_filtered(w / "era5_levels.parquet", ["chip_id", "time", *LEVEL_COLS], chips, min_time=lv_min, max_time=lv_max)
    extra = _read_parquet_filtered(w / "era5_extra.parquet", ["chip_id", "time", *EXTRA_COLS], chips, min_time=min_time, max_time=max_time)
    static = _read_parquet_filtered(w / "era5_static.parquet", ["chip_id", "ground_m"], chips)
    l_chips = land_chips if land_chips is not None else chips
    land = (
        _read_parquet_filtered(w / "era5_land.parquet", ["chip_id", "time", "lat", "lon", *LAND_COLS], l_chips, min_time=min_time, max_time=max_time)
        if include_land else None
    )

    cells_path = db / "cells.parquet"
    cells_df = None
    elev_df = None
    if cells_path.exists():
        c_names = pq.ParquetFile(cells_path).schema.names
        sun_cols = [c for c in c_names if c.startswith("sun_")]
        c_want = ["cell_id", "chip_id", "lat", "lon"] + [c for c in TERRAIN_CELL_COLS if c in c_names] + sun_cols
        cells_all = _read_parquet_filtered(cells_path, ["cell_id", "chip_id", "elev_mean"], chips=chips)
        if cells_all is not None and len(cells_all):
            elev_df = cells_all.groupby("chip_id")["elev_mean"].mean().rename("route_elev").reset_index()
        cells_df = _read_parquet_filtered(cells_path, c_want, chips=chips, cells=cells_needed)

    return {
        "single": single,
        "levels": levels,
        "extra": extra,
        "static": static,
        "land": land,
        "cells": cells_df,
        "elev": elev_df,
    }


# ── Exact 1 h Levels Alignment (with Linear Fallback for Missing Hours) ──────

def align_levels_hourly(
    need_chip_time: pd.DataFrame,
    levels_df: pd.DataFrame | None,
    max_gap_h: float = MAX_LEVEL_INTERP_GAP_H,
) -> tuple[pd.DataFrame, float]:
    """Return hourly pressure-level columns aligned to `need_chip_time` (`chip_id`, `time`).

    Uses exact 1 h matches wherever available; linearly interpolates across gaps <= `max_gap_h`
    for any missing timestamp (e.g. legacy 3-hourly `era5_levels` or CDS edge gaps).
    Returns `(aligned_df, native_1h_share)`.
    """
    keys = need_chip_time[["chip_id", "time"]].drop_duplicates().reset_index(drop=True)
    keys["time"] = pd.to_datetime(keys["time"]).astype("datetime64[ns]")
    if levels_df is None or not len(levels_df):
        return keys, 0.0

    lv = levels_df.copy()
    if "time" not in lv and "time3" in lv:
        lv = lv.rename(columns={"time3": "time"})
    lv["time"] = pd.to_datetime(lv["time"]).astype("datetime64[ns]")
    val_cols = [c for c in lv.columns if c not in ("chip_id", "time")]
    if not val_cols:
        return keys, 0.0
    lv = lv.drop_duplicates(["chip_id", "time"], keep="last")

    merged = keys.merge(lv, on=["chip_id", "time"], how="left")
    probe = "u_700" if "u_700" in val_cols else ("t_700" if "t_700" in val_cols else val_cols[0])
    miss = merged[probe].isna().to_numpy()
    native_share = float(1.0 - miss.mean()) if len(merged) else 0.0
    if not miss.any():
        return merged, native_share

    # Linear interpolation strictly within each chip across gaps <= max_gap_h.
    miss_chips = set(merged.loc[miss, "chip_id"].unique())
    lv_sub = lv[lv["chip_id"].isin(miss_chips)].sort_values(["chip_id", "time"])
    if not len(lv_sub):
        return merged, native_share

    max_gap_s = float(max_gap_h) * 3600.0
    lv_groups = {int(cid): g for cid, g in lv_sub.groupby("chip_id", sort=False)}
    out_arrs = {c: merged[c].to_numpy(dtype=np.float64, copy=True) for c in val_cols}

    miss_idx_by_chip = merged.loc[miss].groupby("chip_id", sort=False).groups
    for cid_val, idxs in miss_idx_by_chip.items():
        src = lv_groups.get(int(cid_val))
        if src is None or len(src) < 2:
            continue
        pos = np.asarray(idxs, dtype=np.int64)
        t_src = src["time"].to_numpy("datetime64[s]").astype(np.int64).astype(np.float64)
        t_req = merged["time"].to_numpy("datetime64[s]")[pos].astype(np.int64).astype(np.float64)

        right = np.searchsorted(t_src, t_req)
        in_bounds = (right > 0) & (right < len(t_src))
        if not in_bounds.any():
            continue
        r_idx = right[in_bounds]
        l_idx = r_idx - 1
        dt = t_src[r_idx] - t_src[l_idx]
        ok_gap = (dt > 0.0) & (dt <= max_gap_s)
        if not ok_gap.any():
            continue

        sel_pos = pos[in_bounds][ok_gap]
        l_i = l_idx[ok_gap]
        r_i = r_idx[ok_gap]
        w = (t_req[in_bounds][ok_gap] - t_src[l_i]) / dt[ok_gap]
        for c in val_cols:
            vals = src[c].to_numpy(dtype=np.float64)
            vl, vr = vals[l_i], vals[r_i]
            both = np.isfinite(vl) & np.isfinite(vr)
            interp = np.where(both, vl + w * (vr - vl), np.nan)
            out_arrs[c][sel_pos] = np.where(np.isnan(out_arrs[c][sel_pos]), interp, out_arrs[c][sel_pos])

    for c in val_cols:
        merged[c] = out_arrs[c].astype(np.float32)
    return merged, native_share


# ── Vectorized Vertical Wind Interpolation ───────────────────────────────────

def interp_wind_to_height(
    df: pd.DataFrame,
    target_asl: np.ndarray,
    levels: tuple[int, ...] = PRESSURE_LEVELS,
) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized vertical interpolation of (u, v) from 10 m, 100 m and pressure levels to `target_asl`.

    Matches wind_check.py semantics: pressure levels with `z <= ground_m + 150 m` are skipped as
    extrapolated below model ground, at least 2 valid height nodes are required, and out-of-range
    altitudes clamp to the lowest/highest valid level.
    """
    n = len(df)
    target = np.asarray(target_asl, dtype=np.float64)
    if n == 0:
        return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

    g = df["ground_m"].to_numpy(dtype=np.float64) if "ground_m" in df else np.full(n, np.nan)
    h_list, u_list, v_list = [], [], []
    if "u10" in df and "v10" in df:
        h_list.append(g + 10.0)
        u_list.append(df["u10"].to_numpy(dtype=np.float64))
        v_list.append(df["v10"].to_numpy(dtype=np.float64))
    if "u100" in df and "v100" in df:
        h_list.append(g + 100.0)
        u_list.append(df["u100"].to_numpy(dtype=np.float64))
        v_list.append(df["v100"].to_numpy(dtype=np.float64))
    for p in levels:
        uk, vk, zk = f"u_{p}", f"v_{p}", f"z_{p}"
        if uk in df and vk in df and zk in df:
            z = df[zk].to_numpy(dtype=np.float64)
            z = np.where(z > g + 150.0, z, np.nan)
            h_list.append(z)
            u_list.append(df[uk].to_numpy(dtype=np.float64))
            v_list.append(df[vk].to_numpy(dtype=np.float64))

    if len(h_list) < 2:
        return np.full(n, np.nan, dtype=np.float64), np.full(n, np.nan, dtype=np.float64)

    H = np.column_stack(h_list)
    U = np.column_stack(u_list)
    V = np.column_stack(v_list)
    valid = np.isfinite(H) & np.isfinite(U) & np.isfinite(V)
    row_ok = (valid.sum(axis=1) >= 2) & np.isfinite(target)

    # Highest valid node <= target, and lowest valid node >= target.
    below = valid & (H <= target[:, None])
    above = valid & (H >= target[:, None])
    has_below = below.any(axis=1)
    has_above = above.any(axis=1)

    h_below_score = np.where(below, H, -np.inf)
    h_above_score = np.where(above, H, np.inf)
    idx_lo = np.argmax(h_below_score, axis=1)
    idx_hi = np.argmin(h_above_score, axis=1)

    rows = np.arange(n)
    h_lo = H[rows, idx_lo]
    u_lo = U[rows, idx_lo]
    v_lo = V[rows, idx_lo]
    h_hi = H[rows, idx_hi]
    u_hi = U[rows, idx_hi]
    v_hi = V[rows, idx_hi]

    mu = np.full(n, np.nan, dtype=np.float64)
    mv = np.full(n, np.nan, dtype=np.float64)

    both = row_ok & has_below & has_above
    if both.any():
        dh = h_hi[both] - h_lo[both]
        w = np.where(dh > 1e-6, (target[both] - h_lo[both]) / np.where(dh > 1e-6, dh, 1.0), 0.0)
        mu[both] = u_lo[both] + w * (u_hi[both] - u_lo[both])
        mv[both] = v_lo[both] + w * (v_hi[both] - v_lo[both])

    only_lo = row_ok & has_below & ~has_above
    if only_lo.any():
        mu[only_lo] = u_lo[only_lo]
        mv[only_lo] = v_lo[only_lo]

    only_hi = row_ok & ~has_below & has_above
    if only_hi.any():
        mu[only_hi] = u_hi[only_hi]
        mv[only_hi] = v_hi[only_hi]

    return mu, mv


# ── MacCready XC Route Kinematics ────────────────────────────────────────────

def maccready_air_xc_kmh(climb_ms: np.ndarray, glider_class: str = "CCC") -> np.ndarray:
    """Optimal still-air XC speed (km/h) for climb rate `climb_ms` (m/s) and glider class polar."""
    v0_k, s0, v1_k, s1 = GLIDER_POLARS.get(glider_class, GLIDER_POLARS["CCC"])
    wc = np.asarray(climb_ms, dtype=np.float64)
    v_grid = np.linspace(v0_k / 3.6, v1_k / 3.6, 11)
    frac = (v_grid - v0_k / 3.6) / max((v1_k - v0_k) / 3.6, 1e-6)
    s_grid = s0 + (s1 - s0) * (frac ** 2)
    wc_pos = np.maximum(wc[:, None], 0.05)
    v_air_xc = v_grid[None, :] * (wc_pos / (wc_pos + s_grid[None, :]))
    best_ms = np.max(v_air_xc, axis=1)
    return np.where(np.isfinite(wc) & (wc > 0.05), best_ms * 3.6, np.nan)


def maccready_segment_groundspeed_kmh(
    climb_ms: np.ndarray,
    u_tail_ms: np.ndarray,
    u_cross_ms: np.ndarray,
    glider_class: str = "CCC",
) -> tuple[np.ndarray, np.ndarray]:
    """Effective track groundspeed (km/h) and unflyable flag per segment.

    Because both climbing and gliding occur in the moving airmass, the airmass XC speed combines
    vectorially with the flight-altitude wind along the leg bearing.
    """
    v_air_kmh = maccready_air_xc_kmh(climb_ms, glider_class=glider_class)
    ut = np.nan_to_num(np.asarray(u_tail_ms, dtype=np.float64), nan=0.0) * 3.6
    uc = np.abs(np.nan_to_num(np.asarray(u_cross_ms, dtype=np.float64), nan=0.0)) * 3.6
    v_eff_sq = v_air_kmh ** 2 - uc ** 2
    can_crab = np.isfinite(v_air_kmh) & (v_eff_sq > 0.0)
    v_track = np.where(can_crab, np.sqrt(np.maximum(v_eff_sq, 0.0)) + ut, 0.0)
    unflyable = (~can_crab) | (v_track < 6.0)
    return np.where(np.isfinite(v_air_kmh), np.maximum(v_track, 4.0), np.nan), unflyable


def maccready_task_summary(
    seg_df: pd.DataFrame,
    climb_ms: np.ndarray,
    u_tail_ms: np.ndarray,
    u_cross_ms: np.ndarray,
    usable_m: np.ndarray,
    task_window_h: dict[str, float],
    glider_class: str = "CCC",
    prefix: str = "mc",
) -> pd.DataFrame:
    """Aggregate segment-level MacCready groundspeeds into task-level time, speed, and margin."""
    v_kmh, unfly = maccready_segment_groundspeed_kmh(climb_ms, u_tail_ms, u_cross_ms, glider_class=glider_class)
    low_air = np.isfinite(usable_m) & (usable_m < 200.0)
    unfly = unfly | low_air

    tmp = pd.DataFrame({
        "task_key": seg_df["task_key"].to_numpy(),
        "seg_km": seg_df["seg_km"].to_numpy(dtype=np.float64),
        "seg_h": seg_df["seg_km"].to_numpy(dtype=np.float64) / np.where(np.isfinite(v_kmh), v_kmh, np.nan),
        "unfly_km": np.where(unfly, seg_df["seg_km"].to_numpy(dtype=np.float64), 0.0),
    })
    by = tmp.groupby("task_key", sort=False)
    tot_km = by["seg_km"].sum()
    tot_h = by["seg_h"].sum(min_count=1)
    unfly_share = by["unfly_km"].sum() / np.maximum(tot_km, 1e-6)
    win_h = pd.Series({k: task_window_h.get(k, 3.0) for k in tot_km.index}, dtype=np.float64)
    out = pd.DataFrame({
        f"{prefix}_speed_kmh": tot_km / np.maximum(tot_h, 1e-6),
        f"{prefix}_time_h": tot_h,
        f"{prefix}_margin_h": win_h - tot_h,
        f"{prefix}_unflyable_share": unfly_share,
    }, index=tot_km.index)
    out.loc[tot_h.isna(), [f"{prefix}_speed_kmh", f"{prefix}_margin_h"]] = np.nan
    return out


# ── ERA5-Land 0.1-deg Nearest-Point Join ─────────────────────────────────────

def _attach_land_nearest(df: pd.DataFrame, land_df: pd.DataFrame | None) -> pd.DataFrame:
    """Join 0.1-deg ERA5-Land values to `(lat, lon, time)` via rounded 0.1-deg grid coordinates."""
    if land_df is None or not len(land_df):
        return df
    l_cols = [c for c in LAND_COLS if c in land_df]
    if not l_cols:
        return df
    ld = land_df[["lat", "lon", "time", *l_cols]].copy()
    ld["lat1"] = np.rint(ld["lat"].to_numpy(dtype=np.float64) * 10.0).astype(np.int32)
    ld["lon1"] = np.rint(ld["lon"].to_numpy(dtype=np.float64) * 10.0).astype(np.int32)
    ld["time"] = pd.to_datetime(ld["time"]).astype("datetime64[ns]")
    ld = ld.drop_duplicates(["lat1", "lon1", "time"], keep="last")
    ren = {c: f"land_{c}" for c in l_cols}
    ld = ld[["lat1", "lon1", "time", *l_cols]].rename(columns=ren)

    out = df.copy()
    out["lat1"] = np.rint(out["lat"].to_numpy(dtype=np.float64) * 10.0).astype(np.int32)
    out["lon1"] = np.rint(out["lon"].to_numpy(dtype=np.float64) * 10.0).astype(np.int32)
    out = out.merge(ld, on=["lat1", "lon1", "time"], how="left").drop(columns=["lat1", "lon1"])
    return out


# ── Core Hourly Space-Time Route Feature Extraction ──────────────────────────

def _enrich_chip_hours(
    chip_times: pd.DataFrame,
    tables: dict[str, pd.DataFrame | None],
) -> tuple[pd.DataFrame, float]:
    """Attach single, 1 h aligned levels, extra, static, and derived columns to unique `(chip_id, time)`."""
    ct = chip_times[["chip_id", "time"]].drop_duplicates().reset_index(drop=True)
    ct["time"] = pd.to_datetime(ct["time"]).astype("datetime64[ns]")

    s = tables.get("single")
    if s is not None and len(s):
        scols = ["chip_id", "time"] + [c for c in SINGLE_COLS if c in s]
        s2 = s[scols].copy()
        s2["time"] = pd.to_datetime(s2["time"]).astype("datetime64[ns]")
        ct = ct.merge(s2.drop_duplicates(["chip_id", "time"], keep="last"), on=["chip_id", "time"], how="left")

    lv_aligned, native_1h_share = align_levels_hourly(ct, tables.get("levels"))
    lv_val_cols = [c for c in lv_aligned.columns if c not in ("chip_id", "time")]
    if lv_val_cols:
        ct = ct.merge(lv_aligned, on=["chip_id", "time"], how="left")

    ex = tables.get("extra")
    if ex is not None and len(ex):
        ecols = ["chip_id", "time"] + [c for c in EXTRA_COLS if c in ex]
        e2 = ex[ecols].copy()
        e2["time"] = pd.to_datetime(e2["time"]).astype("datetime64[ns]")
        ct = ct.merge(e2.drop_duplicates(["chip_id", "time"], keep="last"), on=["chip_id", "time"], how="left")

    st = tables.get("static")
    if st is not None and len(st) and "ground_m" in st:
        ct = ct.merge(st[["chip_id", "ground_m"]].drop_duplicates("chip_id"), on="chip_id", how="left")
    ev = tables.get("elev")
    if ev is not None and len(ev) and "route_elev" in ev:
        ct = ct.merge(ev[["chip_id", "route_elev"]].drop_duplicates("chip_id"), on="chip_id", how="left")

    if "ground_m" not in ct:
        ct["ground_m"] = ct["route_elev"] if "route_elev" in ct else 0.0
    elif "route_elev" in ct:
        ct["ground_m"] = ct["ground_m"].fillna(ct["route_elev"])
    if "route_elev" not in ct:
        ct["route_elev"] = ct["ground_m"]
    else:
        ct["route_elev"] = ct["route_elev"].fillna(ct["ground_m"])

    g = lambda c: ct[c].to_numpy(dtype=np.float64) if c in ct else np.full(len(ct), np.nan)  # noqa: E731
    ground = g("ground_m")
    relev = g("route_elev")
    blh = g("blh_m")
    cb = g("cloud_base_m")
    blh_asl = ground + blh
    cb_asl = ground + cb
    eff_top_asl = np.where(np.isfinite(cb_asl), np.minimum(blh_asl, cb_asl), blh_asl)

    ct["eff_top_asl"] = eff_top_asl
    ct["usable_m"] = eff_top_asl - relev
    ct["cloud_base_above_m"] = cb_asl - relev
    ct["spread_c"] = g("t2m_c") - g("dewpoint_c")
    ct["skin_minus_air_c"] = g("skin_temp_c") - g("t2m_c")
    ct["wind10"] = np.hypot(g("u10"), g("v10"))
    ct["wind100"] = np.hypot(g("u100"), g("v100"))

    dz_850_700 = (g("z_700") - g("z_850")) / 1000.0
    ct["lapse_850_700"] = np.where(dz_850_700 > 0.2, (g("t_850") - g("t_700")) / np.where(dz_850_700 > 0.2, dz_850_700, 1.0), np.nan)
    dz_700_600 = (g("z_600") - g("z_700")) / 1000.0
    ct["lapse_700_600"] = np.where(dz_700_600 > 0.2, (g("t_700") - g("t_600")) / np.where(dz_700_600 > 0.2, dz_700_600, 1.0), np.nan)
    for col in ("rh_850", "rh_700"):
        if col in ct:
            ct[col] = ct[col].clip(0.0, 100.0)
        else:
            ct[col] = np.nan
    for col in SINGLE_COLS + EXTRA_COLS:
        if col not in ct:
            ct[col] = np.nan

    # Radiation-based convective velocity proxy (available in both ERA5 and Open-Meteo).
    rad = g("sunshine_wm2")
    cld = g("cloud_total")
    ct["wstar_rad"] = (
        np.cbrt(np.maximum(blh, 0.0) * np.maximum(rad, 0.0) * np.clip(1.0 - 0.5 * np.nan_to_num(cld, nan=0.0), 0.1, 1.0))
        / 100.0
    )
    shf = g("sensible_heat_wm2")
    ct["wstar_shf"] = np.where(np.isfinite(shf) & np.isfinite(blh), np.cbrt(np.maximum(blh, 0.0) * np.maximum(shf, 0.0) * 0.00003), np.nan)

    # Expected flight altitude wind (60% of effective usable height above ground, >= 200 m AGL).
    h_flt_asl = np.clip(relev + 0.60 * np.maximum(ct["usable_m"].to_numpy(dtype=np.float64), 0.0), relev + 200.0, np.maximum(eff_top_asl, relev + 200.0))
    u_flt, v_flt = interp_wind_to_height(ct, h_flt_asl)
    # Graceful fallback to 100 m wind if pressure levels are entirely absent for a row.
    u_flt = np.where(np.isfinite(u_flt), u_flt, g("u100"))
    v_flt = np.where(np.isfinite(v_flt), v_flt, g("v100"))
    ct["h_flt_asl"] = h_flt_asl
    ct["u_flt"] = u_flt
    ct["v_flt"] = v_flt
    ct["wind_flt"] = np.hypot(u_flt, v_flt)
    ct["shear_sfc_flt"] = np.hypot(u_flt - g("u10"), v_flt - g("v10"))
    return ct, native_1h_share


def extract_hourly_route_features(
    tasks_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    tables: dict[str, pd.DataFrame | None],
    include_terrain: bool = True,
    include_land: bool = True,
    return_panels: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build hourly (1 h) along-route feature table for all tasks in `tasks_df`.

    Combines:
      1. Fixed relative hourly slots (h0..h4) across the task window + diurnal trends & bottlenecks.
      2. Space-time ETA-coupled along-route legs (leader 28 km/h & pack 18 km/h) with flight-altitude
         wind projected onto each segment's exact bearing at its estimated hour of arrival.
      3. Optional 1 km route terrain + hourly sun-on-slope + cell-level clearance (pass bottlenecks).
      4. Optional 0.1-deg ERA5-Land surface fluxes matched to each 5 km route segment & hour.

    Returns `(features_df indexed by task_key, diagnostics_dict)`.
    When `return_panels=True`, `diagnostics_dict` also contains `"lead_panel"` and `"pack_panel"`
    (segment-level DataFrames ready for in-fold thermal sub-model scoring).
    """
    start_s, end_s = task_windows(tasks_df)
    win_df = pd.DataFrame({
        "task_key": tasks_df["task_key"].to_numpy(),
        "start_utc": pd.to_datetime(start_s).astype("datetime64[ns]"),
        "end_utc": pd.to_datetime(end_s).astype("datetime64[ns]"),
    })
    win_df["start_h"] = win_df["start_utc"].dt.floor("h")
    win_df["end_h"] = win_df["end_utc"].dt.floor("h")
    win_df["window_h"] = ((win_df["end_utc"] - win_df["start_utc"]).dt.total_seconds() / 3600.0).clip(0.5, MAX_WINDOW_H)
    win_map = dict(zip(win_df["task_key"], win_df["window_h"]))

    seg = segments_df.merge(win_df, on="task_key", how="inner")
    if not len(seg):
        empty = pd.DataFrame(index=pd.Index(tasks_df["task_key"], name="task_key"))
        return empty, {"levels_native_1h_share": 0.0}

    # Build relative hour grid (task_key, chip_id, rel_h, time) for Vista A (h0..h4).
    task_chips = seg[["task_key", "chip_id", "start_h", "end_h"]].drop_duplicates(["task_key", "chip_id"])
    # Also compute mean leg bearing weights per (task_key, chip_id) for slot tailwind.
    has_brg = seg["leg_bearing_rad"].notna()
    chip_brg = (
        seg[has_brg].assign(
            sb=np.sin(seg.loc[has_brg, "leg_bearing_rad"]) * seg.loc[has_brg, "seg_km"],
            cb=np.cos(seg.loc[has_brg, "leg_bearing_rad"]) * seg.loc[has_brg, "seg_km"],
        )
        .groupby(["task_key", "chip_id"], sort=False)
        .agg(sb=("sb", "sum"), cb=("cb", "sum"), km=("seg_km", "sum"))
        .reset_index()
    )
    chip_brg["sin_b"] = chip_brg["sb"] / np.maximum(chip_brg["km"], 1e-6)
    chip_brg["cos_b"] = chip_brg["cb"] / np.maximum(chip_brg["km"], 1e-6)
    task_chips = task_chips.merge(chip_brg[["task_key", "chip_id", "sin_b", "cos_b"]], on=["task_key", "chip_id"], how="left")

    slot_frames = []
    for k in SLOT_HOURS:
        sub = task_chips.copy()
        sub["rel_h"] = np.int8(k)
        sub["time"] = sub["start_h"] + pd.Timedelta(hours=k)
        sub = sub[sub["time"] <= sub["end_h"]]
        if len(sub):
            slot_frames.append(sub)
    slot_grid = pd.concat(slot_frames, ignore_index=True)

    # Compute leader and pack ETA timestamps per 5 km segment (clamped to [start_utc, end_utc]).
    eta_lead_h = np.minimum(seg["cum_km"].to_numpy(dtype=np.float64) / LEADER_KMH, seg["window_h"].to_numpy(dtype=np.float64))
    eta_pack_h = np.minimum(seg["cum_km"].to_numpy(dtype=np.float64) / PACK_KMH, seg["window_h"].to_numpy(dtype=np.float64))
    seg["time_lead"] = (seg["start_utc"] + pd.to_timedelta(eta_lead_h, unit="h")).dt.round("h").astype("datetime64[ns]")
    seg["time_pack"] = (seg["start_utc"] + pd.to_timedelta(eta_pack_h, unit="h")).dt.round("h").astype("datetime64[ns]")
    seg["time_lead"] = np.minimum(np.maximum(seg["time_lead"], seg["start_h"]), seg["end_h"])
    seg["time_pack"] = np.minimum(np.maximum(seg["time_pack"], seg["start_h"]), seg["end_h"])

    # Union of all unique (chip_id, time) needed across slots + leader + pack.
    all_ct = pd.concat([
        slot_grid[["chip_id", "time"]],
        seg[["chip_id", "time_lead"]].rename(columns={"time_lead": "time"}),
        seg[["chip_id", "time_pack"]].rename(columns={"time_pack": "time"}),
    ], ignore_index=True).drop_duplicates()

    wx_ct, native_1h_share = _enrich_chip_hours(all_ct, tables)

    # ── Vista A: Hourly slots (h0..h4) + Diurnal Dynamics ────────────────────
    sg = slot_grid.merge(wx_ct, on=["chip_id", "time"], how="left")
    sg["tail_flt"] = sg["u_flt"] * sg["sin_b"] + sg["v_flt"] * sg["cos_b"]

    # Per (task_key, rel_h) route-spatial summaries.
    hr = sg.groupby(["task_key", "rel_h"], sort=False).agg(
        blh=("blh_m", "mean"),
        usable=("usable_m", "mean"),
        usable_min=("usable_m", "min"),
        cb_above=("cloud_base_above_m", "mean"),
        sun=("sunshine_wm2", "mean"),
        cloud=("cloud_total", "mean"),
        cloud_low=("cloud_low", "mean"),
        rain=("rain_mmh", "mean"),
        cape=("cape", "mean"),
        cin=("cin", "mean"),
        spread=("spread_c", "mean"),
        t2m=("t2m_c", "mean"),
        lapse_850_700=("lapse_850_700", "mean"),
        lapse_700_600=("lapse_700_600", "mean"),
        rh_850=("rh_850", "mean"),
        rh_700=("rh_700", "mean"),
        skin_minus_air=("skin_minus_air_c", "mean"),
        soil_moisture=("soil_moisture", "mean"),
        freezing_level=("freezing_level_m", "mean"),
        wstar_rad=("wstar_rad", "mean"),
        wstar_shf=("wstar_shf", "mean"),
        sensible_heat=("sensible_heat_wm2", "mean"),
        wind100=("wind100", "mean"),
        gust10=("gust10", "max"),
        wind_flt=("wind_flt", "mean"),
        tail_flt=("tail_flt", "mean"),
        shear_sfc_flt=("shear_sfc_flt", "mean"),
    ).reset_index()

    task_index = pd.Index(tasks_df["task_key"].unique(), name="task_key")
    cols: dict[str, Any] = {}

    def _re(s: Any) -> Any:
        return s.reindex(task_index) if isinstance(s, pd.Series) else s

    # Pivot fixed hourly slots h0..h4.
    for v in SLOT_VARS:
        piv = hr.pivot(index="task_key", columns="rel_h", values=v)
        for k in SLOT_HOURS:
            cols[f"{v}_h{k}"] = _re(piv[k]) if k in piv.columns else np.nan

    # Diurnal dynamics & bottlenecks across the hourly series.
    hr_sorted = hr.sort_values(["task_key", "rel_h"])
    by_hr = hr_sorted.groupby("task_key", sort=False)
    cols["blh_mean"] = _re(by_hr["blh"].mean())
    cols["blh_max"] = _re(by_hr["blh"].max())
    cols["blh_min"] = _re(by_hr["blh"].min())
    blh_last = _re(by_hr["blh"].last())
    cols["blh_growth_0_2"] = cols["blh_h2"] - cols["blh_h0"] if isinstance(cols["blh_h2"], pd.Series) and isinstance(cols["blh_h0"], pd.Series) else np.nan
    cols["blh_decay_late"] = blh_last - cols["blh_max"]

    cols["usable_mean"] = _re(by_hr["usable"].mean())
    cols["usable_min_h"] = _re(by_hr["usable_min"].min())
    cols["usable_p10_h"] = _re(by_hr["usable"].quantile(0.10))
    cols["hours_usable_gt_500"] = _re(hr_sorted.assign(ok=(hr_sorted["usable"] > 500.0).astype(float)).groupby("task_key")["ok"].sum())
    cols["hours_usable_gt_800"] = _re(hr_sorted.assign(ok=(hr_sorted["usable"] > 800.0).astype(float)).groupby("task_key")["ok"].sum())

    cols["cloud_base_above_mean"] = _re(by_hr["cb_above"].mean())
    cols["cloud_base_above_min"] = _re(by_hr["cb_above"].min())
    cols["sunshine_mean"] = _re(by_hr["sun"].mean())
    cols["sunshine_min_h"] = _re(by_hr["sun"].min())
    cols["cloud_total_mean"] = _re(by_hr["cloud"].mean())
    cols["cloud_low_mean"] = _re(by_hr["cloud_low"].mean())
    cols["cloud_buildup"] = _re(by_hr["cloud"].last()) - _re(by_hr["cloud"].first())

    cols["rain_total_mm"] = _re(by_hr["rain"].sum(min_count=1))
    cols["rain_max_mmh"] = _re(by_hr["rain"].max())
    cols["rain_hours"] = _re(hr_sorted.assign(wet=(hr_sorted["rain"] > 0.2).astype(float)).groupby("task_key")["wet"].sum())
    wet_rows = hr_sorted[hr_sorted["rain"] > 0.2]
    cols["rain_onset_rel_h"] = _re(wet_rows.groupby("task_key")["rel_h"].min()) if len(wet_rows) else np.nan

    # Rain the day before over the same chips & hours.
    s_tbl = tables.get("single")
    if s_tbl is not None and len(s_tbl) and "rain_mmh" in s_tbl:
        prev = slot_grid[["task_key", "chip_id", "time"]].copy()
        prev["time"] = prev["time"] - pd.Timedelta(days=1)
        sr = s_tbl[["chip_id", "time", "rain_mmh"]].copy()
        sr["time"] = pd.to_datetime(sr["time"]).astype("datetime64[ns]")
        rp = prev.merge(sr.drop_duplicates(["chip_id", "time"], keep="last"), on=["chip_id", "time"], how="left")
        cols["rain_day_before_mm"] = _re(rp.groupby("task_key")["rain_mmh"].sum(min_count=1))
    else:
        cols["rain_day_before_mm"] = np.nan

    cols["cape_mean"] = _re(by_hr["cape"].mean())
    cols["cape_max_h"] = _re(by_hr["cape"].max())
    cols["cin_mean"] = _re(by_hr["cin"].mean())
    cin_piv = hr.pivot(index="task_key", columns="rel_h", values="cin")
    cols["cin_h0"] = _re(cin_piv[0]) if 0 in cin_piv.columns else np.nan
    cols["spread_mean"] = _re(by_hr["spread"].mean())
    cols["t2m_mean"] = _re(by_hr["t2m"].mean())
    cols["lapse_850_700_mean"] = _re(by_hr["lapse_850_700"].mean())
    cols["lapse_850_700_max_h"] = _re(by_hr["lapse_850_700"].max())
    cols["lapse_700_600_mean"] = _re(by_hr["lapse_700_600"].mean())
    cols["rh_850_mean"] = _re(by_hr["rh_850"].mean())
    cols["rh_700_mean"] = _re(by_hr["rh_700"].mean())
    cols["rh_700_max_h"] = _re(by_hr["rh_700"].max())
    cols["skin_minus_air_mean"] = _re(by_hr["skin_minus_air"].mean())
    cols["soil_moisture_mean"] = _re(by_hr["soil_moisture"].mean())
    cols["freezing_level_mean"] = _re(by_hr["freezing_level"].mean())

    cols["wstar_rad_mean"] = _re(by_hr["wstar_rad"].mean())
    cols["wstar_rad_min_h"] = _re(by_hr["wstar_rad"].min())
    cols["wstar_rad_max_h"] = _re(by_hr["wstar_rad"].max())
    cols["wstar_rad_decay"] = _re(by_hr["wstar_rad"].last()) - cols["wstar_rad_max_h"]

    cols["wind100_mean"] = _re(by_hr["wind100"].mean())
    cols["gust10_max_h"] = _re(by_hr["gust10"].max())
    cols["wind_flt_mean"] = _re(by_hr["wind_flt"].mean())
    cols["wind_flt_max_h"] = _re(by_hr["wind_flt"].max())
    cols["wind_flt_pickup"] = _re(by_hr["wind_flt"].last()) - _re(by_hr["wind_flt"].first())
    cols["shear_sfc_flt_mean"] = _re(by_hr["shear_sfc_flt"].mean())

    cols["sensible_heat_mean"] = _re(by_hr["sensible_heat"].mean())
    cols["sensible_heat_min_h"] = _re(by_hr["sensible_heat"].min())
    cols["wstar_shf_mean"] = _re(by_hr["wstar_shf"].mean())
    cols["route_chips"] = _re(seg.groupby("task_key")["chip_id"].nunique())
    cols["route_n_legs"] = _re(seg.groupby("task_key")["n_legs"].max())
    cols["has_route"] = _re(seg.groupby("task_key")["has_route"].any())

    # ── Vista B: Space × Time ETA-Coupled Route Segments (Leader & Pack) ─────
    cells_df = tables.get("cells") if include_terrain else None
    sun_cols = [c for c in cells_df.columns if c.startswith("sun_")] if cells_df is not None else []

    def _build_eta_panel(time_col: str) -> pd.DataFrame:
        p = seg.merge(
            wx_ct, left_on=["chip_id", time_col], right_on=["chip_id", "time"], how="left",
        )
        b = p["leg_bearing_rad"].to_numpy(dtype=np.float64)
        uf = p["u_flt"].to_numpy(dtype=np.float64)
        vf = p["v_flt"].to_numpy(dtype=np.float64)
        p["seg_tail"] = uf * np.sin(b) + vf * np.cos(b)
        p["seg_cross"] = np.abs(-uf * np.cos(b) + vf * np.sin(b))

        if cells_df is not None and len(cells_df):
            c_keep = ["cell_id"] + [c for c in TERRAIN_CELL_COLS if c in cells_df] + sun_cols
            p = p.merge(cells_df[c_keep].drop_duplicates("cell_id"), on="cell_id", how="left")
            if "elev_mean" in p:
                p["cell_usable_m"] = p["eff_top_asl"] - p["elev_mean"]
            if "facing_east" in p and "facing_north" in p:
                p["seg_upslope"] = -(uf * p["facing_east"].to_numpy(dtype=np.float64) + vf * p["facing_north"].to_numpy(dtype=np.float64))
            if sun_cols and "flat_share" in p:
                t_arr = p["time"].to_numpy("datetime64[s]")
                la_arr = p["lat"].to_numpy(dtype=np.float64)
                lo_arr = p["lon"].to_numpy(dtype=np.float64)
                az, el = terrain.sun_position(la_arr, lo_arr, t_arr)
                p["sun_elev"] = el.astype(np.float32)
                p["solar_hour"] = (((t_arr.astype(np.int64) % 86400) / 3600.0 + lo_arr / 15.0) % 24.0).astype(np.float32)
                p["sun_direct"] = terrain.sun_on_cell(p, az, el).astype(np.float32)
                # Fast 2-hour trapezoid sun accumulation back 10 hours (6 steps instead of 13).
                energy = np.zeros(len(p), dtype=np.float64)
                steps = (0, 2, 4, 6, 8, 10)
                for k_h in steps:
                    a_k, e_k = terrain.sun_position(la_arr, lo_arr, t_arr - np.timedelta64(3600 * k_h, "s"))
                    w_k = 1.0 if k_h in (0, 10) else 2.0
                    energy += terrain.sun_on_cell(p, a_k, e_k) * w_k
                p["sun_energy"] = energy.astype(np.float32)
                p = p.drop(columns=sun_cols)
        else:
            t_arr = p["time"].to_numpy("datetime64[s]")
            lo_arr = p["lon"].to_numpy(dtype=np.float64)
            p["solar_hour"] = (((t_arr.astype(np.int64) % 86400) / 3600.0 + lo_arr / 15.0) % 24.0).astype(np.float32)

        if include_land and tables.get("land") is not None:
            p = _attach_land_nearest(p, tables.get("land"))
        return p

    lead_p = _build_eta_panel("time_lead")
    pack_p = _build_eta_panel("time_pack")

    def _wmean(df_p: pd.DataFrame, col: str) -> pd.Series:
        if col not in df_p:
            return pd.Series(np.nan, index=task_index)
        v = df_p[col].to_numpy(dtype=np.float64)
        w = np.where(np.isfinite(v), df_p["seg_km"].to_numpy(dtype=np.float64), 0.0)
        tmp = pd.DataFrame({"task_key": df_p["task_key"].to_numpy(), "vw": np.where(np.isfinite(v), v * w, 0.0), "w": w})
        g_t = tmp.groupby("task_key", sort=False).sum()
        return _re(g_t["vw"] / g_t["w"].where(g_t["w"] > 0.0))

    def _leg_wind_stats(df_p: pd.DataFrame, tag: str) -> None:
        cols[f"tail_{tag}_mean"] = _wmean(df_p, "seg_tail")
        if tag == "lead":
            cols["cross_lead_mean"] = _wmean(df_p, "seg_cross")
        rt_only = df_p[df_p["has_route"] & df_p["seg_tail"].notna()]
        if len(rt_only):
            leg_t = rt_only.groupby(["task_key", "leg_idx"], sort=False).apply(
                lambda x: float(np.average(x["seg_tail"], weights=np.maximum(x["seg_km"], 1e-6))),
                include_groups=False,
            )
            leg_df = leg_t.rename("leg_tail").reset_index()
            cols[f"tail_{tag}_min_leg"] = _re(leg_df.groupby("task_key")["leg_tail"].min())
            last_idx = leg_df.groupby("task_key")["leg_idx"].idxmax()
            cols[f"tail_{tag}_last_leg"] = _re(leg_df.loc[last_idx].set_index("task_key")["leg_tail"])
            hw = rt_only.assign(hw_km=np.where(rt_only["seg_tail"] < -1.0, rt_only["seg_km"], 0.0))
            g_hw = hw.groupby("task_key")[["hw_km", "seg_km"]].sum()
            cols[f"headwind_share_{tag}"] = _re(g_hw["hw_km"] / np.maximum(g_hw["seg_km"], 1e-6))
        else:
            cols[f"tail_{tag}_min_leg"] = np.nan
            cols[f"tail_{tag}_last_leg"] = np.nan
            cols[f"headwind_share_{tag}"] = np.nan

    _leg_wind_stats(lead_p, "lead")
    _leg_wind_stats(pack_p, "pack")

    cols["usable_lead_mean"] = _wmean(lead_p, "usable_m")
    cols["usable_lead_min"] = _re(lead_p.groupby("task_key")["usable_m"].min())
    cols["usable_pack_mean"] = _wmean(pack_p, "usable_m")
    cols["usable_pack_min"] = _re(pack_p.groupby("task_key")["usable_m"].min())

    lead_early = _re(lead_p[lead_p["frac_route"] <= 0.35].groupby("task_key")["usable_m"].mean())
    pack_late = _re(pack_p[pack_p["frac_route"] >= 0.65].groupby("task_key")["usable_m"].mean())
    cols["pack_late_collapse_m"] = pack_late - lead_early

    cols["rain_lead_max"] = _re(lead_p.groupby("task_key")["rain_mmh"].max())
    cols["rain_pack_max"] = _re(pack_p.groupby("task_key")["rain_mmh"].max())
    cols["cloud_pack_last_third"] = _re(pack_p[pack_p["frac_route"] >= 0.65].groupby("task_key")["cloud_total"].mean())

    cols["wstar_lead_mean"] = _wmean(lead_p, "wstar_rad")
    cols["wstar_pack_mean"] = _wmean(pack_p, "wstar_rad")
    cols["wstar_lead_min"] = _re(lead_p.groupby("task_key")["wstar_rad"].min())

    # Preliminary MacCready kinematics from wstar_rad (calibrated empirical scale ~0.85 * wstar_rad, clipped 0.3..5.0 m/s).
    climb_rad_lead = np.clip(0.85 * lead_p["wstar_rad"].to_numpy(dtype=np.float64), 0.3, 5.0)
    climb_rad_pack = np.clip(0.85 * pack_p["wstar_rad"].to_numpy(dtype=np.float64), 0.3, 5.0)
    mc_lead = maccready_task_summary(
        lead_p, climb_rad_lead, lead_p["seg_tail"].to_numpy(dtype=np.float64),
        lead_p["seg_cross"].to_numpy(dtype=np.float64), lead_p["usable_m"].to_numpy(dtype=np.float64),
        win_map, glider_class="CCC", prefix="mc_rad_lead",
    )
    mc_pack = maccready_task_summary(
        pack_p, climb_rad_pack, pack_p["seg_tail"].to_numpy(dtype=np.float64),
        pack_p["seg_cross"].to_numpy(dtype=np.float64), pack_p["usable_m"].to_numpy(dtype=np.float64),
        win_map, glider_class="EN-C", prefix="mc_rad_pack",
    )
    cols["mc_rad_speed_lead_kmh"] = _re(mc_lead["mc_rad_lead_speed_kmh"])
    cols["mc_rad_margin_lead_h"] = _re(mc_lead["mc_rad_lead_margin_h"])
    cols["mc_rad_margin_pack_h"] = _re(mc_pack["mc_rad_pack_margin_h"])
    cols["mc_rad_unflyable_share"] = _re(mc_lead["mc_rad_lead_unflyable_share"])

    # ── M3a: 1 km Terrain & Sun-on-Slope along route ─────────────────────────
    if include_terrain and "elev_mean" in lead_p:
        cols["route_elev_mean"] = _wmean(lead_p, "elev_mean")
        cols["route_elev_max"] = _re(lead_p.groupby("task_key")["elev_max"].max() if "elev_max" in lead_p else lead_p.groupby("task_key")["elev_mean"].max())
        cols["route_elev_p90"] = _re(lead_p.groupby("task_key")["elev_mean"].quantile(0.90))
        cols["route_elev_std"] = _re(lead_p.groupby("task_key")["elev_mean"].std())
        cols["route_relief_5km_mean"] = _wmean(lead_p, "relief_5km")
        cols["route_relief_5km_max"] = _re(lead_p.groupby("task_key")["relief_5km"].max()) if "relief_5km" in lead_p else np.nan
        cols["route_tpi_5km_mean"] = _wmean(lead_p, "tpi_5km")
        cols["route_tpi_20km_mean"] = _wmean(lead_p, "tpi_20km")
        cols["route_slope_mean"] = _wmean(lead_p, "slope_mean")
        cols["route_ridge_share"] = _wmean(lead_p, "ridge_share")
        cols["route_valley_share"] = _wmean(lead_p, "valley_share")
        cols["route_sun_direct_lead"] = _wmean(lead_p, "sun_direct")
        cols["route_sun_energy_lead"] = _wmean(lead_p, "sun_energy")
        cols["route_sun_direct_pack"] = _wmean(pack_p, "sun_direct")
        cols["route_upslope_flt_lead"] = _wmean(lead_p, "seg_upslope")
        if "seg_upslope" in lead_p:
            lee = lead_p.assign(lee_km=np.where((lead_p["seg_upslope"] < -1.0) & (lead_p["wind_flt"] >= 4.0), lead_p["seg_km"], 0.0))
            g_lee = lee.groupby("task_key")[["lee_km", "seg_km"]].sum()
            cols["route_lee_share_lead"] = _re(g_lee["lee_km"] / np.maximum(g_lee["seg_km"], 1e-6))
        else:
            cols["route_lee_share_lead"] = np.nan
        cols["cell_usable_min_lead"] = _re(lead_p.groupby("task_key")["cell_usable_m"].min()) if "cell_usable_m" in lead_p else np.nan
        cols["cell_usable_p10_lead"] = _re(lead_p.groupby("task_key")["cell_usable_m"].quantile(0.10)) if "cell_usable_m" in lead_p else np.nan
        cols["cell_usable_min_pack"] = _re(pack_p.groupby("task_key")["cell_usable_m"].min()) if "cell_usable_m" in pack_p else np.nan

    # ── M3b: 0.1-deg ERA5-Land Surface Features ──────────────────────────────
    if include_land and "land_sensible_heat_wm2" in lead_p:
        cols["land_sensible_heat_lead"] = _wmean(lead_p, "land_sensible_heat_wm2")
        cols["land_sensible_heat_min_h"] = _re(lead_p.groupby("task_key")["land_sensible_heat_wm2"].min())
        lh_early = _re(lead_p[lead_p["frac_route"] <= 0.35].groupby("task_key")["land_sensible_heat_wm2"].mean())
        lh_late = _re(pack_p[pack_p["frac_route"] >= 0.65].groupby("task_key")["land_sensible_heat_wm2"].mean())
        cols["land_sensible_heat_decay"] = lh_late - lh_early
        cols["land_sunshine_lead"] = _wmean(lead_p, "land_sunshine_wm2")
        if "land_skin_temp_c" in lead_p and "t2m_c" in lead_p:
            lead_p["land_skin_minus_air"] = lead_p["land_skin_temp_c"] - lead_p["t2m_c"]
            cols["land_skin_minus_air_lead"] = _wmean(lead_p, "land_skin_minus_air")
        cols["land_soil_moisture_lead"] = _wmean(lead_p, "land_soil_moisture")
        cols["land_albedo_mean"] = _wmean(lead_p, "land_albedo")
        cols["land_snow_max"] = _re(lead_p.groupby("task_key")["land_snow_cover"].max()) if "land_snow_cover" in lead_p else np.nan

    all_cols = HOURLY_CORE_FEATURES + TERRAIN_ROUTE_FEATURES + ERA5_LAND_FEATURES
    out = pd.DataFrame({c: cols.get(c, np.nan) for c in [*all_cols, "has_route"]}, index=task_index)

    diag: dict[str, Any] = {
        "levels_native_1h_share": round(native_1h_share, 4),
        "n_chip_hours": int(len(all_ct)),
        "task_window_h": win_map,
    }
    if return_panels:
        diag["lead_panel"] = lead_p
        diag["pack_panel"] = pack_p
    return out[all_cols + ["has_route"]], diag


# ── Hourly Chip-Level Quantile Mapping (Forecast -> ERA5) ────────────────────

def fit_quantile_map(
    src: np.ndarray,
    ref: np.ndarray,
    min_pairs: int = 30,
    probs: np.ndarray = QM_PROBS,
) -> dict[str, Any] | None:
    """Fit a monotone piecewise-linear quantile map `src -> ref` on paired finite samples."""
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
    """Apply a fitted monotone quantile map with constant boundary offset outside knots."""
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
    lo = x < ux[0]
    hi = x > ux[-1]
    if lo.any():
        mapped[lo] = x[lo] + float(qmap["shift_lo"])
    if hi.any():
        mapped[hi] = x[hi] + float(qmap["shift_hi"])
    out[ok] = mapped
    return out


def _norm_time_col(df: pd.DataFrame | None) -> pd.DataFrame | None:
    if df is None or not len(df):
        return None
    out = df.copy()
    if "time" not in out and "time3" in out:
        out = out.rename(columns={"time3": "time"})
    out["time"] = pd.to_datetime(out["time"]).astype("datetime64[ns]")
    return out


def fit_hourly_qmaps(
    fc_tables: dict[str, pd.DataFrame | None],
    era_tables: dict[str, pd.DataFrame | None],
    chip_times: pd.DataFrame,
    min_pairs: int = 50,
) -> dict[str, dict[str, Any]]:
    """Fit hourly chip-level Quantile Maps (Forecast -> ERA5) strictly on training `chip_times`.

    Scalars map directly; wind levels map speed magnitude while preserving direction so
    projected tailwind/crosswind vectors stay directionally consistent.
    """
    keys = chip_times[["chip_id", "time"]].drop_duplicates().copy()
    keys["time"] = pd.to_datetime(keys["time"]).astype("datetime64[ns]")
    qmaps: dict[str, dict[str, Any]] = {}

    def _fit_table(tname: str, scalar_cols: list[str], wind_specs: list[tuple[str, str, str]]) -> None:
        fc_t = _norm_time_col(fc_tables.get(tname))
        er_t = _norm_time_col(era_tables.get(tname))
        if fc_t is None or er_t is None:
            return
        if tname == "levels":
            er_t, _ = align_levels_hourly(keys, er_t)
        f_sub = keys.merge(fc_t.drop_duplicates(["chip_id", "time"], keep="last"), on=["chip_id", "time"], how="inner")
        e_sub = keys.merge(er_t.drop_duplicates(["chip_id", "time"], keep="last"), on=["chip_id", "time"], how="inner")
        m = f_sub.merge(e_sub, on=["chip_id", "time"], how="inner", suffixes=("_fc", "_era"))
        if len(m) < min_pairs:
            return
        for c in scalar_cols:
            cf, ce = f"{c}_fc", f"{c}_era"
            if cf in m and ce in m:
                qm = fit_quantile_map(m[cf].to_numpy(dtype=np.float64), m[ce].to_numpy(dtype=np.float64), min_pairs=min_pairs)
                if qm is not None:
                    qmaps[f"{tname}.{c}"] = qm
        for uc, vc, wname in wind_specs:
            if {f"{uc}_fc", f"{vc}_fc", f"{uc}_era", f"{vc}_era"} <= set(m.columns):
                sf = np.hypot(m[f"{uc}_fc"].to_numpy(dtype=np.float64), m[f"{vc}_fc"].to_numpy(dtype=np.float64))
                se = np.hypot(m[f"{uc}_era"].to_numpy(dtype=np.float64), m[f"{vc}_era"].to_numpy(dtype=np.float64))
                qm = fit_quantile_map(sf, se, min_pairs=min_pairs)
                if qm is not None:
                    qmaps[f"{tname}.{wname}"] = qm

    _fit_table("single", HOURLY_QM_THERMO_SINGLE, HOURLY_QM_WIND_SINGLE)
    _fit_table("extra", HOURLY_QM_THERMO_EXTRA, [])
    _fit_table("levels", HOURLY_QM_THERMO_LEVELS, HOURLY_QM_WIND_LEVELS)
    return qmaps


def apply_hourly_qmaps(
    fc_tables: dict[str, pd.DataFrame | None],
    qmaps: dict[str, dict[str, Any]],
    include_wind: bool = True,
) -> dict[str, pd.DataFrame | None]:
    """Return a copy of `fc_tables` with hourly chip-level Quantile Maps applied.

    Non-negative physical variables (`blh_m`, `cloud_base_m`, `sunshine_wm2`, `rain_mmh`, `cape`,
    `cin`, `gust10`, `freezing_level_m`, wind speeds) are clamped >= 0; `cloud_*` to [0, 1];
    `rh_*` to [0, 100].
    """
    out = dict(fc_tables)
    nonneg = {"blh_m", "cloud_base_m", "sunshine_wm2", "rain_mmh", "cape", "cin", "gust10", "soil_moisture", "freezing_level_m"}
    unit01 = {"cloud_total", "cloud_low"}
    pct100 = {"rh_850", "rh_700"}

    def _apply_table(tname: str, scalar_cols: list[str], wind_specs: list[tuple[str, str, str]]) -> None:
        df = out.get(tname)
        if df is None or not len(df):
            return
        d = df.copy()
        for c in scalar_cols:
            qm = qmaps.get(f"{tname}.{c}")
            if qm is not None and c in d:
                vals = apply_quantile_map(d[c].to_numpy(dtype=np.float64), qm)
                if c in nonneg:
                    vals = np.where(np.isfinite(vals), np.maximum(vals, 0.0), np.nan)
                elif c in unit01:
                    vals = np.where(np.isfinite(vals), np.clip(vals, 0.0, 1.0), np.nan)
                elif c in pct100:
                    vals = np.where(np.isfinite(vals), np.clip(vals, 0.0, 100.0), np.nan)
                d[c] = vals
        if include_wind:
            for uc, vc, wname in wind_specs:
                qm = qmaps.get(f"{tname}.{wname}")
                if qm is not None and uc in d and vc in d:
                    u = d[uc].to_numpy(dtype=np.float64)
                    v = d[vc].to_numpy(dtype=np.float64)
                    spd = np.hypot(u, v)
                    mapped_spd = np.where(np.isfinite(spd), np.maximum(apply_quantile_map(spd, qm), 0.0), np.nan)
                    scale = np.where(spd > 1e-6, mapped_spd / np.where(spd > 1e-6, spd, 1.0), 1.0)
                    d[uc] = np.where(np.isfinite(u) & np.isfinite(mapped_spd), u * scale, u)
                    d[vc] = np.where(np.isfinite(v) & np.isfinite(mapped_spd), v * scale, v)
        out[tname] = d

    _apply_table("single", HOURLY_QM_THERMO_SINGLE, HOURLY_QM_WIND_SINGLE)
    _apply_table("extra", HOURLY_QM_THERMO_EXTRA, [])
    _apply_table("levels", HOURLY_QM_THERMO_LEVELS, HOURLY_QM_WIND_LEVELS)
    return out
