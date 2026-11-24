#!/usr/bin/env python3
"""Build a high-precision IGC atmospheric ground-truth dataset from cleaned 5 s fixes.

Upgrades the competition IGC database into a calibrated physical sensor benchmark without
re-parsing raw IGC archives, plus an optional `--check-1hz N` audit that validates 5 s fixes
against raw 1 Hz tracks on a stratified sample of N flights.

Reads existing tables in DB (`flights`, `flight_checks`, `thermals`, `glides`, `tracks`, `tasks`,
`results`, `result_gliders`, `event_sites`, `sites`, `cells`, `fixes/*.parquet`) and optional GLO-30
DEM tiles + ERA5 weather tables, and writes DB/truth/:

    truth_flights.parquet       per-flight dual-DEM calibration check and quality tier (gold/silver)
    truth_thermals.parquet      1 row per thermal: calibrated ASL/AGL top & base, core60/core30/mean
                                climb rate, hodograph + integer-turn + old wind & QC flags
    truth_wind_groups.parquet   pooled clean wind by (site_id, chip_id, hour, band) with vertical
                                slicing, comparing old vs hodograph vs integer-turn vs clean vs ERA5
    truth_glide_wind.parquet    independent multi-heading glide wind controlled by glider class
    truth_ceiling.parquet       calibrated multi-pilot ASL top-of-lift per (site_id, chip_id, hour)
                                with airspace-cap flag, real thermal centroid, and ERA5 BLH/LCL
    truth_climb_groups.parquet  pooled core60/core30/mean climb rates and per-class glide metrics
    truth_daily.parquet         site-day summary (max ceiling ASL, core60 p75, free-air wind)
    truth_tasks.parquet         official task outcomes (winner_kmh, goal_share, stopped) joined with
                                IGC-measured route weather during the task window
    audit_1hz_vs_5s.json        quantitative 1 Hz raw (GPS + baro) vs 5 s fixes sensor audit
    summary.json                overall retention, bias/RMSE improvement vs ERA5, and diagnostics

Resumable in batches via DB/_truth_parts/ (safe to stop and rerun in Colab).

    python3 scripts/build_igc_truth.py /content/drive/MyDrive/IGCS/database \\
        --dem /content/drive/MyDrive/IGCS/dem/glo30 --local /content/dem_glo30 \\
        --workers 4 --check-1hz 300 --root /content/drive/MyDrive/IGCS
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from pta import flight as pta_flight  # noqa: E402
from pta import grid, igc, igc_truth, terrain  # noqa: E402

LEVELS = [925, 850, 700, 600, 500]
CALM_MS = 3.0
MIN_WIND_SLICES = 3
MIN_WIND_FLIGHTS = 2
MIN_CEIL_THERMALS = 5
MIN_CEIL_PILOTS = 3
ARCHIVE_FOLDERS = ("flymaster_igc", "airscore_igc", "airtribune_igc", "civlcomps_igc")


def safe(task_key: str) -> str:
    return task_key.replace(":", "_")


def relocate(path: str, root: Path | None) -> str:
    """Resolve archive path under `root` when Drive mount location differs from catalog machine."""
    if root is None or not path:
        return str(path)
    p = Path(path)
    for i, part in enumerate(p.parts):
        if part in ARCHIVE_FOLDERS:
            return str(root.joinpath(*p.parts[i:]))
    return str(path)


def height_band(agl: np.ndarray, above_crest: np.ndarray) -> np.ndarray:
    return np.where(
        agl < 400.0,
        "near the ground",
        np.where((agl >= 1000.0) | (above_crest > 0.0), "free air", "middle"),
    )


def _bearing_rad(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


def _hhmm(s: Any) -> float:
    try:
        parts = str(s).split(":")
        h, m = int(parts[0]), int(parts[1])
        sec = int(parts[2]) if len(parts) > 2 else 0
        return h + m / 60.0 + sec / 3600.0
    except (ValueError, TypeError, IndexError):
        return float("nan")


# ── Step 1: Dual-DEM altitude audit & flight quality tiers ───────────────────

def _copy_dem_tiles(dem_dir: Path, local_dir: Path, lats: np.ndarray, lons: np.ndarray) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    ok = np.isfinite(lats) & np.isfinite(lons)
    pairs = sorted({(math.floor(la), math.floor(lo)) for la, lo in zip(lats[ok], lons[ok])})
    todo = []
    for la, lo in pairs:
        name = f"{terrain.tile_name(la, lo)}.tif"
        src = dem_dir / name
        dst = local_dir / name
        if src.exists() and not (dst.exists() and dst.stat().st_size == src.stat().st_size):
            todo.append((src, dst))
    if todo:
        print(f"Copying {len(todo)} DEM tiles to {local_dir}...", flush=True)

        def _cp(pair: tuple[Path, Path]) -> None:
            src, dst = pair
            tmp = dst.with_suffix(".tif.part")
            shutil.copyfile(src, tmp)
            tmp.rename(dst)

        with ThreadPoolExecutor(16) as pool:
            list(pool.map(_cp, todo))
    return local_dir


def build_truth_flights(
    db: Path,
    dem_dir: Path | None,
    local_dem: Path | None,
    task_subset: set[str] | None = None,
) -> pd.DataFrame:
    """Build per-flight calibration audit (`gold` / `silver` / `excluded`) and UTC date anchors."""
    fl_cols = [
        "track_id", "task_key", "result_id", "pilot_key", "source", "error",
        "same_flight_as", "shared_path", "step_s", "quality",
        "t_takeoff", "lat_takeoff", "lon_takeoff", "alt_takeoff",
        "t_landing", "lat_landing", "lon_landing", "alt_landing",
        "alt_max", "duration_s",
    ]
    import pyarrow.parquet as pq

    fl_have = set(pq.ParquetFile(db / "flights.parquet").schema.names)
    fl = pd.read_parquet(db / "flights.parquet", columns=[c for c in fl_cols if c in fl_have])
    if task_subset is not None:
        fl = fl[fl["task_key"].isin(task_subset)].copy()

    ok = pd.Series(True, index=fl.index)
    if "error" in fl:
        ok &= fl["error"].isna()
    if "same_flight_as" in fl:
        ok &= fl["same_flight_as"].isna()
    if "shared_path" in fl:
        ok &= ~fl["shared_path"].fillna(False).astype(bool)
    fl = fl[ok].copy()

    fc = pd.read_parquet(
        db / "flight_checks.parquet",
        columns=["track_id", "agl_offset", "landing_agl_raw", "is_drive", "dem_coverage", "agl_median"],
    )
    fl = fl.merge(fc, on="track_id", how="left")
    fl = fl[~fl["is_drive"].fillna(False).astype(bool)].copy()

    # Attach site_id, event_key, and authoritative UTC date (prefer IGC header date if valid, else task date).
    tasks = pd.read_parquet(db / "tasks.parquet", columns=["task_key", "event_key", "date"])
    ev_sites = pd.read_parquet(db / "event_sites.parquet", columns=["event_key", "site_id"])
    tasks = tasks.merge(ev_sites, on="event_key", how="left")
    fl = fl.merge(tasks.rename(columns={"date": "task_date"}), on="task_key", how="left")

    if (db / "tracks.parquet").exists():
        tr_have = set(pq.ParquetFile(db / "tracks.parquet").schema.names)
        if "date" in tr_have:
            tr = pd.read_parquet(db / "tracks.parquet", columns=["track_id", "date"]).drop_duplicates("track_id")
            fl = fl.merge(tr.rename(columns={"date": "igc_date"}), on="track_id", how="left")
    if "igc_date" not in fl:
        fl["igc_date"] = None
    fl["flight_date"] = pd.to_datetime(fl["igc_date"], errors="coerce").fillna(
        pd.to_datetime(fl["task_date"], errors="coerce")
    )
    fl = fl.dropna(subset=["flight_date", "site_id"]).copy()

    # Single vectorized DEM sample at takeoff to cross-check altitude drift against landing calibration.
    if dem_dir is not None and dem_dir.exists():
        use_dem = _copy_dem_tiles(dem_dir, local_dem, fl["lat_takeoff"].to_numpy(float), fl["lon_takeoff"].to_numpy(float)) if local_dem else dem_dir
        dem = terrain.DEM(use_dem, keep=12)
        fl["dem_takeoff"] = dem.sample(fl["lat_takeoff"].to_numpy(float), fl["lon_takeoff"].to_numpy(float))
    elif (db / "cells.parquet").exists():
        cells = pd.read_parquet(db / "cells.parquet", columns=["cell_id", "elev_mean"])
        fl["to_cell"] = grid.cell_id(fl["lat_takeoff"].to_numpy(float), fl["lon_takeoff"].to_numpy(float))
        fl = fl.merge(cells.rename(columns={"cell_id": "to_cell", "elev_mean": "dem_takeoff"}), on="to_cell", how="left").drop(columns=["to_cell"])
    else:
        fl["dem_takeoff"] = np.nan

    fl["takeoff_agl_raw"] = fl["alt_takeoff"].astype(float) - fl["dem_takeoff"].astype(float)
    fl["alt_consistency_m"] = fl["takeoff_agl_raw"] - fl["landing_agl_raw"].astype(float)
    fl["agl_offset"] = fl["agl_offset"].fillna(0.0).astype(float)

    cal_landing = (
        fl["landing_agl_raw"].notna()
        & (fl["landing_agl_raw"].abs() <= igc_truth.MAX_OFFSET_M)
        & (fl["dem_coverage"].fillna(0.0) >= igc_truth.MIN_DEM_COV)
    )
    step = fl["step_s"].fillna(99.0).astype(float)
    is_gold = (
        cal_landing
        & (step <= 2.0)
        & fl["alt_consistency_m"].notna()
        & (fl["alt_consistency_m"].abs() <= igc_truth.GOLD_DRIFT_M)
    )
    is_silver = cal_landing & (step <= 5.0) & (~is_gold)
    fl["cal_tier"] = np.where(is_gold, "gold", np.where(is_silver, "silver", "excluded"))

    keep_cols = [
        "track_id", "task_key", "event_key", "site_id", "result_id", "pilot_key", "source",
        "flight_date", "task_date", "step_s", "quality", "cal_tier",
        "t_takeoff", "t_landing", "duration_s",
        "lat_takeoff", "lon_takeoff", "alt_takeoff", "dem_takeoff",
        "takeoff_agl_raw", "landing_agl_raw", "agl_offset", "alt_consistency_m",
        "dem_coverage", "alt_max",
    ]
    return fl[[c for c in keep_cols if c in fl]].reset_index(drop=True)


# ── Step 2: Per-task thermal & wind extraction from 5 s fixes ────────────────

def _process_task_job(job: tuple[str, str, pd.DataFrame, pd.DataFrame]) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    task_key, fixes_path, fl_task, th_task = job
    th_rows: list[dict[str, Any]] = []
    wind_rows: list[dict[str, Any]] = []
    p = Path(fixes_path)
    if not p.exists() or not len(th_task):
        return task_key, th_rows, wind_rows

    track_ids = set(fl_task["track_id"])
    fx = pd.read_parquet(p, columns=["track_id", "t", "lat", "lon", "alt"])
    fx = fx[fx["track_id"].isin(track_ids)]
    if not len(fx):
        return task_key, th_rows, wind_rows

    fl_meta = fl_task.set_index("track_id").to_dict(orient="index")
    by_track = {tid: g.sort_values("t") for tid, g in fx.groupby("track_id", sort=False)}

    for r in th_task.itertuples(index=False):
        g = by_track.get(r.track_id)
        if g is None or len(g) < 3:
            continue
        meta = fl_meta[r.track_id]
        t_arr = g["t"].to_numpy(np.float64)
        # Binary search slice on sorted timestamps (much faster than boolean masks).
        i0 = int(np.searchsorted(t_arr, float(r.t_start), side="left"))
        i1 = int(np.searchsorted(t_arr, float(r.t_end), side="right"))
        if i1 - i0 < 3:
            continue

        offset = float(meta["agl_offset"])
        g_base = float(r.alt_base - r.agl_base - offset) if pd.notna(r.agl_base) else np.nan
        g_top = float(r.alt_top - r.agl_top - offset) if pd.notna(r.agl_top) else g_base

        summary, slices = igc_truth.extract_thermal_sensors(
            t_arr[i0:i1],
            g["lat"].to_numpy(np.float64)[i0:i1],
            g["lon"].to_numpy(np.float64)[i0:i1],
            g["alt"].to_numpy(np.float64)[i0:i1],
            agl_offset=offset,
            ground_base=g_base,
            ground_top=g_top,
            split_deep=True,
        )
        if not summary:
            continue

        common = {
            "track_id": r.track_id,
            "task_key": task_key,
            "event_key": meta["event_key"],
            "site_id": meta["site_id"],
            "pilot_key": meta["pilot_key"],
            "flight_date": meta["flight_date"],
            "cal_tier": meta["cal_tier"],
            "seq": int(r.seq),
            "turns_raw": float(r.turns),
        }
        cid = int(grid.chip_id(summary["lat"], summary["lon"]))
        th_rows.append({
            **common,
            "chip_id": cid,
            "cell_id": int(grid.cell_id(summary["lat"], summary["lon"])),
            "old_wind_u": float(r.wind_u) if pd.notna(r.wind_u) else np.nan,
            "old_wind_v": float(r.wind_v) if pd.notna(r.wind_v) else np.nan,
            **summary,
        })
        for sl in slices:
            wind_rows.append({
                **common,
                "chip_id": int(grid.chip_id(sl["lat"], sl["lon"])),
                "old_wind_u": float(r.wind_u) if (sl["n_slices"] == 1 and pd.notna(r.wind_u)) else sl["u_old"],
                "old_wind_v": float(r.wind_v) if (sl["n_slices"] == 1 and pd.notna(r.wind_v)) else sl["v_old"],
                **sl,
            })

    return task_key, th_rows, wind_rows


def _write_batch(parts: Path, th_batch: list[dict[str, Any]], w_batch: list[dict[str, Any]], keys: list[str]) -> None:
    name = f"batch_{int(time.time() * 1000)}"
    for tbl, rows in (("thermals", th_batch), ("wind_obs", w_batch)):
        if rows:
            tmp = parts / f"{name}.{tbl}.tmp"
            pd.DataFrame(rows).to_parquet(tmp, index=False)
            tmp.rename(parts / f"{name}.{tbl}.parquet")
    with open(parts / "done_tasks.txt", "a", encoding="utf-8") as f:
        f.write("".join(f"{k}\n" for k in keys))


def _gather_parts(parts: Path, suffix: str) -> pd.DataFrame:
    files = sorted(parts.glob(f"batch_*.{suffix}.parquet"))
    if not files:
        return pd.DataFrame()
    with ThreadPoolExecutor(16) as pool:
        frames = [df for df in pool.map(pd.read_parquet, files) if len(df)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ── Step 3: ERA5 join helper for apples-to-apples benchmark comparison ───────

def attach_era5_wind(groups: pd.DataFrame, db: Path) -> pd.DataFrame:
    """Interpolate ERA5 wind to each wind group's exact calibrated altitude `h_asl`."""
    w_dir = db / "weather"
    if not len(groups) or not (w_dir / "era5_single.parquet").exists():
        return groups

    chips = set(groups["chip_id"].unique())
    single = pd.read_parquet(w_dir / "era5_single.parquet", columns=["chip_id", "time", "u10", "v10", "u100", "v100"])
    single = single[single["chip_id"].isin(chips)].rename(columns={"time": "hour"})
    single["hour"] = pd.to_datetime(single["hour"]).astype("datetime64[ns]")
    g = groups.merge(single.drop_duplicates(["chip_id", "hour"]), on=["chip_id", "hour"], how="left")

    if (w_dir / "era5_static.parquet").exists():
        st = pd.read_parquet(w_dir / "era5_static.parquet", columns=["chip_id", "ground_m"]).drop_duplicates("chip_id")
        g = g.merge(st, on="chip_id", how="left")
    else:
        g["ground_m"] = g["chip_ground_m"]
    g["ground_m"] = g["ground_m"].fillna(g["chip_ground_m"])

    has_levels = (w_dir / "era5_levels.parquet").exists()
    levels: list[int] = []
    if has_levels:
        import pyarrow.parquet as pq

        have = set(pq.ParquetFile(w_dir / "era5_levels.parquet").schema.names)
        levels = [p for p in LEVELS if {f"u_{p}", f"v_{p}", f"z_{p}"} <= have]
        cols = [f"{v}_{p}" for p in levels for v in ("u", "v", "z")]
        lv = pd.read_parquet(w_dir / "era5_levels.parquet", columns=["chip_id", "time", *cols])
        lv = lv[lv["chip_id"].isin(chips)].rename(columns={"time": "hour3"})
        lv["hour3"] = pd.to_datetime(lv["hour3"]).astype("datetime64[ns]")
        g["hour3"] = g["hour"].dt.round("3h")
        g = g.merge(lv.drop_duplicates(["chip_id", "hour3"]), on=["chip_id", "hour3"], how="left")

    gm = g["ground_m"].to_numpy(float)
    H = [gm + 10.0, gm + 100.0]
    U = [g["u10"].to_numpy(float), g["u100"].to_numpy(float)]
    V = [g["v10"].to_numpy(float), g["v100"].to_numpy(float)]
    for p in levels:
        z = g[f"z_{p}"].to_numpy(float)
        z = np.where(z > gm + 150.0, z, np.nan)
        H.append(z)
        U.append(g[f"u_{p}"].to_numpy(float))
        V.append(g[f"v_{p}"].to_numpy(float))

    H_m, U_m, V_m = np.column_stack(H), np.column_stack(U), np.column_stack(V)
    h = g["h_asl"].to_numpy(float)
    eu = np.full(len(g), np.nan, dtype=np.float64)
    ev = np.full(len(g), np.nan, dtype=np.float64)
    for i in range(len(g)):
        if math.isnan(h[i]):
            continue
        ok = ~(np.isnan(H_m[i]) | np.isnan(U_m[i]) | np.isnan(V_m[i]))
        if ok.sum() < 2:
            continue
        hh = H_m[i][ok]
        order = np.argsort(hh)
        hh = hh[order]
        eu[i] = float(np.interp(h[i], hh, U_m[i][ok][order]))
        ev[i] = float(np.interp(h[i], hh, V_m[i][ok][order]))

    drop_cols = [c for c in ["u10", "v10", "u100", "v100", "hour3"] + [f"{v}_{p}" for p in levels for v in ("u", "v", "z")] if c in g]
    g = g.drop(columns=drop_cols)
    g["era5_u"] = eu.astype(np.float32)
    g["era5_v"] = ev.astype(np.float32)
    return g


def wind_score_dict(pu: np.ndarray, pv: np.ndarray, mu: np.ndarray, mv: np.ndarray) -> dict[str, Any]:
    ok = np.isfinite(pu) & np.isfinite(pv) & np.isfinite(mu) & np.isfinite(mv)
    if not ok.any():
        return {"groups": 0}
    pu, pv, mu, mv = pu[ok], pv[ok], mu[ok], mv[ok]
    err_sq = (pu - mu) ** 2 + (pv - mv) ** 2
    vec = np.sqrt(err_sq)
    pspd, mspd = np.hypot(pu, pv), np.hypot(mu, mv)
    pdir = (np.degrees(np.arctan2(-pu, -pv)) + 360.0) % 360.0
    mdir = (np.degrees(np.arctan2(-mu, -mv)) + 360.0) % 360.0
    derr = np.abs((pdir - mdir + 180.0) % 360.0 - 180.0)
    windy = mspd >= CALM_MS
    corr = float(np.corrcoef(pspd, mspd)[0, 1]) if len(pspd) > 2 and np.std(pspd) > 0 and np.std(mspd) > 0 else None
    return {
        "groups": int(len(vec)),
        "vector_rmse_ms": round(float(np.sqrt(np.mean(err_sq))), 3),
        "vector_median_ms": round(float(np.median(vec)), 3),
        "speed_bias_ms": round(float(np.median(pspd - mspd)), 3),
        "speed_corr": round(corr, 3) if corr is not None else None,
        "dir_error_deg": round(float(np.median(derr[windy])), 1) if windy.any() else None,
        "dir_within_30": round(float(np.mean(derr[windy] <= 30.0)), 4) if windy.any() else None,
    }


# ── Step 4: Pool wind, glide-wind, ceiling, climb, daily & task truth tables ─

def build_wind_groups(wind_obs: pd.DataFrame, cells_chip: pd.DataFrame, db: Path, skip_era5: bool) -> pd.DataFrame:
    if not len(wind_obs):
        return pd.DataFrame()
    w = wind_obs[wind_obs["wind_qc_ok"]].copy()
    if not len(w):
        return pd.DataFrame()

    w["time"] = pd.to_datetime(w["flight_date"]) + pd.to_timedelta(w["t_mid"], unit="s")
    w["hour"] = w["time"].dt.round("h").astype("datetime64[ns]")
    w["solar_hour"] = ((w["time"].dt.hour + w["time"].dt.minute / 60.0 + w["lon"] / 15.0) % 24.0).round()
    w = w.merge(cells_chip, on="chip_id", how="left")
    w["above_crest"] = w["wind_alt_cal"] - w["crest_m"]
    w["band"] = height_band(w["agl_cal"].to_numpy(float), w["above_crest"].to_numpy(float))

    keys = ["site_id", "chip_id", "hour", "band"]
    g = w.groupby(keys, sort=False).agg(
        event_key=("event_key", "first"),
        n_obs=("u_clean", "size"),
        n_flights=("track_id", "nunique"),
        n_gold=("cal_tier", lambda s: int((s == "gold").sum())),
        clean_u=("u_clean", "median"),
        clean_v=("v_clean", "median"),
        hodo_u=("u_hodo", "median"),
        hodo_v=("v_hodo", "median"),
        int_u=("u_int", "median"),
        int_v=("v_int", "median"),
        old_u=("old_wind_u", "median"),
        old_v=("old_wind_v", "median"),
        h_asl=("wind_alt_cal", "median"),
        agl=("agl_cal", "median"),
        centroid_lat=("lat", "median"),
        centroid_lon=("lon", "median"),
        tas_fit_med=("tas_fit", "median"),
        hodo_resid_med=("hodo_resid_ms", "median"),
        solar_hour=("solar_hour", "median"),
        chip_ground_m=("chip_ground_m", "first"),
    ).reset_index()

    g = g[(g["n_obs"] >= MIN_WIND_SLICES) & (g["n_flights"] >= MIN_WIND_FLIGHTS)].reset_index(drop=True)
    g["solar_hour"] = g["solar_hour"].round()
    g["clean_speed_ms"] = np.hypot(g["clean_u"], g["clean_v"]).astype(np.float32)
    g["clean_from_deg"] = ((np.degrees(np.arctan2(-g["clean_u"], -g["clean_v"])) + 360.0) % 360.0).astype(np.float32)
    if not skip_era5:
        g = attach_era5_wind(g, db)
    return g


def build_glide_wind_and_metrics(
    db: Path,
    fl_cal: pd.DataFrame,
    cells_chip: pd.DataFrame,
    task_subset: set[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build independent multi-heading glide wind groups + per-(site, chip, hour) class glide stats."""
    gp = db / "glides.parquet"
    if not gp.exists() or not len(fl_cal):
        return pd.DataFrame(), pd.DataFrame()

    gl = pd.read_parquet(gp)
    if task_subset is not None:
        gl = gl[gl["task_key"].isin(task_subset)]
    gl = gl.merge(
        fl_cal[["track_id", "site_id", "event_key", "result_id", "flight_date", "agl_offset"]],
        on="track_id",
        how="inner",
    )
    if not len(gl):
        return pd.DataFrame(), pd.DataFrame()

    if (db / "result_gliders.parquet").exists():
        rg = pd.read_parquet(db / "result_gliders.parquet", columns=["result_id", "glider_class"]).drop_duplicates("result_id")
        gl = gl.merge(rg, on="result_id", how="left")
    else:
        gl["glider_class"] = "unknown"
    gl["glider_class"] = gl["glider_class"].fillna("unknown")

    gl = gl[
        (gl["duration_s"] >= 60.0)
        & (gl["distance_m"] >= 500.0)
        & (gl["speed_kmh"].between(15.0, 85.0))
    ].copy()
    if not len(gl):
        return pd.DataFrame(), pd.DataFrame()

    gl["lat_mid"] = 0.5 * (gl["lat_start"] + gl["lat_end"])
    gl["lon_mid"] = 0.5 * (gl["lon_start"] + gl["lon_end"])
    gl["chip_id"] = grid.chip_id(gl["lat_mid"].to_numpy(float), gl["lon_mid"].to_numpy(float))
    t_mid = 0.5 * (gl["t_start"] + gl["t_end"])
    gl["time"] = pd.to_datetime(gl["flight_date"]) + pd.to_timedelta(t_mid, unit="s")
    gl["hour"] = gl["time"].dt.round("h").astype("datetime64[ns]")
    gl["bearing_rad"] = _bearing_rad(
        gl["lat_start"].to_numpy(float), gl["lon_start"].to_numpy(float),
        gl["lat_end"].to_numpy(float), gl["lon_end"].to_numpy(float),
    )
    gl["speed_ms"] = gl["speed_kmh"] / 3.6
    gl["sink_ms"] = gl["alt_loss_m"] / gl["duration_s"]
    gl["h_asl"] = 0.5 * (gl["alt_start"] + gl["alt_end"]) - gl["agl_offset"]
    gl = gl.merge(cells_chip, on="chip_id", how="left")
    gl["agl_est"] = gl["h_asl"] - gl["chip_ground_m"]
    gl["band"] = height_band(gl["agl_est"].to_numpy(float), (gl["h_asl"] - gl["crest_m"]).to_numpy(float))

    # Multi-heading Huber glide wind on known competition glider classes.
    gl_w = gl[gl["glider_class"].isin(["CCC", "EN-D", "EN-C"])]
    gw_rows = []
    for (sid, cid, hr, band), grp in gl_w.groupby(["site_id", "chip_id", "hour", "band"], sort=False):
        if len(grp) < 8 or grp["track_id"].nunique() < 3:
            continue
        fit = igc_truth.fit_glide_wind_huber(
            grp["bearing_rad"].to_numpy(float),
            grp["speed_ms"].to_numpy(float),
            grp["glider_class"].to_numpy(),
        )
        if fit is None:
            continue
        gw_rows.append({
            "site_id": sid,
            "chip_id": int(cid),
            "hour": hr,
            "band": band,
            "event_key": grp["event_key"].iloc[0],
            "n_flights": int(grp["track_id"].nunique()),
            "h_asl": float(grp["h_asl"].median()),
            "glide_u": fit["glide_u"],
            "glide_v": fit["glide_v"],
            "glide_speed_ms": fit["glide_speed_ms"],
            "n_glides": fit["n_glides"],
            "spread_deg": fit["spread_deg"],
            "resid_ms": fit["resid_ms"],
            "v_air_ccc_ms": fit["v_air_by_class"].get("CCC", np.nan),
        })
    glide_wind_df = pd.DataFrame(gw_rows)

    # Per-(site, chip, hour) glider-class groundspeed / glide ratio summary.
    cls_frames = []
    for cls, tag in (("CCC", "ccc"), ("EN-D", "end"), ("EN-C", "enc")):
        sub = gl[gl["glider_class"] == cls]
        if not len(sub):
            continue
        agg = sub.groupby(["site_id", "chip_id", "hour"], sort=False).agg(
            **{
                f"glide_speed_{tag}_p50": ("speed_kmh", "median"),
                f"glide_gr_{tag}_p50": ("glide_ratio", "median"),
                f"glide_sink_{tag}_p50": ("sink_ms", "median"),
                f"glide_n_{tag}": ("speed_kmh", "size"),
            }
        ).reset_index()
        agg = agg[agg[f"glide_n_{tag}"] >= 3]
        cls_frames.append(agg)

    if cls_frames:
        glide_metrics = cls_frames[0]
        for f_extra in cls_frames[1:]:
            glide_metrics = glide_metrics.merge(f_extra, on=["site_id", "chip_id", "hour"], how="outer")
    else:
        glide_metrics = pd.DataFrame(columns=["site_id", "chip_id", "hour"])

    return glide_wind_df, glide_metrics


def build_ceiling_and_climb_groups(
    th_df: pd.DataFrame,
    glide_metrics: pd.DataFrame,
    cells_chip: pd.DataFrame,
    db: Path,
    skip_era5: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Pool multi-pilot calibrated ASL ceilings and core60/core30/mean climb speeds."""
    if not len(th_df):
        return pd.DataFrame(), pd.DataFrame()

    t = th_df.copy()
    t["time"] = pd.to_datetime(t["flight_date"]) + pd.to_timedelta(t["t_mid"], unit="s")
    t["hour"] = t["time"].dt.round("h").astype("datetime64[ns]")
    t["solar_hour"] = ((t["time"].dt.hour + t["time"].dt.minute / 60.0 + t["lon"] / 15.0) % 24.0).round()
    t = t.merge(cells_chip, on="chip_id", how="left")

    # 1. Ceiling: sustained thermals (>= 150 m gain), >= 3 distinct calibrated pilots, >= 5 thermals.
    sust = t[(t["gain_m"] >= igc_truth.MIN_CEILING_GAIN_M) & (t["rate_mean_ms"] >= 0.5)].copy()
    ceil_rows = []
    for (sid, cid, hr), grp in sust.groupby(["site_id", "chip_id", "hour"], sort=False):
        n_pilots = int(grp["track_id"].nunique())
        if n_pilots < MIN_CEIL_PILOTS or len(grp) < MIN_CEIL_THERMALS:
            continue
        pilot_tops = grp.groupby("track_id")["alt_top_cal"].max().to_numpy(float)
        all_tops = grp["alt_top_cal"].to_numpy(float)
        p25, p75, p90, p95 = np.percentile(pilot_tops, [25, 75, 90, 95])
        ceil_rows.append({
            "site_id": sid,
            "chip_id": int(cid),
            "hour": hr,
            "event_key": grp["event_key"].iloc[0],
            "n_pilots": n_pilots,
            "n_gold_pilots": int(grp.loc[grp["cal_tier"] == "gold", "track_id"].nunique()),
            "n_thermals": int(len(grp)),
            "ceiling_p90_asl": float(p90),
            "ceiling_p95_asl": float(p95),
            "ceiling_max_asl": float(np.max(pilot_tops)),
            "ceiling_all_p90_asl": float(np.percentile(all_tops, 90)),
            "ceiling_iqr_m": float(p75 - p25),
            "ceiling_p90_agl_dem": float(np.nanpercentile(grp["agl_top_cal"].to_numpy(float), 90)) if grp["agl_top_cal"].notna().any() else np.nan,
            "airspace_cap_flag": igc_truth.detect_airspace_cap(pilot_tops),
            "centroid_lat": float(grp["lat"].median()),
            "centroid_lon": float(grp["lon"].median()),
            "solar_hour": float(round(float(grp["solar_hour"].median()))),
            "chip_ground_m": float(grp["chip_ground_m"].iloc[0]),
        })
    ceil_df = pd.DataFrame(ceil_rows)

    if len(ceil_df) and not skip_era5:
        w_dir = db / "weather"
        if (w_dir / "era5_single.parquet").exists():
            chips = set(ceil_df["chip_id"].unique())
            era = pd.read_parquet(w_dir / "era5_single.parquet", columns=["chip_id", "time", "blh_m", "cloud_base_m", "cloud_total"])
            era = era[era["chip_id"].isin(chips)].rename(columns={"time": "hour"})
            era["hour"] = pd.to_datetime(era["hour"]).astype("datetime64[ns]")
            ceil_df = ceil_df.merge(era.drop_duplicates(["chip_id", "hour"]), on=["chip_id", "hour"], how="left")
        if (w_dir / "era5_static.parquet").exists():
            st = pd.read_parquet(w_dir / "era5_static.parquet", columns=["chip_id", "ground_m"]).drop_duplicates("chip_id")
            ceil_df = ceil_df.merge(st, on="chip_id", how="left")
        else:
            ceil_df["ground_m"] = ceil_df["chip_ground_m"]
        ceil_df["ground_m"] = ceil_df["ground_m"].fillna(ceil_df["chip_ground_m"])
        if "blh_m" in ceil_df:
            ceil_df["era5_blh_asl"] = ceil_df["ground_m"] + ceil_df["blh_m"]
            ceil_df["ceiling_above_era5"] = ceil_df["ceiling_p90_asl"] - ceil_df["ground_m"]
        if "cloud_base_m" in ceil_df:
            ceil_df["era5_cb_asl"] = ceil_df["ground_m"] + ceil_df["cloud_base_m"]
            ceil_df["era5_min_blh_cb_asl"] = np.where(
                ceil_df["era5_cb_asl"].notna(),
                np.minimum(ceil_df["era5_blh_asl"], ceil_df["era5_cb_asl"]),
                ceil_df["era5_blh_asl"],
            )

    # 2. Climb strength groups per (site_id, chip_id, hour): >= 3 pilots, >= 5 thermals.
    climb_rows = []
    for (sid, cid, hr), grp in t.groupby(["site_id", "chip_id", "hour"], sort=False):
        n_pilots = int(grp["track_id"].nunique())
        if n_pilots < MIN_CEIL_PILOTS or len(grp) < MIN_CEIL_THERMALS:
            continue
        c60 = grp["rate_core60_ms"].dropna().to_numpy(float)
        c30 = grp["rate_core30_ms"].dropna().to_numpy(float)
        rm = grp["rate_mean_ms"].dropna().to_numpy(float)
        if len(rm) < MIN_CEIL_THERMALS:
            continue
        climb_rows.append({
            "site_id": sid,
            "chip_id": int(cid),
            "hour": hr,
            "event_key": grp["event_key"].iloc[0],
            "n_pilots": n_pilots,
            "n_thermals": int(len(grp)),
            "n_core60": int(len(c60)),
            "core60_p50_ms": float(np.percentile(c60, 50)) if len(c60) >= 3 else np.nan,
            "core60_p75_ms": float(np.percentile(c60, 75)) if len(c60) >= 3 else np.nan,
            "core60_p90_ms": float(np.percentile(c60, 90)) if len(c60) >= 3 else np.nan,
            "core30_p75_ms": float(np.percentile(c30, 75)) if len(c30) >= 3 else np.nan,
            "mean_rate_p50_ms": float(np.percentile(rm, 50)),
            "mean_rate_p75_ms": float(np.percentile(rm, 75)),
            "mid_asl_med": float(grp["alt_mid_cal"].median()),
            "mid_agl_med": float(grp["agl_mid_cal"].median()) if grp["agl_mid_cal"].notna().any() else np.nan,
            "centroid_lat": float(grp["lat"].median()),
            "centroid_lon": float(grp["lon"].median()),
            "solar_hour": float(round(float(grp["solar_hour"].median()))),
        })
    climb_df = pd.DataFrame(climb_rows)
    if len(climb_df) and len(glide_metrics):
        climb_df = climb_df.merge(glide_metrics, on=["site_id", "chip_id", "hour"], how="left")
    return ceil_df, climb_df


def build_daily_and_tasks(
    db: Path,
    fl_cal: pd.DataFrame,
    th_df: pd.DataFrame,
    wind_groups: pd.DataFrame,
    ceil_df: pd.DataFrame,
    climb_df: pd.DataFrame,
    task_subset: set[str] | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build site-day physical summary (`truth_daily`) and task-level truth (`truth_tasks`)."""
    # Daily summary keyed by (site_id, local_date)
    daily_parts = []
    if len(ceil_df):
        c = ceil_df[~ceil_df["airspace_cap_flag"]].copy()
        if len(c):
            c["local_date"] = (c["hour"] + pd.to_timedelta(c["centroid_lon"] / 15.0, unit="h")).dt.strftime("%Y-%m-%d")
            idx_max = c.groupby(["site_id", "local_date"])["ceiling_p90_asl"].idxmax()
            d_ceil = c.loc[idx_max, ["site_id", "local_date", "event_key", "ceiling_p90_asl", "ceiling_p95_asl", "solar_hour", "n_pilots"]].rename(
                columns={"ceiling_p90_asl": "day_ceiling_p90_asl", "ceiling_p95_asl": "day_ceiling_p95_asl",
                         "solar_hour": "peak_ceiling_solar_h", "n_pilots": "ceil_pilots"}
            )
            if "era5_blh_asl" in c:
                d_era = c.groupby(["site_id", "local_date"]).agg(
                    day_era5_blh_max_asl=("era5_blh_asl", "max"),
                    day_era5_min_blh_cb_asl=("era5_min_blh_cb_asl", "max"),
                ).reset_index()
                d_ceil = d_ceil.merge(d_era, on=["site_id", "local_date"], how="left")
            daily_parts.append(d_ceil)

    if len(climb_df):
        cl = climb_df.copy()
        cl["local_date"] = (cl["hour"] + pd.to_timedelta(cl["centroid_lon"] / 15.0, unit="h")).dt.strftime("%Y-%m-%d")
        d_cl = cl.groupby(["site_id", "local_date"]).agg(
            day_core60_p75_ms=("core60_p75_ms", "max"),
            day_core60_p50_ms=("core60_p50_ms", "median"),
            day_mean_rate_p75_ms=("mean_rate_p75_ms", "max"),
        ).reset_index()
        daily_parts.append(d_cl)

    if len(wind_groups):
        wg = wind_groups[wind_groups["band"].isin(["free air", "middle"])].copy()
        if len(wg):
            wg["local_date"] = (wg["hour"] + pd.to_timedelta(wg["centroid_lon"] / 15.0, unit="h")).dt.strftime("%Y-%m-%d")
            d_w = wg.groupby(["site_id", "local_date"]).agg(
                day_wind_u=("clean_u", "median"),
                day_wind_v=("clean_v", "median"),
                day_wind_ms=("clean_speed_ms", "median"),
            ).reset_index()
            daily_parts.append(d_w)

    if daily_parts:
        daily = daily_parts[0]
        for p in daily_parts[1:]:
            daily = daily.merge(p, on=["site_id", "local_date"], how="outer")
    else:
        daily = pd.DataFrame(columns=["site_id", "local_date"])

    # Task-level truth combining official results with IGC-measured conditions during the task.
    tasks = pd.read_parquet(db / "tasks.parquet")
    if task_subset is not None:
        tasks = tasks[tasks["task_key"].isin(task_subset)].copy()
    tasks = tasks[~tasks["training"].fillna(False).astype(bool) & tasks["task_type"].isin(["race", "elapsed"])].copy()

    res = pd.read_parquet(db / "results.parquet", columns=["task_key", "status", "time_s", "distance_km"])
    flew = res[~res["status"].isin(["abs", "dnf", "dsq"])].copy()
    agg = flew.groupby("task_key").agg(
        launched=("status", "size"),
        n_goal=("status", lambda s: int((s == "goal").sum())),
        winner_s=("time_s", lambda s: float(s[s > 0].min()) if (s > 0).any() else np.nan),
        p10_s=("time_s", lambda s: float(np.percentile(s[s > 0], 10)) if (s > 0).sum() >= 5 else np.nan),
        median_goal_s=("time_s", lambda s: float(np.median(s[s > 0])) if (s > 0).any() else np.nan),
    ).reset_index()
    tt = tasks.merge(agg, on="task_key", how="inner")
    tt = tt[tt["launched"] >= 10].copy()
    tt["goal_share"] = tt["n_goal"] / tt["launched"]
    dist = tt["ss_distance_km"].where(tt["ss_distance_km"] > 0, tt["distance_km"])
    tt["ss_km"] = dist
    tt["winner_h"] = tt["winner_s"] / 3600.0
    tt["winner_kmh"] = dist / tt["winner_h"]
    tt["p10_kmh"] = dist / (tt["p10_s"] / 3600.0)
    tt["median_goal_kmh"] = dist / (tt["median_goal_s"] / 3600.0)
    tt.loc[(tt["winner_kmh"] < 5.0) | (tt["winner_kmh"] > 80.0), ["winner_h", "winner_kmh"]] = np.nan

    ev_sites = pd.read_parquet(db / "event_sites.parquet", columns=["event_key", "site_id"])
    tt = tt.merge(ev_sites, on="event_key", how="left")

    # Aggregate calibrated IGC measurements per task_key directly from truth_thermals + truth_flights.
    if len(fl_cal):
        n_cal = fl_cal.groupby("task_key").agg(
            n_cal_flights=("track_id", "nunique"),
            n_gold_flights=("cal_tier", lambda s: int((s == "gold").sum())),
        ).reset_index()
        tt = tt.merge(n_cal, on="task_key", how="left")

    if len(th_df):
        th_task = th_df.groupby("task_key").agg(
            task_n_thermals=("alt_top_cal", "size"),
            task_ceiling_p90_asl=("alt_top_cal", lambda s: float(np.percentile(s, 90)) if len(s) >= 5 else np.nan),
            task_ceiling_p95_asl=("alt_top_cal", lambda s: float(np.percentile(s, 95)) if len(s) >= 5 else np.nan),
            task_core60_p75_ms=("rate_core60_ms", lambda s: float(np.nanpercentile(s, 75)) if s.notna().sum() >= 5 else np.nan),
            task_core60_p50_ms=("rate_core60_ms", lambda s: float(np.nanpercentile(s, 50)) if s.notna().sum() >= 5 else np.nan),
            task_mean_rate_p50_ms=("rate_mean_ms", "median"),
            task_wind_u=("u_clean", lambda s: float(np.nanmedian(s)) if s.notna().sum() >= 3 else np.nan),
            task_wind_v=("v_clean", lambda s: float(np.nanmedian(s)) if s.notna().sum() >= 3 else np.nan),
        ).reset_index()
        th_task["task_wind_ms"] = np.hypot(th_task["task_wind_u"], th_task["task_wind_v"])
        tt = tt.merge(th_task, on="task_key", how="left")

    # Compute measured tailwind along the real turnpoint route when coordinates exist.
    if (db / "turnpoints.parquet").exists() and "task_wind_u" in tt:
        tp = pd.read_parquet(db / "turnpoints.parquet", columns=["task_key", "seq", "lat", "lon"])
        tp = tp[tp["task_key"].isin(tt["task_key"])].dropna(subset=["lat", "lon"]).sort_values(["task_key", "seq"])
        wind_map = tt.set_index("task_key")[["task_wind_u", "task_wind_v"]].to_dict(orient="index")
        tw_map, cw_map = {}, {}
        for key, grp in tp.groupby("task_key", sort=False):
            if len(grp) < 2:
                continue
            wm = wind_map.get(key)
            if not wm or pd.isna(wm["task_wind_u"]):
                continue
            la, lo = grp["lat"].to_numpy(float), grp["lon"].to_numpy(float)
            dlat, dlon = np.radians(np.diff(la)), np.radians(np.diff(lo))
            p1, p2 = np.radians(la[:-1]), np.radians(la[1:])
            a = np.sin(dlat / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlon / 2) ** 2
            L = 6371.0 * 2.0 * np.arcsin(np.sqrt(np.minimum(a, 1.0)))
            valid = L >= 0.5
            if not valid.any():
                continue
            L = L[valid]
            b = _bearing_rad(la[:-1][valid], lo[:-1][valid], la[1:][valid], lo[1:][valid])
            uu, vv = float(wm["task_wind_u"]), float(wm["task_wind_v"])
            tw_map[key] = float((L * (uu * np.sin(b) + vv * np.cos(b))).sum() / L.sum())
            cw_map[key] = float((L * np.abs(-uu * np.cos(b) + vv * np.sin(b))).sum() / L.sum())
        tt["task_tailwind_measured_ms"] = tt["task_key"].map(tw_map)
        tt["task_crosswind_measured_ms"] = tt["task_key"].map(cw_map)

    keep_cols = [
        "task_key", "event_key", "site_id", "date", "task_type", "stopped",
        "ss_km", "launched", "n_goal", "goal_share", "winner_h", "winner_kmh", "p10_kmh", "median_goal_kmh",
        "n_cal_flights", "n_gold_flights", "task_n_thermals",
        "task_ceiling_p90_asl", "task_ceiling_p95_asl",
        "task_core60_p75_ms", "task_core60_p50_ms", "task_mean_rate_p50_ms",
        "task_wind_u", "task_wind_v", "task_wind_ms",
        "task_tailwind_measured_ms", "task_crosswind_measured_ms",
    ]
    return daily, tt[[c for c in keep_cols if c in tt]].reset_index(drop=True)


# ── Step 5: 1 Hz raw IGC vs 5 s fixes quantitative audit ─────────────────────

def run_1hz_audit(
    db: Path,
    fl_cal: pd.DataFrame,
    th_all: pd.DataFrame,
    out_dir: Path,
    n_sample: int,
    root: Path | None,
    seed: int = 42,
) -> dict[str, Any]:
    """Compare raw 1 Hz IGC tracks (GPS + barometric) against 5 s fixes on identical thermals."""
    if n_sample <= 0 or not (db / "tracks.parquet").exists():
        return {"skipped": True}

    cand = fl_cal[(fl_cal["cal_tier"] == "gold") & (fl_cal["step_s"] <= 1.1)].copy()
    if not len(cand):
        cand = fl_cal[fl_cal["step_s"] <= 1.1].copy()
    if not len(cand):
        return {"skipped": True, "reason": "no 1 Hz calibrated flights found"}

    tr = pd.read_parquet(db / "tracks.parquet", columns=["track_id", "zip_path", "member"]).drop_duplicates("track_id")
    cand = cand.merge(tr, on="track_id", how="inner").dropna(subset=["zip_path", "member"])
    if not len(cand):
        return {"skipped": True, "reason": "no track zip paths found"}

    # Stratified pick across distinct tasks/sites (prefer up to 2 flights per task; top up if few tasks).
    rng = np.random.default_rng(seed)
    shuffled = cand.iloc[rng.permutation(len(cand))]
    strat = shuffled.groupby("task_key", sort=False).head(2)
    if len(strat) < n_sample and len(shuffled) > len(strat):
        rest = shuffled.loc[~shuffled.index.isin(strat.index)].head(n_sample - len(strat))
        cand = pd.concat([strat, rest], ignore_index=True)
    else:
        cand = strat.head(n_sample)
    th_sub = th_all[th_all["track_id"].isin(set(cand["track_id"]))].copy()
    if not len(th_sub):
        return {"skipped": True, "reason": "no thermals on sampled 1 Hz flights"}

    rows: list[dict[str, Any]] = []
    n_flights_read = 0
    for zip_path, grp in cand.groupby("zip_path", sort=False):
        resolved = relocate(str(zip_path), root)
        if not Path(resolved).exists():
            continue
        try:
            zf = zipfile.ZipFile(resolved)
        except (OSError, zipfile.BadZipFile):
            continue
        with zf:
            for fl_r in grp.itertuples(index=False):
                fp = db / "fixes" / f"{safe(fl_r.task_key)}.parquet"
                if not fp.exists():
                    continue
                try:
                    raw_fx = igc.fixes(zf.read(fl_r.member))
                except Exception:
                    continue
                if len(raw_fx["t"]) < 30:
                    continue
                t1, lat1, lon1, alt1 = pta_flight.clean(raw_fx)
                if len(t1) < 30 or float(np.median(np.diff(t1))) > 1.5:
                    continue

                # Check if valid barometric altitude is also recorded in this raw IGC.
                baro_raw = raw_fx["alt_baro"].astype(float)
                has_baro = bool((baro_raw != 0).mean() > 0.8 and np.ptp(baro_raw) > 50.0)
                if has_baro:
                    t_b, _, _, alt_b = pta_flight.clean({**raw_fx, "alt_gps": raw_fx["alt_baro"]})
                else:
                    t_b, alt_b = np.empty(0), np.empty(0)

                fx5 = pd.read_parquet(fp, columns=["track_id", "t", "lat", "lon", "alt"])
                fx5 = fx5[fx5["track_id"] == fl_r.track_id].sort_values("t")
                if len(fx5) < 10:
                    continue
                t5 = fx5["t"].to_numpy(np.float64)
                lat5, lon5, alt5 = fx5["lat"].to_numpy(np.float64), fx5["lon"].to_numpy(np.float64), fx5["alt"].to_numpy(np.float64)

                fl_th = th_sub[th_sub["track_id"] == fl_r.track_id]
                if not len(fl_th):
                    continue
                n_flights_read += 1
                off = float(fl_r.agl_offset)

                for tr_r in fl_th.itertuples(index=False):
                    a1, b1 = int(np.searchsorted(t1, float(tr_r.t_start))), int(np.searchsorted(t1, float(tr_r.t_end), side="right"))
                    a5, b5 = int(np.searchsorted(t5, float(tr_r.t_start))), int(np.searchsorted(t5, float(tr_r.t_end), side="right"))
                    if (b1 - a1) < 20 or (b5 - a5) < 6:
                        continue
                    s1, _ = igc_truth.extract_thermal_sensors(t1[a1:b1], lat1[a1:b1], lon1[a1:b1], alt1[a1:b1], agl_offset=off, split_deep=False)
                    s5, _ = igc_truth.extract_thermal_sensors(t5[a5:b5], lat5[a5:b5], lon5[a5:b5], alt5[a5:b5], agl_offset=off, split_deep=False)
                    if not s1 or not s5:
                        continue

                    core60_baro = np.nan
                    if has_baro and len(t_b):
                        ab, bb = int(np.searchsorted(t_b, float(tr_r.t_start))), int(np.searchsorted(t_b, float(tr_r.t_end), side="right"))
                        if (bb - ab) >= 20:
                            core60_baro = igc_truth.peak_climb_rate(t_b[ab:bb], alt_b[ab:bb], igc_truth.CORE60_S, igc_truth.MAX_CORE60_MS)

                    rows.append({
                        "track_id": fl_r.track_id,
                        "task_key": fl_r.task_key,
                        "seq": int(tr_r.seq),
                        "u_clean_1hz": s1["u_clean"], "v_clean_1hz": s1["v_clean"],
                        "u_clean_5s": s5["u_clean"], "v_clean_5s": s5["v_clean"],
                        "u_hodo_1hz": s1["u_hodo"], "v_hodo_1hz": s1["v_hodo"],
                        "u_hodo_5s": s5["u_hodo"], "v_hodo_5s": s5["v_hodo"],
                        "u_int_1hz": s1["u_int"], "v_int_1hz": s1["v_int"],
                        "u_int_5s": s5["u_int"], "v_int_5s": s5["v_int"],
                        "core60_1hz": s1["rate_core60_ms"], "core60_5s": s5["rate_core60_ms"],
                        "core60_baro_1hz": core60_baro,
                        "core30_1hz": s1["rate_core30_ms"], "core30_5s": s5["rate_core30_ms"],
                        "top_1hz": s1["alt_top_cal"], "top_5s": s5["alt_top_cal"],
                    })

    if not rows:
        return {"skipped": True, "reason": "no raw IGC zip archives accessible under --root"}

    df = pd.DataFrame(rows)
    df.to_parquet(out_dir / "audit_1hz_vs_5s.parquet", index=False)

    def _vec_cmp(u1: np.ndarray, v1: np.ndarray, u5: np.ndarray, v5: np.ndarray) -> dict[str, Any]:
        ok = np.isfinite(u1) & np.isfinite(v1) & np.isfinite(u5) & np.isfinite(v5)
        if not ok.any():
            return {"n": 0}
        diff = np.hypot(u5[ok] - u1[ok], v5[ok] - v1[ok])
        spd_bias = np.hypot(u5[ok], v5[ok]) - np.hypot(u1[ok], v1[ok])
        return {
            "n": int(ok.sum()),
            "vector_rmse_ms": round(float(np.sqrt(np.mean(diff * diff))), 4),
            "vector_median_ms": round(float(np.median(diff)), 4),
            "speed_bias_5s_minus_1hz_ms": round(float(np.median(spd_bias)), 4),
        }

    def _scalar_cmp(a: np.ndarray, b: np.ndarray) -> dict[str, Any]:
        ok = np.isfinite(a) & np.isfinite(b)
        if not ok.any():
            return {"n": 0}
        d = b[ok] - a[ok]
        corr = float(np.corrcoef(a[ok], b[ok])[0, 1]) if ok.sum() > 2 else None
        return {
            "n": int(ok.sum()),
            "mae": round(float(np.mean(np.abs(d))), 4),
            "rmse": round(float(np.sqrt(np.mean(d * d))), 4),
            "bias_5s_minus_1hz": round(float(np.median(d)), 4),
            "p95_abs_diff": round(float(np.percentile(np.abs(d), 95)), 4),
            "corr": round(corr, 4) if corr is not None else None,
        }

    w_clean = _vec_cmp(df["u_clean_1hz"].to_numpy(), df["v_clean_1hz"].to_numpy(), df["u_clean_5s"].to_numpy(), df["v_clean_5s"].to_numpy())
    c60 = _scalar_cmp(df["core60_1hz"].to_numpy(), df["core60_5s"].to_numpy())
    c60_baro = _scalar_cmp(df["core60_baro_1hz"].to_numpy(), df["core60_5s"].to_numpy())
    c30 = _scalar_cmp(df["core30_1hz"].to_numpy(), df["core30_5s"].to_numpy())
    top = _scalar_cmp(df["top_1hz"].to_numpy(), df["top_5s"].to_numpy())

    ok_5s = (
        (w_clean.get("vector_rmse_ms", 99.0) <= 0.25)
        and (c60.get("mae", 99.0) <= 0.10)
        and (top.get("mae", 99.0) <= 5.0)
    )
    return {
        "n_flights_audited": int(n_flights_read),
        "n_thermals_matched": int(len(df)),
        "wind_clean_1hz_vs_5s": w_clean,
        "wind_hodo_1hz_vs_5s": _vec_cmp(df["u_hodo_1hz"].to_numpy(), df["v_hodo_1hz"].to_numpy(), df["u_hodo_5s"].to_numpy(), df["v_hodo_5s"].to_numpy()),
        "wind_int_1hz_vs_5s": _vec_cmp(df["u_int_1hz"].to_numpy(), df["v_int_1hz"].to_numpy(), df["u_int_5s"].to_numpy(), df["v_int_5s"].to_numpy()),
        "core60_gps_1hz_vs_5s": c60,
        "core60_baro_1hz_vs_5s_gps": c60_baro,
        "core30_gps_1hz_vs_5s": c30,
        "alt_top_1hz_vs_5s_m": top,
        "verdict": "5s_sufficient" if ok_5s else "inspect_1hz_difference",
    }


# ── Main CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder (e.g. /content/drive/MyDrive/IGCS/database)")
    parser.add_argument("--out", type=Path, help="output folder (default DB/truth)")
    parser.add_argument("--dem", type=Path, help="folder with Copernicus GLO-30 DEM tiles (for takeoff audit)")
    parser.add_argument("--local", type=Path, help="local scratch folder to mirror DEM tiles (faster than Drive)")
    parser.add_argument("--root", type=Path, help="project root containing *_igc zip folders (for --check-1hz)")
    parser.add_argument("--workers", type=int, default=4, help="worker processes for task extraction (default 4)")
    parser.add_argument("--limit", type=int, default=None, help="process only the first N tasks (smoke test)")
    parser.add_argument("--check-1hz", type=int, default=0, help="audit N raw 1 Hz flights vs 5 s fixes (default 0)")
    parser.add_argument("--skip-era5-check", action="store_true", help="skip ERA5 benchmark join on wind & ceiling")
    parser.add_argument("--redo", action="store_true", help="discard cached task batches in DB/_truth_parts")
    args = parser.parse_args()

    out = args.out or (args.db / "truth")
    parts = args.db / "_truth_parts"
    if args.redo:
        shutil.rmtree(parts, ignore_errors=True)
    out.mkdir(parents=True, exist_ok=True)
    parts.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    all_tasks = sorted(pd.read_parquet(args.db / "thermals.parquet", columns=["task_key"])["task_key"].unique())
    task_subset = set(all_tasks[: args.limit]) if args.limit else None

    print("1/5 Auditing flight altitude calibration (takeoff + landing vs DEM)...", flush=True)
    fl = build_truth_flights(args.db, args.dem, args.local, task_subset=task_subset)
    fl.to_parquet(out / "truth_flights.parquet", index=False)
    tier_counts = fl["cal_tier"].value_counts().to_dict()
    print(f"    Flights: {len(fl):,} total -> {tier_counts}", flush=True)

    fl_cal = fl[fl["cal_tier"].isin(["gold", "silver"])].copy()
    cal_ids = set(fl_cal["track_id"])

    print("2/5 Extracting high-precision thermal sensors from 5 s fixes...", flush=True)
    th_cols = ["track_id", "task_key", "seq", "t_start", "t_end", "alt_base", "alt_top", "agl_base", "agl_top", "turns", "wind_u", "wind_v"]
    th_src = pd.read_parquet(args.db / "thermals.parquet", columns=th_cols)
    th_src = th_src[th_src["track_id"].isin(cal_ids)]
    if task_subset is not None:
        th_src = th_src[th_src["task_key"].isin(task_subset)]

    done_file = parts / "done_tasks.txt"
    done = set(done_file.read_text(encoding="utf-8").split()) if done_file.exists() else set()
    todo_tasks = sorted(set(th_src["task_key"].unique()) - done)
    print(f"    Tasks: {len(done):,} cached, {len(todo_tasks):,} to process ({th_src['task_key'].isin(todo_tasks).sum():,} thermals)", flush=True)

    if todo_tasks:
        fl_by = dict(tuple(fl_cal[fl_cal["task_key"].isin(todo_tasks)].groupby("task_key")))
        th_by = dict(tuple(th_src[th_src["task_key"].isin(todo_tasks)].groupby("task_key")))
        jobs = [
            (k, str(args.db / "fixes" / f"{safe(k)}.parquet"), fl_by[k], th_by[k])
            for k in todo_tasks
            if k in fl_by and k in th_by
        ]
        th_batch: list[dict[str, Any]] = []
        w_batch: list[dict[str, Any]] = []
        keys_batch: list[str] = []
        t_ext = time.time()
        if args.workers <= 1:
            results_iter = (_process_task_job(j) for j in jobs)
            for n, (tk, th_r, w_r) in enumerate(results_iter, 1):
                th_batch.extend(th_r)
                w_batch.extend(w_r)
                keys_batch.append(tk)
                if len(keys_batch) >= 100 or n == len(jobs):
                    _write_batch(parts, th_batch, w_batch, keys_batch)
                    th_batch, w_batch, keys_batch = [], [], []
                if n % 100 == 0 or n == len(jobs):
                    print(f"    ... {n:,}/{len(jobs):,} tasks ({(time.time() - t_ext) / 60:.1f} min)", flush=True)
        else:
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futs = [pool.submit(_process_task_job, j) for j in jobs]
                for n, fut in enumerate(as_completed(futs), 1):
                    tk, th_r, w_r = fut.result()
                    th_batch.extend(th_r)
                    w_batch.extend(w_r)
                    keys_batch.append(tk)
                    if len(keys_batch) >= 100 or n == len(jobs):
                        _write_batch(parts, th_batch, w_batch, keys_batch)
                        th_batch, w_batch, keys_batch = [], [], []
                    if n % 100 == 0 or n == len(jobs):
                        print(f"    ... {n:,}/{len(jobs):,} tasks ({(time.time() - t_ext) / 60:.1f} min)", flush=True)

    print("3/5 Gathering thermal tables and pooling sensors...", flush=True)
    th_df = _gather_parts(parts, "thermals")
    wind_obs = _gather_parts(parts, "wind_obs")
    if len(th_df):
        th_df = th_df.drop_duplicates(["track_id", "seq"], keep="last")
        th_df = th_df[th_df["track_id"].isin(cal_ids)].reset_index(drop=True)
    if len(wind_obs):
        wind_obs = wind_obs.drop_duplicates(["track_id", "seq", "slice_idx"], keep="last")
        wind_obs = wind_obs[wind_obs["track_id"].isin(cal_ids)].reset_index(drop=True)
    th_df.to_parquet(out / "truth_thermals.parquet", index=False)
    print(
        f"    truth_thermals: {len(th_df):,} rows; wind_obs: {len(wind_obs):,} slices "
        f"(QC pass: {wind_obs['wind_qc_ok'].mean():.1%})" if len(wind_obs) else "    truth_thermals: 0 rows",
        flush=True,
    )

    cells = pd.read_parquet(args.db / "cells.parquet", columns=["chip_id", "elev_mean", "elev_max"])
    cells_chip = cells.groupby("chip_id").agg(chip_ground_m=("elev_mean", "mean"), crest_m=("elev_max", "max")).reset_index()

    wind_groups = build_wind_groups(wind_obs, cells_chip, args.db, skip_era5=args.skip_era5_check)
    wind_groups.to_parquet(out / "truth_wind_groups.parquet", index=False)

    glide_wind, glide_metrics = build_glide_wind_and_metrics(args.db, fl_cal, cells_chip, task_subset=task_subset)
    glide_wind.to_parquet(out / "truth_glide_wind.parquet", index=False)

    ceil_df, climb_df = build_ceiling_and_climb_groups(th_df, glide_metrics, cells_chip, args.db, skip_era5=args.skip_era5_check)
    ceil_df.to_parquet(out / "truth_ceiling.parquet", index=False)
    climb_df.to_parquet(out / "truth_climb_groups.parquet", index=False)

    print("4/5 Building daily & task-level physical truth tables...", flush=True)
    daily_df, tasks_df = build_daily_and_tasks(args.db, fl_cal, th_df, wind_groups, ceil_df, climb_df, task_subset=task_subset)
    daily_df.to_parquet(out / "truth_daily.parquet", index=False)
    tasks_df.to_parquet(out / "truth_tasks.parquet", index=False)

    print("5/5 Running diagnostics & optional 1 Hz raw vs 5 s audit...", flush=True)
    audit_1hz = run_1hz_audit(args.db, fl_cal, th_df, out, n_sample=args.check_1hz, root=args.root)
    (out / "audit_1hz_vs_5s.json").write_text(json.dumps(audit_1hz, indent=2), encoding="utf-8")

    # Summary diagnostics: compare old vs clean thermal wind against ERA5 and glide wind.
    summary: dict[str, Any] = {
        "elapsed_s": round(time.time() - t0, 1),
        "flights": {"total": int(len(fl)), "tiers": {k: int(v) for k, v in tier_counts.items()}},
        "counts": {
            "truth_thermals": int(len(th_df)),
            "wind_obs_slices": int(len(wind_obs)),
            "wind_obs_qc_pass": int(wind_obs["wind_qc_ok"].sum()) if len(wind_obs) else 0,
            "truth_wind_groups": int(len(wind_groups)),
            "truth_glide_wind": int(len(glide_wind)),
            "truth_ceiling": int(len(ceil_df)),
            "truth_ceiling_airspace_flagged": int(ceil_df["airspace_cap_flag"].sum()) if len(ceil_df) else 0,
            "truth_climb_groups": int(len(climb_df)),
            "truth_daily": int(len(daily_df)),
            "truth_tasks": int(len(tasks_df)),
        },
        "audit_1hz_vs_5s": audit_1hz,
    }

    if len(wind_groups) and "era5_u" in wind_groups:
        vs_era5: dict[str, Any] = {}
        for band_name, b_list in (
            ("near the ground", ["near the ground"]),
            ("middle", ["middle"]),
            ("free air", ["free air"]),
            ("free+middle", ["free air", "middle"]),
        ):
            sub = wind_groups[wind_groups["band"].isin(b_list)]
            eu, ev = sub["era5_u"].to_numpy(float), sub["era5_v"].to_numpy(float)
            vs_era5[band_name] = {
                "clean_vs_era5": wind_score_dict(sub["clean_u"].to_numpy(float), sub["clean_v"].to_numpy(float), eu, ev),
                "hodo_vs_era5": wind_score_dict(sub["hodo_u"].to_numpy(float), sub["hodo_v"].to_numpy(float), eu, ev),
                "int_vs_era5": wind_score_dict(sub["int_u"].to_numpy(float), sub["int_v"].to_numpy(float), eu, ev),
                "old_vs_era5": wind_score_dict(sub["old_u"].to_numpy(float), sub["old_v"].to_numpy(float), eu, ev),
            }
        summary["wind_vs_era5"] = vs_era5

    if len(wind_groups) and len(glide_wind):
        joined = wind_groups.merge(
            glide_wind[["site_id", "chip_id", "hour", "band", "glide_u", "glide_v"]],
            on=["site_id", "chip_id", "hour", "band"],
            how="inner",
        )
        if len(joined):
            gu, gv = joined["glide_u"].to_numpy(float), joined["glide_v"].to_numpy(float)
            summary["thermal_vs_glide_wind"] = {
                "clean_vs_glide": wind_score_dict(joined["clean_u"].to_numpy(float), joined["clean_v"].to_numpy(float), gu, gv),
                "old_vs_glide": wind_score_dict(joined["old_u"].to_numpy(float), joined["old_v"].to_numpy(float), gu, gv),
                "era5_vs_glide": wind_score_dict(joined["era5_u"].to_numpy(float), joined["era5_v"].to_numpy(float), gu, gv)
                if "era5_u" in joined else {},
            }

    if len(ceil_df) and "era5_blh_asl" in ceil_df:
        c_all = ceil_df.dropna(subset=["ceiling_p90_asl", "era5_blh_asl"])
        c_nocap = c_all[~c_all["airspace_cap_flag"]]
        summary["ceiling_vs_era5"] = {
            "all_chip_hours": int(len(c_all)),
            "nocap_chip_hours": int(len(c_nocap)),
            "corr_asl_blh_all": round(float(c_all["ceiling_p90_asl"].corr(c_all["era5_blh_asl"])), 4) if len(c_all) > 2 else None,
            "corr_asl_blh_nocap": round(float(c_nocap["ceiling_p90_asl"].corr(c_nocap["era5_blh_asl"])), 4) if len(c_nocap) > 2 else None,
            "corr_above_era5_nocap": round(float(c_nocap["ceiling_above_era5"].corr(c_nocap["blh_m"])), 4) if len(c_nocap) > 2 else None,
            "corr_min_blh_cb_nocap": round(float(c_nocap["ceiling_p90_asl"].corr(c_nocap["era5_min_blh_cb_asl"])), 4) if len(c_nocap) > 2 else None,
            "median_diff_blh_minus_ceil_m": round(float((c_nocap["era5_blh_asl"] - c_nocap["ceiling_p90_asl"]).median()), 1) if len(c_nocap) else None,
        }

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved IGC truth tables to {out} in {summary['elapsed_s']:.1f} s", flush=True)
    print(json.dumps(summary["counts"], indent=2), flush=True)
    if "wind_vs_era5" in summary and "free+middle" in summary["wind_vs_era5"]:
        fm = summary["wind_vs_era5"]["free+middle"]
        print("\nFree+middle wind vs ERA5 (same groups):", flush=True)
        for k, m in fm.items():
            if m.get("groups"):
                print(f"  {k:16s}: RMSE={m['vector_rmse_ms']:.3f} m/s  bias={m['speed_bias_ms']:+.3f} m/s  dir={m['dir_error_deg']} deg  <30deg={m['dir_within_30']:.1%}", flush=True)
    if not audit_1hz.get("skipped"):
        print(f"\n1 Hz vs 5 s audit verdict: {audit_1hz.get('verdict')} ({audit_1hz.get('n_thermals_matched')} thermals)", flush=True)


if __name__ == "__main__":
    main()
