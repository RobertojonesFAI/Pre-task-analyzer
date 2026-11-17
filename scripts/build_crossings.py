#!/usr/bin/env python3
"""Labels for the thermal and landout models: every flight's height above ground, the cells it
crossed, its thermals with the wind the pilot measured and where each thermal started, and where
it landed.

Reads the flights (notebook 05: flights, climbs and the 5 s fixes) and the GLO-30 tiles (notebook
06), and writes to DB:

    flight_checks   one row per flight: height above ground (median, share near the ground) and
                    whether it is really a drive (a "flight" that stays on the ground)
    crossings       one row per flight and cell it crossed: when, how long, height above ground in
                    and out, time circling and climbing there, best climb rate
    thermals        one row per thermal: base and top above the ground, the wind the pilot measured
                    (the drift of the circles), and the trigger point traced back upwind
    landings        one row per flight: landing cell, time and height above ground, with the
                    pilot's official status (goal, ess, lo)

Copies of another file's flight (same_flight_as) are left out. Each task is cached in
DB/_crossing_parts (in batches), so a second run only does what is missing.

    python3 scripts/build_crossings.py /content/drive/MyDrive/IGCS/database \\
        --dem /content/drive/MyDrive/IGCS/dem/glo30 --local /content/dem_glo30 --workers 8
"""

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pta import grid, terrain  # noqa: E402

TABLES = ("flight_checks", "crossings", "thermals", "landings")
NEAR_GROUND_M = 50        # a fix this close to the ground counts as on the ground
DRIVE_SHARE = 0.6         # a "flight" with this share of its fixes on the ground is a drive
CLIMB_VZ = 0.5            # m/s: circling at least this fast counts as climbing
MIN_WIND_TURNS = 2        # a thermal needs this many turns for its drift to measure the wind
MAX_RISE_S = 1800         # tracing a thermal back: at most this long from the ground to its base
MIN_WIND_S = 60           # ...and at least this long
MAX_WIND_MS = 30          # a drift faster than this is a GPS or logging error, not wind


def safe(task_key):
    return task_key.replace(":", "_")


def analyse_task(job):
    """All four tables for one task's flights."""
    task_key, fixes_path, flights, climbs, dem_dir = job
    out = {t: [] for t in TABLES}
    if not Path(fixes_path).exists():
        return task_key, out
    fx = pd.read_parquet(fixes_path)
    fx = fx[fx["track_id"].isin(flights["track_id"])]
    if not len(fx):
        return task_key, out
    dem = terrain.DEM(dem_dir)
    fx["ground"] = dem.sample(fx["lat"].to_numpy(), fx["lon"].to_numpy())
    fx["agl"] = fx["alt"].astype("float32") - fx["ground"]
    fx["cell_id"] = grid.cell_id(fx["lat"].to_numpy(), fx["lon"].to_numpy())
    by_track = dict(tuple(fx.groupby("track_id", sort=False)))
    climbs_by = dict(tuple(climbs.groupby("track_id", sort=False))) if len(climbs) else {}

    for f in flights.itertuples(index=False):
        g = by_track.get(f.track_id)
        if g is None or not len(g):
            continue
        g = g.sort_values("t")
        agl = g["agl"].to_numpy()
        known = np.isfinite(agl)
        near = float((agl[known] < NEAR_GROUND_M).mean()) if known.any() else np.nan
        is_drive = bool(known.any() and near >= DRIVE_SHARE)
        out["flight_checks"].append({
            "track_id": f.track_id, "task_key": task_key,
            "agl_median": float(np.nanmedian(agl)) if known.any() else np.nan,
            "agl_p10": float(np.nanpercentile(agl, 10)) if known.any() else np.nan,
            "near_ground_share": near, "is_drive": is_drive, "dem_coverage": float(known.mean()),
        })

        # landing
        last = g.iloc[-1]
        out["landings"].append({
            "track_id": f.track_id, "task_key": task_key, "result_id": f.result_id,
            "t": int(last["t"]), "lat": float(last["lat"]), "lon": float(last["lon"]),
            "cell_id": int(last["cell_id"]), "agl": float(last["agl"]) if np.isfinite(last["agl"]) else np.nan,
            "is_drive": is_drive,
        })
        if is_drive:
            continue

        # cell crossings: runs of fixes in the same cell
        cell = g["cell_id"].to_numpy()
        start = np.r_[0, np.flatnonzero(cell[1:] != cell[:-1]) + 1]
        end = np.r_[start[1:], len(g)]
        t = g["t"].to_numpy()
        circ = g["circling"].to_numpy()
        vz = g["vz"].to_numpy()
        climbing = circ & (vz >= CLIMB_VZ)
        step = float(np.median(np.diff(t))) if len(t) > 1 else 5.0
        for a, b in zip(start, end):
            seg_agl = agl[a:b]
            out["crossings"].append({
                "track_id": f.track_id, "task_key": task_key, "cell_id": int(cell[a]),
                "t_in": int(t[a]), "t_out": int(t[b - 1] + step), "duration_s": float((b - a) * step),
                "agl_in": float(seg_agl[0]), "agl_out": float(seg_agl[-1]),
                "agl_min": float(np.nanmin(seg_agl)) if np.isfinite(seg_agl).any() else np.nan,
                "circling_s": float(circ[a:b].sum() * step), "climbing_s": float(climbing[a:b].sum() * step),
                "best_vz": float(np.nanmax(vz[a:b])),
                "mean_vz_climbing": float(vz[a:b][climbing[a:b]].mean()) if climbing[a:b].any() else np.nan,
            })

        # thermals: wind from the drift of the circles, base and top above ground, trigger point
        for c in climbs_by.get(f.track_id, pd.DataFrame()).itertuples(index=False):
            if not c.is_thermal:
                continue
            seg = g[(g["t"] >= c.t_start) & (g["t"] <= c.t_end)]
            if len(seg) < 3:
                continue
            ground_base = float(seg["ground"].iloc[0])
            row = {"track_id": f.track_id, "task_key": task_key, "seq": int(c.seq), "t_start": int(c.t_start),
                   "t_end": int(c.t_end), "lat": float(c.lat), "lon": float(c.lon), "cell_id": int(grid.cell_id(c.lat, c.lon)),
                   "alt_base": float(c.alt_base), "alt_top": float(c.alt_top),
                   "agl_base": float(c.alt_base - ground_base) if np.isfinite(ground_base) else np.nan,
                   "agl_top": float(c.alt_top - seg["ground"].iloc[-1]) if np.isfinite(seg["ground"].iloc[-1]) else np.nan,
                   "rate_ms": float(c.rate_ms), "turns": float(c.turns),
                   "wind_u": np.nan, "wind_v": np.nan, "wind_alt": np.nan,
                   "trigger_lat": np.nan, "trigger_lon": np.nan, "trigger_cell": None}
            if c.turns >= MIN_WIND_TURNS:
                # over whole turns the glider's own speed cancels out: what is left is the wind
                k = np.pi / 180
                dy = (seg["lat"].iloc[-1] - seg["lat"].iloc[0]) * 111320.0
                dx = (seg["lon"].iloc[-1] - seg["lon"].iloc[0]) * 111320.0 * np.cos(c.lat * k)
                dt = float(seg["t"].iloc[-1] - seg["t"].iloc[0])
                if dt >= MIN_WIND_S and np.hypot(dx, dy) / dt <= MAX_WIND_MS:
                    u, v = dx / dt, dy / dt
                    row.update(wind_u=u, wind_v=v, wind_alt=float(seg["alt"].mean()))
                    if np.isfinite(row["agl_base"]) and c.rate_ms > 0:
                        rise = min(max(row["agl_base"], 0.0) / c.rate_ms, MAX_RISE_S)
                        lat0 = float(seg["lat"].iloc[0]) - v * rise / 111320.0
                        lon0 = float(seg["lon"].iloc[0]) - u * rise / (111320.0 * np.cos(c.lat * k))
                        row.update(trigger_lat=lat0, trigger_lon=lon0, trigger_cell=int(grid.cell_id(lat0, lon0)))
            out["thermals"].append(row)
    return task_key, out


def write_batch(parts, batch, keys):
    """One part file per table for many tasks, then the tasks are added to done_tasks.txt."""
    try:
        name = f"batch_{int(time.time() * 1000)}"
        for table in TABLES:
            rows = [r for b in batch for r in b[table]]
            if rows:
                tmp = parts / f"{name}.{table}.tmp"
                pd.DataFrame(rows).to_parquet(tmp, index=False)
                tmp.rename(parts / f"{name}.{table}.parquet")
        with open(parts / "done_tasks.txt", "a") as f:
            f.write("".join(f"{k}\n" for k in keys))
    except OSError as e:
        sys.exit(f"\nCould not write to {parts} ({e}).\nIf Google Drive disconnected, run the Drive cell "
                 "again and then this cell: finished tasks are kept.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder")
    parser.add_argument("--dem", type=Path, required=True, help="folder with the GLO-30 tiles")
    parser.add_argument("--local", type=Path, help="local folder to copy the tiles to first (faster than a Drive mount)")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--redo", action="store_true", help="discard the cache and start again")
    args = parser.parse_args()
    parts = args.db / "_crossing_parts"
    if args.redo:
        shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir(parents=True, exist_ok=True)

    flights = pd.read_parquet(args.db / "flights.parquet")
    ok = flights["error"].isna() if "error" in flights else pd.Series(True, index=flights.index)
    if "same_flight_as" in flights:
        ok &= flights["same_flight_as"].isna()
    flights = flights[ok][["track_id", "task_key", "result_id"]]
    climbs = pd.read_parquet(args.db / "climbs.parquet")
    climbs = climbs[climbs["is_thermal"] & climbs["track_id"].isin(flights["track_id"])]

    done_file = parts / "done_tasks.txt"
    done = set(done_file.read_text().split()) if done_file.exists() else set()
    tasks = sorted(set(flights["task_key"]) - done)
    print(f"Crossings: {len(done):,} tasks done before, {len(tasks):,} to do "
          f"({flights['task_key'].isin(tasks).sum():,} flights)", flush=True)

    dem_dir = args.dem
    if args.local and tasks:
        # copy the tiles in parallel (one at a time from a Drive mount is very slow), skipping
        # those already copied, and only the tiles under the flights
        args.local.mkdir(parents=True, exist_ok=True)
        chips = pd.read_parquet(args.db / "site_chips.parquet")["chip_id"].unique()
        names = {terrain.tile_name(*t) + ".tif" for c in chips for t in terrain.tiles_for_box(*grid.chip_bounds(c))}
        todo_tiles = [args.dem / n for n in sorted(names) if (args.dem / n).exists()
                      and not ((args.local / n).exists() and (args.local / n).stat().st_size == (args.dem / n).stat().st_size)]
        print(f"Copying {len(todo_tiles)} tiles to {args.local} ({len(names)} needed)", flush=True)

        def copy(src):
            tmp = args.local / (src.name + ".part")
            shutil.copyfile(src, tmp)
            tmp.rename(args.local / src.name)

        started = time.time()
        with ThreadPoolExecutor(16) as pool:
            for n, _ in enumerate(pool.map(copy, todo_tiles), 1):
                if n % 50 == 0 or n == len(todo_tiles):
                    print(f"... {n}/{len(todo_tiles)} tiles, {(time.time() - started) / 60:.0f} min", flush=True)
        dem_dir = args.local

    f_by, c_by = dict(tuple(flights.groupby("task_key"))), dict(tuple(climbs.groupby("task_key")))
    jobs = [(k, str(args.db / "fixes" / f"{safe(k)}.parquet"), f_by[k], c_by.get(k, climbs.iloc[:0]), dem_dir)
            for k in tasks]
    started, batch, keys = time.time(), [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(analyse_task, j) for j in jobs]
        for n, f in enumerate(as_completed(futures), 1):
            key, out = f.result()
            batch.append(out)
            keys.append(key)
            if len(keys) >= 100 or n == len(jobs):
                write_batch(parts, batch, keys)
                batch, keys = [], []
            if n % 100 == 0 or n == len(jobs):
                print(f"... {n:,}/{len(jobs):,} tasks, {(time.time() - started) / 60:.0f} min", flush=True)

    print("\nCollecting the tables", flush=True)
    for table in TABLES:
        files = sorted(parts.glob(f"batch_*.{table}.parquet"))
        with ThreadPoolExecutor(16) as pool:
            frames = [x for x in pool.map(pd.read_parquet, files) if len(x)]
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if len(df):
            key = ["track_id", "seq"] if table == "thermals" else ["track_id", "cell_id", "t_in"] if table == "crossings" else ["track_id"]
            df = df.drop_duplicates(key, keep="last")
            df = df[df["track_id"].isin(flights["track_id"])]
        df.to_parquet(args.db / f"{table}.parquet", index=False)
        print(f"  {table:<14}{len(df):>12,}", flush=True)

    checks = pd.read_parquet(args.db / "flight_checks.parquet")
    th = pd.read_parquet(args.db / "thermals.parquet")
    land = pd.read_parquet(args.db / "landings.parquet")
    print(f"\nDrives found (a 'flight' on the ground): {int(checks['is_drive'].sum()):,} of {len(checks):,}; "
          f"DEM coverage {checks['dem_coverage'].mean():.1%}")
    print(f"Median height above ground in flight: {checks.loc[~checks['is_drive'], 'agl_median'].median():.0f} m")
    if len(th):
        print(f"Thermals: base {th['agl_base'].median():.0f} m and top {th['agl_top'].median():.0f} m above ground "
              f"(median); wind measured in {th['wind_u'].notna().mean():.0%} of them, "
              f"median {np.hypot(th['wind_u'], th['wind_v']).median():.1f} m/s")
    if len(land):
        print(f"Landings: median {land.loc[~land['is_drive'], 'agl'].median():.0f} m above ground (should be close to 0)")


if __name__ == "__main__":
    main()
