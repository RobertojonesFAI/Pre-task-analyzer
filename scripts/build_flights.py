#!/usr/bin/env python3
"""Analyse every unique competition track: the flight, its climbs and glides, and its fixes.

Reads the catalog tables in DB (built by build_catalog.py) and the task packages they point to,
and writes to DB:

    flights      one row per unique track: takeoff, landing, duration, climb statistics, or why no
                 flight was found in it
    climbs       one row per circling climb (thermals and weak circling)
    glides       one row per stretch between climbs
    fixes/       one Parquet file per task with every flight's fixes at 5 s (time, position,
                 altitude, vertical speed, circling)

Each task is cached in DB/_flight_parts, so a second run only reads tasks it has not seen.
--redo discards the cache (after the flight analysis itself changes).

    python3 scripts/build_flights.py /content/drive/MyDrive/IGCS/database --root /content/drive/MyDrive/IGCS
"""

import argparse
import shutil
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pta import flight, igc  # noqa: E402

TABLES = ("flights", "climbs", "glides")
NOT_ERRORS = ("no flight found", "no valid fixes")  # what the track holds, not a failure to retry


def safe(task_key):
    return task_key.replace(":", "_")


def relocate(path, root):
    """The package path under root: the catalog keeps the paths of the machine that built it, and
    Drive may be mounted elsewhere now."""
    if root is None:
        return path
    p = Path(path)
    for i, part in enumerate(p.parts):
        if part in ("flymaster_igc", "airscore_igc", "airtribune_igc"):
            return str(root.joinpath(*p.parts[i:]))
    return path


def analyse_task(job):
    """Analyse the tracks of one task package and write its parts. Returns (task_key, counts)."""
    task_key, zip_path, members, parts, fixes_dir = job
    flights, climbs, glides, kept = [], [], [], []
    try:
        z = zipfile.ZipFile(zip_path)
    except (OSError, zipfile.BadZipFile) as e:
        z, error = None, f"package: {type(e).__name__}"
    for track_id, member in members:
        row = {"track_id": track_id, "task_key": task_key}
        if z is None:
            flights.append({**row, "error": error})
            continue
        try:
            fx = igc.fixes(z.read(member))
            if len(fx["t"]) == 0:
                flights.append({**row, "error": "no valid fixes"})
                continue
            f, c, g, k = flight.analyse(fx)
        except Exception as e:  # a damaged file must not stop the run
            flights.append({**row, "error": f"{type(e).__name__}: {str(e)[:150]}"})
            continue
        if f is None:
            flights.append({**row, "error": "no flight found"})
            continue
        flights.append({**row, **f})
        climbs += [{**row, "seq": i, **x} for i, x in enumerate(c)]
        glides += [{**row, "seq": i, **x} for i, x in enumerate(g)]
        kept.append(pd.DataFrame({"track_id": track_id, **k}))

    name = safe(task_key)
    for table, rows in zip(TABLES, (flights, climbs, glides)):
        df = pd.DataFrame(rows) if rows else pd.DataFrame({"track_id": pd.Series(dtype=str),
                                                           "task_key": pd.Series(dtype=str)})
        tmp = parts / f"{name}.{table}.tmp"
        df.to_parquet(tmp, index=False)
        tmp.rename(parts / f"{name}.{table}.parquet")
    if kept:
        tmp = fixes_dir / f"{name}.tmp"
        pd.concat(kept, ignore_index=True).to_parquet(tmp, index=False)
        tmp.rename(fixes_dir / f"{name}.parquet")
    # written last: a task counts as done only when all its parts are there
    (parts / f"{name}.done").write_text(str(len(members)))
    return task_key, len(members), sum("error" not in r for r in flights)


SOURCE_PRIORITY = {"airscore": 0, "flymaster": 1, "airtribune": 2, "civlcomps": 3}
SAME_START_S = 600       # candidate copies of one flight: takeoffs this close in time...
SAME_START_M = 1000      # ...and space,
SAME_END_S = 900         # and landings this close
SAME_END_M = 2000
SAME_TRACK_M = 100       # the same flight: median distance between the two tracks over their common time
SAME_OVERLAP = 0.5       # (overlapping for at least half of the shorter flight)


def _track_distance(a, b):
    """Median distance (m) between two tracks over their common time, and the share of the shorter
    one that the common time covers. a and b are DataFrames with t, lat, lon (5 s fixes)."""
    t0, t1 = max(a["t"].iloc[0], b["t"].iloc[0]), min(a["t"].iloc[-1], b["t"].iloc[-1])
    shorter = min(a["t"].iloc[-1] - a["t"].iloc[0], b["t"].iloc[-1] - b["t"].iloc[0])
    if t1 <= t0 or shorter <= 0:
        return None, 0.0
    t = a["t"].to_numpy()
    t = t[(t >= t0) & (t <= t1)]
    if len(t) < 10:
        return None, 0.0
    bt = b["t"].to_numpy()
    lat_b, lon_b = np.interp(t, bt, b["lat"].to_numpy()), np.interp(t, bt, b["lon"].to_numpy())
    sel = (a["t"] >= t0) & (a["t"] <= t1)
    d = flight._dist(a.loc[sel, "lat"].to_numpy(), a.loc[sel, "lon"].to_numpy(), lat_b, lon_b)
    return float(np.median(d)), (t1 - t0) / shorter


def mark_same_flights(flights, tasks, fixes_dir, threads=16):
    """Copies of one flight in different files: the same competition published by two sources (a
    live-tracker file on one, the pilot's instrument on the other) or a pilot's second instrument.
    Their fixes differ, so the catalog's hash does not catch them. Candidates take off and land
    close together; they are the same flight when the two tracks stay within SAME_TRACK_M of each
    other (median) over most of their common time, which pilots flying together in a gaggle do
    not, and are not known to be two different pilots. Returns ({track_id: track_id of the copy
    kept}, track ids that share one path with other pilots' tracks). The second are mostly pilots
    who did not fly and whose trackers recorded the drive down from launch in one car; the terrain
    step checks their height above the ground."""
    f = flights[flights["error"].isna()].copy() if "error" in flights else flights.copy()
    f["day"] = pd.to_datetime(f["task_key"].map(tasks.set_index("task_key")["date"]), errors="coerce")
    f = f.dropna(subset=["day", "t_takeoff", "t_landing"])
    f["start"] = f["t_takeoff"] % 86400
    f = f.sort_values("start").reset_index(drop=True)
    start, day = f["start"].to_numpy(), f["day"].to_numpy()
    lat0, lon0, lat1, lon1 = (f[c].to_numpy() for c in ("lat_takeoff", "lon_takeoff", "lat_landing", "lon_landing"))
    end = f["t_landing"].to_numpy() - f["t_takeoff"].to_numpy() + start
    pairs = []
    for i in range(len(f)):
        j = i + 1
        while j < len(f) and start[j] - start[i] <= SAME_START_S:
            if (abs((day[j] - day[i]) / np.timedelta64(1, "D")) <= 1
                    and abs(end[j] - end[i]) <= SAME_END_S
                    and flight._dist(lat0[i], lon0[i], lat0[j], lon0[j]) <= SAME_START_M
                    and flight._dist(lat1[i], lon1[i], lat1[j], lon1[j]) <= SAME_END_M):
                pairs.append((i, j))
            j += 1
    if not pairs:
        return {}, set()

    tasks_needed = sorted({f.at[k, "task_key"] for p in pairs for k in p})
    wanted = {f.at[k, "track_id"] for p in pairs for k in p}

    def load(task_key):
        path = fixes_dir / f"{safe(task_key)}.parquet"
        if not path.exists():
            return {}
        fx = pd.read_parquet(path, columns=["track_id", "t", "lat", "lon"])
        fx = fx[fx["track_id"].isin(wanted)]
        return {k: g.reset_index(drop=True) for k, g in fx.groupby("track_id")}

    tracks = {}
    with ThreadPoolExecutor(threads) as pool:
        for d in pool.map(load, tasks_needed):
            tracks.update(d)

    parent = {}

    def find(x):
        while parent.get(x, x) != x:
            x = parent[x]
        return x

    def different_pilots(i, j):
        """Two pilots, known to be different: two results of one task, or two CIVL IDs (pilots
        driven down from launch in one car record the same path)."""
        ri, rj = f.at[i, "result_id"], f.at[j, "result_id"]
        if f.at[i, "task_key"] == f.at[j, "task_key"] and pd.notna(ri) and pd.notna(rj) and ri != rj:
            return True
        pi, pj = str(f.at[i, "pilot_key"]), str(f.at[j, "pilot_key"])
        return pi.startswith("civl:") and pj.startswith("civl:") and pi != pj

    for i, j in pairs:
        if different_pilots(i, j):
            continue
        a, b = tracks.get(f.at[i, "track_id"]), tracks.get(f.at[j, "track_id"])
        if a is None or b is None:
            continue
        dist, overlap = _track_distance(a, b)
        if dist is not None and dist <= SAME_TRACK_M and overlap >= SAME_OVERLAP:
            parent[find(f.at[i, "track_id"])] = find(f.at[j, "track_id"])

    groups = {}
    for tid in list(parent) + [p for p in parent.values()]:
        groups.setdefault(find(tid), set()).add(tid)
    info = f.set_index("track_id")
    keep_order = lambda tid: (pd.isna(info.at[tid, "result_id"]), info.at[tid, "step_s"],  # noqa: E731
                              SOURCE_PRIORITY.get(info.at[tid, "source"], 9), tid)
    same, shared = {}, set()
    for members in groups.values():
        pilots = {str(info.at[t, "pilot_key"]) for t in members if pd.notna(info.at[t, "pilot_key"])}
        results = {info.at[t, "result_id"] for t in members if pd.notna(info.at[t, "result_id"])}
        if len(pilots) > 1 or len(results) > 1 and len({info.at[t, "task_key"] for t in members}) == 1:
            shared |= members  # several pilots on one path: driven down from launch together, not copies
            continue
        kept = min(members, key=keep_order)
        same.update({tid: kept for tid in members if tid != kept})
    return same, shared


def gather(parts, table, keys, threads=16):
    """One table from the per-task parts. Reading thousands of small files from a Drive mount is
    slow one at a time but fast in parallel, so they are read by a pool of threads."""
    started = time.time()
    with ThreadPoolExecutor(threads) as pool:
        frames = list(pool.map(lambda k: pd.read_parquet(parts / f"{safe(k)}.{table}.parquet"), keys))
    frames = [f.dropna(axis=1, how="all") for f in frames if len(f)]
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    print(f"  read {table}: {len(out):,} rows from {len(keys):,} tasks, {time.time() - started:.0f} s", flush=True)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder written by build_catalog.py")
    parser.add_argument("--workers", type=int, default=2, help="processes analysing tracks (default 2)")
    parser.add_argument("--redo", action="store_true", help="discard the cache and analyse every track again")
    parser.add_argument("--retry-errors", action="store_true",
                        help="analyse again the tasks with tracks that failed with an error (not tracks with no flight or no fixes)")
    parser.add_argument("--root", type=Path, help="folder holding flymaster_igc, airscore_igc and airtribune_igc, "
                        "if it is not where it was when the catalog was built")
    parser.add_argument("--limit", type=int, help="analyse only this many tasks (for a test)")
    args = parser.parse_args()
    parts, fixes_dir = args.db / "_flight_parts", args.db / "fixes"
    if args.redo:
        shutil.rmtree(parts, ignore_errors=True)
        shutil.rmtree(fixes_dir, ignore_errors=True)
    parts.mkdir(parents=True, exist_ok=True)
    fixes_dir.mkdir(parents=True, exist_ok=True)

    tracks = pd.read_parquet(args.db / "tracks.parquet")
    unique = tracks[tracks["duplicate_of"].isna() & tracks["member"].notna()]
    if "error" in unique:
        unique = unique[unique["error"].isna()]
    jobs = [(task_key, relocate(g["zip_path"].iloc[0], args.root), list(zip(g["track_id"], g["member"])), parts, fixes_dir)
            for task_key, g in unique.groupby("task_key", sort=True)]
    if args.limit:
        jobs = jobs[:args.limit]
    if args.retry_errors:
        def failed(job):
            path = parts / f"{safe(job[0])}.flights.parquet"
            if not path.exists():
                return False
            err = pd.read_parquet(path).get("error")
            return err is not None and bool((err.notna() & ~err.isin(NOT_ERRORS)).any())

        with ThreadPoolExecutor(16) as pool:
            retry = [j for j, bad in zip(jobs, pool.map(failed, jobs)) if bad]
        for j in retry:
            (parts / f"{safe(j[0])}.done").unlink(missing_ok=True)
        print(f"Retrying {len(retry)} tasks with errors", flush=True)
    todo = [j for j in jobs if not (parts / f"{safe(j[0])}.done").exists()]
    print(f"Flights: {len(jobs) - len(todo)} tasks already analysed, {len(todo)} to analyse "
          f"({sum(len(j[2]) for j in todo):,} tracks)", flush=True)

    started, n_tracks, n_flights = time.time(), 0, 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(analyse_task, j) for j in todo]
        for n, future in enumerate(as_completed(futures), 1):
            try:
                _, tracks_done, flights_done = future.result()
            except Exception as e:  # e.g. Drive hiccup writing a part; the next run retries the task
                print(f"    a task failed: {type(e).__name__}: {e}", flush=True)
                continue
            n_tracks, n_flights = n_tracks + tracks_done, n_flights + flights_done
            if n % 25 == 0 or n == len(todo):
                print(f"... {n}/{len(todo)} tasks, {n_tracks:,} tracks, {n_flights:,} flights, "
                      f"{(time.time() - started) / 60:.0f} min", flush=True)

    print("\nCollecting the tables from the task parts", flush=True)
    keys = [j[0] for j in jobs if (parts / f"{safe(j[0])}.done").exists()]
    links = unique[["track_id", "result_id", "pilot_key", "source"]]
    # inner join: tracks dropped from the catalog since a task was analysed are left out
    flights = gather(parts, "flights", keys).merge(links, on="track_id", how="inner")
    climbs, glides = gather(parts, "climbs", keys), gather(parts, "glides", keys)
    if len(climbs):
        climbs = climbs[climbs["track_id"].isin(links["track_id"])]
    if len(glides):
        glides = glides[glides["track_id"].isin(links["track_id"])]
    started = time.time()
    same, shared = mark_same_flights(flights, pd.read_parquet(args.db / "tasks.parquet"), fixes_dir)
    flights["same_flight_as"] = flights["track_id"].map(same)
    flights["shared_path"] = flights["track_id"].isin(shared)
    print(f"  same flight in two files: {len(same):,} copies marked; {len(shared):,} tracks share one path "
          f"with other pilots (likely a car ride), {time.time() - started:.0f} s", flush=True)
    for name, df in [("flights", flights), ("climbs", climbs), ("glides", glides)]:
        df.to_parquet(args.db / f"{name}.parquet", index=False)
        print(f"  wrote {name}.parquet", flush=True)

    ok = flights["error"].isna() if "error" in flights else pd.Series(True, index=flights.index)
    print("\nTables written to", args.db)
    print(f"  flights     {int(ok.sum()):>8,}  (of {len(flights):,} tracks; {int((~ok).sum()):,} without a flight; "
          f"{int(flights['same_flight_as'].notna().sum()):,} copies of another file's flight)")
    if len(climbs):
        print(f"  climbs      {len(climbs):>8,}  ({int(climbs['is_thermal'].sum()):,} thermals)")
    print(f"  glides      {len(glides):>8,}")
    print(f"  fixes       {len(list(fixes_dir.glob('*.parquet'))):>8,}  task files")
    if ok.any():
        f = flights[ok]
        print(f"\nMedian flight {f['duration_s'].median() / 3600:.1f} h, "
              f"{f['n_thermals'].median():.0f} thermals, "
              f"climb rate {f['mean_thermal_rate_ms'].median():.1f} m/s, "
              f"circling {f['circling_share'].median():.0%} of the time")
        print("Quality (fix interval):", f["quality"].value_counts().to_dict())
    if (~ok).any():
        why = flights.loc[~ok, "error"].str.split(":").str[0]
        print("Tracks without a flight:", why.value_counts().head(5).to_dict())
        print(pd.crosstab(flights.loc[~ok, "source"], why).to_string())


if __name__ == "__main__":
    main()
