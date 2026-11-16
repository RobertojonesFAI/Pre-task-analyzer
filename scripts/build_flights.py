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
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pta import flight, igc  # noqa: E402

TABLES = ("flights", "climbs", "glides")


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


def gather(parts, table, keys):
    frames = [pd.read_parquet(parts / f"{safe(k)}.{table}.parquet") for k in keys]
    frames = [f.dropna(axis=1, how="all") for f in frames if len(f)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder written by build_catalog.py")
    parser.add_argument("--workers", type=int, default=2, help="processes analysing tracks (default 2)")
    parser.add_argument("--redo", action="store_true", help="discard the cache and analyse every track again")
    parser.add_argument("--retry-errors", action="store_true",
                        help="analyse again the tasks with tracks that failed with an error (not those with no flight)")
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
        retried = 0
        for j in jobs:
            path = parts / f"{safe(j[0])}.flights.parquet"
            if path.exists():
                err = pd.read_parquet(path, columns=None).get("error")
                if err is not None and (err.notna() & (err != "no flight found")).any():
                    (parts / f"{safe(j[0])}.done").unlink(missing_ok=True)
                    retried += 1
        print(f"Retrying {retried} tasks with errors", flush=True)
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

    keys = [j[0] for j in jobs if (parts / f"{safe(j[0])}.done").exists()]
    links = unique[["track_id", "result_id", "pilot_key", "source"]]
    flights = gather(parts, "flights", keys).merge(links, on="track_id", how="left")
    climbs, glides = gather(parts, "climbs", keys), gather(parts, "glides", keys)
    for name, df in [("flights", flights), ("climbs", climbs), ("glides", glides)]:
        df.to_parquet(args.db / f"{name}.parquet", index=False)

    ok = flights["error"].isna() if "error" in flights else pd.Series(True, index=flights.index)
    print("\nTables written to", args.db)
    print(f"  flights     {int(ok.sum()):>8,}  (of {len(flights):,} tracks; {int((~ok).sum()):,} without a flight)")
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
