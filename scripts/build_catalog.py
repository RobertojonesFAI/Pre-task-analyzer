#!/usr/bin/env python3
"""Build the catalog tables of the competition database from the downloaded files.

Reads ROOT/flymaster_igc, ROOT/airscore_igc and ROOT/airtribune_igc (as written by the download
scripts) and writes Parquet tables to OUT:

    events       one row per competition
    tasks        one row per task (turnpoints as JSON too)
    turnpoints   one row per turnpoint of a task
    results      one row per pilot in a task's official results
    pilots       one row per pilot (CIVL ID when known, otherwise name + nation)
    tracks       one row per IGC file: where it is, header fields, a fix summary, the result it
                 belongs to, and the track it duplicates (same fixes) if any

Reading the IGC files is the slow part. Each task's track rows are cached in OUT/_parts, so a
second run only reads tasks it has not seen.

    python3 scripts/build_catalog.py /content/drive/MyDrive/IGCS --out /content/drive/MyDrive/IGCS/database
"""

import argparse
import csv
import difflib
import hashlib
import json
import sys
import time
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pta import igc, sources  # noqa: E402
from pta.util import norm_name  # noqa: E402

SOURCE_PRIORITY = {"airscore": 0, "flymaster": 1, "airtribune": 2}  # which copy of a duplicate to keep
TRACK_ID = {"flymaster": sources.flymaster_track_id, "airtribune": sources.airtribune_track_id,
            "airscore": lambda member: None}


def clean(df):
    """Make object columns with mixed value types plain strings, so Parquet can store them."""
    for c in df.columns:
        if df[c].dtype == object:
            kinds = {type(v) for v in df[c].dropna()}
            if len(kinds) > 1 or (kinds and kinds != {str} and kinds != {bool}):
                df[c] = df[c].map(lambda v: None if v is None or (isinstance(v, float) and pd.isna(v)) else str(v))
    return df


def parse(fn, *args):
    """Run one task parser; a page that cannot be read gives a task without results instead of
    stopping the build."""
    try:
        return fn(*args)
    except Exception as e:
        print(f"    could not read the results of {fn.__name__}{args[:2]}: {type(e).__name__}: {e}", flush=True)
        return None


def read_csv(path):
    if not Path(path).exists():
        return []
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ── Tasks and results ───────────────────────────────────────────────────────

def collect_tasks(root):
    """(events, tasks, results, jobs) for every task of the three sources that has an IGC package."""
    events, tasks, results, jobs = [], [], [], []

    fm = root / "flymaster_igc"
    by_group = defaultdict(list)
    for r in read_csv(fm / "flymaster_catalog.csv"):
        if r.get("discipline") == "PG" and r.get("zip") == "1" and (fm / r["group"] / f"{r['task']}.zip").exists():
            by_group[r["group"]].append(r)
    for group, rows in by_group.items():
        first = None
        for r in rows:
            task, res = parse(sources.flymaster_task, group, r["task"], fm / group, r) or \
                sources.flymaster_task(group, r["task"], Path("/nonexistent"), r)
            tasks.append(task)
            results += res
            jobs.append(("flymaster", task["task_key"], str(fm / group / f"{r['task']}.zip")))
            first = first or task
        events.append(sources.flymaster_event(group, rows, first))
    print(f"Flymaster: {len(by_group)} competitions, {sum(map(len, by_group.values()))} tasks", flush=True)

    asc = root / "airscore_igc"
    comps = {}
    if (asc / "airscore_comps.json").exists():
        data = json.loads((asc / "airscore_comps.json").read_text())
        comps = {str(c["comp_id"]): c for c in data.get("data", data)}
    by_comp = defaultdict(list)
    for r in read_csv(asc / "airscore_catalog.csv"):
        if (asc / r["comp_id"] / f"{r['task']}.zip").exists():
            by_comp[r["comp_id"]].append(r)
    for comp_id, rows in by_comp.items():
        first = None
        for r in rows:
            task, res = parse(sources.airscore_task, comp_id, r["task"], asc / comp_id) or \
                sources.airscore_task(comp_id, r["task"], Path("/nonexistent"))
            tasks.append(task)
            results += res
            jobs.append(("airscore", task["task_key"], str(asc / comp_id / f"{r['task']}.zip")))
            first = first or task
        events.append(sources.airscore_event(comps.get(comp_id, {"comp_id": comp_id, "comp_name": rows[0]["comp_name"]}), first))
    print(f"AirScore: {len(by_comp)} competitions, {sum(map(len, by_comp.values()))} tasks", flush=True)

    at = root / "airtribune_igc"
    listed = sources.airtribune_events((at / "events_past.html").read_text(encoding="utf-8", errors="replace")) \
        if (at / "events_past.html").exists() else {}
    by_event = defaultdict(list)
    for r in read_csv(at / "airtribune_catalog.csv"):
        if (at / r["event_id"] / f"{r['task']}.zip").exists():
            by_event[r["event_id"]].append(r)
    for event_id, rows in by_event.items():
        page = at / event_id / "results.html"
        pilots = sources.airtribune_pilots(page.read_text(encoding="utf-8", errors="replace")) if page.exists() else {}
        first = None
        for r in rows:
            task, res = parse(sources.airtribune_task, event_id, r["task"], at / event_id, pilots, r.get("zip_url", "")) \
                or sources.airtribune_task(event_id, r["task"], Path("/nonexistent"), pilots, r.get("zip_url", ""))
            task["track_kind"] = r.get("kind")
            tasks.append(task)
            results += res
            jobs.append(("airtribune", task["task_key"], str(at / event_id / f"{r['task']}.zip")))
            first = first or task
        e = listed.get(int(event_id)) or {"id": int(event_id), "title": rows[0]["title"], "place": rows[0]["place"],
                                          "start_date": rows[0]["start_date"], "end_date": rows[0]["end_date"]}
        events.append(sources.airtribune_event(e, first))
    print(f"Airtribune: {len(by_event)} competitions, {sum(map(len, by_event.values()))} tasks", flush=True)
    return events, tasks, results, jobs


# ── Tracks ──────────────────────────────────────────────────────────────────

def read_zip(job):
    """Track rows (header fields and fix summary) for every IGC file in one task package."""
    source, task_key, path = job
    rows = []
    try:
        with zipfile.ZipFile(path) as z:
            for member in z.namelist():
                if not member.lower().endswith(".igc"):
                    continue
                try:
                    info = igc.summary(z.read(member))
                except Exception as e:  # a damaged file must not stop the run
                    info = {"error": str(e)[:200]}
                rows.append({"source": source, "task_key": task_key, "zip_path": path, "member": member, **info})
    except zipfile.BadZipFile:
        rows.append({"source": source, "task_key": task_key, "zip_path": path, "member": None, "error": "bad zip"})
    return job, rows


def read_tracks(jobs, parts, workers):
    """All track rows, reading only packages without a cached part."""
    parts.mkdir(parents=True, exist_ok=True)
    part = lambda job: parts / f"{job[1].replace(':', '_')}.parquet"  # noqa: E731
    todo = [j for j in jobs if not part(j).exists()]
    print(f"Tracks: {len(jobs) - len(todo)} packages already read, {len(todo)} to read", flush=True)
    started = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(read_zip, j) for j in todo]
        for n, future in enumerate(as_completed(futures), 1):
            job, rows = future.result()
            clean(pd.DataFrame(rows or [{"source": job[0], "task_key": job[1], "zip_path": job[2]}])).to_parquet(
                part(job), index=False)
            if n % 50 == 0 or n == len(todo):
                print(f"... {n}/{len(todo)} packages, {(time.time() - started) / 60:.0f} min", flush=True)
    frames = [f for f in (pd.read_parquet(part(j)) for j in jobs if part(j).exists()) if len(f.columns) > 3]
    tracks = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return tracks[tracks["member"].notna()].reset_index(drop=True) if "member" in tracks else tracks


def _num(value):
    """Competition number without leading zeros ('0040' -> '40'), or None."""
    s = str(value or "").strip()
    return str(int(s)) if s.isdigit() else None


def name_from_file(member):
    """A pilot name guessed from a track's file name ('LiveTrack Koen Vancampenhoudt.947030...igc')."""
    base = Path(member).name
    base = base.split(".")[0].replace("LiveTrack ", "")
    return norm_name(base.replace("_", " "))


def match_tracks(tracks, results):
    """Link each track to its row in the task's results: by file name (AirScore), by competition id
    (file name or IGC header), then by pilot name. Adds result_id and match columns."""
    by_task = defaultdict(list)
    for r in results.itertuples():
        by_task[r.task_key].append(r)
    result_ids, methods = [], []
    for t in tracks.itertuples():
        rows = by_task.get(t.task_key, [])
        found, how = None, None
        base = Path(t.member).name
        for r in rows:
            if r.track_file and Path(str(r.track_file)).name == base:
                found, how = r, "file"
                break
        if found is None:
            ids = {_num(TRACK_ID[t.source](t.member)), _num(getattr(t, "comp_id", None))} - {None}
            found = next((r for r in rows if r.comp_id in ids), None)
            how = "comp_id" if found is not None else None
        if found is None and rows:
            names = {n for n in [norm_name(getattr(t, "pilot", "")), name_from_file(t.member)] if n}
            found = next((r for r in rows if r.name_norm in names), None)
            how = "name" if found is not None else None
            if found is None and names:
                best = max(((difflib.SequenceMatcher(None, n, r.name_norm).ratio(), r) for n in names for r in rows),
                           key=lambda x: x[0], default=(0, None))
                if best[0] >= 0.9:
                    found, how = best[1], "fuzzy_name"
        result_ids.append(found.result_id if found is not None else None)
        methods.append(how)
    tracks["result_id"] = result_ids
    tracks["match"] = methods
    return tracks


def mark_duplicates(tracks):
    """duplicate_of = track_id of the copy kept among tracks with identical fixes."""
    tracks["priority"] = tracks["source"].map(SOURCE_PRIORITY)
    tracks["unmatched"] = tracks["result_id"].isna()
    order = tracks.sort_values(["unmatched", "priority", "track_id"])
    keep = order.dropna(subset=["sha1"]).drop_duplicates("sha1").set_index("sha1")["track_id"]
    tracks["duplicate_of"] = tracks["sha1"].map(keep)
    tracks.loc[tracks["duplicate_of"] == tracks["track_id"], "duplicate_of"] = None
    return tracks.drop(columns=["priority", "unmatched"])


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("root", type=Path, help="folder with flymaster_igc, airscore_igc and airtribune_igc")
    parser.add_argument("--out", type=Path, help="folder for the tables (default ROOT/database)")
    parser.add_argument("--workers", type=int, default=2, help="processes reading IGC files (default 2)")
    args = parser.parse_args()
    out = args.out or args.root / "database"
    out.mkdir(parents=True, exist_ok=True)

    events, tasks, results, jobs = collect_tasks(args.root)

    results = pd.DataFrame(results)
    results["name_norm"] = results["name"].map(norm_name)
    results["result_id"] = [hashlib.sha1(f"{r.task_key}|{r.comp_id}|{r.name_norm}".encode()).hexdigest()[:16]
                            for r in results.itertuples()]
    results["pilot_key"] = [f"civl:{int(r.civl_id)}" if pd.notna(r.civl_id) and r.civl_id > 0
                            else f"name:{r.name_norm}:{r.nat or ''}" for r in results.itertuples()]

    tracks = read_tracks(jobs, out / "_parts", args.workers)
    tracks["track_id"] = [hashlib.sha1(f"{r.task_key}|{r.member}".encode()).hexdigest()[:16] for r in tracks.itertuples()]
    tracks = match_tracks(tracks, results)
    tracks = mark_duplicates(tracks)
    tracks = tracks.merge(results[["result_id", "pilot_key"]], on="result_id", how="left")

    pilots = (results.sort_values("task_key")
              .groupby("pilot_key")
              .agg(civl_id=("civl_id", "max"), name=("name", "last"),
                   nat=("nat", lambda s: Counter(s.dropna()).most_common(1)[0][0] if s.notna().any() else None),
                   sex=("sex", lambda s: Counter(s.dropna()).most_common(1)[0][0] if s.notna().any() else None),
                   n_results=("result_id", "size"))
              .reset_index())

    turnpoints = pd.DataFrame([{"task_key": t["task_key"], **tp} for t in tasks for tp in t["turnpoints"]])
    tasks = pd.DataFrame(tasks)
    # Tasks whose results gave no date: the date most of their tracks carry, if it falls within the
    # competition's dates (some loggers write a wrong date), otherwise the competition's first day
    ev = pd.DataFrame(events).set_index("event_key")
    start = pd.to_datetime(tasks["event_key"].map(ev["start_date"]), errors="coerce")
    end = pd.to_datetime(tasks["event_key"].map(ev["end_date"]), errors="coerce")
    track_date = pd.Series(dtype=object)
    if "date" in tracks:
        track_date = tracks.dropna(subset=["date"]).groupby("task_key")["date"].agg(lambda s: s.mode().iloc[0])
    guess = pd.to_datetime(tasks["task_key"].map(track_date), errors="coerce")
    inside = guess.between(start - pd.Timedelta(days=2), end + pd.Timedelta(days=2))
    missing = tasks["date"].isna()
    tasks["date_from_tracks"] = missing & inside
    tasks.loc[missing & inside, "date"] = guess[missing & inside].dt.strftime("%Y-%m-%d")
    tasks.loc[missing & ~inside, "date"] = start[missing & ~inside].dt.strftime("%Y-%m-%d")
    tasks["turnpoints"] = tasks["turnpoints"].map(json.dumps)
    counts = tracks[tracks["duplicate_of"].isna()].groupby("task_key").size()
    tasks["n_tracks"] = tasks["task_key"].map(counts).fillna(0).astype(int)
    events = pd.DataFrame(events)

    for name, df in [("events", events), ("tasks", tasks), ("turnpoints", turnpoints), ("results", results),
                     ("pilots", pilots), ("tracks", tracks)]:
        clean(df).to_parquet(out / f"{name}.parquet", index=False)

    unique = tracks["duplicate_of"].isna()
    print("\nTables written to", out)
    print(f"  events      {len(events):>8,}")
    print(f"  tasks       {len(tasks):>8,}  ({int(tasks['results_found'].sum()):,} with results)")
    print(f"  results     {len(results):>8,}")
    print(f"  pilots      {len(pilots):>8,}  ({int(pilots['civl_id'].notna().sum()):,} with CIVL ID)")
    print(f"  tracks      {len(tracks):>8,}  ({int(unique.sum()):,} unique, {int((~unique).sum()):,} duplicates)")
    matched = tracks.loc[unique, "result_id"].notna().mean() if unique.any() else 0
    print(f"  unique tracks linked to a result: {matched:.0%}")
    print("\nUnique tracks by source:")
    print(tracks[unique].groupby("source").size().to_string())


if __name__ == "__main__":
    main()
