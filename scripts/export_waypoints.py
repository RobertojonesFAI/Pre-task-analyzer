"""Waypoints and task definitions for the web page's task editor.

For every site with tasks on the page (`web_data/tasks/<site>.json`, written by
scripts/task_model.py), the waypoints its competitions used (one per name and place) and each
task as it was set: its turnpoints (waypoint, cylinder radius, type), first start gate, deadline
and UTC offset. The editor loads a past task from here, lets the user change it, and exports it to
instruments (XCTrack QR and .xctsk, SeeYou .cup, Flymaster and CompeGPS .wpt, GPX).

Only places and task settings are written, never pilot data. Takes a few seconds:

    python scripts/export_waypoints.py --db "G:/My Drive/IGCS/database" --web web/public/data

Output: `<web>/waypoints/<site>.json`
    {"waypoints": [[name, lat, lon, alt_m], ...],
     "tasks": {task_key: {"tps": [[waypoint index, radius_m, type], ...],
                          "start": "HH:MM", "deadline": "HH:MM", "utc": hours}}}
Times are local; `utc` is local time minus UTC.
"""

import argparse
import json
from pathlib import Path

import pandas as pd

TYPES = {"TAKEOFF": "TAKEOFF", "SSS": "SSS", "ESS": "ESS"}   # GOAL is the last turnpoint; others are ordinary


def hhmm(v):
    if not isinstance(v, str) or ":" not in v:
        return None
    h, m = v.split(":")[:2]
    return f"{int(h):02d}:{int(m):02d}" if h.strip().isdigit() and m.strip().isdigit() else None


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, required=True, help="the database folder")
    ap.add_argument("--web", type=Path, required=True, help="the web data folder (with tasks/)")
    a = ap.parse_args()

    shown = {p.stem: [t["key"] for t in json.loads(p.read_text(encoding="utf-8"))["tasks"]]
             for p in sorted((a.web / "tasks").glob("*.json"))}
    keys = {k for ks in shown.values() for k in ks}
    tasks = pd.read_parquet(a.db / "tasks.parquet")
    tasks = tasks[tasks["task_key"].isin(keys)]
    if "utc_offset_h" not in tasks or tasks["utc_offset_h"].isna().all():
        ev = pd.read_parquet(a.db / "events.parquet", columns=["event_key", "utc_offset_h"])
        tasks = tasks.drop(columns=["utc_offset_h"], errors="ignore").merge(ev, on="event_key", how="left")
    tasks = tasks.set_index("task_key")
    tp = pd.read_parquet(a.db / "turnpoints.parquet", columns=["task_key", "seq", "name", "type", "lat", "lon", "alt_m", "radius_m"])
    tp = tp[tp["task_key"].isin(keys)].dropna(subset=["lat", "lon"]).sort_values(["task_key", "seq"])
    tp["name"] = tp["name"].fillna("").astype(str).str.strip()

    out = a.web / "waypoints"
    out.mkdir(parents=True, exist_ok=True)
    n_wp = n_task = 0
    by_task = dict(tuple(tp.groupby("task_key")))
    for site, site_keys in shown.items():
        index, waypoints, defs = {}, [], {}
        for key in site_keys:
            g = by_task.get(key)
            if g is None or len(g) < 2:
                continue
            tps = []
            for r in g.itertuples():
                name = r.name or f"TP{int(r.seq):02d}"
                k = (name.upper(), round(float(r.lat), 4), round(float(r.lon), 4))
                if k not in index:
                    index[k] = len(waypoints)
                    alt = None if pd.isna(r.alt_m) else int(round(float(r.alt_m)))
                    waypoints.append([name, round(float(r.lat), 5), round(float(r.lon), 5), alt])
                radius = 0 if pd.isna(r.radius_m) else int(round(float(r.radius_m)))
                tps.append([index[k], radius, TYPES.get(str(r.type or "").upper(), "")])
            t = tasks.loc[key] if key in tasks.index else None
            utc = None if t is None or pd.isna(t.get("utc_offset_h")) else float(t["utc_offset_h"])
            defs[key] = {"tps": tps, "start": hhmm(t.get("start_time")) if t is not None else None,
                         "deadline": hhmm(t.get("deadline")) if t is not None else None, "utc": utc}
        if not defs:
            continue
        (out / f"{site}.json").write_text(json.dumps({"waypoints": waypoints, "tasks": defs}, ensure_ascii=False,
                                                     separators=(",", ":")), encoding="utf-8")
        n_wp += len(waypoints)
        n_task += len(defs)
    print(f"{n_task:,} tasks and {n_wp:,} waypoints at {len(list(out.glob('*.json'))):,} sites -> {out}")


if __name__ == "__main__":
    main()
