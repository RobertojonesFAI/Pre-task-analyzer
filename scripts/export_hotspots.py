"""Thermal hotspots for the web page: where pilots circled in thermals, at about 200 m.

Every thermal of the competition flights (`thermals`, notebook 08) is put in a cell of about 200 m
(0.002° of latitude, and the same distance in longitude) at its site. A cell is published when
thermals from at least --min-flights different flights (default 2) went up there; it is drawn at
the mean position of its thermals, so the map follows the thermals rather than a grid.

Only places and aggregates are written, never pilots or tracks. A few seconds to a minute:

    python scripts/export_hotspots.py --db "G:/My Drive/IGCS/database" --web web/public/data

Output: `<web>/hotspots/<site>.json`
    {"cols": ["lat", "lon", "thermals", "flights", "climb_ms", "top_agl_m"], "rows": [...]}
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

STEP_DEG = 0.002          # about 220 m of latitude
MAX_ROWS = 80_000         # a site with more cells keeps the ones with the most flights


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--db", type=Path, required=True, help="the database folder")
    ap.add_argument("--web", type=Path, required=True, help="the web data folder (with sites.json)")
    ap.add_argument("--min-flights", type=int, default=2)
    a = ap.parse_args()

    sites = {s["id"]: s for s in json.loads((a.web / "sites.json").read_text(encoding="utf-8"))}
    th = pd.read_parquet(a.db / "thermals.parquet", columns=["track_id", "task_key", "lat", "lon", "rate_ms", "agl_top"])
    tasks = pd.read_parquet(a.db / "tasks.parquet", columns=["task_key", "event_key"])
    ev = pd.read_parquet(a.db / "event_sites.parquet", columns=["event_key", "site_id"])
    th = th.dropna(subset=["lat", "lon"]).merge(tasks, on="task_key").merge(ev, on="event_key")
    th = th[th["site_id"].isin(sites)]

    out = a.web / "hotspots"
    out.mkdir(parents=True, exist_ok=True)
    total = cells = 0
    for site, g in th.groupby("site_id", sort=False):
        lat0 = float(sites[site]["lat"])
        step_lon = STEP_DEG / max(np.cos(np.radians(lat0)), 0.2)
        g = g.assign(iy=np.floor(g["lat"] / STEP_DEG).astype("int64"), ix=np.floor(g["lon"] / step_lon).astype("int64"))
        c = g.groupby(["iy", "ix"]).agg(lat=("lat", "mean"), lon=("lon", "mean"), thermals=("lat", "size"),
                                        flights=("track_id", "nunique"), climb_ms=("rate_ms", "mean"),
                                        top_agl_m=("agl_top", "median"))
        c = c[c["flights"] >= a.min_flights]
        if len(c) > MAX_ROWS:
            c = c.sort_values(["flights", "thermals"], ascending=False).head(MAX_ROWS)
        if c.empty:
            continue
        rows = np.column_stack([c["lat"].round(5), c["lon"].round(5), c["thermals"], c["flights"],
                                c["climb_ms"].round(2), c["top_agl_m"].round(0)]).tolist()
        rows = [[None if isinstance(v, float) and np.isnan(v) else (int(v) if k in (2, 3, 5) else v) for k, v in enumerate(r)]
                for r in rows]
        (out / f"{site}.json").write_text(json.dumps(
            {"cols": ["lat", "lon", "thermals", "flights", "climb_ms", "top_agl_m"], "step_deg": STEP_DEG, "rows": rows},
            separators=(",", ":")), encoding="utf-8")
        total += int(c["thermals"].sum())
        cells += len(c)
    print(f"{cells:,} cells with {total:,} thermals at {len(list(out.glob('*.json'))):,} sites -> {out}")


if __name__ == "__main__":
    main()
