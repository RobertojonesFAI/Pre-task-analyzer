#!/usr/bin/env python3
"""Data for the web page: sites and, per site, the 1 km cells pilots crossed with how often they
found a climb there and what the thermal model predicts.

Only places and aggregates are exported, never pilot names or tracks: a cell is published when at
least --min-crossings crossings went through it (default 20).

    OUT/summary.json         totals and the model's scores
    OUT/sites.json           one entry per site: name, country, launch, box, competitions, flights, cells
    OUT/cells/<site>.json    {"cols": [...], "rows": [[lat, lon, crossings, climb %, predicted %, climb m/s], ...]}

    python3 scripts/export_web.py /content/drive/MyDrive/IGCS/database --out web/public/data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path)
    parser.add_argument("--out", type=Path, help="output folder (default DB/web_data)")
    parser.add_argument("--min-crossings", type=int, default=20)
    args = parser.parse_args()
    out = args.out or args.db / "web_data"
    (out / "cells").mkdir(parents=True, exist_ok=True)

    sites = pd.read_parquet(args.db / "sites.parquet")
    model_dir = args.db / "models" / "thermal_v0"
    cells = pd.read_parquet(model_dir / "cells.parquet") if (model_dir / "cells.parquet").exists() else pd.DataFrame()
    metrics = json.loads((model_dir / "metrics.json").read_text()) if (model_dir / "metrics.json").exists() else {}

    n_cells = {}
    if len(cells):
        cells = cells[cells["crossings"] >= args.min_crossings]
        for site, g in cells.groupby("site_id"):
            rows = np.column_stack([g["lat"].round(4), g["lon"].round(4), g["crossings"],
                                    (100 * g["climb_share"]).round(1), (100 * g["predicted"]).round(1),
                                    g["climb_rate_ms"].round(2)]).tolist()
            rows = [[None if (isinstance(v, float) and np.isnan(v)) else v for v in r] for r in rows]
            (out / "cells" / f"{site}.json").write_text(json.dumps(
                {"cols": ["lat", "lon", "crossings", "climb_pct", "predicted_pct", "climb_ms"], "rows": rows},
                separators=(",", ":")))
            n_cells[site] = len(g)

    entries = []
    for s in sites.itertuples(index=False):
        entries.append({"id": s.site_id, "name": s.name, "country": s.country, "lat": round(float(s.lat), 4),
                        "lon": round(float(s.lon), 4),
                        "box": [round(float(s.west), 3), round(float(s.south), 3), round(float(s.east), 3), round(float(s.north), 3)],
                        "events": int(s.n_events), "flights": int(s.n_flights),
                        "first": str(s.first_date)[:4], "last": str(s.last_date)[:4], "cells": n_cells.get(s.site_id, 0)})
    (out / "sites.json").write_text(json.dumps(entries, separators=(",", ":"), ensure_ascii=False))

    climb = metrics.get("climb", {})
    summary = {"sites": len(entries), "flights": int(sites["n_flights"].sum()), "competitions": int(sites["n_events"].sum()),
               "cells": int(sum(n_cells.values())), "min_crossings": args.min_crossings,
               "model": {k: {m: round(v[m], 3) for m in ("auc", "log_loss_skill") if m in v} for k, v in climb.items()},
               "climb_rate": {k: round(v, 3) for k, v in metrics.get("climb_rate", {}).items() if isinstance(v, float)},
               "ceiling": {k: (round(v, 2) if isinstance(v, float) else v) for k, v in metrics.get("ceiling", {}).items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    size = sum(p.stat().st_size for p in out.rglob("*.json")) / 1e6
    print(f"web data in {out}: {len(entries)} sites, {summary['cells']:,} cells with {args.min_crossings}+ crossings "
          f"in {len(n_cells)} sites, {size:.1f} MB")


if __name__ == "__main__":
    main()
