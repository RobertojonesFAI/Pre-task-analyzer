#!/usr/bin/env python3
"""Statistics and coverage maps of the database, for the repo's results/ folder.

Only counts, places and aggregates (no pilot names, no tracks), so the files can be public:

    STATS.md               totals, by year, by source, by country, the sites, glider classes and the
                           models' latest scores
    stats.json             the same numbers, for scripts
    sites.geojson          one point per site (GitHub shows .geojson files as an interactive map)
    chips.geojson          the 0.25° weather chips the sites use, shaded by flights
    thermal_cells.geojson  the 1 km cells crossed most often, shaded by how often pilots found a climb
                           (when notebook 09 has run)
    coverage.png           world map of the chips and sites, plus Europe and the Americas

    python3 scripts/repo_stats.py /content/drive/MyDrive/IGCS/database --out /content/repo_results
"""

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pta import grid  # noqa: E402

# one hue, light to dark (flights per chip) and the orange ramp of the web page (climb share)
BLUES = ["#d6e6f9", "#a9cbf2", "#76a9ea", "#4a87dc", "#2a6ac4", "#1d4f99", "#123566"]
ORANGES = ["#fdeee6", "#fbd3bd", "#f6a982", "#f08455", "#eb6834", "#cf5222", "#a33e16"]
MAX_CELLS = 5000


def rows(path):
    import pyarrow.parquet as pq

    return pq.ParquetFile(path).metadata.num_rows if path.exists() else None


def cols(path):
    import pyarrow.parquet as pq

    return set(pq.ParquetFile(path).schema.names) if path.exists() else set()


def read(path, want):
    have = cols(path)
    use = [c for c in want if c in have]
    return pd.read_parquet(path, columns=use) if use else pd.DataFrame()


def shade(values, ramp, lo=None, hi=None, log=False):
    v = np.asarray(values, float)
    v = np.log10(np.maximum(v, 1)) if log else v
    lo = np.nanmin(v) if lo is None else lo
    hi = np.nanmax(v) if hi is None else hi
    k = np.clip(((v - lo) / max(hi - lo, 1e-9) * (len(ramp) - 1)).round(), 0, len(ramp) - 1)
    return [ramp[int(i)] if np.isfinite(i) else "#cccccc" for i in k]


def square(lat, lon, half):
    return [[[round(lon - half, 4), round(lat - half, 4)], [round(lon + half, 4), round(lat - half, 4)],
             [round(lon + half, 4), round(lat + half, 4)], [round(lon - half, 4), round(lat + half, 4)],
             [round(lon - half, 4), round(lat - half, 4)]]]


def collect(db):
    s = {"updated": date.today().isoformat(), "totals": {}}
    t = s["totals"]
    events = read(db / "events.parquet", ["event_key", "country", "start_date"])
    tasks = read(db / "tasks.parquet", ["task_key", "event_key", "date", "results_found"])
    t["competitions"] = len(events)
    t["tasks"] = len(tasks)
    if "results_found" in tasks:
        t["tasks_with_results"] = int(tasks["results_found"].fillna(False).astype(bool).sum())
    t["results"] = rows(db / "results.parquet")
    t["pilots"] = rows(db / "pilots.parquet")
    tracks = read(db / "tracks.parquet", ["track_id", "source", "duplicate_of", "error"])
    if len(tracks):
        t["tracks"] = len(tracks)
        uniq = tracks[tracks["duplicate_of"].isna()] if "duplicate_of" in tracks else tracks
        t["unique_tracks"] = len(uniq)
        if "source" in uniq:
            s["tracks_by_source"] = uniq["source"].value_counts().to_dict()
    flights = read(db / "flights.parquet", ["track_id", "task_key", "error", "same_flight_as", "duration_s",
                                            "n_thermals", "mean_thermal_rate_ms"])
    if len(flights):
        ok = flights["error"].isna() if "error" in flights else pd.Series(True, index=flights.index)
        if "same_flight_as" in flights:
            ok &= flights["same_flight_as"].isna()
        f = flights[ok]
        t["flights"] = len(f)
        s["flight_medians"] = {"hours": round(float(f["duration_s"].median()) / 3600, 2),
                               "thermals": float(f["n_thermals"].median()),
                               "climb_ms": round(float(f["mean_thermal_rate_ms"].median()), 2)}
        f = f.merge(tasks[["task_key", "event_key", "date"]], on="task_key", how="left")
        year = pd.to_datetime(f["date"], errors="coerce").dt.year
        by_year = pd.DataFrame({"flights": f.groupby(year).size()})
        ty = pd.to_datetime(tasks["date"], errors="coerce").dt.year
        by_year["tasks"] = tasks.groupby(ty).size()
        by_year["competitions"] = tasks.groupby(ty)["event_key"].nunique()
        s["by_year"] = {int(y): {k: int(v) for k, v in r.items() if pd.notna(v)} for y, r in by_year.iterrows() if pd.notna(y)}
    if len(events) and "country" in events:
        s["competitions_by_country"] = events["country"].fillna("?").value_counts().head(20).to_dict()
    for name in ("sites", "site_chips", "cells", "flight_checks", "crossings", "thermals", "landings", "climbs", "glides"):
        n = rows(db / f"{name}.parquet")
        if n is not None:
            t[name] = n
    checks = read(db / "flight_checks.parquet", ["is_drive"])
    if len(checks):
        t["drives"] = int(checks["is_drive"].fillna(False).astype(bool).sum())
    s["weather"] = {}
    for name in ("single", "levels", "extra", "land", "static"):
        p = db / "weather" / f"era5_{name}.parquet"
        if p.exists():
            d = read(p, ["chip_id", "time"])
            s["weather"][f"era5_{name}"] = {"rows": len(d), "chips": int(d["chip_id"].nunique()),
                                            **({"days": int(d["time"].dt.normalize().nunique())} if "time" in d else {})}
    g = read(db / "result_gliders.parquet", ["glider_class", "class_source", "harness_key"])
    if len(g):
        s["glider_classes"] = {k: round(v, 4) for k, v in g["glider_class"].value_counts(normalize=True).items()}
        s["glider_class_found"] = round(float((g["class_source"] != "unknown").mean()), 4)
        s["harness_written"] = round(float((g["harness_key"].fillna("") != "").mean()), 4) if "harness_key" in g else None
    pr = read(db / "pilot_ratings.parquet", ["rating", "tasks"])
    if len(pr):
        s["ratings"] = {"pilots": len(pr), "with_20_tasks": int((pr["tasks"] >= 20).sum()),
                        "min": round(float(pr["rating"].min())), "median": round(float(pr["rating"].median())),
                        "max": round(float(pr["rating"].max()))}
    models = {}
    for name, f in (("thermal_v0", "metrics.json"), ("task_v0", "metrics.json"), ("wind_check", "summary.json")):
        p = db / "models" / name / f
        if p.exists():
            m = json.loads(p.read_text())
            if name == "thermal_v0":
                m = {"climb": m.get("climb"), "climb_rate": m.get("climb_rate"), "ceiling": m.get("ceiling"), "rows": m.get("rows")}
            models[name] = m
    s["models"] = models
    return s


def sites_table(db):
    sites = pd.read_parquet(db / "sites.parquet")
    return sites.sort_values("n_flights", ascending=False)


def chips_table(db):
    c = pd.read_parquet(db / "site_chips.parquet")
    names = pd.read_parquet(db / "sites.parquet", columns=["site_id", "name"]).set_index("site_id")["name"]
    c["site"] = c["site_id"].map(names)
    return c.groupby("chip_id").agg(lat=("era5_lat", "first"), lon=("era5_lon", "first"), flights=("n_flights", "sum"),
                                    fixes=("n_fixes", "sum"), in_area=("in_area", "max"),
                                    sites=("site", lambda x: ", ".join(sorted(set(map(str, x)))))).reset_index()


def write_geojson(path, features):
    path.write_text(json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":"), ensure_ascii=False))


def geojsons(db, out):
    sites = sites_table(db)
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [round(float(r.lon), 4), round(float(r.lat), 4)]},
              "properties": {"name": r.name, "country": r.country, "competitions": int(r.n_events), "flights": int(r.n_flights),
                             "chips": int(r.n_chips), "years": f"{str(r.first_date)[:4]}–{str(r.last_date)[:4]}",
                             "marker-color": "#cf5222", "marker-size": "small"}}
             for r in sites.itertuples(index=False)]
    write_geojson(out / "sites.geojson", feats)
    chips = chips_table(db)
    colors = shade(chips["flights"], BLUES, log=True)
    feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": square(r.lat, r.lon, 0.125)},
              "properties": {"sites": r.sites, "flights": int(r.flights), "fixes": int(r.fixes),
                             "competition_area": bool(r.in_area), "fill": col, "fill-opacity": 0.6,
                             "stroke": col, "stroke-width": 0.5}}
             for r, col in zip(chips.itertuples(index=False), colors)]
    write_geojson(out / "chips.geojson", feats)
    n_cells = 0
    p = db / "models" / "thermal_v0" / "cells.parquet"
    if p.exists():
        cells = pd.read_parquet(p).sort_values("crossings", ascending=False).head(MAX_CELLS)
        colors = shade(100 * cells["climb_share"], ORANGES, lo=0, hi=75)
        feats = [{"type": "Feature", "geometry": {"type": "Polygon", "coordinates": square(r.lat, r.lon, 0.005)},
                  "properties": {"crossings": int(r.crossings), "found_a_climb": f"{100 * r.climb_share:.0f}%",
                                 "model": f"{100 * r.predicted:.0f}%", "fill": col, "fill-opacity": 0.8, "stroke-width": 0}}
                 for r, col in zip(cells.itertuples(index=False), colors)]
        write_geojson(out / "thermal_cells.geojson", feats)
        n_cells = len(feats)
    return sites, chips, n_cells


def coverage_png(sites, chips, out):
    """World, Europe and Americas maps: coastlines from cartopy when it can get them (Colab can),
    else plain longitude/latitude axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection

    views = [("world", (-180, 180, -60, 75)), ("europe", (-12, 45, 34, 60)), ("americas", (-125, -30, -40, 50))]
    colors = shade(chips["flights"], BLUES, log=True)

    def draw(name, box, with_coast):
        w, e, s_, n = box
        fig = plt.figure(figsize=(12, 6.2) if name == "world" else (9, 6.5), dpi=150)
        kw = {}
        if with_coast:
            import cartopy.crs as ccrs
            import cartopy.feature as cfeature

            ax = plt.axes(projection=ccrs.PlateCarree())
            ax.set_extent([w, e, s_, n], crs=ccrs.PlateCarree())
            ax.add_feature(cfeature.LAND, facecolor="#f3f2ef")
            ax.add_feature(cfeature.OCEAN, facecolor="#ffffff")
            ax.add_feature(cfeature.COASTLINE, linewidth=0.4, edgecolor="#9a9890")
            ax.add_feature(cfeature.BORDERS, linewidth=0.3, edgecolor="#c3c2b7")
            kw = {"transform": ccrs.PlateCarree()}
        else:
            ax = plt.axes()
            ax.set_xlim(w, e)
            ax.set_ylim(s_, n)
            ax.set_aspect("equal")
            ax.set_facecolor("#f3f2ef")
            ax.set_xlabel("longitude")
            ax.set_ylabel("latitude")
        boxes = [plt.Rectangle((lo - 0.125, la - 0.125), 0.25, 0.25) for la, lo in chips[["lat", "lon"]].to_numpy()]
        ax.add_collection(PatchCollection(boxes, facecolor=colors, edgecolor="none", **kw))
        ax.scatter(sites["lon"], sites["lat"], s=8 if name == "world" else 16, color="#cf5222", edgecolor="white",
                   linewidth=0.5, zorder=5, **kw)
        ax.set_title(f"{len(sites)} sites and {len(chips):,} weather chips (0.25°), shaded by flights"
                     if name == "world" else name.capitalize(), fontsize=11, loc="left")
        fig.savefig(out / f"coverage_{name}.png", bbox_inches="tight", facecolor="white")
        plt.close(fig)

    for name, box in views:
        try:
            draw(name, box, with_coast=True)
        except Exception:  # noqa: BLE001  (no cartopy, or no Natural Earth download)
            plt.close("all")
            draw(name, box, with_coast=False)


def markdown(s, sites, n_cells):
    t = s["totals"]
    fmt = lambda v: f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)  # noqa: E731
    lines = [f"# Database statistics", "",
             f"Generated from the database by `scripts/repo_stats.py` (notebook 12) on {s['updated']}. "
             "Counts and aggregates only; pilot names and tracks stay private.", "",
             "![Sites and weather chips](coverage_world.png)", "",
             "Interactive maps (GitHub draws them): [sites](sites.geojson), [weather chips](chips.geojson)"
             + (f", [the {n_cells:,} cells crossed most often, by how often pilots found a climb](thermal_cells.geojson)" if n_cells else "")
             + ". Close-ups: [Europe](coverage_europe.png), [the Americas](coverage_americas.png).", "",
             "## Totals", "", "| | |", "|---|---|"]
    labels = [("competitions", "Competitions"), ("tasks", "Tasks"), ("tasks_with_results", "Tasks with results"),
              ("results", "Results"), ("pilots", "Pilots"), ("tracks", "IGC tracks"), ("unique_tracks", "Unique tracks"),
              ("flights", "Unique flights"), ("drives", "Drives found (not flights)"), ("sites", "Sites"),
              ("site_chips", "Site chips"), ("cells", "1 km cells"), ("climbs", "Climbs"), ("glides", "Glides"),
              ("crossings", "Cell crossings"), ("thermals", "Thermals"), ("landings", "Landings")]
    lines += [f"| {lab} | {fmt(t[k])} |" for k, lab in labels if t.get(k) is not None]
    if "flight_medians" in s:
        m = s["flight_medians"]
        lines += ["", f"Median flight: {m['hours']} h, {m['thermals']:.0f} thermals at {m['climb_ms']} m/s."]
    if s.get("tracks_by_source"):
        lines += ["", "## Unique tracks by source", "", "| Source | Tracks |", "|---|---|"]
        lines += [f"| {k} | {v:,} |" for k, v in s["tracks_by_source"].items()]
    if s.get("by_year"):
        lines += ["", "## By year", "", "| Year | Competitions | Tasks | Flights |", "|---|---|---|---|"]
        lines += [f"| {y} | {r.get('competitions', 0):,} | {r.get('tasks', 0):,} | {r.get('flights', 0):,} |"
                  for y, r in sorted(s["by_year"].items())]
    if s.get("weather"):
        lines += ["", "## Weather tables", "", "| Table | Rows | Chips | Days |", "|---|---|---|---|"]
        lines += [f"| `{k}` | {v['rows']:,} | {v['chips']:,} | {v.get('days', '')} |" for k, v in s["weather"].items()]
    if s.get("competitions_by_country"):
        lines += ["", "## Competitions by country (top 20)", "", "| Country | Competitions |", "|---|---|"]
        lines += [f"| {k} | {v:,} |" for k, v in s["competitions_by_country"].items()]
    lines += ["", "## Sites with the most flights", "", "| Site | Country | Competitions | Flights | Years |", "|---|---|---|---|---|"]
    lines += [f"| {r.name} | {r.country} | {int(r.n_events)} | {int(r.n_flights):,} | {str(r.first_date)[:4]}–{str(r.last_date)[:4]} |"
              for r in sites.head(30).itertuples(index=False)]
    if s.get("glider_classes"):
        lines += ["", "## Gliders", "", f"Class found for {s['glider_class_found']:.1%} of results"
                  + (f"; harness written for {s['harness_written']:.1%}" if s.get("harness_written") is not None else "") + ".", "",
                  "| Class | Share of results |", "|---|---|"]
        lines += [f"| {k} | {v:.1%} |" for k, v in s["glider_classes"].items()]
    if s.get("ratings"):
        r = s["ratings"]
        lines += ["", f"Pilot ratings: {r['pilots']:,} pilots ({r['with_20_tasks']:,} with 20+ tasks), from {r['min']} to {r['max']} (median {r['median']})."]
    m = s.get("models", {})
    if m:
        lines += ["", "## Models (latest run)", ""]
        th = m.get("thermal_v0") or {}
        if th.get("climb"):
            lines += ["Thermal model, chance of finding a climb when crossing a cell, scored on sites left out:", "",
                      "| Features | AUC | Log-loss skill |", "|---|---|---|"]
            lines += [f"| {k} | {v['auc']:.3f} | {v['log_loss_skill']:+.1%} |" for k, v in th["climb"].items()]
            lines += [""]
        tk = m.get("task_v0") or {}
        if tk:
            lines += ["Task baseline without weather (mean absolute error):", "", "| Target and split | Model | Median of all tasks |", "|---|---|---|"]
            for k, v in tk.items():
                unit = " km/h" if k.startswith("winner") else ""
                f2 = (lambda x: f"{x:.1f}{unit}") if unit else (lambda x: f"{100 * x:.1f} points")
                lines += [f"| {k.replace('|', ', ')} | {f2(v['mae'])} | {f2(v['rule_overall_median'])} |"]
            lines += [""]
        wc = (m.get("wind_check") or {}).get("bands")
        if wc:
            lines += ["Wind pilots measured vs ERA5 at the same height:", "", "| Band | Groups | Direction error | Within 30° | Speed correlation |", "|---|---|---|---|---|"]
            lines += [f"| {k} | {v['groups']:,} | {v['dir_error_deg']:.0f}° | {v['dir_within_30']:.0%} | {v['speed_corr']:.2f} |"
                      for k, v in wc.items() if v.get("dir_error_deg") is not None]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path)
    parser.add_argument("--out", type=Path, help="output folder (default DB/repo_results)")
    args = parser.parse_args()
    out = args.out or args.db / "repo_results"
    out.mkdir(parents=True, exist_ok=True)
    s = collect(args.db)
    sites, chips, n_cells = geojsons(args.db, out)
    coverage_png(sites, chips, out)
    (out / "stats.json").write_text(json.dumps(s, indent=1, default=str, ensure_ascii=False))
    (out / "STATS.md").write_text(markdown(s, sites, n_cells))
    for p in sorted(out.iterdir()):
        print(f"  {p.name:24s} {p.stat().st_size / 1e6:6.2f} MB")
    print(f"Written to {out}")


if __name__ == "__main__":
    main()
