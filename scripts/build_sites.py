#!/usr/bin/env python3
"""Group competitions into flying sites and list the chips each site covers.

A competition's area is the box holding its turnpoints (cylinders up to MAX_RADIUS_KM) and, when
notebook 05 has run, 98 % of its climbs, takeoffs and landings, plus a margin. Competitions whose
areas mostly overlap (by at least MERGE_OVERLAP of the smaller one) are the same site, so ten
years of competitions at one place share one terrain and weather extract.

Writes to DB:

    sites         one row per site: box, chips, competitions, flights
    event_sites   which site each competition belongs to
    site_chips    one row per chip of each site (chip_id, ERA5 point)

    python3 scripts/build_sites.py /content/drive/MyDrive/IGCS/database
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pta import grid  # noqa: E402

MARGIN_DEG = 0.05        # added around each competition's area (about 5 km)
MAX_RADIUS_KM = 10       # turnpoint cylinders count up to this radius (open-distance goals can be huge)
POINTS_PCT = 1           # flight points outside the 1st-99th percentiles are ignored (retrieves, GPS errors)
MAX_SPAN_DEG = 4.0       # an area larger than this is cut down to its flight points (or dropped)
MERGE_OVERLAP = 0.5      # areas overlapping by this share of the smaller one are one site
KM_PER_DEG = 111.32


def read(db, name):
    path = db / f"{name}.parquet"
    return pd.read_parquet(path) if path.exists() else None


def box_of(lat, lon, pct=0):
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    ok = np.isfinite(lat) & np.isfinite(lon) & ((lat != 0) | (lon != 0))
    lat, lon = lat[ok], lon[ok]
    if not len(lat):
        return None
    if pct and len(lat) >= 20:
        return (*np.percentile(lat, [pct, 100 - pct]), *np.percentile(lon, [pct, 100 - pct]))
    return lat.min(), lat.max(), lon.min(), lon.max()


def union(a, b):
    if a is None or b is None:
        return a or b
    return min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])


def area(b):
    return max(b[1] - b[0], 0) * max(b[3] - b[2], 0) * np.cos(np.radians((b[0] + b[1]) / 2))


def overlap(a, b):
    s, n, w, e = max(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), min(a[3], b[3])
    return area((s, n, w, e)) if n > s and e > w else 0.0


def event_boxes(db):
    """{event_key: (south, north, west, east)} and the event table."""
    events, tasks, tps = read(db, "events"), read(db, "tasks"), read(db, "turnpoints")
    flights, climbs = read(db, "flights"), read(db, "climbs")
    task_event = tasks.set_index("task_key")["event_key"]

    route = {}
    if tps is not None and len(tps):
        tps = tps.dropna(subset=["lat", "lon"]).copy()
        tps["event_key"] = tps["task_key"].map(task_event)
        r_deg = tps["radius_m"].fillna(0).clip(upper=MAX_RADIUS_KM * 1000) / 1000 / KM_PER_DEG
        cos = np.cos(np.radians(tps["lat"])).clip(lower=0.2)
        tps["s"], tps["n"] = tps["lat"] - r_deg, tps["lat"] + r_deg
        tps["w"], tps["e"] = tps["lon"] - r_deg / cos, tps["lon"] + r_deg / cos
        for ev, g in tps.groupby("event_key"):
            route[ev] = box_of(np.r_[g["s"], g["n"]], np.r_[g["w"], g["e"]])

    flown = {}
    pts = []
    if flights is not None and len(flights):
        f = flights[flights["error"].isna()] if "error" in flights else flights
        for end in ("takeoff", "landing"):
            pts.append(pd.DataFrame({"task_key": f["task_key"], "lat": f[f"lat_{end}"], "lon": f[f"lon_{end}"]}))
    if climbs is not None and len(climbs):
        pts.append(climbs[["task_key", "lat", "lon"]])
    if pts:
        pts = pd.concat(pts, ignore_index=True)
        pts["event_key"] = pts["task_key"].map(task_event)
        for ev, g in pts.groupby("event_key"):
            flown[ev] = box_of(g["lat"], g["lon"], POINTS_PCT)

    boxes, notes = {}, Counter()
    for ev in events["event_key"]:
        b = union(route.get(ev), flown.get(ev))
        if b is not None and max(b[1] - b[0], b[3] - b[2]) > MAX_SPAN_DEG:
            b = flown.get(ev)  # e.g. a turnpoint with wrong coordinates, far from where pilots flew
            notes["cut to the flown area"] += 1
            if b is not None and max(b[1] - b[0], b[3] - b[2]) > MAX_SPAN_DEG:
                b = None
        if b is None:
            notes["no coordinates"] += 1
            continue
        cos = max(np.cos(np.radians((b[0] + b[1]) / 2)), 0.2)
        boxes[ev] = (b[0] - MARGIN_DEG, b[1] + MARGIN_DEG, b[2] - MARGIN_DEG / cos, b[3] + MARGIN_DEG / cos)
    return boxes, events, notes


def merge(boxes):
    """Groups of events whose boxes overlap enough (union-find)."""
    keys = list(boxes)
    parent = list(range(len(keys)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = boxes[keys[i]], boxes[keys[j]]
            ov = overlap(a, b)
            if ov and ov >= MERGE_OVERLAP * min(area(a), area(b)):
                parent[find(i)] = find(j)
    groups = {}
    for i, k in enumerate(keys):
        groups.setdefault(find(i), []).append(k)
    return list(groups.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder")
    args = parser.parse_args()

    boxes, events, notes = event_boxes(args.db)
    ev = events.set_index("event_key")
    flights = read(args.db, "flights")
    tasks = read(args.db, "tasks")
    n_flights = Counter()
    if flights is not None and len(flights):
        ok = flights["error"].isna() if "error" in flights else slice(None)
        n_flights = flights[ok]["task_key"].map(tasks.set_index("task_key")["event_key"]).value_counts().to_dict()

    sites, links, chips = [], [], []
    for members in merge(boxes):
        b = None
        for k in members:
            b = union(b, boxes[k])
        lat_c, lon_c = (b[0] + b[1]) / 2, (b[2] + b[3]) / 2
        site_id = f"s{int(grid.chip_id(lat_c, lon_c))}"  # stable: the chip at the site's centre
        while any(s["site_id"] == site_id for s in sites):
            site_id += "b"
        places = Counter(str(ev.at[k, "place"]) for k in members if pd.notna(ev.at[k, "place"]))
        names = [ev.at[k, "name"] for k in sorted(members, key=lambda k: str(ev.at[k, "start_date"]))]
        site_chips = grid.chips_in_box(*b)
        dates = pd.to_datetime(ev.loc[members, "start_date"], errors="coerce")
        sites.append({
            "site_id": site_id, "name": places.most_common(1)[0][0] if places else names[-1],
            "country": Counter(ev.loc[members, "country"].dropna()).most_common(1)[0][0]
            if ev.loc[members, "country"].notna().any() else None,
            "lat": lat_c, "lon": lon_c, "south": b[0], "north": b[1], "west": b[2], "east": b[3],
            "n_chips": len(site_chips), "n_events": len(members),
            "n_flights": int(sum(n_flights.get(k, 0) for k in members)),
            "first_date": dates.min().strftime("%Y-%m-%d") if dates.notna().any() else None,
            "last_date": dates.max().strftime("%Y-%m-%d") if dates.notna().any() else None,
            "events": " | ".join(names),
        })
        links += [{"event_key": k, "site_id": site_id} for k in members]
        for c in site_chips:
            clat, clon = grid.chip_centre(c)
            chips.append({"site_id": site_id, "chip_id": c, "era5_lat": clat, "era5_lon": clon})

    sites = pd.DataFrame(sites).sort_values("n_flights", ascending=False)
    pd.DataFrame(sites).to_parquet(args.db / "sites.parquet", index=False)
    pd.DataFrame(links).to_parquet(args.db / "event_sites.parquet", index=False)
    chips = pd.DataFrame(chips)
    chips.to_parquet(args.db / "site_chips.parquet", index=False)

    print(f"Sites: {len(sites):,} from {len(links):,} competitions "
          f"({len(events) - len(links):,} without coordinates or with an area that makes no sense)")
    for k, v in notes.items():
        print(f"  {k}: {v}")
    print(f"Chips: {len(chips):,} site chips, {chips['chip_id'].nunique():,} unique "
          f"(about {chips['chip_id'].nunique() * 0.25 * 0.25 * 111 * 85:,.0f} km²)")
    print("\nSites with the most flights:")
    print(sites.head(15)[["site_id", "name", "country", "n_events", "n_flights", "n_chips", "first_date", "last_date"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
