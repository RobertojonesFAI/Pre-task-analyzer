#!/usr/bin/env python3
"""Group competitions into flying sites and list the chips each site covers.

A competition's area is the box holding its turnpoints (cylinders up to MAX_RADIUS_KM) and 98 %
of its climbs, takeoffs and landings, plus a margin. Competitions that launched within LAUNCH_KM of
each other are the same site (their launch is the median takeoff of their flights), so ten years
of competitions at one place share one terrain and weather extract, while neighbouring sites in a
mountain range stay apart.

A site's chips are every chip of its competitions' areas plus every chip any of its flights
crossed (from the 5 s fixes of notebook 05), so no flown data is left outside the terrain and
weather extracts: a pilot who flew far off the course, or an open-distance flight 200 km downwind,
still gets terrain and weather under every fix. Neighbouring sites can share chips.

Writes to DB:

    sites         one row per site: box, chips, competitions, flights
    event_sites   which site each competition belongs to
    site_chips    one row per chip of each site (chip_id, ERA5 point, fixes and flights in it)

    python3 scripts/build_sites.py /content/drive/MyDrive/IGCS/database
"""

import argparse
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pta import grid  # noqa: E402

MARGIN_DEG = 0.05        # added around each competition's area (about 5 km)
MAX_RADIUS_KM = 10       # turnpoint cylinders count up to this radius (open-distance goals can be huge)
POINTS_PCT = 1           # flight points outside the 1st-99th percentiles are ignored (retrieves, GPS errors)
MAX_SPAN_DEG = 4.0       # an area larger than this is cut down to its flight points, or to LAUNCH_BOX_DEG around the launch
LAUNCH_BOX_DEG = 0.5     # (half-width; the flown chips are added anyway)
MIN_CHIP_FIXES = 12      # a chip crossed for less than a minute in all (12 fixes of 5 s) is not added
LAUNCH_KM = 20           # competitions launching this close to a site's main launch belong to it
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

    launch = {}
    if flights is not None and len(flights):
        f = flights[flights["error"].isna()] if "error" in flights else flights
        med = f.assign(event_key=f["task_key"].map(task_event)).groupby("event_key")[["lat_takeoff", "lon_takeoff"]].median()
        launch = {k: (r.lat_takeoff, r.lon_takeoff) for k, r in med.iterrows()}
    if tps is not None and len(tps):
        first = tps.sort_values("seq").groupby("task_key").first()
        first["event_key"] = first.index.map(task_event)
        for k, r in first.groupby("event_key")[["lat", "lon"]].median().iterrows():
            launch.setdefault(k, (r.lat, r.lon))

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
            notes["area cut to where pilots flew"] += 1
            if (b is None or max(b[1] - b[0], b[3] - b[2]) > MAX_SPAN_DEG) and ev in launch:
                la, lo = launch[ev]  # a very long open-distance course: the launch area, plus the flown chips
                cos = max(np.cos(np.radians(la)), 0.2)
                b = (la - LAUNCH_BOX_DEG, la + LAUNCH_BOX_DEG, lo - LAUNCH_BOX_DEG / cos, lo + LAUNCH_BOX_DEG / cos)
                notes["area cut to the launch (flown chips added)"] += 1
        if b is None:
            notes["no coordinates and no flights"] += 1
            continue
        cos = max(np.cos(np.radians((b[0] + b[1]) / 2)), 0.2)
        boxes[ev] = (b[0] - MARGIN_DEG, b[1] + MARGIN_DEG, b[2] - MARGIN_DEG / cos, b[3] + MARGIN_DEG / cos)
        launch.setdefault(ev, ((b[0] + b[1]) / 2, (b[2] + b[3]) / 2))
    return boxes, {k: launch[k] for k in boxes}, events, notes


def flown_chips(db, threads=16):
    """Fixes and flights per (event, chip), from the 5 s fixes of every flight (notebook 05)."""
    folder = db / "fixes"
    files = sorted(folder.glob("*.parquet")) if folder.exists() else []
    if not files:
        return pd.DataFrame(columns=["event_key", "chip_id", "n_fixes", "n_flights"])
    tasks = read(db, "tasks")
    event_of = {k.replace(":", "_"): e for k, e in zip(tasks["task_key"], tasks["event_key"])}

    def one(path):
        ev = event_of.get(path.stem)
        if ev is None:
            return None
        fx = pd.read_parquet(path, columns=["track_id", "lat", "lon"])
        if not len(fx):
            return None
        fx["chip_id"] = grid.chip_id(fx["lat"].to_numpy(), fx["lon"].to_numpy())
        g = fx.groupby("chip_id").agg(n_fixes=("track_id", "size"), n_flights=("track_id", "nunique")).reset_index()
        g.insert(0, "event_key", ev)
        return g

    started = time.time()
    with ThreadPoolExecutor(threads) as pool:
        parts = [p for p in pool.map(one, files) if p is not None]
    out = pd.concat(parts, ignore_index=True).groupby(["event_key", "chip_id"], as_index=False).sum()
    print(f"Flown chips: {int(out['n_fixes'].sum()):,} fixes in {len(files):,} task files, "
          f"{out['chip_id'].nunique():,} chips crossed, {time.time() - started:.0f} s", flush=True)
    return out


def km(a, b):
    la1, lo1, la2, lo2 = map(np.radians, (a[0], a[1], b[0], b[1]))
    h = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(h))


def merge(launch, weight):
    """Groups of events by launch: the event with the most flights founds a site at its launch, and
    every event launching within LAUNCH_KM of it joins (no chaining from one site to the next)."""
    sites = []  # (launch, members)
    for k in sorted(launch, key=lambda k: -weight.get(k, 0)):
        near = [(km(launch[k], s[0]), s) for s in sites]
        near = [x for x in near if x[0] <= LAUNCH_KM]
        if near:
            min(near, key=lambda x: x[0])[1][1].append(k)
        else:
            sites.append((launch[k], [k]))
    return [(lp, members) for lp, members in sites]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder")
    args = parser.parse_args()

    boxes, launch, events, notes = event_boxes(args.db)
    ev = events.set_index("event_key")
    flights = read(args.db, "flights")
    tasks = read(args.db, "tasks")
    n_flights = Counter()
    if flights is not None and len(flights):
        ok = flights["error"].isna() if "error" in flights else slice(None)
        n_flights = flights[ok]["task_key"].map(tasks.set_index("task_key")["event_key"]).value_counts().to_dict()

    flown = flown_chips(args.db)
    kept = flown[flown["n_fixes"] >= MIN_CHIP_FIXES]
    flown_by_event = {k: g for k, g in kept.groupby("event_key")}
    no_site = set(flown["event_key"]) - set(boxes)

    sites, links, chips = [], [], []
    for (lat_c, lon_c), members in merge(launch, n_flights):
        b = None
        for k in members:
            b = union(b, boxes[k])
        site_id = f"s{int(grid.chip_id(lat_c, lon_c))}"  # stable: the chip of the site's main launch
        while any(s["site_id"] == site_id for s in sites):
            site_id += "b"
        places = Counter(str(ev.at[k, "place"]) for k in members if pd.notna(ev.at[k, "place"]))
        names = [ev.at[k, "name"] for k in sorted(members, key=lambda k: str(ev.at[k, "start_date"]))]
        box_chips = set(grid.chips_in_box(*b))
        crossed = [flown_by_event[k] for k in members if k in flown_by_event]
        crossed = (pd.concat(crossed).groupby("chip_id")[["n_fixes", "n_flights"]].sum()
                   if crossed else pd.DataFrame(columns=["n_fixes", "n_flights"]))
        site_chips = sorted(box_chips | set(crossed.index))
        dates = pd.to_datetime(ev.loc[members, "start_date"], errors="coerce")
        sites.append({
            "site_id": site_id, "name": places.most_common(1)[0][0] if places else names[-1],
            "country": Counter(ev.loc[members, "country"].dropna()).most_common(1)[0][0]
            if ev.loc[members, "country"].notna().any() else None,
            "lat": lat_c, "lon": lon_c, "south": b[0], "north": b[1], "west": b[2], "east": b[3],
            "n_chips": len(site_chips), "n_chips_flown": len(crossed),
            "n_chips_outside_area": len(set(crossed.index) - box_chips), "n_events": len(members),
            "n_flights": int(sum(n_flights.get(k, 0) for k in members)),
            "first_date": dates.min().strftime("%Y-%m-%d") if dates.notna().any() else None,
            "last_date": dates.max().strftime("%Y-%m-%d") if dates.notna().any() else None,
            "events": " | ".join(names),
        })
        links += [{"event_key": k, "site_id": site_id} for k in members]
        for c in site_chips:
            clat, clon = grid.chip_centre(c)
            n = crossed.loc[c] if c in crossed.index else None
            chips.append({"site_id": site_id, "chip_id": c, "era5_lat": clat, "era5_lon": clon,
                          "in_area": c in box_chips,
                          "n_fixes": int(n["n_fixes"]) if n is not None else 0,
                          "n_flights": int(n["n_flights"]) if n is not None else 0})

    sites = pd.DataFrame(sites).sort_values("n_flights", ascending=False)
    pd.DataFrame(sites).to_parquet(args.db / "sites.parquet", index=False)
    pd.DataFrame(links).to_parquet(args.db / "event_sites.parquet", index=False)
    chips = pd.DataFrame(chips)
    chips.to_parquet(args.db / "site_chips.parquet", index=False)

    print(f"Sites: {len(sites):,} from {len(links):,} competitions "
          f"({len(events) - len(links):,} with no turnpoint coordinates and no flights)")
    for k, v in notes.items():
        print(f"  {k}: {v}")
    print(f"Chips: {len(chips):,} site chips, {chips['chip_id'].nunique():,} unique "
          f"(about {chips['chip_id'].nunique() * 0.25 * 0.25 * 111 * 85:,.0f} km²); "
          f"{int((chips['n_fixes'] > 0).sum()):,} crossed by flights, "
          f"{int(((chips['n_fixes'] > 0) & ~chips['in_area']).sum()):,} of them outside the competition areas")
    if len(flown):
        covered = chips["n_fixes"].sum() / flown["n_fixes"].sum()
        print(f"Flight fixes inside the site chips: {covered:.2%} "
              f"(the rest: chips crossed for under a minute in all"
              + (f", or {len(no_site)} competitions without a site" if no_site else "") + ")")
    print("\nSites with the most flights:")
    print(sites.head(20)[["site_id", "name", "country", "n_events", "n_flights", "n_chips", "n_chips_flown",
                          "first_date", "last_date"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
