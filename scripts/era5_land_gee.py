#!/usr/bin/env python3
"""ERA5-Land for every site from Google Earth Engine (ECMWF/ERA5_LAND/HOURLY), without the CDS
queue: the same table as era5.py's era5_land, in hours instead of days.

For each site and each of its competition days (and the day before), one request: the ERA5-Land
points (0.1°) of the site's chips, every hour from SOLAR_FROM to SOLAR_TO local solar time
(ee.ImageCollection.getRegion). Requests run several at a time on Earth Engine's high-volume
endpoint. Earth Engine already gives the hourly fluxes (bands ending in _hourly), so nothing has
to be de-accumulated.

    ERA5-Land band (Earth Engine)              column               unit
    surface_sensible_heat_flux_hourly          sensible_heat_wm2    W/m², up positive
    surface_solar_radiation_downwards_hourly   sunshine_wm2         W/m²
    skin_temperature                           skin_temp_c          °C
    volumetric_soil_water_layer_1              soil_moisture        m³/m³
    forecast_albedo                            albedo               0-1
    snow_cover                                 snow_cover           0-1

Each month goes to WEATHER/gee/land/<year>-<month>.parquet (chip_id, time, lat, lon, columns) with
<year>-<month>.done.json listing the site-days already read, so a second run only reads what is
missing. era5.py merges these files into era5_land.parquet when it converts.

Needs an Earth Engine project (--project): in Colab, run ee.Authenticate() in the notebook first.

    python3 scripts/era5_land_gee.py /content/drive/MyDrive/IGCS/database --project my-ee-project
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from era5 import site_days  # noqa: E402
from pta import grid  # noqa: E402

COLLECTION = "ECMWF/ERA5_LAND/HOURLY"
HIGH_VOLUME = "https://earthengine-highvolume.googleapis.com"
SOLAR_FROM, SOLAR_TO = 5, 21
STEP = 0.1
# Earth Engine band -> (column, conversion: factor, or "K" for kelvin -> °C)
BANDS = {
    "surface_sensible_heat_flux_hourly": ("sensible_heat_wm2", -1 / 3600.0),   # J/m² over the hour, down positive
    "surface_solar_radiation_downwards_hourly": ("sunshine_wm2", 1 / 3600.0),
    "skin_temperature": ("skin_temp_c", "K"),
    "volumetric_soil_water_layer_1": ("soil_moisture", None),
    "forecast_albedo": ("albedo", None),
    "snow_cover": ("snow_cover", 0.01),                                         # percent -> 0-1
}


def init(project):
    import ee

    try:
        ee.Initialize(project=project, opt_url=HIGH_VOLUME)
    except Exception:  # noqa: BLE001  (no Earth Engine credentials saved: use the Colab/gcloud login)
        import google.auth

        creds, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/earthengine",
                                               "https://www.googleapis.com/auth/cloud-platform"])
        ee.Initialize(creds, project=project, opt_url=HIGH_VOLUME)
    return ee


def land_points(chips):
    """ERA5-Land points (0.1° grid) inside or just around each chip -> unique (lat, lon) array."""
    pts = set()
    for c in chips:
        s, n, w, e = grid.chip_bounds(c)
        for la in np.arange(np.floor((s - STEP / 2) / STEP), np.ceil((n + STEP / 2) / STEP) + 1) * STEP:
            for lo in np.arange(np.floor((w - STEP / 2) / STEP), np.ceil((e + STEP / 2) / STEP) + 1) * STEP:
                if s - STEP <= la <= n + STEP and w - STEP <= lo <= e + STEP:
                    pts.add((round(float(la), 1), round(float(lo), 1)))
    return sorted(pts)


def requests(db, solar_from, solar_to):
    """One request per site and day: its points and the UTC span of the day's thermal hours."""
    chips = pd.read_parquet(db / "site_chips.parquet", columns=["site_id", "chip_id", "era5_lon"])
    site_chips = {s: g["chip_id"].tolist() for s, g in chips.groupby("site_id")}
    site_lon = chips.groupby("site_id")["era5_lon"].mean()
    points = {s: land_points(c) for s, c in site_chips.items()}
    out = []
    for site, days in site_days(db).items():
        if site not in points:
            continue
        shift = site_lon[site] / 15.0
        for d in sorted(days):
            start = (d + pd.Timedelta(hours=solar_from - shift)).floor("h")
            end = (d + pd.Timedelta(hours=solar_to - shift)).ceil("h") + pd.Timedelta(hours=1)
            out.append({"key": f"{site}|{d:%Y-%m-%d}", "site": site, "month": f"{d:%Y-%m}",
                        "start": start, "end": end, "points": points[site]})
    return out


def fetch(ee, req, bands, retries=5):
    """getRegion for one request -> tidy rows."""
    coll = ee.ImageCollection(COLLECTION).filterDate(req["start"].strftime("%Y-%m-%dT%H:%M:%S"),
                                                     req["end"].strftime("%Y-%m-%dT%H:%M:%S")).select(bands)
    geom = ee.Geometry.MultiPoint([[lon, lat] for lat, lon in req["points"]])
    for attempt in range(retries):
        try:
            rows = coll.getRegion(geom, 11132).getInfo()
            break
        except Exception:  # noqa: BLE001  (quota, timeouts: wait and try again)
            if attempt == retries - 1:
                raise
            time.sleep(10 * (attempt + 1) + np.random.uniform(0, 5))
    head, body = rows[0], rows[1:]
    if not body:
        return None   # no images for these hours: Earth Engine does not have them yet
    df = pd.DataFrame(body, columns=head)
    out = pd.DataFrame({"time": pd.to_datetime(df["time"], unit="ms"),
                        "lat": df["latitude"].astype(float).round(1), "lon": df["longitude"].astype(float).round(1)})
    for band in bands:
        col, conv = BANDS[band]
        x = pd.to_numeric(df[band], errors="coerce").astype("float64")
        if conv == "K":
            x = x - 273.15
        elif conv is not None:
            x = x * conv
        out[col] = x.astype("float32")
    out["chip_id"] = grid.chip_id(out["lat"].to_numpy(), out["lon"].to_numpy())
    return out.dropna(subset=[BANDS[b][0] for b in bands], how="all")   # sea points have no land data


def really_done(out, month, reqs):
    """Site-days listed as done whose rows are in the month's file (a first version marked days
    Earth Engine did not have yet as done, with no rows)."""
    done_file, path = out / f"{month}.done.json", out / f"{month}.parquet"
    listed = set(json.loads(done_file.read_text())) if done_file.exists() else set()
    if not listed or not path.exists():
        return set()
    d = pd.read_parquet(path, columns=["lat", "lon", "time"])
    have = set(zip(d["lat"].round(1), d["lon"].round(1), d["time"].dt.floor("h")))
    ok = set()
    for r in reqs:
        if r["key"] not in listed:
            continue
        mid = (r["start"] + (r["end"] - r["start"]) / 2).floor("h")
        if any((la, lo, mid) in have for la, lo in r["points"]):
            ok.add(r["key"])
    return ok


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path)
    parser.add_argument("--project", required=True, help="Google Cloud project registered for Earth Engine")
    parser.add_argument("--weather", type=Path)
    parser.add_argument("--threads", type=int, default=16, help="requests to Earth Engine at a time (default 16)")
    parser.add_argument("--solar", default=f"{SOLAR_FROM}-{SOLAR_TO}")
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    out = (args.weather or args.db / "weather") / "gee" / "land"
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in args.solar.split("-"))

    reqs = requests(args.db, lo, hi)
    by_month = {}
    for r in reqs:
        by_month.setdefault(r["month"], []).append(r)
    todo = {}
    for m, rs in by_month.items():
        done = really_done(out, m, rs)
        left = [r for r in rs if r["key"] not in done]
        if left:
            todo[m] = left
    n = sum(len(v) for v in todo.values())
    pts = sum(len(r["points"]) for v in todo.values() for r in v)
    print(f"ERA5-Land from Earth Engine: {len(reqs):,} site-days in {len(by_month)} months; {n:,} to read in "
          f"{len(todo)} months ({pts:,} point-days)", flush=True)
    if args.plan_only or not todo:
        return

    ee = init(args.project)
    have = ee.ImageCollection(COLLECTION).first().bandNames().getInfo()
    bands = [b for b in BANDS if b in have]
    missing = [b for b in BANDS if b not in have]
    if missing:
        print(f"  not in {COLLECTION}, left out: {', '.join(missing)}", flush=True)
    started, done_n, seen, failed, empty = time.time(), 0, 0, 0, 0
    for m in sorted(todo):
        t0 = time.time()
        frames, ok_keys = [], []
        with ThreadPoolExecutor(args.threads) as pool:
            futs = {pool.submit(fetch, ee, r, bands): r for r in todo[m]}
            for f in as_completed(futs):
                r = futs[f]
                try:
                    got = f.result()
                    if got is None:
                        empty += 1
                        continue
                    frames.append(got)
                    ok_keys.append(r["key"])
                except Exception as e:  # noqa: BLE001
                    failed += 1
                    print(f"    {r['key']} failed: {str(e)[:200]}", flush=True)
        path = out / f"{m}.parquet"
        parts = ([pd.read_parquet(path)] if path.exists() else []) + [x for x in frames if len(x)]
        if parts:
            df = pd.concat(parts, ignore_index=True)
            df = df.drop_duplicates(["lat", "lon", "time"], keep="last").sort_values(["chip_id", "time", "lat", "lon"])
            tmp = path.with_suffix(".part")
            df.to_parquet(tmp, index=False)
            tmp.replace(path)
        done_file = out / f"{m}.done.json"
        prev = set(json.loads(done_file.read_text())) if done_file.exists() else set()
        done_file.write_text(json.dumps(sorted(prev | set(ok_keys))))
        done_n += len(ok_keys)
        seen += len(todo[m])
        rate = seen / max(time.time() - started, 1e-9)
        print(f"... {m}: {len(ok_keys)}/{len(todo[m])} site-days, {time.time() - t0:.0f} s | {seen:,}/{n:,}, "
              f"about {(n - seen) / max(rate, 1e-9) / 3600:.1f} h left" + (f", {failed} failed" if failed else "")
              + (f", {empty} not in Earth Engine yet" if empty else ""), flush=True)
    print(f"Done in {(time.time() - started) / 3600:.1f} h"
          + (f"; {failed} site-days failed, run again to retry them" if failed else "")
          + (f"; {empty} site-days are after Earth Engine's last date: notebook 07 (CDS, --datasets land) "
             f"can get those" if empty else "")
          + ". Run era5.py --convert-only --tables land to build era5_land.parquet.", flush=True)


if __name__ == "__main__":
    main()
