#!/usr/bin/env python3
"""Hourly ERA5 and ERA5-Land weather for every site, on the days of its competitions.

Asks the Copernicus Climate Data Store (CDS) for the hours of each competition day and the day
before it (morning heating starts the evening before for soil and boundary layer; ERA5-Land
accumulations need the previous hour). Sites flown in the same month are grouped into regional
boxes of up to BOX_DEG degrees, so one request serves several sites. Three datasets:

    single     ERA5 single levels (0.25°): boundary-layer height, cloud base, CAPE, CIN, heat
               fluxes, sunshine, cloud cover, temperature, dew point, wind at 10 and 100 m,
               gusts, pressure, rain
    levels     ERA5 pressure levels (0.25°): wind, temperature, humidity and height at 925, 850,
               700, 600 and 500 hPa
    land       ERA5-Land (0.1°): sensible heat flux, skin temperature, soil moisture, albedo,
               snow cover, sunshine

Single levels can instead be read from ARCO-ERA5 on Google Cloud Storage (era5_arco.py, much faster
than the CDS queue); run this script with --datasets levels,land and the conversion merges both.

Raw files go to WEATHER/raw/<dataset>/, so a second run only asks for what is missing. They are
then converted to tidy tables:

    WEATHER/era5_single.parquet   one row per chip (ERA5 point) and UTC hour
    WEATHER/era5_levels.parquet   the same, levels as columns (u_850, t_700, ...)
    WEATHER/era5_land.parquet     one row per 0.1° ERA5-Land point and UTC hour

Fluxes are converted to W/m² (positive upward for the heat fluxes, the way pilots think of them:
the ground heating the air), temperatures to °C, pressure to hPa, rain to mm/h, geopotential to
metres. The CDS key is read from the CDSAPI_KEY environment variable.

    CDSAPI_KEY=... python3 scripts/era5.py /content/drive/MyDrive/IGCS/database --parallel 6
"""

import argparse
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pta import grid  # noqa: E402

CDS_URL = "https://cds.climate.copernicus.eu/api"
BOX_DEG = 25.0           # sites flown in one month are grouped into boxes up to this size
HOURS = [f"{h:02d}:00" for h in range(24)]
LEVELS = ["925", "850", "700", "600", "500"]

DATASETS = {
    "single": {
        "name": "reanalysis-era5-single-levels",
        "variables": [
            "boundary_layer_height", "cloud_base_height", "convective_available_potential_energy",
            "convective_inhibition", "k_index", "surface_sensible_heat_flux", "surface_latent_heat_flux",
            "surface_solar_radiation_downwards", "total_cloud_cover", "low_cloud_cover",
            "medium_cloud_cover", "high_cloud_cover", "2m_temperature", "2m_dewpoint_temperature",
            "10m_u_component_of_wind", "10m_v_component_of_wind", "100m_u_component_of_wind",
            "100m_v_component_of_wind", "instantaneous_10m_wind_gust", "surface_pressure",
            "total_precipitation",
        ],
        "extra": {"product_type": ["reanalysis"]},
        "step": 0.25,
    },
    "levels": {
        "name": "reanalysis-era5-pressure-levels",
        "variables": ["u_component_of_wind", "v_component_of_wind", "temperature", "relative_humidity",
                      "geopotential"],
        "extra": {"product_type": ["reanalysis"], "pressure_level": LEVELS},
        "step": 0.25,
    },
    "land": {
        "name": "reanalysis-era5-land",
        "variables": ["surface_sensible_heat_flux", "skin_temperature", "volumetric_soil_water_layer_1",
                      "forecast_albedo", "snow_cover", "surface_solar_radiation_downwards"],
        "extra": {},
        "step": 0.1,
    },
}

# netCDF short name -> (column, conversion)
ACCUM_HOURLY = 1 / 3600.0     # J/m² over one hour -> W/m²
CONVERT = {
    "blh": ("blh_m", None), "cbh": ("cloud_base_m", None), "cape": ("cape", None), "cin": ("cin", None),
    "kx": ("k_index", None),
    "sshf": ("sensible_heat_wm2", -ACCUM_HOURLY), "slhf": ("latent_heat_wm2", -ACCUM_HOURLY),
    "ssrd": ("sunshine_wm2", ACCUM_HOURLY),
    "tcc": ("cloud_total", None), "lcc": ("cloud_low", None), "mcc": ("cloud_mid", None), "hcc": ("cloud_high", None),
    "t2m": ("t2m_c", "K"), "d2m": ("dewpoint_c", "K"), "skt": ("skin_temp_c", "K"),
    "u10": ("u10", None), "v10": ("v10", None), "u100": ("u100", None), "v100": ("v100", None),
    "i10fg": ("gust10", None), "sp": ("pressure_hpa", 0.01), "tp": ("rain_mmh", 1000.0),
    "swvl1": ("soil_moisture", None), "fal": ("albedo", None), "snowc": ("snow_cover", 0.01),
    "u": ("u", None), "v": ("v", None), "t": ("t", "K"), "r": ("rh", None), "z": ("z", 1 / 9.80665),
}
LAND_ACCUM = {"sshf", "ssrd"}   # ERA5-Land accumulates from 00 UTC through the day


# ── Plan ─────────────────────────────────────────────────────────────────────

def site_days(db):
    """{site_id: set of dates} (competition days and the day before each)."""
    tasks = pd.read_parquet(db / "tasks.parquet", columns=["task_key", "event_key", "date"])
    links = pd.read_parquet(db / "event_sites.parquet")
    t = tasks.merge(links, on="event_key").dropna(subset=["date"])
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"])
    days = {}
    for site, g in t.groupby("site_id"):
        d = set(g["date"].dt.normalize())
        days[site] = d | {x - pd.Timedelta(days=1) for x in d}
    return days


def site_boxes(db):
    """{site_id: (south, north, west, east)} of the ERA5 points of the site's chips."""
    chips = pd.read_parquet(db / "site_chips.parquet")
    return {s: (g["era5_lat"].min(), g["era5_lat"].max(), g["era5_lon"].min(), g["era5_lon"].max())
            for s, g in chips.groupby("site_id")}


def plan(db, box_deg=BOX_DEG, datasets=tuple(DATASETS)):
    """Requests: one per (dataset, month, regional group of sites). The CDS works through a few
    requests per user at a time, so fewer, larger requests finish sooner. Ordered by dataset
    (most useful first), then month."""
    days, boxes = site_days(db), site_boxes(db)
    by_month = {}
    for site, ds in days.items():
        if site not in boxes:
            continue
        for d in ds:
            by_month.setdefault((d.year, d.month), {}).setdefault(site, set()).add(d.day)
    jobs = []
    for (year, month), sites in sorted(by_month.items()):
        groups = []  # [box, {site: days}]
        for site in sorted(sites, key=lambda s: boxes[s][2]):
            b = boxes[site]
            for g in groups:
                u = (min(g[0][0], b[0]), max(g[0][1], b[1]), min(g[0][2], b[2]), max(g[0][3], b[3]))
                if u[1] - u[0] <= box_deg and u[3] - u[2] <= box_deg:
                    g[0] = u
                    g[1][site] = sites[site]
                    break
            else:
                groups.append([b, {site: sites[site]}])
        for i, (box, members) in enumerate(groups):
            all_days = sorted(set().union(*members.values()))
            for key in datasets:
                jobs.append({"dataset": key, "year": year, "month": month, "days": all_days, "box": box,
                             "sites": sorted(members), "name": f"{year}-{month:02d}_b{int(box_deg)}_{i:02d}"})
    order = {k: i for i, k in enumerate(DATASETS)}
    return sorted(jobs, key=lambda j: (order[j["dataset"]], j["year"], j["month"], j["name"]))


def request(job):
    d = DATASETS[job["dataset"]]
    s, n, w, e = job["box"]
    pad = d["step"] if job["dataset"] == "land" else 0.0   # ERA5-Land points around the chips' edges
    return {
        **d["extra"], "variable": d["variables"], "year": [str(job["year"])], "month": [f"{job['month']:02d}"],
        "day": [f"{x:02d}" for x in job["days"]], "time": HOURS,
        "area": [round(n + pad + 0.125 * (pad > 0), 2), round(w - pad - 0.125 * (pad > 0), 2),
                 round(s - pad - 0.125 * (pad > 0), 2), round(e + pad + 0.125 * (pad > 0), 2)],
        "data_format": "netcdf", "download_format": "unarchived",
    }


# ── Download ─────────────────────────────────────────────────────────────────

def raw_path(weather, job):
    return weather / "raw" / job["dataset"] / f"{job['name']}.nc"


def have(weather, job):
    """Already downloaded: this request's file, or files from an earlier run with another box size
    whose boxes, together, cover all of this request's sites for the month (recorded in the
    .sites.json written next to each file)."""
    if raw_path(weather, job).exists():
        return True
    import json

    folder = weather / "raw" / job["dataset"]
    covered = set()
    for meta in folder.glob(f"{job['year']}-{job['month']:02d}_*.sites.json"):
        if meta.with_name(meta.name.replace(".sites.json", ".nc")).exists():
            covered |= set(json.loads(meta.read_text()))
    return set(job["sites"]) <= covered


def note_old_files(db, weather):
    """Files from the first runs (12° boxes, named <year>-<month>_<n>.nc) get their .sites.json,
    so a run with other boxes does not ask for them again."""
    import json
    import re

    old = [p for p in (weather / "raw").glob("*/*.nc") if re.fullmatch(r"\d{4}-\d{2}_\d{2}\.nc", p.name)
           and not p.with_name(p.name.replace(".nc", ".sites.json")).exists()]
    if not old:
        return
    sites = {f"{j['dataset']}/{j['name'].replace('_b12_', '_')}": j["sites"] for j in plan(db, 12.0)}
    for p in old:
        key = f"{p.parent.name}/{p.stem}"
        if key in sites:
            p.with_name(p.name.replace(".nc", ".sites.json")).write_text(json.dumps(sites[key]))


def download(job, weather, key):
    import cdsapi

    dest = raw_path(weather, job)
    if dest.exists():
        return job, "cached"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".part")
    client = cdsapi.Client(url=CDS_URL, key=key, quiet=True, progress=False)
    client.retrieve(DATASETS[job["dataset"]]["name"], request(job), str(tmp))
    import json
    dest.with_name(dest.name.replace(".nc", ".sites.json")).write_text(json.dumps(job["sites"]))
    tmp.rename(dest)  # may hold a zip of several netCDF files; read() handles both
    return job, "ok"


# ── Convert ──────────────────────────────────────────────────────────────────

def _datasets(path):
    """The netCDF datasets in a downloaded file (the CDS sends a zip when variables come from
    several streams, even when asked for unarchived files)."""
    import tempfile

    import xarray as xr

    if not zipfile.is_zipfile(path):
        return [xr.load_dataset(path)]
    out = []
    with zipfile.ZipFile(path) as z, tempfile.TemporaryDirectory() as tmp:
        for m in z.namelist():
            if m.endswith(".nc"):
                out.append(xr.load_dataset(z.extract(m, tmp)))
    return out


def _tidy(ds, dataset):
    """One row per point and hour, columns converted (see CONVERT)."""
    ds = ds.rename({k: v for k, v in {"valid_time": "time", "pressure_level": "level"}.items() if k in ds.dims or k in ds})
    ds = ds.drop_vars([v for v in ("number", "expver") if v in ds.variables and v not in ds.dims], errors="ignore")
    df = ds.to_dataframe().reset_index()
    keep = ["time", "latitude", "longitude"] + (["level"] if "level" in df else [])
    out = df[keep].copy()
    for short in ds.data_vars:
        if short not in CONVERT:
            continue
        col, conv = CONVERT[short]
        x = df[short].astype("float64")
        if conv == "K":
            x = x - 273.15
        elif isinstance(conv, float) and not (dataset == "land" and short in LAND_ACCUM):
            x = x * conv
        out[col] = x
    if "level" in out:
        out["level"] = out["level"].astype(float).astype(int)
        values = [c for c in out.columns if c not in ("time", "latitude", "longitude", "level")]
        out = out.pivot_table(index=["time", "latitude", "longitude"], columns="level", values=values)
        out.columns = [f"{c}_{lvl}" for c, lvl in out.columns]
        out = out.reset_index()
    return out


def _deaccumulate_land(df):
    """ERA5-Land fluxes accumulate from 00 UTC: hourly value = this hour's total minus the
    previous hour's, except at 01 UTC (the first hour of the day's accumulation)."""
    df = df.sort_values(["latitude", "longitude", "time"])
    for col, sign in (("sensible_heat_wm2", -1.0), ("sunshine_wm2", 1.0)):
        if col not in df:
            continue
        prev = df.groupby(["latitude", "longitude"])[col].shift(1)
        gap = df.groupby(["latitude", "longitude"])["time"].diff() != pd.Timedelta(hours=1)
        hourly = np.where(df["time"].dt.hour == 1, df[col], df[col] - prev)
        hourly = np.where(gap & (df["time"].dt.hour != 1), np.nan, hourly)
        df[col] = sign * hourly / 3600.0
    return df


def convert(weather, wanted_chips):
    """Tidy tables from every raw file."""
    for dataset in DATASETS:
        files = sorted((weather / "raw" / dataset).glob("*.nc"))
        arco = sorted((weather / "arco" / dataset).glob("*.parquet"))  # from era5_arco.py (single levels)
        if not files and not arco:
            continue
        frames = []
        for path in files:
            parts = [_tidy(ds, dataset) for ds in _datasets(path)]
            df = parts[0]
            for p in parts[1:]:
                df = df.merge(p, on=["time", "latitude", "longitude"], how="outer")
            frames.append(df)
        df = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["time", "latitude", "longitude"])
        df = df.drop_duplicates(["time", "latitude", "longitude"], keep="last")
        if dataset == "land":
            df = _deaccumulate_land(df)
            df = df.rename(columns={"latitude": "lat", "longitude": "lon"})
            df["chip_id"] = grid.chip_id(df["lat"].to_numpy(), df["lon"].to_numpy())
            df = df[df["chip_id"].isin(wanted_chips)]
        else:
            df["chip_id"] = grid.chip_id(df["latitude"].round(2).to_numpy(), df["longitude"].round(2).to_numpy())
            df = df[df["chip_id"].isin(wanted_chips)].drop(columns=["latitude", "longitude"])
            if arco:  # the same variables read from ARCO-ERA5; together they cover every hour
                df = pd.concat(([df] if frames else []) + [pd.read_parquet(p) for p in arco], ignore_index=True)
                df = df[df["chip_id"].isin(wanted_chips)].drop_duplicates(["chip_id", "time"], keep="last")
        cols = ["chip_id", "time"] + [c for c in df.columns if c not in ("chip_id", "time")]
        df = df[cols].sort_values(["chip_id", "time"]).reset_index(drop=True)
        for c in df.columns[2:]:
            if df[c].dtype == "float64":
                df[c] = df[c].astype("float32")
        df.to_parquet(weather / f"era5_{dataset}.parquet", index=False)
        print(f"  era5_{dataset}.parquet: {len(df):,} rows, {df['chip_id'].nunique():,} chips, "
              f"{df['time'].dt.normalize().nunique():,} days", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder (with sites from build_sites.py)")
    parser.add_argument("--weather", type=Path, help="output folder (default DB/weather)")
    parser.add_argument("--parallel", type=int, default=8, help="requests queued at the CDS at a time (default 8; "
                        "the CDS finishes queued requests in batches, so fewer in the queue means fewer per hour)")
    parser.add_argument("--box", type=float, default=BOX_DEG, help=f"largest regional box, degrees (default {BOX_DEG:g})")
    parser.add_argument("--datasets", default=",".join(DATASETS), help="which datasets, in this order (default single,levels,land)")
    parser.add_argument("--plan-only", action="store_true", help="print the requests and stop")
    parser.add_argument("--convert-only", action="store_true", help="only rebuild the tables from the raw files")
    args = parser.parse_args()
    weather = args.weather or args.db / "weather"
    weather.mkdir(parents=True, exist_ok=True)

    datasets = tuple(d for d in args.datasets.split(",") if d in DATASETS)
    note_old_files(args.db, weather)
    jobs = plan(args.db, args.box, datasets)
    todo = [j for j in jobs if not have(weather, j)]
    months = len({(j["year"], j["month"]) for j in jobs})
    print(f"ERA5: {len(jobs) // len(datasets):,} regional requests in {months} months × {len(datasets)} datasets; "
          f"{len(jobs) - len(todo):,} files already here, {len(todo):,} to ask for", flush=True)
    if args.plan_only:
        for b in (12, 20, 25, 35):
            print(f"  boxes of {b}°: {len(plan(args.db, b, datasets[:1])):,} requests per dataset")
        for j in [j for j in jobs if j["dataset"] == datasets[0]][:10]:
            print(f"  {j['name']}: {len(j['sites'])} sites, {len(j['days'])} days, box {tuple(round(x, 2) for x in j['box'])}")
        return

    if todo and not args.convert_only:
        key = os.environ.get("CDSAPI_KEY")
        if not key:
            sys.exit("No CDS key: set the CDSAPI_KEY environment variable (Colab: the CDSAPI_KEY secret).")
        started, done, failed = time.time(), 0, 0
        with ThreadPoolExecutor(args.parallel) as pool:
            futures = {pool.submit(download, j, weather, key): j for j in todo}
            for f in as_completed(futures):
                j = futures[f]
                try:
                    f.result()
                    done += 1
                except Exception as e:  # noqa: BLE001  one failed request must not stop the others
                    failed += 1
                    with open(weather / "failed.txt", "a") as log:
                        log.write(f"{time.strftime('%Y-%m-%d %H:%M')} {j['dataset']} {j['name']}: {str(e)[:300]}\n")
                    print(f"    {j['dataset']} {j['name']} failed: {str(e)[:200]}", flush=True)
                size = raw_path(weather, j).stat().st_size / 1e6 if raw_path(weather, j).exists() else 0
                print(f"... {done + failed}/{len(todo)}  {j['dataset']} {j['name']} ({size:.1f} MB)"
                      f"{' FAILED' if not raw_path(weather, j).exists() else ''}, "
                      f"{(time.time() - started) / 60:.0f} min", flush=True)
        if failed:
            print(f"{failed} requests failed (see {weather / 'failed.txt'}); run again to retry them.")

    print("\nConverting", flush=True)
    wanted = set(pd.read_parquet(args.db / "site_chips.parquet")["chip_id"])
    convert(weather, wanted)


if __name__ == "__main__":
    main()
