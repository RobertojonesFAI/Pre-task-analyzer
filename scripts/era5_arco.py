#!/usr/bin/env python3
"""ERA5 single levels for every site, read from the ARCO-ERA5 copy on Google Cloud Storage.

The same 21 variables that era5.py asks the CDS for (see era5.DATASETS["single"]), without the CDS
queue. ARCO-ERA5 (Google Research) stores the whole ERA5 archive as Zarr, one global field per
variable and hour, open to anyone. Reading one field costs the same whatever part of the world is
kept, so the work is set by the hours needed, not by the number of sites:

    for each site, its competition days and the day before each, from SOLAR_FROM to SOLAR_TO local
    solar time (thermal hours with a margin); hours needed by several sites are read once

Each month is written to WEATHER/arco/single/<year>-<month>.parquet in the tidy form of
era5_single.parquet (chip_id, time, converted columns); era5.py merges these files into
era5_single.parquet when it converts. A second run reads only the hours a month file is missing
(new sites or days are added, nothing is read twice). Hours after the last date in ARCO are left
for the CDS (era5.py).

    python3 scripts/era5_arco.py /content/drive/MyDrive/IGCS/database --threads 32
"""

import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from era5 import CONVERT, DATASETS, site_days  # noqa: E402

ARCO_URL = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
SOLAR_FROM, SOLAR_TO = 5, 21

# ARCO variable (CDS name) -> netCDF short name used in CONVERT
SHORT = {
    "boundary_layer_height": "blh", "cloud_base_height": "cbh", "convective_available_potential_energy": "cape",
    "convective_inhibition": "cin", "k_index": "kx", "surface_sensible_heat_flux": "sshf",
    "surface_latent_heat_flux": "slhf", "surface_solar_radiation_downwards": "ssrd", "total_cloud_cover": "tcc",
    "low_cloud_cover": "lcc", "medium_cloud_cover": "mcc", "high_cloud_cover": "hcc", "2m_temperature": "t2m",
    "2m_dewpoint_temperature": "d2m", "10m_u_component_of_wind": "u10", "10m_v_component_of_wind": "v10",
    "100m_u_component_of_wind": "u100", "100m_v_component_of_wind": "v100",
    "instantaneous_10m_wind_gust": "i10fg", "surface_pressure": "sp", "total_precipitation": "tp",
}
assert set(SHORT) == set(DATASETS["single"]["variables"])


def convert_values(name, x):
    """(column, values) for an ARCO variable, in the units of era5_single.parquet."""
    col, conv = CONVERT[SHORT[name]]
    x = x.astype("float64")
    if conv == "K":
        x = x - 273.15
    elif isinstance(conv, float):
        x = x * conv
    return col, x.astype("float32")


# ── What to read ─────────────────────────────────────────────────────────────

def needs(db, solar_from=SOLAR_FROM, solar_to=SOLAR_TO):
    """{UTC hour: array of chip_ids} for every site's days, local solar hours solar_from..solar_to."""
    chips = pd.read_parquet(db / "site_chips.parquet", columns=["site_id", "chip_id", "era5_lon"])
    site_chips = {s: g["chip_id"].to_numpy() for s, g in chips.groupby("site_id")}
    site_lon = chips.groupby("site_id")["era5_lon"].mean()
    hours = {}
    for site, days in site_days(db).items():
        if site not in site_chips:
            continue
        shift = site_lon[site] / 15.0  # solar time = UTC + longitude / 15
        for d in days:
            start = (d + pd.Timedelta(hours=solar_from - shift)).floor("h")
            end = (d + pd.Timedelta(hours=solar_to - shift)).ceil("h")
            for t in pd.date_range(start, end, freq="h"):
                hours.setdefault(t, []).append(site_chips[site])
    return {t: np.unique(np.concatenate(v)) for t, v in sorted(hours.items())}


def arco_rc(chip_ids):
    """Row and column of each chip's ERA5 point in an ARCO field (latitude 90 → -90, longitude 0 → 359.75)."""
    from pta import grid

    lat, lon = grid.chip_centre(np.asarray(chip_ids))
    row = np.rint((90.0 - lat) / 0.25).astype(int)
    col = np.rint((lon % 360.0) / 0.25).astype(int) % 1440
    return row, col


# ── Reading ──────────────────────────────────────────────────────────────────

def open_arco():
    import xarray as xr

    return xr.open_zarr(ARCO_URL, chunks=None, storage_options={"token": "anon"})


def last_hour(ds):
    """The last hour with data (the time axis runs to 2050; later hours are empty)."""
    for key in ("valid_time_stop_era5t", "valid_time_stop"):
        if key in ds.attrs:
            try:
                return pd.Timestamp(ds.attrs[key]) + pd.Timedelta(hours=23)
            except (ValueError, TypeError):
                pass
    return None


def read_hours(ds, wanted, threads):
    """Tidy rows for {hour: chip_ids}: one global field per variable and hour, read `threads` at a time."""
    t0 = pd.Timestamp(ds.time.values[0])
    jobs = [(t, name) for t in wanted for name in SHORT]

    def one(job):
        t, name = job
        i = int((t - t0) / pd.Timedelta(hours=1))
        field = ds[name].isel(time=i).values
        row, col = arco_rc(wanted[t])
        return t, name, field[row, col]

    with ThreadPoolExecutor(threads) as pool:
        got = list(pool.map(one, jobs))
    frames = {}
    for t, name, values in got:
        f = frames.setdefault(t, {"chip_id": wanted[t], "time": np.full(len(wanted[t]), t.to_datetime64())})
        col, x = convert_values(name, values)
        f[col] = x
    if not frames:
        return pd.DataFrame()
    df = pd.concat([pd.DataFrame(f) for f in frames.values()], ignore_index=True)
    cols = [c for c in df.columns if c not in ("chip_id", "time")]
    return df.dropna(subset=cols, how="all")  # an hour ARCO does not have yet


def missing(path, wanted):
    """The hours of `wanted` the month file lacks (or lacks some chips of)."""
    if not path.exists():
        return dict(wanted)
    have = pd.read_parquet(path, columns=["chip_id", "time"])
    have = have.groupby("time")["chip_id"].apply(set).to_dict()
    return {t: c for t, c in wanted.items() if not set(c.tolist()) <= have.get(t, set())}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder (with sites from build_sites.py)")
    parser.add_argument("--weather", type=Path, help="output folder (default DB/weather)")
    parser.add_argument("--threads", type=int, default=32, help="fields read at a time (default 32)")
    parser.add_argument("--solar", default=f"{SOLAR_FROM}-{SOLAR_TO}",
                        help=f"local solar hours to keep (default {SOLAR_FROM}-{SOLAR_TO})")
    parser.add_argument("--plan-only", action="store_true", help="print how much there is to read and stop")
    args = parser.parse_args()
    out = (args.weather or args.db / "weather") / "arco" / "single"
    out.mkdir(parents=True, exist_ok=True)
    lo, hi = (int(x) for x in args.solar.split("-"))

    wanted = needs(args.db, lo, hi)
    ds = open_arco()
    absent = [n for n in SHORT if n not in ds]
    if absent:
        sys.exit(f"ARCO-ERA5 does not have: {', '.join(absent)}")
    stop = last_hour(ds)
    late = {t for t in wanted if stop is not None and t > stop}
    by_month = {}
    for t, c in wanted.items():
        if t not in late:
            by_month.setdefault((t.year, t.month), {})[t] = c
    todo = {m: missing(out / f"{m[0]}-{m[1]:02d}.parquet", h) for m, h in by_month.items()}
    todo = {m: h for m, h in todo.items() if h}
    n_hours = sum(len(h) for h in todo.values())
    print(f"ARCO-ERA5: {len(wanted):,} UTC hours needed ({len({t.normalize() for t in wanted}):,} days, "
          f"{len(by_month):,} months); {n_hours:,} hours in {len(todo):,} months to read "
          f"= {n_hours * len(SHORT):,} global fields", flush=True)
    if late:
        print(f"  {len(late):,} hours are after ARCO's last date ({stop:%Y-%m-%d}); era5.py gets those from the CDS",
              flush=True)
    if args.plan_only or not todo:
        return

    started, done = time.time(), 0
    for (year, month), hours in sorted(todo.items()):
        t = time.time()
        df = read_hours(ds, hours, args.threads)
        path = out / f"{year}-{month:02d}.parquet"
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
            df = df.drop_duplicates(["chip_id", "time"], keep="last")
        df = df.sort_values(["chip_id", "time"]).reset_index(drop=True)
        tmp = path.with_suffix(".part")
        df.to_parquet(tmp, index=False)
        tmp.replace(path)
        done += len(hours)
        took = time.time() - t
        rate = done / (time.time() - started)
        print(f"... {year}-{month:02d}: {len(hours)} hours, {df['chip_id'].nunique()} chips, "
              f"{took:.0f} s ({len(hours) * len(SHORT) / max(took, 1e-9):.1f} fields/s) | "
              f"{done:,}/{n_hours:,} hours, about {(n_hours - done) / rate / 3600:.1f} h left", flush=True)
    print(f"Done in {(time.time() - started) / 3600:.1f} h. Run era5.py (or notebook 07) to merge them into "
          f"era5_single.parquet.", flush=True)


if __name__ == "__main__":
    main()
