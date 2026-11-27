#!/usr/bin/env python3
"""ERA5 for every site, read from the ARCO-ERA5 copy on Google Cloud Storage, without the CDS queue.

ARCO-ERA5 (Google Research) stores the whole ERA5 archive as Zarr, one global field per variable
and hour (all 37 pressure levels together), open to anyone. Reading a field costs the same whatever
part of the world is kept, so the work is set by the hours needed, not by the number of sites:

    for each site, its competition days and the day before each, from SOLAR_FROM to SOLAR_TO local
    solar time (thermal hours with a margin); hours needed by several sites are read once

Products (--product):

    single   the 21 single-level variables era5.py asks the CDS for (era5.DATASETS["single"])
    levels   wind, temperature, humidity and height at 925, 850, 700, 600 and 500 hPa, as in
             era5_levels, plus specific humidity (q_<hPa>, g/kg); relative humidity is computed
             from it, since ARCO does not store it. Each read brings all 37 levels of the world (about 150 MB);
             use `--every 1` for hourly profiles (skipping any 3-hourly timestamps already saved)
             and `--chunk-hours 50` to checkpoint inside each month.
    extra    more single-level variables for the ground and the air mass: skin temperature, top
             soil moisture, albedo, snow depth, water vapour, freezing level, convective rain,
             friction velocity (ERA5-Land has the first four at 0.1°; these are at 0.25°)
    static   fields that do not change with time, once per chip: ERA5's own ground height (the
             height its boundary layer is measured from), land-sea mask, vegetation cover and
             type, soil type, sub-grid relief → WEATHER/era5_static.parquet

Each month is written to WEATHER/arco/<product>/<year>-<month>.parquet in the tidy form of the
era5_* tables (chip_id, time, converted columns); era5.py merges them into era5_single,
era5_levels and era5_extra when it converts. A second run reads only the hours a month file is
missing (new sites, new days, or filling in 1 h timestamps around an earlier `--every 3` run;
nothing is read twice). Hours after the last date in ARCO are left for the CDS (era5.py).

    python3 scripts/era5_arco.py /content/drive/MyDrive/IGCS/database --product levels --every 1 --threads 20 --chunk-hours 50 --solar 7-20 --no-prev-day
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
G = 9.80665

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

# product -> {ARCO variable: (column, conversion)}; conversion: None, "K" (to °C) or a factor
PRODUCTS = {
    "single": {name: CONVERT[short] for name, short in SHORT.items()},
    # ARCO has specific humidity, not relative humidity: rh_<hPa> is computed from it (see add_rh)
    "levels": {"u_component_of_wind": ("u", None), "v_component_of_wind": ("v", None),
               "temperature": ("t", "K"), "specific_humidity": ("q", 1000.0), "geopotential": ("z", 1 / G)},
    "extra": {"skin_temperature": ("skin_temp_c", "K"), "volumetric_soil_water_layer_1": ("soil_moisture", None),
              "forecast_albedo": ("albedo", None), "snow_depth": ("snow_depth_m", None),
              "total_column_water_vapour": ("water_vapour_kgm2", None), "zero_degree_level": ("freezing_level_m", None),
              "convective_precipitation": ("conv_rain_mmh", 1000.0), "friction_velocity": ("friction_velocity", None)},
}
LEVELS = [925, 850, 700, 600, 500]
STATIC = {"geopotential_at_surface": ("ground_m", 1 / G), "land_sea_mask": ("land_share", None),
          "high_vegetation_cover": ("veg_high", None), "low_vegetation_cover": ("veg_low", None),
          "type_of_high_vegetation": ("veg_high_type", None), "type_of_low_vegetation": ("veg_low_type", None),
          "soil_type": ("soil_type", None), "standard_deviation_of_orography": ("relief_std_m", None),
          "lake_cover": ("lake_share", None)}


def add_rh(df, levels):
    """Relative humidity (%) at each level from specific humidity (q_<hPa>, g/kg) and temperature
    (t_<hPa>, °C), with the saturation pressure ERA5 itself uses: over water above 0 °C, over ice
    below -23 °C, a mix in between (IFS documentation, part IV, 7.4.2)."""
    for lvl in levels:
        if f"q_{lvl}" not in df or f"t_{lvl}" not in df:
            continue
        q = df[f"q_{lvl}"].to_numpy("float64") / 1000.0
        t = df[f"t_{lvl}"].to_numpy("float64") + 273.15
        e = q * lvl * 100.0 / (0.621981 + 0.378019 * q)            # vapour pressure, Pa
        es_water = 611.21 * np.exp(17.502 * (t - 273.16) / (t - 32.19))
        es_ice = 611.21 * np.exp(22.587 * (t - 273.16) / (t + 0.7))
        a = np.clip((t - 250.16) / (273.16 - 250.16), 0.0, 1.0) ** 2
        df[f"rh_{lvl}"] = (100.0 * e / (a * es_water + (1 - a) * es_ice)).astype("float32")
    return df


def convert_values(x, conv):
    x = x.astype("float64")
    if conv == "K":
        x = x - 273.15
    elif isinstance(conv, float):
        x = x * conv
    return x.astype("float32")


# ── What to read ─────────────────────────────────────────────────────────────

def _site_days(db, include_prev_day=True):
    """{site_id: set of dates} for each site's competition days (optionally plus the day before)."""
    if include_prev_day:
        return site_days(db)
    tasks = pd.read_parquet(db / "tasks.parquet", columns=["task_key", "event_key", "date"])
    links = pd.read_parquet(db / "event_sites.parquet")
    t = tasks.merge(links, on="event_key").dropna(subset=["date"])
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date"])
    return {site: set(g["date"].dt.normalize()) for site, g in t.groupby("site_id")}


def needs(db, solar_from=SOLAR_FROM, solar_to=SOLAR_TO, every=1, include_prev_day=True, from_year=None, to_year=None):
    """{UTC hour: array of chip_ids} for every site's days, local solar hours solar_from..solar_to,
    UTC hours that are multiples of `every`."""
    chips = pd.read_parquet(db / "site_chips.parquet", columns=["site_id", "chip_id", "era5_lon"])
    site_chips = {s: g["chip_id"].to_numpy() for s, g in chips.groupby("site_id")}
    site_lon = chips.groupby("site_id")["era5_lon"].mean()
    hours = {}
    for site, days in _site_days(db, include_prev_day=include_prev_day).items():
        if site not in site_chips:
            continue
        shift = site_lon[site] / 15.0  # solar time = UTC + longitude / 15
        for d in days:
            if from_year is not None and d.year < from_year:
                continue
            if to_year is not None and d.year > to_year:
                continue
            start = (d + pd.Timedelta(hours=solar_from - shift)).floor("h")
            end = (d + pd.Timedelta(hours=solar_to - shift)).ceil("h")
            for t in pd.date_range(start, end, freq="h"):
                if t.hour % every == 0:
                    hours.setdefault(pd.Timestamp(t), []).append(site_chips[site])
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


def read_hours(ds, wanted, variables, threads, levels=None):
    """Tidy rows for {hour: chip_ids}: one global field per variable and hour, `threads` at a time.
    With levels, one column per variable and level (u_850, t_700, ...)."""
    t0 = pd.Timestamp(ds.time.values[0])
    jobs = [(t, name) for t in wanted for name in variables]
    if levels:
        all_levels = [int(x) for x in ds["level"].values]
        lev_idx = [all_levels.index(x) for x in levels]

    def one(job):
        t, name = job
        i = int((t - t0) / pd.Timedelta(hours=1))
        row, col = arco_rc(wanted[t])
        if levels:
            field = ds[name].isel(time=i).values  # all 37 levels: Zarr stores them in one chunk
            return t, name, field[lev_idx][:, row, col]
        return t, name, ds[name].isel(time=i).values[row, col]

    with ThreadPoolExecutor(threads) as pool:
        got = list(pool.map(one, jobs))
    frames = {}
    for t, name, values in got:
        f = frames.setdefault(t, {"chip_id": wanted[t], "time": np.full(len(wanted[t]), t.to_datetime64())})
        col, conv = variables[name]
        if levels:
            for k, lvl in enumerate(levels):
                f[f"{col}_{lvl}"] = convert_values(values[k], conv)
        else:
            f[col] = convert_values(values, conv)
    if not frames:
        return pd.DataFrame()
    df = pd.concat([pd.DataFrame(f) for f in frames.values()], ignore_index=True)
    if levels:
        df = add_rh(df, levels)
    cols = [c for c in df.columns if c not in ("chip_id", "time")]
    return df.dropna(subset=cols, how="all")  # an hour ARCO does not have yet


def missing(path, wanted):
    """The hours of `wanted` the month file lacks (or lacks some chips of)."""
    if not path.exists():
        return dict(wanted)
    have_df = pd.read_parquet(path, columns=["chip_id", "time"])
    have_df["time"] = pd.to_datetime(have_df["time"]).astype("datetime64[ns]")
    have = {pd.Timestamp(k): set(v.tolist()) for k, v in have_df.groupby("time")["chip_id"]}
    return {t: c for t, c in wanted.items() if not set(c.tolist()) <= have.get(pd.Timestamp(t), set())}


def _merge_save_month(path, new_df):
    """Atomically merge `new_df` into an existing month parquet, deduplicating on (chip_id, time)."""
    if new_df is None or not len(new_df):
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
    new_df = new_df.copy()
    new_df["time"] = pd.to_datetime(new_df["time"]).astype("datetime64[ns]")
    if path.exists():
        old = pd.read_parquet(path)
        old["time"] = pd.to_datetime(old["time"]).astype("datetime64[ns]")
        df = pd.concat([old, new_df], ignore_index=True)
        df = df.drop_duplicates(["chip_id", "time"], keep="last")
    else:
        df = new_df
    df = df.sort_values(["chip_id", "time"]).reset_index(drop=True)
    tmp = path.with_suffix(".part")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)
    return df


def _hour_chunks(hours_dict, chunk_hours):
    """Split `{time: chip_ids}` into ordered sub-dicts of at most `chunk_hours` entries."""
    items = sorted(hours_dict.items())
    if not chunk_hours or chunk_hours <= 0 or len(items) <= chunk_hours:
        return [dict(items)]
    return [dict(items[i:i + chunk_hours]) for i in range(0, len(items), chunk_hours)]


def present(ds, variables, required):
    absent = [n for n in variables if n not in ds]
    if absent and required:
        sys.exit(f"ARCO-ERA5 does not have: {', '.join(absent)}")
    if absent:
        print(f"  not in ARCO-ERA5, left out: {', '.join(absent)}", flush=True)
    return {n: v for n, v in variables.items() if n in ds}


def static(db, weather, ds):
    """era5_static.parquet: one row per site chip with the fields that do not change."""
    chips = np.unique(pd.read_parquet(db / "site_chips.parquet", columns=["chip_id"])["chip_id"].to_numpy())
    row, col = arco_rc(chips)
    out = pd.DataFrame({"chip_id": chips})
    t_ok = int((pd.Timestamp("2020-01-01") - pd.Timestamp(ds.time.values[0])) / pd.Timedelta(hours=1))
    for name, (column, conv) in present(ds, STATIC, required=False).items():
        v = ds[name]
        field = (v.isel(time=t_ok) if "time" in v.dims else v).values
        out[column] = convert_values(field[row, col], conv)
    out.to_parquet(weather / "era5_static.parquet", index=False)
    print(f"era5_static.parquet: {len(out):,} chips, columns {', '.join(out.columns[1:])}", flush=True)
    print(out.drop(columns="chip_id").describe().T[["mean", "min", "max"]].round(2).to_string(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder (with sites from build_sites.py)")
    parser.add_argument("--weather", type=Path, help="output folder (default DB/weather)")
    parser.add_argument("--product", default="single", choices=[*PRODUCTS, "static"], help="what to read (default single)")
    parser.add_argument("--threads", type=int, default=32, help="fields read at a time (default 32; 12 for levels, "
                        "each field of all levels is about 150 MB in memory)")
    parser.add_argument("--every", type=int, default=1, help="read UTC hours that are multiples of this (default 1)")
    parser.add_argument("--solar", default=f"{SOLAR_FROM}-{SOLAR_TO}",
                        help=f"local solar hours to keep (default {SOLAR_FROM}-{SOLAR_TO})")
    parser.add_argument("--no-prev-day", action="store_true",
                        help="read competition days only (skip the day before each competition)")
    parser.add_argument("--from-year", type=int, default=None, help="earliest competition year to include")
    parser.add_argument("--to-year", type=int, default=None, help="latest competition year to include")
    parser.add_argument("--chunk-hours", type=int, default=None,
                        help="checkpoint within each month every N hours (default 50 for levels, 0 = whole month)")
    parser.add_argument("--plan-only", action="store_true", help="print how much there is to read and stop")
    args = parser.parse_args()
    weather = args.weather or args.db / "weather"
    weather.mkdir(parents=True, exist_ok=True)
    ds = open_arco()
    if args.product == "static":
        static(args.db, weather, ds)
        return

    out = weather / "arco" / args.product
    out.mkdir(parents=True, exist_ok=True)
    variables = present(ds, PRODUCTS[args.product], required=args.product != "extra")
    levels = LEVELS if args.product == "levels" else None
    lo, hi = (int(x) for x in args.solar.split("-"))
    chunk_hours = args.chunk_hours if args.chunk_hours is not None else (50 if levels else 0)

    wanted = needs(args.db, lo, hi, args.every, include_prev_day=not args.no_prev_day,
                   from_year=args.from_year, to_year=args.to_year)
    stop = last_hour(ds)
    late = {t for t in wanted if stop is not None and t > stop}
    by_month = {}
    for t, c in wanted.items():
        if t not in late:
            by_month.setdefault((t.year, t.month), {})[t] = c
    todo = {m: missing(out / f"{m[0]}-{m[1]:02d}.parquet", h) for m, h in by_month.items()}
    todo = {m: h for m, h in todo.items() if h}
    n_hours = sum(len(h) for h in todo.values())
    span_note = f", years {args.from_year or '..'}-{args.to_year or '..'}" if (args.from_year or args.to_year) else ""
    day_note = ", comp days only" if args.no_prev_day else ""
    print(f"ARCO-ERA5 {args.product}: {len(wanted):,} UTC hours needed ({len({t.normalize() for t in wanted}):,} days, "
          f"{len(by_month):,} months, every {args.every} h, solar {lo}-{hi}{day_note}{span_note}); "
          f"{n_hours:,} hours in {len(todo):,} months to read "
          f"= {n_hours * len(variables):,} global fields{' of 37 levels' if levels else ''}", flush=True)
    if late:
        print(f"  {len(late):,} hours are after ARCO's last date ({stop:%Y-%m-%d}); era5.py gets those from the CDS",
              flush=True)
    if args.plan_only or not todo:
        return

    started, done = time.time(), 0
    for (year, month), hours in sorted(todo.items()):
        path = out / f"{year}-{month:02d}.parquet"
        chunks = _hour_chunks(hours, chunk_hours)
        for ci, chunk in enumerate(chunks, 1):
            t = time.time()
            new_df = read_hours(ds, chunk, variables, args.threads, levels)
            df = _merge_save_month(path, new_df)
            done += len(chunk)
            took = time.time() - t
            rate = done / max(time.time() - started, 1e-9)
            tag = f" [{ci}/{len(chunks)}]" if len(chunks) > 1 else ""
            n_chips = df["chip_id"].nunique() if len(df) else 0
            print(f"... {year}-{month:02d}{tag}: {len(chunk)} hours, {n_chips} chips, "
                  f"{took:.0f} s ({len(chunk) * len(variables) / max(took, 1e-9):.1f} fields/s) | "
                  f"{done:,}/{n_hours:,} hours, about {(n_hours - done) / rate / 3600:.1f} h left", flush=True)
    print(f"Done in {(time.time() - started) / 3600:.1f} h. Run era5.py --convert-only to merge them into the "
          f"era5_* tables.", flush=True)


if __name__ == "__main__":
    main()
