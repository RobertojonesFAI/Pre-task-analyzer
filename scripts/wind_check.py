#!/usr/bin/env python3
"""The wind pilots measured against the wind of the weather model, at the same place, hour and
height.

Every thermal of notebook 08 with at least two full turns has the wind the pilot measured (the
drift of the circles). ERA5's wind is interpolated to the height of each measurement: 10 m and
100 m above ERA5's ground, then the pressure levels (925, 850, 700, 600, 500 hPa, at their
geopotential heights; levels below ERA5's ground are skipped), at the nearest hour (every 3rd hour
for levels read from ARCO).

The measurements are pooled by site, chip, hour and height band (median of u and v, at least 3
thermals from 2 flights), so single circles and each thermal's own flow weigh little. Bands:

    near the ground   below 400 m above the ground: the wind already shaped by the terrain.
                      This is what a terrain-flow model (WindNinja, the CFD surrogate) has to get
                      right; ERA5 does not resolve it.
    middle            400 to 1000 m
    free air          above 1000 m over the ground, or above the highest ground of the chip: the
                      prevailing wind, the input a terrain-flow model takes from the weather model

Scores per band: vector error, speed bias (pilots minus model), direction error where the model
wind is at least 3 m/s, and the share within 30°; also by site, by solar hour and by ERA5 height.

Writes DB/models/wind_check/: groups.parquet (one row per pooled group with both winds),
by_site.csv, summary.json.

    python3 scripts/wind_check.py /content/drive/MyDrive/IGCS/database
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pta import grid  # noqa: E402

LEVELS = [925, 850, 700, 600, 500]
MIN_THERMALS, MIN_FLIGHTS = 3, 2
CALM_MS = 3.0             # below this the direction of the model wind means little


def band(agl, above_crest):
    return np.where(agl < 400, "near the ground",
                    np.where((agl >= 1000) | (above_crest > 0), "free air", "middle"))


def measurements(db):
    th = pd.read_parquet(db / "thermals.parquet", columns=["track_id", "task_key", "t_start", "t_end", "lat", "lon",
                                                           "alt_base", "agl_base", "wind_u", "wind_v", "wind_alt"])
    th = th.dropna(subset=["wind_u", "wind_v", "wind_alt"])
    off = pd.read_parquet(db / "flight_checks.parquet", columns=["track_id", "agl_offset", "is_drive"])
    th = th.merge(off, on="track_id", how="left")
    th = th[~th["is_drive"].fillna(False).astype(bool)]
    tasks = pd.read_parquet(db / "tasks.parquet", columns=["task_key", "event_key", "date"])
    tasks = tasks.merge(pd.read_parquet(db / "event_sites.parquet", columns=["event_key", "site_id"]), on="event_key")
    th = th.merge(tasks[["task_key", "date", "site_id"]], on="task_key")
    mid = (th["t_start"] + th["t_end"]) / 2
    th["time"] = pd.to_datetime(th["date"], errors="coerce") + pd.to_timedelta(mid, unit="s")
    th = th.dropna(subset=["time"])
    th["hour"] = th["time"].dt.round("h").astype("datetime64[ns]")
    th["hour3"] = th["time"].dt.round("3h").astype("datetime64[ns]")
    th["h_asl"] = th["wind_alt"] - th["agl_offset"].fillna(0)            # calibrated altitude of the measurement
    th["agl"] = th["agl_base"] + (th["wind_alt"] - th["alt_base"])       # height above the ground under it
    th["chip_id"] = grid.chip_id(th["lat"].to_numpy(), th["lon"].to_numpy())
    th["solar_hour"] = ((th["time"].dt.hour + th["time"].dt.minute / 60 + th["lon"] / 15) % 24).round()
    return th.reset_index(drop=True)


def model_wind(th, db):
    """ERA5's u and v interpolated to each measurement's height (above sea level)."""
    w = db / "weather"
    single = pd.read_parquet(w / "era5_single.parquet", columns=["chip_id", "time", "u10", "v10", "u100", "v100"])
    single = single[single["chip_id"].isin(th["chip_id"].unique())].rename(columns={"time": "hour"})
    single["hour"] = single["hour"].astype("datetime64[ns]")
    th = th.merge(single.drop_duplicates(["chip_id", "hour"]), on=["chip_id", "hour"], how="left")
    if (w / "era5_static.parquet").exists():
        th = th.merge(pd.read_parquet(w / "era5_static.parquet", columns=["chip_id", "ground_m"]), on="chip_id", how="left")
    else:
        cells = pd.read_parquet(db / "cells.parquet", columns=["chip_id", "elev_mean"])
        th = th.merge(cells.groupby("chip_id")["elev_mean"].mean().rename("ground_m").reset_index(), on="chip_id", how="left")
    cells = pd.read_parquet(db / "cells.parquet", columns=["chip_id", "elev_max"])
    th = th.merge(cells.groupby("chip_id")["elev_max"].max().rename("crest_m").reset_index(), on="chip_id", how="left")
    has_levels = (w / "era5_levels.parquet").exists()
    levels = []
    if has_levels:
        import pyarrow.parquet as pq

        names = set(pq.ParquetFile(w / "era5_levels.parquet").schema.names)
        levels = [p for p in LEVELS if {f"u_{p}", f"v_{p}", f"z_{p}"} <= names]
        cols = [f"{v}_{p}" for p in levels for v in ("u", "v", "z")]
        lv = pd.read_parquet(w / "era5_levels.parquet", columns=["chip_id", "time", *cols])
        lv = lv[lv["chip_id"].isin(th["chip_id"].unique())].rename(columns={"time": "hour3"})
        lv["hour3"] = lv["hour3"].astype("datetime64[ns]")
        th = th.merge(lv.drop_duplicates(["chip_id", "hour3"]), on=["chip_id", "hour3"], how="left")

    g = th["ground_m"].to_numpy(float)
    H = [g + 10, g + 100]
    U = [th["u10"].to_numpy(float), th["u100"].to_numpy(float)]
    V = [th["v10"].to_numpy(float), th["v100"].to_numpy(float)]
    if has_levels:
        for p in levels:
            z = th[f"z_{p}"].to_numpy(float)
            z = np.where(z > g + 150, z, np.nan)          # a level under ERA5's ground is extrapolated: skip it
            H.append(z)
            U.append(th[f"u_{p}"].to_numpy(float))
            V.append(th[f"v_{p}"].to_numpy(float))
    H, U, V = np.column_stack(H), np.column_stack(U), np.column_stack(V)
    h = th["h_asl"].to_numpy(float)
    mu, mv, top = np.full(len(th), np.nan), np.full(len(th), np.nan), np.full(len(th), np.nan)
    for i in range(len(th)):
        ok = ~(np.isnan(H[i]) | np.isnan(U[i]) | np.isnan(V[i]))
        if ok.sum() < 2 or np.isnan(h[i]):
            continue
        hh = H[i][ok]
        order = np.argsort(hh)
        hh = hh[order]
        mu[i] = np.interp(h[i], hh, U[i][ok][order])
        mv[i] = np.interp(h[i], hh, V[i][ok][order])
        top[i] = hh[-1]
    th["model_u"], th["model_v"] = mu, mv
    th["above_model_top"] = h > top    # higher than the highest ERA5 height available: value of the top level
    th["above_crest"] = th["h_asl"] - th["crest_m"]
    return th, has_levels


def pool(th):
    th = th.dropna(subset=["model_u", "model_v"]).copy()
    th["band"] = band(th["agl"].to_numpy(), th["above_crest"].to_numpy())
    keys = ["site_id", "chip_id", "hour", "band"]
    g = th.groupby(keys).agg(n=("wind_u", "size"), flights=("track_id", "nunique"),
                             pilot_u=("wind_u", "median"), pilot_v=("wind_v", "median"),
                             model_u=("model_u", "median"), model_v=("model_v", "median"),
                             agl=("agl", "median"), h_asl=("h_asl", "median"), solar_hour=("solar_hour", "median"))
    g = g[(g["n"] >= MIN_THERMALS) & (g["flights"] >= MIN_FLIGHTS)].reset_index()
    g["solar_hour"] = g["solar_hour"].round()
    g["pilot_speed"] = np.hypot(g["pilot_u"], g["pilot_v"])
    g["model_speed"] = np.hypot(g["model_u"], g["model_v"])
    g["vector_error"] = np.hypot(g["pilot_u"] - g["model_u"], g["pilot_v"] - g["model_v"])
    # direction the wind comes FROM, degrees clockwise from north
    g["pilot_from"] = (np.degrees(np.arctan2(-g["pilot_u"], -g["pilot_v"])) + 360) % 360
    g["model_from"] = (np.degrees(np.arctan2(-g["model_u"], -g["model_v"])) + 360) % 360
    d = (g["pilot_from"] - g["model_from"] + 180) % 360 - 180
    g["dir_error"] = np.where(g["model_speed"] >= CALM_MS, np.abs(d), np.nan)
    return g


def scores(g):
    with np.errstate(all="ignore"):
        return _scores(g)


def _scores(g):
    return {"groups": int(len(g)), "measurements": int(g["n"].sum()),
            "vector_error_ms": float(g["vector_error"].median()),
            "speed_bias_ms": float((g["pilot_speed"] - g["model_speed"]).median()),
            "speed_corr": float(g["pilot_speed"].corr(g["model_speed"])) if len(g) > 2 else None,
            "dir_error_deg": float(g["dir_error"].median()) if g["dir_error"].notna().any() else None,
            "dir_within_30": float((g["dir_error"] <= 30).sum() / g["dir_error"].notna().sum()) if g["dir_error"].notna().any() else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.db / "models" / "wind_check"
    out.mkdir(parents=True, exist_ok=True)

    th = measurements(args.db)
    print(f"{len(th):,} thermals with a measured wind, {th['site_id'].nunique()} sites", flush=True)
    th, has_levels = model_wind(th, args.db)
    print(f"ERA5 at the same height: 10 m, 100 m{' and the pressure levels' if has_levels else ' only (no era5_levels yet: run 07c)'}; "
          f"{th['model_u'].notna().mean():.0%} of the measurements have it", flush=True)
    g = pool(th)
    g.to_parquet(out / "groups.parquet", index=False)
    print(f"{len(g):,} groups (site, chip, hour, height band) with {MIN_THERMALS}+ thermals from {MIN_FLIGHTS}+ flights\n")

    summary = {"bands": {}, "has_levels": has_levels}
    rows = []
    for b in ("near the ground", "middle", "free air"):
        s = scores(g[g["band"] == b])
        summary["bands"][b] = s
        rows.append({"band": b, **s})
    t = pd.DataFrame(rows).set_index("band")
    show = pd.DataFrame({
        "groups": t["groups"].map("{:,}".format),
        "vector error": t["vector_error_ms"].map("{:.1f} m/s".format),
        "speed bias": t["speed_bias_ms"].map("{:+.1f} m/s".format),
        "speed corr": t["speed_corr"].map(lambda v: "–" if v is None or pd.isna(v) else f"{v:.2f}"),
        "dir error": t["dir_error_deg"].map(lambda v: "–" if v is None or pd.isna(v) else f"{v:.0f}°"),
        "within 30°": t["dir_within_30"].map(lambda v: "–" if v is None or pd.isna(v) else f"{v:.0%}")})
    print("Pilots vs ERA5 (medians; bias > 0: pilots measured more wind than the model):")
    print(show.to_string())

    free = g[g["band"] == "free air"]
    by_hour = free.groupby("solar_hour").apply(lambda x: pd.Series(scores(x)), include_groups=False)
    if len(by_hour):
        print("\nFree air by solar hour:")
        print(by_hour[["groups", "vector_error_ms", "speed_bias_ms", "dir_error_deg"]].round(1).to_string())
    by_site = g.groupby(["site_id", "band"]).apply(lambda x: pd.Series(scores(x)), include_groups=False).reset_index()
    sites = pd.read_parquet(args.db / "sites.parquet", columns=["site_id", "name", "country"])
    by_site = by_site.merge(sites, on="site_id", how="left")
    by_site.to_csv(out / "by_site.csv", index=False)
    top = by_site[(by_site["band"] == "free air") & (by_site["groups"] >= 30)].sort_values("groups", ascending=False)
    if len(top):
        print("\nFree air, sites with the most groups:")
        print(top.head(15)[["name", "country", "groups", "vector_error_ms", "speed_bias_ms", "dir_error_deg", "dir_within_30"]]
              .round(2).to_string(index=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
