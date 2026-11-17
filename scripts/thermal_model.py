#!/usr/bin/env python3
"""Thermal model v0: the chance that a pilot crossing a 1 km cell finds a climb there, from the
terrain, the sun and the weather, tested on sites the model never saw.

One example per crossing (a flight's run of fixes in one cell, table `crossings`), so busy cells
do not look more thermic just because more pilots pass. The label is `climb`: at least 30 s of
climbing (circling at 0.5 m/s or more) in the cell. Nothing that happens inside the cell is used as
a feature (time spent, height on leaving): only what is known when the pilot arrives.

Features (no latitude, longitude or site name, so the model has to learn physics, not places):

    pilot      height above the ground on arrival
    terrain    elevation, relief, slope, aspect, curvature, ridges and valleys (cells)
    sun        sun elevation, solar time, direct sun on the cell's slopes now, and accumulated
               since sunrise (pta.terrain.sun_on_cell, without mountain shadows)
    weather    ERA5 at the cell's chip and hour: boundary layer, cloud base, CAPE/CIN, heat flux,
               sunshine, cloud, temperature spread, wind and wind against the slope, usable height
               (top of the boundary layer above the cell); era5_extra, era5_static and
               era5_levels are used when they exist (ground temperature, soil moisture, lapse
               rate, humidity and wind aloft)

Models are compared by adding the groups one at a time (pilot + terrain, + sun, + weather), with
5-fold cross-validation grouped by site: each fold is scored on sites left out of training.
Also checks the pilots' ceilings against ERA5's boundary layer (`--ceiling`).

Writes DB/models/thermal_v0/: metrics.json, importance.csv, model.txt (LightGBM, all data),
oof.parquet (out-of-fold predictions per crossing: p_terrain, p_sun, p_full), cells.parquet (per cell: crossings, observed
and predicted climb share, mean climb rate), ceiling.parquet.

    python3 scripts/thermal_model.py /content/drive/MyDrive/IGCS/database --max-rows 4000000
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pta import grid, terrain  # noqa: E402

CLIMB_S = 30          # seconds climbing in the cell that count as finding a climb
MIN_CROSS_S = 20      # crossings shorter than this only clip a corner of the cell
SUN_HOURS = 12        # hours back to accumulate sun on the slope

TERRAIN = ["elev_mean", "elev_std", "slope_mean", "slope_p90", "steep_share", "facing_east", "facing_north",
           "curvature", "ridge_share", "valley_share", "tpi_5km", "tpi_20km", "relief_5km", "flat_share"]
SUN = ["sun_elev", "solar_hour", "sun_direct", "sun_energy"]
PILOT = ["agl_in"]
SINGLE = ["blh_m", "cloud_base_m", "cape", "cin", "sensible_heat_wm2", "latent_heat_wm2", "sunshine_wm2",
          "cloud_total", "cloud_low", "t2m_c", "dewpoint_c", "u10", "v10", "u100", "v100", "gust10", "rain_mmh"]
WEATHER = ["blh_m", "cloud_base_m", "cape", "cin", "sensible_heat_wm2", "latent_heat_wm2", "sunshine_wm2",
           "cloud_total", "cloud_low", "t2m_c", "spread_c", "wind10", "wind100", "gust10", "rain_mmh",
           "upslope100", "usable_m", "agl_in_frac", "cloud_base_above_m"]
EXTRA = ["skin_minus_air_c", "soil_moisture", "albedo", "snow_depth_m", "water_vapour_kgm2", "freezing_level_m",
         "conv_rain_mmh"]
LEVELS = ["lapse_850_700", "lapse_700_500", "rh_700", "rh_850", "wind_700", "shear_700"]
NAMES = {"pilot + terrain": "p_terrain", "+ sun": "p_sun", "+ weather": "p_full"}   # columns of oof.parquet
GROUPS = {
    "pilot + terrain": PILOT + TERRAIN,
    "+ sun": PILOT + TERRAIN + SUN,
    "+ weather": PILOT + TERRAIN + SUN + WEATHER + EXTRA + LEVELS,
}


def log(msg):
    print(msg, flush=True)


# ── Training table ───────────────────────────────────────────────────────────

def crossings(db, max_rows, seed):
    """Crossings of real flights (no drives), with the task's date, the site and the label."""
    cols = ["track_id", "task_key", "cell_id", "t_in", "duration_s", "agl_in", "climbing_s", "mean_vz_climbing"]
    cx = pd.read_parquet(db / "crossings.parquet", columns=cols)
    n_all = len(cx)
    checks = pd.read_parquet(db / "flight_checks.parquet", columns=["track_id", "is_drive"])
    drives = set(checks.loc[checks["is_drive"].fillna(False).astype(bool), "track_id"])
    cx = cx[(cx["duration_s"] >= MIN_CROSS_S) & cx["agl_in"].notna() & ~cx["track_id"].isin(drives)]
    if max_rows and len(cx) > max_rows:
        cx = cx.sample(max_rows, random_state=seed)
    tasks = pd.read_parquet(db / "tasks.parquet", columns=["task_key", "event_key", "date"])
    sites = pd.read_parquet(db / "event_sites.parquet", columns=["event_key", "site_id"])
    tasks = tasks.merge(sites, on="event_key", how="left")
    tasks["date"] = pd.to_datetime(tasks["date"], errors="coerce")
    cx = cx.merge(tasks[["task_key", "date", "site_id"]], on="task_key", how="inner")
    cx = cx.dropna(subset=["date", "site_id"])
    cx["time"] = cx["date"] + pd.to_timedelta(cx["t_in"], unit="s")
    cx["hour"] = cx["time"].dt.round("h")
    cx["climb"] = (cx["climbing_s"] >= CLIMB_S).astype("int8")
    cx["chip_id"] = grid.chip_of_cell(cx["cell_id"].to_numpy())
    log(f"crossings: {n_all:,} in all, {len(cx):,} used ({cx['site_id'].nunique()} sites, "
        f"{cx['task_key'].nunique():,} tasks); {cx['climb'].mean():.1%} found a climb")
    return cx.reset_index(drop=True)


def add_terrain_and_sun(cx, db):
    import pyarrow.parquet as pq

    names = pq.ParquetFile(db / "cells.parquet").schema.names
    sun_cols = [c for c in names if c.startswith("sun_")]
    want = ["cell_id", "lat", "lon"] + [c for c in TERRAIN if c in names] + sun_cols
    cells = pd.read_parquet(db / "cells.parquet", columns=want)
    cells = cells[cells["cell_id"].isin(cx["cell_id"].unique())]
    cx = cx.merge(cells, on="cell_id", how="inner")
    lat, lon = cx["lat"].to_numpy(), cx["lon"].to_numpy()
    t = cx["time"].to_numpy("datetime64[s]")
    az, el = terrain.sun_position(lat, lon, t)
    cx["sun_elev"] = el.astype("float32")
    cx["solar_hour"] = (((t.astype("int64") % 86400) / 3600.0 + lon / 15.0) % 24).astype("float32")
    cx["sun_direct"] = terrain.sun_on_cell(cx, az, el).astype("float32")
    energy = np.zeros(len(cx))
    for k in range(SUN_HOURS + 1):   # sun on the slope over the last hours (trapezoids of 1 h)
        a, e = terrain.sun_position(lat, lon, t - np.timedelta64(3600 * k, "s"))
        energy += terrain.sun_on_cell(cx, a, e) * (0.5 if k in (0, SUN_HOURS) else 1.0)
    cx["sun_energy"] = energy.astype("float32")   # hours of full sun on a surface facing it
    return cx.drop(columns=sun_cols)


def _join(cx, path, cols, time_key=None):
    """Left-join a weather table on chip_id (and its `time` to cx[time_key]), keeping `cols`."""
    if not path.exists():
        return cx, False
    import pyarrow.parquet as pq

    names = pq.ParquetFile(path).schema.names
    keys = ["chip_id"] + (["time"] if time_key else [])
    w = pd.read_parquet(path, columns=keys + [c for c in cols if c in names and c not in keys])
    w = w[w["chip_id"].isin(cx["chip_id"].unique())]
    if time_key:
        w = w.rename(columns={"time": time_key}).drop_duplicates(["chip_id", time_key])
        w[time_key] = pd.to_datetime(w[time_key]).astype("datetime64[ns]")   # same time unit on both sides
        cx[time_key] = pd.to_datetime(cx[time_key]).astype("datetime64[ns]")
        return cx.merge(w, on=["chip_id", time_key], how="left"), True
    return cx.merge(w.drop_duplicates("chip_id"), on="chip_id", how="left"), True


def add_weather(cx, db):
    w = db / "weather"
    cx, ok = _join(cx, w / "era5_single.parquet", SINGLE, "hour")
    if not ok:
        log("  no era5_single: weather features left empty")
    cx, has_extra = _join(cx, w / "era5_extra.parquet", ["skin_temp_c", "soil_moisture", "albedo", "snow_depth_m",
                                                          "water_vapour_kgm2", "freezing_level_m", "conv_rain_mmh"], "hour")
    cx, has_static = _join(cx, w / "era5_static.parquet", ["ground_m"])
    cx["hour3"] = cx["hour"].dt.round("3h")   # ARCO pressure levels are every 3rd hour
    lv = [f"{v}_{p}" for v in ("t", "z", "u", "v", "rh") for p in (850, 700, 500)]
    cx, has_levels = _join(cx, w / "era5_levels.parquet", lv, "hour3")
    log(f"  weather: single {'yes' if ok else 'no'}, extra {'yes' if has_extra else 'no'}, "
        f"static {'yes' if has_static else 'no'}, levels {'yes' if has_levels else 'no'}")

    g = lambda c: cx[c].to_numpy("float64") if c in cx else np.full(len(cx), np.nan)  # noqa: E731
    cx["spread_c"] = g("t2m_c") - g("dewpoint_c")
    cx["wind10"] = np.hypot(g("u10"), g("v10"))
    cx["wind100"] = np.hypot(g("u100"), g("v100"))
    # wind blowing up the cell's slope (facing_* is the way the slope faces; > 0 windward, < 0 lee)
    cx["upslope100"] = -(g("u100") * g("facing_east") + g("v100") * g("facing_north"))
    if has_static:
        ground = g("ground_m")
    else:   # ERA5's ground is smooth: the mean height of the chip's cells is close to it
        ground = cx.groupby("chip_id")["elev_mean"].transform("mean").to_numpy("float64")
    cx["usable_m"] = ground + g("blh_m") - g("elev_mean")              # top of the boundary layer above the cell
    cx["agl_in_frac"] = g("agl_in") / np.maximum(cx["usable_m"].to_numpy(), 100.0)
    cx["cloud_base_above_m"] = ground + g("cloud_base_m") - g("elev_mean")
    cx["skin_minus_air_c"] = g("skin_temp_c") - g("t2m_c")
    km = lambda a, b: (g(f"z_{b}") - g(f"z_{a}")) / 1000.0  # noqa: E731
    cx["lapse_850_700"] = (g("t_850") - g("t_700")) / km(850, 700)
    cx["lapse_700_500"] = (g("t_700") - g("t_500")) / km(700, 500)
    cx["wind_700"] = np.hypot(g("u_700"), g("v_700"))
    cx["shear_700"] = np.hypot(g("u_700") - g("u10"), g("v_700") - g("v10"))
    for c in WEATHER + EXTRA + LEVELS:
        if c not in cx:
            cx[c] = np.nan
        cx[c] = cx[c].astype("float32")
    return cx


def table(db, max_rows, seed):
    t0 = time.time()
    cx = crossings(db, max_rows, seed)
    cx = add_terrain_and_sun(cx, db)
    cx = add_weather(cx, db)
    log(f"training table: {len(cx):,} rows, {time.time() - t0:.0f} s")
    return cx


# ── Models ───────────────────────────────────────────────────────────────────

def make_model(n_rounds):
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(n_estimators=n_rounds, learning_rate=0.05, num_leaves=63, min_child_samples=200,
                                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(max_iter=n_rounds, learning_rate=0.05, max_leaf_nodes=63,
                                              min_samples_leaf=200)


def scores(y, p, base):
    from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

    p = np.clip(p, 1e-6, 1 - 1e-6)
    return {"auc": float(roc_auc_score(y, p)), "log_loss": float(log_loss(y, p)),
            "brier": float(brier_score_loss(y, p)),
            "log_loss_skill": float(1 - log_loss(y, p) / log_loss(y, np.full(len(y), base)))}


def site_folds(sites, k, seed):
    """Sites split into k folds of about equal crossings."""
    counts = sites.value_counts().sample(frac=1, random_state=seed)
    fold, load = {}, np.zeros(k)
    for s, n in counts.sort_values(ascending=False).items():
        f = int(np.argmin(load))
        fold[s] = f
        load[f] += n
    return sites.map(fold).to_numpy()


def compare(cx, folds, n_rounds, out):
    y = cx["climb"].to_numpy()
    metrics, oof = {}, {}
    for name, feats in GROUPS.items():
        feats = [f for f in feats if cx[f].notna().any()]
        p = np.zeros(len(cx))
        for f in np.unique(folds):
            tr, te = folds != f, folds == f
            m = make_model(n_rounds)
            m.fit(cx.loc[tr, feats], y[tr])
            p[te] = m.predict_proba(cx.loc[te, feats])[:, 1]
        base = y.mean()
        metrics[name] = {**scores(y, p, base), "features": len(feats)}
        oof[name] = p
        s = metrics[name]
        log(f"  {name:16s} AUC {s['auc']:.3f}  log-loss {s['log_loss']:.4f}  skill {s['log_loss_skill']:+.1%}  "
            f"Brier {s['brier']:.4f}  ({len(feats)} features)")
    return metrics, oof


def calibration(y, p, bins=10):
    q = pd.qcut(p, bins, duplicates="drop")
    return pd.DataFrame({"p": p, "y": y}).groupby(q, observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"),
                                                                       n=("y", "size")).reset_index(drop=True)


def climb_strength(cx, folds, n_rounds):
    """How strong the climb is when there is one: R² of mean_vz_climbing on held-out sites."""
    try:
        import lightgbm as lgb

        make = lambda: lgb.LGBMRegressor(n_estimators=n_rounds, learning_rate=0.05, num_leaves=63,  # noqa: E731
                                         min_child_samples=200, subsample=0.8, subsample_freq=1,
                                         colsample_bytree=0.8, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        make = lambda: HistGradientBoostingRegressor(max_iter=n_rounds, learning_rate=0.05)  # noqa: E731
    ok = (cx["climb"] == 1) & cx["mean_vz_climbing"].between(0.3, 8)
    d, f = cx[ok], folds[ok.to_numpy()]
    feats = [c for c in GROUPS["+ weather"] if d[c].notna().any()]
    y, p = d["mean_vz_climbing"].to_numpy(), np.zeros(len(d))
    for k in np.unique(f):
        tr, te = f != k, f == k
        m = make()
        m.fit(d.loc[tr, feats], y[tr])
        p[te] = m.predict(d.loc[te, feats])
    r2 = 1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()
    mae = float(np.abs(y - p).mean())
    log(f"  climb rate when there is a climb: R² {r2:.3f}, mean error {mae:.2f} m/s "
        f"(mean {y.mean():.2f} m/s, {len(d):,} climbs)")
    return {"r2": float(r2), "mae_ms": mae, "n": int(len(d)), "mean_ms": float(y.mean())}


# ── Ceiling check ────────────────────────────────────────────────────────────

def ceiling(db, out):
    """The pilots' ceiling (90th percentile of thermal tops per chip and hour, above ERA5's ground)
    against ERA5's boundary layer and cloud base, overall and day to day within each site."""
    w = db / "weather"
    if not (w / "era5_single.parquet").exists():
        log("ceiling: no era5_single")
        return {}
    th = pd.read_parquet(db / "thermals.parquet", columns=["track_id", "task_key", "t_start", "lat", "lon", "alt_top"])
    off = pd.read_parquet(db / "flight_checks.parquet", columns=["track_id", "agl_offset"])
    th = th.merge(off, on="track_id", how="left")
    th["top"] = th["alt_top"] - th["agl_offset"].fillna(0)        # calibrated altitude (logger error removed)
    tasks = pd.read_parquet(db / "tasks.parquet", columns=["task_key", "event_key", "date"])
    tasks = tasks.merge(pd.read_parquet(db / "event_sites.parquet", columns=["event_key", "site_id"]), on="event_key")
    th = th.merge(tasks, on="task_key")
    th["time"] = (pd.to_datetime(th["date"], errors="coerce") + pd.to_timedelta(th["t_start"], unit="s")).dt.round("h")
    th["chip_id"] = grid.chip_id(th["lat"].to_numpy(), th["lon"].to_numpy())
    by = th.groupby(["site_id", "chip_id", "time"])["top"]
    g = pd.concat([by.quantile(0.9).rename("ceiling_asl"), by.size().rename("n")], axis=1).reset_index()
    g = g[g["n"] >= 5]
    era = pd.read_parquet(w / "era5_single.parquet", columns=["chip_id", "time", "blh_m", "cloud_base_m"])
    g = g.merge(era, on=["chip_id", "time"])
    if (w / "era5_static.parquet").exists():
        g = g.merge(pd.read_parquet(w / "era5_static.parquet", columns=["chip_id", "ground_m"]), on="chip_id")
    else:
        cells = pd.read_parquet(db / "cells.parquet", columns=["chip_id", "elev_mean"])
        g = g.merge(cells.groupby("chip_id")["elev_mean"].mean().rename("ground_m").reset_index(), on="chip_id")
    g["ceiling_m"] = g["ceiling_asl"] - g["ground_m"]
    g.to_parquet(out / "ceiling.parquet", index=False)
    r = {"chip_hours": int(len(g)), "median_ceiling_m": float(g["ceiling_m"].median()),
         "median_blh_m": float(g["blh_m"].median()), "corr_blh": float(g["ceiling_m"].corr(g["blh_m"]))}
    cb = g.dropna(subset=["cloud_base_m"])
    r["corr_cloud_base"] = float(cb["ceiling_m"].corr(cb["cloud_base_m"])) if len(cb) > 10 else None
    # day to day within a site: the day's highest ceiling against the day's highest boundary layer
    g["day"] = g["time"].dt.normalize()
    d = g.groupby(["site_id", "day"]).agg(ceiling=("ceiling_m", "max"), blh=("blh_m", "max")).reset_index()
    within = d.groupby("site_id").filter(lambda x: len(x) >= 5)
    per_site = within.groupby("site_id").apply(lambda x: x["ceiling"].corr(x["blh"]), include_groups=False)
    r["sites_with_5_days"] = int(per_site.notna().sum())
    r["median_within_site_corr"] = float(per_site.median()) if len(per_site) else None
    log(f"ceiling: {r['chip_hours']:,} chip-hours with 5+ thermals; median ceiling {r['median_ceiling_m']:.0f} m "
        f"vs ERA5 boundary layer {r['median_blh_m']:.0f} m above ERA5's ground; correlation {r['corr_blh']:.2f} "
        f"(cloud base {r['corr_cloud_base'] if r['corr_cloud_base'] is None else round(r['corr_cloud_base'], 2)}); "
        f"day to day within a site (median of {r['sites_with_5_days']} sites): "
        f"{r['median_within_site_corr'] if r['median_within_site_corr'] is None else round(r['median_within_site_corr'], 2)}")
    return r


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder")
    parser.add_argument("--out", type=Path, help="output folder (default DB/models/thermal_v0)")
    parser.add_argument("--max-rows", type=int, default=4_000_000, help="crossings sampled for training (default 4M)")
    parser.add_argument("--rounds", type=int, default=300, help="boosting rounds (default 300)")
    parser.add_argument("--folds", type=int, default=5, help="site folds (default 5)")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--ceiling-only", action="store_true", help="only the ceiling check")
    args = parser.parse_args()
    out = args.out or args.db / "models" / "thermal_v0"
    out.mkdir(parents=True, exist_ok=True)

    metrics = {"ceiling": ceiling(args.db, out)}
    if args.ceiling_only:
        (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
        return

    cx = table(args.db, args.max_rows, args.seed)
    folds = site_folds(cx["site_id"], args.folds, args.seed)
    log(f"\nChance of finding a climb in a cell, scored on sites left out ({args.folds} folds, "
        f"base rate {cx['climb'].mean():.1%}):")
    m, oof = compare(cx, folds, args.rounds, out)
    metrics["climb"] = m
    metrics["climb_rate"] = climb_strength(cx, folds, args.rounds)
    y, p = cx["climb"].to_numpy(), oof["+ weather"]
    cal = calibration(y, p)
    log("\nCalibration of the full model (held-out sites):\n" + cal.round(3).to_string(index=False))
    metrics["calibration"] = cal.to_dict(orient="list")

    feats = [f for f in GROUPS["+ weather"] if cx[f].notna().any()]
    model = make_model(args.rounds)
    model.fit(cx[feats], y)
    if hasattr(model, "booster_"):
        model.booster_.save_model(str(out / "model.txt"))
        imp = pd.DataFrame({"feature": feats, "gain": model.booster_.feature_importance("gain")})
        imp["share"] = imp["gain"] / imp["gain"].sum()
        imp = imp.sort_values("share", ascending=False)
        imp.to_csv(out / "importance.csv", index=False)
        log("\nWhat the model uses most (share of gain):\n"
            + imp.head(15).assign(share=lambda x: x["share"].map("{:.1%}".format))[["feature", "share"]].to_string(index=False))
    pd.DataFrame({"cell_id": cx["cell_id"], "site_id": cx["site_id"], "time": cx["time"], "agl_in": cx["agl_in"],
                  "climb": y, **{NAMES[k]: v.astype("float32") for k, v in oof.items()}}
                 ).to_parquet(out / "oof.parquet", index=False)
    cells = cx.assign(p=p).groupby("cell_id").agg(
        site_id=("site_id", "first"), lat=("lat", "first"), lon=("lon", "first"), crossings=("climb", "size"),
        climb_share=("climb", "mean"), predicted=("p", "mean"), climb_rate_ms=("mean_vz_climbing", "mean")).reset_index()
    cells.to_parquet(out / "cells.parquet", index=False)
    metrics["features"] = feats
    metrics["rows"] = int(len(cx))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str))
    log(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
