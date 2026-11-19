#!/usr/bin/env python3
"""Task model v1: the winner's speed and time and the share of pilots in goal, from what a task
committee knows before the task *and the weather along the route*.

Same tasks, targets and features as the baseline (task_baseline.py), plus the weather of the day
over the route, during the task: the hours from the first start gate to the deadline (at most 5 h;
13 to 16 h local solar time when the times are not published), at every ERA5 chip the route
crosses (turnpoint to turnpoint, a point every 5 km; the site's competition area when the route has
no coordinates):

    boundary layer   height (mean and highest hour), above the route's ground too (usable height),
                     cloud base, CAPE, CIN, sensible heat flux, sunshine, cloud cover, rain
    air              temperature spread (dry air: blue days), lapse rate 850-700 hPa, humidity aloft
    wind             speed at 100 m and 700 hPa, and its component along the route: tailwind and
                     crosswind, leg by leg, weighted by leg length
    ground           skin minus air temperature, soil moisture, freezing level; rain the day before

ERA5 is the reanalysis of the day; for a real task the forecast takes its place.

Both models are scored on the same folds (competitions left out; sites left out), so the
difference is what the weather adds. For the web page it also writes the model as JSON (evaluated
in the browser) and, per site, its tasks with the weather, the predictions made without seeing the
competition, and what happened.

Writes DB/models/task_v1/ (metrics.json, predictions.parquet, importance.csv, features.parquet) and
DB/web_data/task_model.json, DB/web_data/tasks/<site>.json.

    python3 scripts/task_model.py /content/drive/MyDrive/IGCS/database
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from pta import grid  # noqa: E402
from task_baseline import FEATURES as BASE, hhmm, tasks_table  # noqa: E402

STEP_KM = 5.0
MAX_WINDOW_H = 5
DEFAULT_WINDOW = (13, 16)   # local solar hours when the task has no published times
WEATHER = ["blh_mean", "blh_max", "usable_m", "cloud_base_m", "cloud_base_above_m", "cape", "cin",
           "sensible_heat_wm2", "sunshine_wm2", "cloud_total", "cloud_low", "rain_mm", "spread_c", "t2m_c",
           "wind100", "gust10", "wind_700", "wind_850", "tailwind_100", "crosswind_100", "tailwind_700",
           "crosswind_700", "lapse_850_700", "rh_700", "rh_850", "skin_minus_air_c", "soil_moisture",
           "freezing_level_m", "rain_day_before_mm", "route_chips"]
TARGETS = {"winner_kmh": "winner's speed", "goal_share": "goal share"}


def km_between(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371.0 * 2 * np.arcsin(np.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Direction of travel, radians clockwise from north."""
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    return np.arctan2(np.sin(dl) * np.cos(p2), np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl))


def routes(db, t):
    """{task_key: (chip_ids, [(leg km, bearing)])} from the turnpoints; empty when no coordinates."""
    out = {}
    tp = None
    if (db / "turnpoints.parquet").exists():
        tp = pd.read_parquet(db / "turnpoints.parquet", columns=["task_key", "seq", "lat", "lon"])
        tp = tp[tp["task_key"].isin(t["task_key"])].dropna(subset=["lat", "lon"])
    if tp is None or not len(tp):
        return out
    for key, g in tp.sort_values(["task_key", "seq"]).groupby("task_key"):
        la, lo = g["lat"].to_numpy(float), g["lon"].to_numpy(float)
        if len(la) < 2:
            continue
        pts_la, pts_lo, legs = [la[0]], [lo[0]], []
        for i in range(len(la) - 1):
            d = km_between(la[i], lo[i], la[i + 1], lo[i + 1])
            if d < 0.5:
                continue
            legs.append((d, bearing(la[i], lo[i], la[i + 1], lo[i + 1])))
            n = max(int(d // STEP_KM), 1)
            f = np.arange(1, n + 1) / n
            pts_la.extend(la[i] + f * (la[i + 1] - la[i]))
            pts_lo.extend(lo[i] + f * (lo[i + 1] - lo[i]))
        out[key] = (np.unique(grid.chip_id(np.array(pts_la), np.array(pts_lo))), legs)
    return out


def windows(t):
    """UTC start and end of each task's flying window."""
    off = t["utc_offset_h"].astype(float).where(t["utc_offset_h"].notna(), (t["lon"] / 15).round())
    start_l, end_l = t["start_time"].map(hhmm), t["deadline"].map(hhmm)
    solar = t["lon"] / 15 - off            # local clock minus local solar time
    start_l = start_l.where(start_l.notna(), DEFAULT_WINDOW[0] + solar)
    end_l = end_l.where(end_l.notna() & (end_l > start_l), DEFAULT_WINDOW[1] + solar)
    end_l = np.minimum(end_l, start_l + MAX_WINDOW_H)
    day = pd.to_datetime(t["date"])
    start = day + pd.to_timedelta(start_l - off, unit="h")
    end = day + pd.to_timedelta(end_l - off, unit="h")
    return start, end


def _read(path, cols, chips):
    if not path.exists():
        return None
    import pyarrow.parquet as pq

    have = set(pq.ParquetFile(path).schema.names)
    use = [c for c in cols if c in have]
    d = pd.read_parquet(path, columns=use)
    d = d[d["chip_id"].isin(chips)]
    if "time" in d:
        d["time"] = pd.to_datetime(d["time"]).astype("datetime64[ns]")
    return d


def weather(db, t):
    """One row of weather features per task."""
    rt = routes(db, t)
    site_chips = pd.read_parquet(db / "site_chips.parquet", columns=["site_id", "chip_id", "in_area"])
    area = {s: g.loc[g["in_area"].fillna(False).astype(bool), "chip_id"].to_numpy()
            for s, g in site_chips.groupby("site_id")}
    start, end = windows(t)
    rows = []
    for i, r in enumerate(t.itertuples(index=False)):
        chips = rt[r.task_key][0] if r.task_key in rt else area.get(r.site_id, np.array([], int))
        if not len(chips):
            continue
        hours = pd.date_range(start.iloc[i].floor("h"), end.iloc[i].floor("h"), freq="h")
        for h in hours:
            for c in chips:
                rows.append((r.task_key, int(c), h))
    need = pd.DataFrame(rows, columns=["task_key", "chip_id", "time"])
    need["time"] = need["time"].astype("datetime64[ns]")
    need["time3"] = need["time"].dt.round("3h")
    chips = set(need["chip_id"])
    w = db / "weather"
    s = _read(w / "era5_single.parquet", ["chip_id", "time", "blh_m", "cloud_base_m", "cape", "cin", "sensible_heat_wm2",
                                          "sunshine_wm2", "cloud_total", "cloud_low", "rain_mmh", "t2m_c", "dewpoint_c",
                                          "u100", "v100", "u10", "v10", "gust10"], chips)
    d = need.merge(s, on=["chip_id", "time"], how="left") if s is not None else need
    lv = _read(w / "era5_levels.parquet", ["chip_id", "time", "t_850", "t_700", "z_850", "z_700", "u_850", "v_850",
                                           "u_700", "v_700", "rh_700", "rh_850"], chips)
    if lv is not None:
        d = d.merge(lv.rename(columns={"time": "time3"}).drop_duplicates(["chip_id", "time3"]), on=["chip_id", "time3"], how="left")
    ex = _read(w / "era5_extra.parquet", ["chip_id", "time", "skin_temp_c", "soil_moisture", "freezing_level_m"], chips)
    if ex is not None:
        d = d.merge(ex, on=["chip_id", "time"], how="left")
    st = _read(w / "era5_static.parquet", ["chip_id", "ground_m"], chips)
    if st is not None:
        d = d.merge(st, on="chip_id", how="left")
    cells = pd.read_parquet(db / "cells.parquet", columns=["chip_id", "elev_mean"])
    elev = cells[cells["chip_id"].isin(chips)].groupby("chip_id")["elev_mean"].mean().rename("route_elev")
    d = d.merge(elev.reset_index(), on="chip_id", how="left")
    if "ground_m" not in d:
        d["ground_m"] = d["route_elev"]

    g = lambda c: d[c].to_numpy(float) if c in d else np.full(len(d), np.nan)  # noqa: E731
    d["usable_m"] = g("ground_m") + g("blh_m") - g("route_elev")
    d["cloud_base_above_m"] = g("ground_m") + g("cloud_base_m") - g("route_elev")
    d["spread_c"] = g("t2m_c") - g("dewpoint_c")
    d["wind100"] = np.hypot(g("u100"), g("v100"))
    d["wind_700"] = np.hypot(g("u_700"), g("v_700"))
    d["wind_850"] = np.hypot(g("u_850"), g("v_850"))
    d["lapse_850_700"] = (g("t_850") - g("t_700")) / ((g("z_700") - g("z_850")) / 1000.0)
    d["skin_minus_air_c"] = g("skin_temp_c") - g("t2m_c")
    for c in ("rh_700", "rh_850"):
        if c in d:
            d[c] = d[c].clip(0, 100)
    mean_cols = [k for k in ["usable_m", "cloud_base_m", "cloud_base_above_m", "cape", "cin", "sensible_heat_wm2",
                             "sunshine_wm2", "cloud_total", "cloud_low", "spread_c", "t2m_c", "wind100", "gust10",
                             "wind_700", "wind_850", "lapse_850_700", "rh_700", "rh_850", "skin_minus_air_c",
                             "soil_moisture", "freezing_level_m", "u100", "v100", "u_700", "v_700"] if k in d]
    by = d.groupby("task_key")
    out = by[mean_cols].mean()
    out["route_chips"] = by["chip_id"].nunique()
    if "blh_m" in d:
        out["blh_mean"] = by["blh_m"].mean()
    if "rain_mmh" in d:
        out["rain_mm"] = by["rain_mmh"].sum(min_count=1)
    # the highest hour of the boundary layer over the route
    if "blh_m" in d:
        out["blh_max"] = d.groupby(["task_key", "time"])["blh_m"].mean().groupby("task_key").max()
    # wind along the route, leg by leg
    for lvl, (u, v) in {"100": ("u100", "v100"), "700": ("u_700", "v_700")}.items():
        tail, cross = {}, {}
        for key, (_, legs) in rt.items():
            if key not in out.index or u not in out or not legs or np.isnan(out.at[key, u]):
                continue
            uu, vv = out.at[key, u], out.at[key, v]
            L = np.array([leg[0] for leg in legs])
            b = np.array([leg[1] for leg in legs])
            tail[key] = float((L * (uu * np.sin(b) + vv * np.cos(b))).sum() / L.sum())
            cross[key] = float((L * np.abs(-uu * np.cos(b) + vv * np.sin(b))).sum() / L.sum())
        out[f"tailwind_{lvl}"] = pd.Series(tail)
        out[f"crosswind_{lvl}"] = pd.Series(cross)
    # rain the day before (soil and early convection)
    if s is not None and "rain_mmh" in s:
        prev = need.assign(time=need["time"] - pd.Timedelta(days=1))[["task_key", "chip_id", "time"]].drop_duplicates()
        rp = prev.merge(s[["chip_id", "time", "rain_mmh"]], on=["chip_id", "time"], how="left")
        out["rain_day_before_mm"] = rp.groupby("task_key")["rain_mmh"].sum(min_count=1)
    for c in WEATHER:
        if c not in out:
            out[c] = np.nan
    out["has_route"] = out.index.isin(list(rt))
    return out[WEATHER + ["has_route"]]


# ── Models ───────────────────────────────────────────────────────────────────

def model():
    import lightgbm as lgb

    return lgb.LGBMRegressor(n_estimators=400, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                             subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def cv(t, target, groups, feats, k=5):
    from sklearn.model_selection import GroupKFold

    d = t.dropna(subset=[target])
    p = np.full(len(d), np.nan)
    for tr, te in GroupKFold(k).split(d, groups=d[groups]):
        m = model()
        m.fit(d.iloc[tr][feats], d.iloc[tr][target])
        p[te] = m.predict(d.iloc[te][feats])
    return d.index.to_numpy(), p


def compact(booster, feats):
    """LightGBM trees as nested arrays for the browser: [feature, threshold, missing_left, left, right]
    for a split, a number for a leaf. Prediction = sum over trees."""
    def node(n):
        if "leaf_value" in n:
            return round(float(n["leaf_value"]), 6)
        missing = n.get("missing_type", "None")
        # NaN goes the default way; with missing_type None NaN is treated as 0 by LightGBM
        nan_left = bool(n.get("default_left", True)) if missing in ("NaN", "Zero") else (0.0 <= float(n["threshold"]))
        return [int(n["split_feature"]), round(float(n["threshold"]), 6), int(nan_left), node(n["left_child"]), node(n["right_child"])]
    dump = booster.dump_model()
    return {"features": feats, "trees": [node(t["tree_structure"]) for t in dump["tree_info"]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--web", type=Path, help="web data folder (default DB/web_data)")
    args = parser.parse_args()
    out = args.out or args.db / "models" / "task_v1"
    web = args.web or args.db / "web_data"
    out.mkdir(parents=True, exist_ok=True)
    (web / "tasks").mkdir(parents=True, exist_ok=True)

    t = tasks_table(args.db)
    wx = weather(args.db, t)
    t = t.merge(wx, left_on="task_key", right_index=True, how="left")
    print(f"{len(t):,} tasks ({t['winner_kmh'].notna().sum():,} with a winner's time); weather for "
          f"{t['blh_mean'].notna().mean():.0%}, along the route for {t['has_route'].fillna(False).astype(bool).mean():.0%} "
          f"(the others: the competition area)", flush=True)
    t[["task_key"] + WEATHER].to_parquet(out / "features.parquet", index=False)

    base = [f for f in BASE if f in t and t[f].notna().any()]
    full = base + [f for f in WEATHER if t[f].notna().any()]
    metrics, preds = {}, t[["task_key", "event_key", "site_id", "date", "ss_km", "winner_h", "winner_kmh", "goal_share"]].copy()
    for target in TARGETS:
        for split, label in (("event_key", "new competition"), ("site_id", "new site")):
            res = {}
            for name, feats in (("without weather", base), ("with weather", full)):
                idx, p = cv(t, target, split, feats)
                y = t.loc[idx, target].to_numpy()
                r = {"mae": float(np.mean(np.abs(y - p)))}
                if target == "winner_kmh":
                    h, km = t.loc[idx, "winner_h"].to_numpy(), t.loc[idx, "ss_km"].to_numpy()
                    err = np.abs(h - km / p) * 60
                    r.update(time_mae_min=float(np.nanmean(err)), within_15_min=float(np.nanmean(err <= 15)),
                             within_30_min=float(np.nanmean(err <= 30)))
                else:
                    r.update(within_10_points=float(np.mean(np.abs(y - p) <= 0.10)))
                res[name] = r
                if name == "with weather" and split == "event_key":
                    preds.loc[idx, f"pred_{target}"] = p
                if name == "without weather" and split == "event_key":
                    preds.loc[idx, f"base_{target}"] = p
            metrics[f"{target}|{label}"] = res
            a, b = res["without weather"], res["with weather"]
            if target == "winner_kmh":
                print(f"  winner, {label:15s} time error {a['time_mae_min']:.0f} -> {b['time_mae_min']:.0f} min "
                      f"(within 15 min: {a['within_15_min']:.0%} -> {b['within_15_min']:.0%}; within 30: "
                      f"{a['within_30_min']:.0%} -> {b['within_30_min']:.0%}); speed {a['mae']:.1f} -> {b['mae']:.1f} km/h", flush=True)
            else:
                print(f"  goal share, {label:11s} error {100 * a['mae']:.1f} -> {100 * b['mae']:.1f} points "
                      f"(within 10 points: {a['within_10_points']:.0%} -> {b['within_10_points']:.0%})", flush=True)

    models, importance = {}, []
    for target in TARGETS:
        d = t.dropna(subset=[target])
        m = model().fit(d[full], d[target])
        models[target] = compact(m.booster_, full)
        imp = pd.DataFrame({"target": target, "feature": full, "gain": m.booster_.feature_importance("gain")})
        imp["share"] = imp["gain"] / imp["gain"].sum()
        importance.append(imp.sort_values("share", ascending=False))
    importance = pd.concat(importance)
    importance.to_csv(out / "importance.csv", index=False)
    print("\nWhat the models use most (share of gain):")
    for target in TARGETS:
        top = importance[importance["target"] == target].head(10)
        print(f"  {TARGETS[target]}: " + ", ".join(f"{r.feature} {r.share:.0%}" for r in top.itertuples()))
    preds.to_parquet(out / "predictions.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))

    # ── web: the model and each site's tasks ─────────────────────────────────
    (web / "task_model.json").write_text(json.dumps({"targets": models, "metrics": metrics}, separators=(",", ":")))
    events = pd.read_parquet(args.db / "events.parquet", columns=["event_key", "name"]).rename(columns={"name": "competition"})
    tk = pd.read_parquet(args.db / "tasks.parquet", columns=["task_key", "name"]).rename(columns={"name": "task"})
    w = t.merge(preds[["task_key", "pred_winner_kmh", "pred_goal_share", "base_winner_kmh", "base_goal_share"]], on="task_key", how="left")
    w = w.merge(events, on="event_key", how="left").merge(tk, on="task_key", how="left")
    n_sites = 0
    for site, g in w.groupby("site_id"):
        g = g.sort_values("date", ascending=False)
        rows = []
        for r in g.itertuples(index=False):
            rows.append({"key": r.task_key, "date": str(r.date)[:10], "competition": r.competition, "task": r.task,
                         "km": round(float(r.ss_km), 1) if pd.notna(r.ss_km) else None, "pilots": int(r.launched),
                         "winner_h": None if pd.isna(r.winner_h) else round(float(r.winner_h), 3),
                         "goal": round(float(r.goal_share), 3),
                         "pred_kmh": None if pd.isna(r.pred_winner_kmh) else round(float(r.pred_winner_kmh), 2),
                         "pred_goal": None if pd.isna(r.pred_goal_share) else round(float(r.pred_goal_share), 3),
                         "base_kmh": None if pd.isna(r.base_winner_kmh) else round(float(r.base_winner_kmh), 2),
                         "base_goal": None if pd.isna(r.base_goal_share) else round(float(r.base_goal_share), 3),
                         "x": [None if pd.isna(v) else round(float(v), 4) for v in (getattr(r, f) for f in full)]})
        (web / "tasks" / f"{site}.json").write_text(json.dumps({"features": full, "tasks": rows}, separators=(",", ":"),
                                                               ensure_ascii=False))
        n_sites += 1
    print(f"\nWeb: the models ({sum(len(m['trees']) for m in models.values())} trees) and the tasks of {n_sites} sites "
          f"in {web}\nSaved to {out}")


if __name__ == "__main__":
    main()
