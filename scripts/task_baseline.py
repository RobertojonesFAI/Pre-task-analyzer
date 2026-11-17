#!/usr/bin/env python3
"""Task model baseline, without weather: the winner's time and the share of pilots in goal, from
what a task committee knows before the task.

This is the bar the weather has to beat. Features, all known before launch:

    route      speed-section distance, number of turnpoints, task type
    timing     start time (local solar hours), time from start to deadline
    season     solar declination and day length at the launch (the same physics in both
               hemispheres), launch elevation, distance from the equator
    field      pilots launched, mean rating of the field's top 10 and top 30 before the task
               (pilot_ratings.py), share of CCC and EN-D gliders (gliders.py)
    site       the site's own history (median winner speed and goal share in earlier tasks);
               empty for a site's first task

Two scores: on competitions left out (grouped by event, sites known) and on sites left out
(grouped by site, a new site). Compared with simple rules: the median winner speed over all
tasks, and that of the same site before.

Writes DB/models/task_v0/: metrics.json, predictions.parquet, importance.csv.

    python3 scripts/task_baseline.py /content/drive/MyDrive/IGCS/database
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def hhmm(s):
    try:
        h, m, *rest = str(s).split(":")
        return int(h) + int(m) / 60 + (int(rest[0]) / 3600 if rest else 0)
    except (ValueError, TypeError):
        return np.nan


def tasks_table(db):
    t = pd.read_parquet(db / "tasks.parquet")
    t = t[~t["training"].fillna(False).astype(bool) & t["task_type"].isin(["race", "elapsed"])]
    r = pd.read_parquet(db / "results.parquet", columns=["result_id", "task_key", "status", "time_s"])
    flew = r[~r["status"].isin(["abs", "dnf", "dsq"])]
    agg = flew.groupby("task_key").agg(launched=("status", "size"), goal=("status", lambda s: (s == "goal").sum()),
                                       winner_s=("time_s", lambda s: s[s > 0].min()))
    t = t.merge(agg.reset_index(), on="task_key")
    t = t[(t["launched"] >= 10) & ~t["stopped"].fillna(False).astype(bool)]
    t["goal_share"] = t["goal"] / t["launched"]
    dist = t["ss_distance_km"].where(t["ss_distance_km"] > 0, t["distance_km"])
    t["ss_km"] = dist
    t["winner_h"] = t["winner_s"] / 3600
    t["winner_kmh"] = dist / t["winner_h"]
    t.loc[(t["winner_kmh"] < 5) | (t["winner_kmh"] > 80), ["winner_h", "winner_kmh"]] = np.nan   # timing errors

    ev = pd.read_parquet(db / "event_sites.parquet").merge(
        pd.read_parquet(db / "sites.parquet", columns=["site_id", "lat", "lon"]), on="site_id")
    t = t.merge(ev, on="event_key", how="left")
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date", "site_id"])
    doy = t["date"].dt.dayofyear.to_numpy()
    decl = 23.44 * np.sin(np.radians(360 / 365 * (doy - 81)))
    lat = np.radians(t["lat"].to_numpy(float))
    cos_h = np.clip(-np.tan(lat) * np.tan(np.radians(decl)), -1, 1)
    t["day_length_h"] = 2 * np.degrees(np.arccos(cos_h)) / 15
    t["sun_decl_local"] = decl * np.sign(t["lat"].to_numpy(float))   # + in the local summer
    t["abs_lat"] = t["lat"].abs()
    off = t["utc_offset_h"].astype(float)
    t["start_solar_h"] = t["start_time"].map(hhmm) - off + t["lon"] / 15
    t["window_h"] = t["deadline"].map(hhmm) - t["start_time"].map(hhmm)
    if "turnpoints" in t:
        t["n_turnpoints"] = t["turnpoints"].map(lambda x: len(json.loads(x)) if isinstance(x, str) and x.startswith("[") else np.nan)
    else:
        t["n_turnpoints"] = np.nan
    t["is_race"] = (t["task_type"] == "race").astype(int)

    if (db / "task_fields.parquet").exists():
        t = t.merge(pd.read_parquet(db / "task_fields.parquet"), on="task_key", how="left")
    if (db / "result_gliders.parquet").exists():
        g = pd.read_parquet(db / "result_gliders.parquet", columns=["result_id", "glider_class"]).merge(
            flew[["result_id", "task_key"]], on="result_id")
        share = g.groupby("task_key")["glider_class"].agg(
            ccc_share=lambda c: (c == "CCC").mean(), en_d_share=lambda c: (c == "EN-D").mean())
        t = t.merge(share.reset_index(), on="task_key", how="left")
    # the site's history: medians of its earlier tasks only
    t = t.sort_values("date")
    for col in ("winner_kmh", "goal_share"):
        t[f"site_prev_{col}"] = t.groupby("site_id")[col].transform(lambda s: s.shift().expanding().median())
    t["site_prev_tasks"] = t.groupby("site_id").cumcount()
    return t.reset_index(drop=True)


FEATURES = ["ss_km", "n_turnpoints", "is_race", "start_solar_h", "window_h", "day_length_h", "sun_decl_local",
            "abs_lat", "launched", "field_top10", "field_top30", "n_rated", "ccc_share", "en_d_share",
            "site_prev_winner_kmh", "site_prev_goal_share", "site_prev_tasks"]


def model():
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15, min_child_samples=20,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.03, max_leaf_nodes=15)


def cv(t, target, groups, k=5, seed=7):
    from sklearn.model_selection import GroupKFold

    d = t.dropna(subset=[target])
    feats = [f for f in FEATURES if f in d and d[f].notna().any()]
    p = np.full(len(d), np.nan)
    for tr, te in GroupKFold(k).split(d, groups=d[groups]):
        m = model()
        m.fit(d.iloc[tr][feats], d.iloc[tr][target])
        p[te] = m.predict(d.iloc[te][feats])
    return d.index.to_numpy(), p, feats


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    out = args.out or args.db / "models" / "task_v0"
    out.mkdir(parents=True, exist_ok=True)
    t = tasks_table(args.db)
    n_time = t["winner_kmh"].notna().sum()
    print(f"{len(t):,} tasks with results and 10+ pilots ({n_time:,} with a winner's time), "
          f"{t['site_id'].nunique()} sites, {t['event_key'].nunique()} competitions")

    metrics, preds = {}, t[["task_key", "event_key", "site_id", "date", "ss_km", "winner_h", "winner_kmh", "goal_share"]].copy()
    for target in ("winner_kmh", "goal_share"):
        for split in ("event_key", "site_id"):
            idx, p, feats = cv(t, target, split)
            y = t.loc[idx, target].to_numpy()
            name = f"{target}|{'new competition' if split == 'event_key' else 'new site'}"
            res = {"mae": float(np.mean(np.abs(y - p))), "n": int(len(y)),
                   "rule_overall_median": float(np.mean(np.abs(y - np.median(y))))}
            prev = t.loc[idx, f"site_prev_{target}"].to_numpy()
            ok = ~np.isnan(prev)
            res["rule_site_history"] = float(np.mean(np.abs(y[ok] - prev[ok]))) if ok.any() else None
            res["model_where_site_history"] = float(np.mean(np.abs(y[ok] - p[ok]))) if ok.any() else None
            if target == "winner_kmh":   # the same errors as time, which is what committees think in
                h = t.loc[idx, "winner_h"].to_numpy()
                res["time_mae_min"] = float(np.mean(np.abs(h - t.loc[idx, "ss_km"].to_numpy() / p)) * 60)
                res["time_mae_min_rule"] = float(np.mean(np.abs(h - t.loc[idx, "ss_km"].to_numpy() / np.median(y))) * 60)
            metrics[name] = res
            preds.loc[idx, f"pred_{target}_{split}"] = p
            unit = "km/h" if target == "winner_kmh" else ""
            fmt = (lambda v: f"{v:.1f} {unit}") if unit else (lambda v: f"{v:.1%}")
            line = (f"  {name:32s} model error {fmt(res['mae'])} | overall median {fmt(res['rule_overall_median'])}")
            if res["rule_site_history"] is not None:
                line += (f" | site history {fmt(res['rule_site_history'])} "
                         f"(model on the same tasks {fmt(res['model_where_site_history'])})")
            if target == "winner_kmh":
                line += f" | winner's time off by {res['time_mae_min']:.0f} min (rule {res['time_mae_min_rule']:.0f} min)"
            print(line)
    m = model().fit(t.dropna(subset=["winner_kmh"])[feats], t.dropna(subset=["winner_kmh"])["winner_kmh"])
    if hasattr(m, "booster_"):
        imp = pd.DataFrame({"feature": feats, "gain": m.booster_.feature_importance("gain")})
        imp["share"] = imp["gain"] / imp["gain"].sum()
        imp = imp.sort_values("share", ascending=False)
        imp.to_csv(out / "importance.csv", index=False)
        print("\nWinner's speed, what the model uses most:\n"
              + imp.head(10).assign(share=lambda x: x["share"].map("{:.1%}".format))[["feature", "share"]].to_string(index=False))
    preds.to_parquet(out / "predictions.parquet", index=False)
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1))
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
