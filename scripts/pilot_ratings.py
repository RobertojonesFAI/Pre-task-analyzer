#!/usr/bin/env python3
"""Pilot ratings from the official results, and the strength of each task's field.

An Elo rating for many players at once: every task is a set of head-to-head games between all the
pilots who flew it, won by the one ranked higher. Tasks are processed in date order and each
pilot's rating is recorded *before* the task, so models of a task only use what was known before
it was flown (no peeking at the result).

    expected(i) = mean over the others j of 1 / (1 + 10^((R_j - R_i) / 400))
    actual(i)   = share of the others ranked below i
    R_i        += K * (actual - expected), K larger for pilots with few tasks

Order in a task: the official rank when published, else goal pilots by time, then everyone by
distance. Absent, did-not-fly and disqualified pilots are left out; training tasks too.

Writes DB/ratings.parquet (task_key, pilot_key, rating_before, tasks_before, place, n_pilots),
DB/pilot_ratings.parquet (pilot_key, rating, tasks, last_date) and DB/task_fields.parquet
(task_key, n_rated, field_top10, field_top30, field_mean: mean ratings before the task).

    python3 scripts/pilot_ratings.py /content/drive/MyDrive/IGCS/database
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

START = 1500.0
K_NEW, K_SETTLED, SETTLED_AFTER = 48.0, 24.0, 20   # faster moves for a pilot's first tasks
LEFT_OUT = {"abs", "dnf", "dsq"}


def order(task):
    """Place of each pilot in a task (1 = best; ties share a place)."""
    t = task.copy()
    if t["rank"].notna().all() and (t["rank"] > 0).all() and t["rank"].nunique() > 1:
        return t["rank"].rank(method="min").to_numpy()
    goal = t["status"].eq("goal")
    t["k1"] = np.where(goal, 0, 1)
    t["k2"] = np.where(goal, t["time_s"].fillna(np.inf), -t["distance_km"].fillna(0))
    return t[["k1", "k2"]].apply(tuple, axis=1).rank(method="min").to_numpy()


def rate(results, tasks):
    r = results[~results["status"].isin(LEFT_OUT) & results["pilot_key"].notna()]
    r = r.merge(tasks[["task_key", "date"]], on="task_key")
    r["date"] = pd.to_datetime(r["date"], errors="coerce")
    r = r.dropna(subset=["date"]).sort_values(["date", "task_key"])
    rating, played, last = {}, {}, {}
    rows, fields = [], []
    for task_key, t in r.groupby("task_key", sort=False):
        t = t.drop_duplicates("pilot_key")
        n = len(t)
        if n < 5:
            continue
        keys = t["pilot_key"].to_numpy()
        R = np.array([rating.get(k, START) for k in keys])
        games = np.array([played.get(k, 0) for k in keys])
        place = order(t)
        diff = (R[None, :] - R[:, None]) / 400.0                    # [i, j]: R_j - R_i
        expected = (1.0 / (1.0 + 10.0 ** diff)).sum(1) - 0.5        # minus the game against oneself
        actual = (place[None, :] > place[:, None]).sum(1) + 0.5 * ((place[None, :] == place[:, None]).sum(1) - 1)
        k = np.where(games < SETTLED_AFTER, K_NEW, K_SETTLED)
        new = R + k * (actual - expected) / (n - 1) * np.sqrt(n - 1) / 2   # one task weighs like a few games
        top = np.sort(R)[::-1]
        fields.append({"task_key": task_key, "n_rated": int((games > 0).sum()), "field_top10": float(top[:10].mean()),
                       "field_top30": float(top[:30].mean()), "field_mean": float(R.mean())})
        date = t["date"].iloc[0]
        for i, kk in enumerate(keys):
            rows.append((task_key, kk, R[i], int(games[i]), int(place[i]), n))
            rating[kk], played[kk], last[kk] = new[i], games[i] + 1, date
    ratings = pd.DataFrame(rows, columns=["task_key", "pilot_key", "rating_before", "tasks_before", "place", "n_pilots"])
    pilots = pd.DataFrame({"pilot_key": list(rating), "rating": list(rating.values()),
                           "tasks": [played[k] for k in rating], "last_date": [last[k] for k in rating]})
    return ratings, pilots.sort_values("rating", ascending=False), pd.DataFrame(fields)


def check(ratings):
    """Does a higher rating before the task predict a better place? Share of pilot pairs ordered right."""
    ok, total = 0, 0
    for _, t in ratings[ratings["tasks_before"] >= 5].groupby("task_key"):
        if len(t) < 5:
            continue
        R, P = t["rating_before"].to_numpy(), t["place"].to_numpy()
        dr, dp = R[:, None] - R[None, :], P[:, None] - P[None, :]
        m = (dr != 0) & (dp != 0)
        ok += ((dr > 0) == (dp < 0))[m].sum()
        total += m.sum()
    return ok / total if total else float("nan")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path)
    args = parser.parse_args()
    results = pd.read_parquet(args.db / "results.parquet",
                              columns=["task_key", "pilot_key", "status", "rank", "time_s", "distance_km"])
    tasks = pd.read_parquet(args.db / "tasks.parquet", columns=["task_key", "date", "training"])
    tasks = tasks[~tasks["training"].fillna(False).astype(bool)]
    ratings, pilots, fields = rate(results, tasks)
    ratings.to_parquet(args.db / "ratings.parquet", index=False)
    pilots.to_parquet(args.db / "pilot_ratings.parquet", index=False)
    fields.to_parquet(args.db / "task_fields.parquet", index=False)
    print(f"{len(fields):,} tasks rated, {len(pilots):,} pilots; ratings from {pilots['rating'].min():.0f} "
          f"to {pilots['rating'].max():.0f} (median {pilots['rating'].median():.0f})")
    print(f"pilots with 20+ tasks: {(pilots['tasks'] >= 20).sum():,}")
    acc = check(ratings)
    print(f"check: in {acc:.1%} of pairs of pilots (both with 5+ tasks before), the one rated higher before the task "
          f"placed better (50% = no better than chance)")
    print("\nStrongest fields (mean rating of the top 30 before the task):")
    print(fields.sort_values("field_top30", ascending=False).head(10).round(0).to_string(index=False))


if __name__ == "__main__":
    main()
