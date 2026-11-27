#!/usr/bin/env python3
"""Task model v2: hourly (1 h) space-time along-route weather, 1 km terrain and sun-on-slope,
0.1-deg ERA5-Land, leakage-free in-fold thermal sub-models (core60, ceiling ASL, climb chance)
with MacCready kinematics, and a pilot-level goal/speed model (~173k results) aggregated up to
task goal share and winner time.

Compares a full ablation ladder on identical 5-fold grouped cross-validation splits
(whole competitions held out; whole sites held out) with competition/site-clustered paired
bootstrap 95% confidence intervals against task_v1 (5 h averaged weather):

    M0_baseline         task_v0 without weather (route, timing, season, field, site history)
    M1_task_v1_5h       task_v1 (5 h window-averaged ERA5 weather)
    M2_hourly_1h        1 h ERA5 slots (h0..h4) + diurnal dynamics + ETA-coupled legs + flight wind
    M3_forecast_compat  M2 + 1 km route terrain, passes, hourly sun-on-slope (100% forecast-ready)
    M3_terrain_land     M3_forecast_compat + 0.1-deg ERA5-Land & sensible heat flux (ERA5-full)
    M4_thermal_mc       M3 + leakage-free in-fold thermal sub-models (core60, ceiling, climb prob)
                        + MacCready route speed and time margin
    M5_pilot_agg        pilot-level model (~173k results: Elo, causal history, glider, harness,
                        pilot x hourly weather) aggregated to task goal share and top speed
    M5_blend            in-fold convex blend of M4 task model + M5 pilot model

Also trains an auxiliary P(stopped) classifier on all tasks (including stopped tasks) without
altering the primary benchmark task set.

Writes DB/models/task_v2/: metrics.json, ablation.csv, paired_bootstrap.csv, features_task.parquet,
predictions.parquet, pilot_predictions.parquet, importance.csv, and models/*.txt + meta.json.

    python3 scripts/task_model_v2.py /content/drive/MyDrive/IGCS/database
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from pta import grid, route_weather as RW, terrain  # noqa: E402
import task_baseline as TB  # noqa: E402
import task_model as TM  # noqa: E402

BOOTSTRAP_ITERS = 2000
DEFAULT_THERMAL_ROWS = 800_000

STRICT_BASE_FEATURES = [
    "ss_km", "n_turnpoints", "is_race", "start_solar_h", "window_h", "day_length_h",
    "sun_decl_local", "abs_lat", "launched", "field_top10", "field_top30", "n_rated",
    "ccc_share", "en_d_share",
    "site_prev_winner_kmh_strict", "site_prev_goal_share_strict", "site_prev_tasks_strict",
]

# Shared hourly weather columns available in both ERA5 and Open-Meteo for in-fold thermal sub-models.
SUB_WX_FEATURES = [
    "blh_m", "cloud_base_m", "usable_m", "cloud_base_above_m", "cape", "cin",
    "sunshine_wm2", "cloud_total", "cloud_low", "rain_mmh", "t2m_c", "spread_c",
    "wind10", "wind100", "gust10", "wind_flt", "shear_sfc_flt",
    "skin_minus_air_c", "soil_moisture", "freezing_level_m",
    "lapse_850_700", "lapse_700_600", "rh_850", "rh_700", "wstar_rad",
    "ground_m", "route_elev", "solar_hour",
]

SUB_CLIMB_TERRAIN = [
    "elev_mean", "elev_std", "slope_mean", "slope_p90", "steep_share",
    "facing_east", "facing_north", "curvature", "ridge_share", "valley_share",
    "tpi_5km", "tpi_20km", "relief_5km", "flat_share",
    "sun_elev", "sun_direct", "sun_energy",
]

THERMAL_SUB_FEATURES = [
    "pred_core60_lead_mean", "pred_core60_lead_min", "pred_core60_pack_mean", "pred_core60_pack_min",
    "pred_usable_lead_mean", "pred_usable_lead_min", "pred_usable_lead_p10", "pred_usable_pack_min",
    "pred_p_climb_low_mean", "pred_p_climb_low_min_leg",
    "pred_p_climb_cruise_mean", "pred_p_climb_cruise_min_leg",
    "mc_sub_speed_lead_kmh", "mc_sub_time_lead_h", "mc_sub_margin_lead_h", "mc_sub_margin_pack_h",
    "mc_sub_unflyable_share",
    "mc_sub_margin_ccc_h", "mc_sub_margin_end_h", "mc_sub_margin_enc_h",
]

GLIDER_ORD = {"CCC": 5.0, "EN-D": 4.0, "EN-C": 3.0, "EN-B": 2.0, "EN-A": 1.0}
GLIDER_VBAR_MS = {"CCC": 63.0 / 3.6, "EN-D": 58.0 / 3.6, "EN-C": 53.0 / 3.6, "EN-B": 48.0 / 3.6, "EN-A": 44.0 / 3.6}

PILOT_FEATURES = [
    "rating_before", "tasks_before", "rating_vs_field_mean", "rating_vs_top10", "rating_pct_in_field",
    "pilot_prev_tasks", "pilot_prev_goal_rate", "pilot_prev_speed_ratio", "pilot_prev_site_tasks",
    "glider_class_ord", "is_ccc", "is_en_d", "is_en_c_low", "harness_code", "glider_vs_field_ccc",
    "pilot_mc_margin_h", "pilot_headwind_penetration",
]


def log(msg: str) -> None:
    print(msg, flush=True)


# ── Estimators (LightGBM with sklearn HistGB fallback for local Mac tests) ───

def make_task_regressor(n_estimators: int = 400, seed: int = 7):
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=0.03, num_leaves=15, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            random_state=seed, verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=min(n_estimators, 250), learning_rate=0.04, max_leaf_nodes=15,
            min_samples_leaf=20, l2_regularization=2.0, random_state=seed,
        )


def make_sub_regressor(n_estimators: int = 250, seed: int = 7):
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=0.04, num_leaves=31, min_child_samples=25,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
            random_state=seed, verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=min(n_estimators, 180), learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=25, random_state=seed,
        )


def make_pilot_classifier(n_estimators: int = 400, seed: int = 7):
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.03, num_leaves=31, min_child_samples=100,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            random_state=seed, verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=min(n_estimators, 200), learning_rate=0.04, max_leaf_nodes=31,
            min_samples_leaf=50, l2_regularization=2.0, random_state=seed,
        )


def make_pilot_regressor(n_estimators: int = 350, seed: int = 7):
    try:
        import lightgbm as lgb

        return lgb.LGBMRegressor(
            n_estimators=n_estimators, learning_rate=0.03, num_leaves=31, min_child_samples=80,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.7, reg_alpha=0.5, reg_lambda=2.0,
            random_state=seed, verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor

        return HistGradientBoostingRegressor(
            max_iter=min(n_estimators, 200), learning_rate=0.04, max_leaf_nodes=31,
            min_samples_leaf=40, l2_regularization=2.0, random_state=seed,
        )


def make_binary_classifier(n_estimators: int = 250, min_child: int = 100, seed: int = 7):
    try:
        import lightgbm as lgb

        return lgb.LGBMClassifier(
            n_estimators=n_estimators, learning_rate=0.04, num_leaves=31, min_child_samples=min_child,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1,
        )
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=min(n_estimators, 160), learning_rate=0.05, max_leaf_nodes=31,
            min_samples_leaf=min(min_child, 50), random_state=seed,
        )


def _save_model_if_lgb(model: Any, path: Path) -> bool:
    if hasattr(model, "booster_"):
        model.booster_.save_model(str(path))
        return True
    return False


def _feature_importance(model: Any, feats: list[str], target: str) -> pd.DataFrame:
    if hasattr(model, "booster_"):
        gain = np.asarray(model.booster_.feature_importance("gain"), dtype=float)
    elif hasattr(model, "feature_importances_"):
        gain = np.asarray(model.feature_importances_, dtype=float)
    else:
        return pd.DataFrame(columns=["target", "feature", "gain", "share"])
    total = float(gain.sum())
    share = gain / total if total > 0 else np.zeros_like(gain)
    return pd.DataFrame({"target": target, "feature": feats, "gain": gain, "share": share}).sort_values("share", ascending=False)


# ── Strict Prior-Competition History (Zero Same-Event Leakage) ───────────────

def add_strict_site_history(tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Compute site history strictly from earlier competitions (`event_key` never sees its own tasks)."""
    t = tasks_df.copy()
    t["date"] = pd.to_datetime(t["date"])
    ev_start = t.groupby("event_key")["date"].transform("min")
    t["_ev_start"] = ev_start

    prev_kmh = np.full(len(t), np.nan, dtype=np.float64)
    prev_goal = np.full(len(t), np.nan, dtype=np.float64)
    prev_n = np.zeros(len(t), dtype=np.float64)

    for _, grp in t.groupby("site_id", sort=False):
        # Order events chronologically by their start date (tie-broken deterministically by event_key).
        ev_order = (
            grp.groupby("event_key", sort=False)
            .agg(d=("_ev_start", "min"))
            .reset_index()
            .sort_values(["d", "event_key"])
        )
        seen_kmh: list[float] = []
        seen_goal: list[float] = []
        seen_n = 0
        for ev_key in ev_order["event_key"]:
            sub = grp[grp["event_key"] == ev_key]
            idxs = sub.index.to_numpy()
            if seen_kmh:
                prev_kmh[idxs] = float(np.median(seen_kmh))
            if seen_goal:
                prev_goal[idxs] = float(np.median(seen_goal))
            prev_n[idxs] = float(seen_n)
            # Only add this competition's outcomes AFTER recording predictions for all of its tasks.
            seen_kmh.extend(sub["winner_kmh"].dropna().tolist())
            seen_goal.extend(sub["goal_share"].dropna().tolist())
            seen_n += len(sub)

    t["site_prev_winner_kmh_strict"] = prev_kmh
    t["site_prev_goal_share_strict"] = prev_goal
    t["site_prev_tasks_strict"] = prev_n
    return t.drop(columns=["_ev_start"])


def build_base_task_table(db: Path) -> pd.DataFrame:
    """Standard task table from task_baseline.py enriched with strict prior-competition site history."""
    t = TB.tasks_table(db)
    return add_strict_site_history(t)


def build_all_tasks_for_stopped(db: Path) -> pd.DataFrame:
    """All non-training race/elapsed tasks with >= 10 launched pilots, INCLUDING stopped tasks."""
    t = pd.read_parquet(db / "tasks.parquet")
    t = t[~t["training"].fillna(False).astype(bool) & t["task_type"].isin(["race", "elapsed"])].copy()
    r = pd.read_parquet(db / "results.parquet", columns=["result_id", "task_key", "status"])
    flew = r[~r["status"].isin(["abs", "dnf", "dsq"])]
    launched = flew.groupby("task_key").size().rename("launched")
    t = t.merge(launched.reset_index(), on="task_key")
    t = t[t["launched"] >= 10].copy()
    t["is_stopped"] = t["stopped"].fillna(False).astype(int)
    dist = t["ss_distance_km"].where(t["ss_distance_km"] > 0, t["distance_km"])
    t["ss_km"] = dist
    ev = pd.read_parquet(db / "event_sites.parquet").merge(
        pd.read_parquet(db / "sites.parquet", columns=["site_id", "lat", "lon"]), on="site_id"
    )
    t = t.merge(ev, on="event_key", how="left")
    t["date"] = pd.to_datetime(t["date"], errors="coerce")
    t = t.dropna(subset=["date", "site_id"]).reset_index(drop=True)
    doy = t["date"].dt.dayofyear.to_numpy()
    decl = 23.44 * np.sin(np.radians(360.0 / 365.0 * (doy - 81)))
    lat = np.radians(t["lat"].to_numpy(float))
    cos_h = np.clip(-np.tan(lat) * np.tan(np.radians(decl)), -1.0, 1.0)
    t["day_length_h"] = 2.0 * np.degrees(np.arccos(cos_h)) / 15.0
    t["sun_decl_local"] = decl * np.sign(t["lat"].to_numpy(float))
    t["abs_lat"] = t["lat"].abs()
    off = t["utc_offset_h"].astype(float)
    t["start_solar_h"] = t["start_time"].map(TB.hhmm) - off + t["lon"] / 15.0
    t["window_h"] = t["deadline"].map(TB.hhmm) - t["start_time"].map(TB.hhmm)
    return t


# ── In-Fold Leakage-Free Thermal Sub-Models (M4) ─────────────────────────────

def load_thermal_sub_datasets(
    db: Path,
    era_tables: dict[str, pd.DataFrame | None],
    thermal_rows: int = DEFAULT_THERMAL_ROWS,
    seed: int = 7,
) -> dict[str, pd.DataFrame]:
    """Load and enrich datasets for the three in-fold thermal sub-models (core60, ceiling, climb_prob)."""
    out: dict[str, pd.DataFrame] = {}
    truth_dir = db / "truth"
    cells_all = None
    if (db / "cells.parquet").exists():
        import pyarrow.parquet as pq

        c_names = set(pq.ParquetFile(db / "cells.parquet").schema.names)
        chip_cols = [c for c in ("chip_id", "elev_mean", "elev_std", "relief_5km") if c in c_names]
        if len(chip_cols) > 1:
            c_raw = pd.read_parquet(db / "cells.parquet", columns=chip_cols)
            cells_all = c_raw.groupby("chip_id").mean().reset_index().rename(
                columns={"elev_mean": "chip_elev_mean", "elev_std": "chip_elev_std", "relief_5km": "chip_relief_5km"}
            )

    # 1. Core 60 s climb strength (`truth_climb_groups.parquet`).
    cl_p = truth_dir / "truth_climb_groups.parquet"
    if cl_p.exists():
        cl = pd.read_parquet(cl_p, columns=["site_id", "chip_id", "hour", "event_key", "core60_p75_ms", "solar_hour"])
        cl = cl.dropna(subset=["core60_p75_ms", "event_key", "site_id"]).copy()
        cl["time"] = pd.to_datetime(cl["hour"]).astype("datetime64[ns]")
        if len(cl):
            wx, _ = RW._enrich_chip_hours(cl[["chip_id", "time"]], era_tables)
            cl = cl.merge(wx, on=["chip_id", "time"], how="left")
            if cells_all is not None:
                cl = cl.merge(cells_all, on="chip_id", how="left")
            out["core60"] = cl.reset_index(drop=True)

    # 2. Calibrated ASL top-of-lift (`truth_ceiling.parquet`, excluding airspace caps).
    ce_p = truth_dir / "truth_ceiling.parquet"
    if ce_p.exists():
        ce = pd.read_parquet(ce_p, columns=["site_id", "chip_id", "hour", "event_key", "ceiling_p90_asl", "airspace_cap_flag", "solar_hour"])
        ce = ce[~ce["airspace_cap_flag"].fillna(False).astype(bool)].dropna(subset=["ceiling_p90_asl", "event_key", "site_id"]).copy()
        ce["time"] = pd.to_datetime(ce["hour"]).astype("datetime64[ns]")
        if len(ce):
            wx, _ = RW._enrich_chip_hours(ce[["chip_id", "time"]], era_tables)
            ce = ce.merge(wx, on=["chip_id", "time"], how="left")
            if cells_all is not None:
                ce = ce.merge(cells_all, on="chip_id", how="left")
            ce["ceiling_resid_m"] = ce["ceiling_p90_asl"] - ce["eff_top_asl"]
            ce = ce.dropna(subset=["ceiling_resid_m"])
            out["ceiling"] = ce.reset_index(drop=True)

    # 3. Crossing climb probability (`crossings.parquet`, compact sample).
    cx_p = db / "crossings.parquet"
    if thermal_rows > 0 and cx_p.exists() and (db / "cells.parquet").exists():
        import pyarrow.parquet as pq

        cx = pd.read_parquet(cx_p, columns=["track_id", "task_key", "cell_id", "t_in", "duration_s", "agl_in", "climbing_s"])
        if (db / "flight_checks.parquet").exists():
            chk = pd.read_parquet(db / "flight_checks.parquet", columns=["track_id", "is_drive"])
            drives = set(chk.loc[chk["is_drive"].fillna(False).astype(bool), "track_id"])
            cx = cx[~cx["track_id"].isin(drives)]
        cx = cx[(cx["duration_s"] >= 20.0) & cx["agl_in"].notna()]
        if len(cx) > thermal_rows:
            cx = cx.sample(thermal_rows, random_state=seed)
        tasks = pd.read_parquet(db / "tasks.parquet", columns=["task_key", "event_key", "date"])
        ev = pd.read_parquet(db / "event_sites.parquet", columns=["event_key", "site_id"])
        tasks = tasks.merge(ev, on="event_key", how="left")
        tasks["date"] = pd.to_datetime(tasks["date"], errors="coerce")
        cx = cx.merge(tasks[["task_key", "event_key", "date", "site_id"]], on="task_key", how="inner")
        cx = cx.dropna(subset=["date", "site_id", "event_key"]).copy()
        if len(cx):
            cx["exact_time"] = cx["date"] + pd.to_timedelta(cx["t_in"], unit="s")
            cx["time"] = cx["exact_time"].dt.round("h").astype("datetime64[ns]")
            cx["climb"] = (cx["climbing_s"] >= 30.0).astype("int8")
            cx["chip_id"] = grid.chip_of_cell(cx["cell_id"].to_numpy()).astype(np.int64)

            c_names = pq.ParquetFile(db / "cells.parquet").schema.names
            sun_cols = [c for c in c_names if c.startswith("sun_")]
            want = ["cell_id", "lat", "lon"] + [c for c in RW.TERRAIN_CELL_COLS if c in c_names] + sun_cols
            cells = pd.read_parquet(db / "cells.parquet", columns=want)
            cells = cells[cells["cell_id"].isin(cx["cell_id"].unique())]
            cx = cx.merge(cells, on="cell_id", how="inner")
            if len(cx):
                la = cx["lat"].to_numpy(dtype=np.float64)
                lo = cx["lon"].to_numpy(dtype=np.float64)
                ts = cx["exact_time"].to_numpy("datetime64[s]")
                az, el = terrain.sun_position(la, lo, ts)
                cx["sun_elev"] = el.astype(np.float32)
                cx["solar_hour"] = (((ts.astype(np.int64) % 86400) / 3600.0 + lo / 15.0) % 24.0).astype(np.float32)
                if sun_cols and "flat_share" in cx:
                    cx["sun_direct"] = terrain.sun_on_cell(cx, az, el).astype(np.float32)
                    energy = np.zeros(len(cx), dtype=np.float64)
                    for k_h in (0, 2, 4, 6, 8, 10):
                        a_k, e_k = terrain.sun_position(la, lo, ts - np.timedelta64(3600 * k_h, "s"))
                        w_k = 1.0 if k_h in (0, 10) else 2.0
                        energy += terrain.sun_on_cell(cx, a_k, e_k) * w_k
                    cx["sun_energy"] = energy.astype(np.float32)
                    cx = cx.drop(columns=sun_cols)
                wx, _ = RW._enrich_chip_hours(cx[["chip_id", "time"]], era_tables)
                cx = cx.merge(wx, on=["chip_id", "time"], how="left")
                cx["agl_in_frac"] = cx["agl_in"].to_numpy(float) / np.maximum(cx["usable_m"].to_numpy(float), 100.0)
                if "facing_east" in cx and "facing_north" in cx:
                    cx["seg_upslope"] = -(cx["u_flt"] * cx["facing_east"] + cx["v_flt"] * cx["facing_north"])
                out["climb_prob"] = cx.reset_index(drop=True)

    return out


class ThermalSubBundle:
    """Holds the three leakage-free thermal sub-models fitted strictly on a training fold."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.core60_model: Any = None
        self.core60_feats: list[str] = []
        self.ceiling_model: Any = None
        self.ceiling_feats: list[str] = []
        self.climb_model: Any = None
        self.climb_feats: list[str] = []

    def fit(
        self,
        sub_data: dict[str, pd.DataFrame],
        group_col: str | None = None,
        train_groups: set[Any] | None = None,
    ) -> "ThermalSubBundle":
        def _filter(df: pd.DataFrame | None) -> pd.DataFrame:
            if df is None or not len(df):
                return pd.DataFrame()
            if group_col is None or train_groups is None:
                return df
            return df[df[group_col].isin(train_groups)]

        # 1. Core60 climb strength
        d_c = _filter(sub_data.get("core60"))
        if len(d_c) >= 40:
            cands = [c for c in SUB_WX_FEATURES + ["chip_elev_mean", "chip_elev_std", "chip_relief_5km"] if c in d_c and d_c[c].notna().any()]
            if cands:
                self.core60_feats = cands
                self.core60_model = make_sub_regressor(n_estimators=220, seed=self.seed)
                self.core60_model.fit(d_c[cands], d_c["core60_p75_ms"].to_numpy(dtype=np.float64))

        # 2. Ceiling ASL residual relative to min(BLH, LCL)
        d_z = _filter(sub_data.get("ceiling"))
        if len(d_z) >= 40:
            cands = [c for c in SUB_WX_FEATURES + ["chip_elev_mean", "chip_elev_std", "chip_relief_5km"] if c in d_z and d_z[c].notna().any()]
            if cands:
                self.ceiling_feats = cands
                self.ceiling_model = make_sub_regressor(n_estimators=220, seed=self.seed)
                self.ceiling_model.fit(d_z[cands], d_z["ceiling_resid_m"].to_numpy(dtype=np.float64))

        # 3. Crossing climb probability
        d_p = _filter(sub_data.get("climb_prob"))
        if len(d_p) >= 200 and d_p["climb"].nunique() == 2:
            cands = [
                c for c in ["agl_in", "agl_in_frac", "seg_upslope"] + SUB_CLIMB_TERRAIN + SUB_WX_FEATURES
                if c in d_p and d_p[c].notna().any()
            ]
            if cands:
                self.climb_feats = cands
                self.climb_model = make_binary_classifier(n_estimators=200, min_child=150, seed=self.seed)
                self.climb_model.fit(d_p[cands], d_p["climb"].to_numpy(dtype=np.int8))

        return self

    def _align_X(self, df: pd.DataFrame, feats: list[str]) -> pd.DataFrame:
        X = pd.DataFrame(index=df.index)
        for c in feats:
            if c in df:
                X[c] = df[c].to_numpy(dtype=np.float64)
            elif c == "chip_elev_mean" and "route_elev" in df:
                X[c] = df["route_elev"].to_numpy(dtype=np.float64)
            elif c == "chip_elev_std" and "elev_std" in df:
                X[c] = df["elev_std"].to_numpy(dtype=np.float64)
            elif c == "chip_relief_5km" and "relief_5km" in df:
                X[c] = df["relief_5km"].to_numpy(dtype=np.float64)
            else:
                X[c] = np.nan
        return X

    def predict_task_features(
        self,
        lead_panel: pd.DataFrame,
        pack_panel: pd.DataFrame,
        task_keys: pd.Index,
        task_window_h: dict[str, float],
    ) -> pd.DataFrame:
        """Score segment-hour panels with the fitted sub-models and aggregate to task features."""
        out = pd.DataFrame(index=pd.Index(task_keys, name="task_key"))

        def _wmean(df_p: pd.DataFrame, vals: np.ndarray) -> pd.Series:
            v = np.asarray(vals, dtype=np.float64)
            w = np.where(np.isfinite(v), df_p["seg_km"].to_numpy(dtype=np.float64), 0.0)
            tmp = pd.DataFrame({"task_key": df_p["task_key"].to_numpy(), "vw": np.where(np.isfinite(v), v * w, 0.0), "w": w})
            g_t = tmp.groupby("task_key", sort=False).sum()
            return g_t["vw"] / g_t["w"].where(g_t["w"] > 0.0)

        # Core60 predictions along leader & pack trajectories.
        c60_lead = np.clip(0.85 * lead_panel["wstar_rad"].to_numpy(dtype=np.float64), 0.3, 5.0)
        c60_pack = np.clip(0.85 * pack_panel["wstar_rad"].to_numpy(dtype=np.float64), 0.3, 5.0)
        if self.core60_model is not None and len(lead_panel):
            c60_lead = np.clip(self.core60_model.predict(self._align_X(lead_panel, self.core60_feats)), 0.25, 7.0)
            c60_pack = np.clip(self.core60_model.predict(self._align_X(pack_panel, self.core60_feats)), 0.25, 7.0)
            out["pred_core60_lead_mean"] = _wmean(lead_panel, c60_lead)
            out["pred_core60_lead_min"] = pd.Series(c60_lead, index=lead_panel.index).groupby(lead_panel["task_key"]).min()
            out["pred_core60_pack_mean"] = _wmean(pack_panel, c60_pack)
            out["pred_core60_pack_min"] = pd.Series(c60_pack, index=pack_panel.index).groupby(pack_panel["task_key"]).min()

        # Calibrated ASL ceiling & usable height predictions.
        u_lead = lead_panel["cell_usable_m"].to_numpy(dtype=np.float64) if "cell_usable_m" in lead_panel else lead_panel["usable_m"].to_numpy(dtype=np.float64)
        u_pack = pack_panel["cell_usable_m"].to_numpy(dtype=np.float64) if "cell_usable_m" in pack_panel else pack_panel["usable_m"].to_numpy(dtype=np.float64)
        if self.ceiling_model is not None and len(lead_panel):
            r_lead = self.ceiling_model.predict(self._align_X(lead_panel, self.ceiling_feats))
            r_pack = self.ceiling_model.predict(self._align_X(pack_panel, self.ceiling_feats))
            elev_l = lead_panel["elev_mean"].to_numpy(dtype=np.float64) if "elev_mean" in lead_panel else lead_panel["route_elev"].to_numpy(dtype=np.float64)
            elev_p = pack_panel["elev_mean"].to_numpy(dtype=np.float64) if "elev_mean" in pack_panel else pack_panel["route_elev"].to_numpy(dtype=np.float64)
            u_lead = lead_panel["eff_top_asl"].to_numpy(dtype=np.float64) + r_lead - elev_l
            u_pack = pack_panel["eff_top_asl"].to_numpy(dtype=np.float64) + r_pack - elev_p
            s_ul = pd.Series(u_lead, index=lead_panel.index)
            s_up = pd.Series(u_pack, index=pack_panel.index)
            out["pred_usable_lead_mean"] = _wmean(lead_panel, u_lead)
            out["pred_usable_lead_min"] = s_ul.groupby(lead_panel["task_key"]).min()
            out["pred_usable_lead_p10"] = s_ul.groupby(lead_panel["task_key"]).quantile(0.10)
            out["pred_usable_pack_min"] = s_up.groupby(pack_panel["task_key"]).min()

        # Climb probability at standardized 300 m (low save) and 700 m (cruise) AGL probes.
        if self.climb_model is not None and len(lead_panel):
            for probe_m, tag in ((300.0, "low"), (700.0, "cruise")):
                Xp = self._align_X(lead_panel, self.climb_feats)
                if "agl_in" in Xp:
                    Xp["agl_in"] = probe_m
                if "agl_in_frac" in Xp:
                    Xp["agl_in_frac"] = probe_m / np.maximum(u_lead, 100.0)
                probs = self.climb_model.predict_proba(Xp)[:, 1]
                out[f"pred_p_climb_{tag}_mean"] = _wmean(lead_panel, probs)
                tmp = pd.DataFrame({
                    "task_key": lead_panel["task_key"].to_numpy(),
                    "leg_idx": lead_panel["leg_idx"].to_numpy(),
                    "p": probs,
                })
                leg_p = tmp.groupby(["task_key", "leg_idx"], sort=False)["p"].mean().reset_index()
                out[f"pred_p_climb_{tag}_min_leg"] = leg_p.groupby("task_key")["p"].min()

        # Upgraded MacCready kinematics using predicted core60 & predicted usable height.
        if len(lead_panel):
            tl = lead_panel["seg_tail"].to_numpy(dtype=np.float64)
            cl = lead_panel["seg_cross"].to_numpy(dtype=np.float64)
            tp = pack_panel["seg_tail"].to_numpy(dtype=np.float64)
            cp = pack_panel["seg_cross"].to_numpy(dtype=np.float64)
            mc_ccc = RW.maccready_task_summary(lead_panel, c60_lead, tl, cl, u_lead, task_window_h, "CCC", "mc_sub")
            mc_end = RW.maccready_task_summary(lead_panel, c60_lead, tl, cl, u_lead, task_window_h, "EN-D", "mc_end")
            mc_enc = RW.maccready_task_summary(pack_panel, c60_pack, tp, cp, u_pack, task_window_h, "EN-C", "mc_enc")
            out["mc_sub_speed_lead_kmh"] = mc_ccc["mc_sub_speed_kmh"]
            out["mc_sub_time_lead_h"] = mc_ccc["mc_sub_time_h"]
            out["mc_sub_margin_lead_h"] = mc_ccc["mc_sub_margin_h"]
            out["mc_sub_margin_pack_h"] = mc_enc["mc_enc_margin_h"]
            out["mc_sub_unflyable_share"] = mc_ccc["mc_sub_unflyable_share"]
            out["mc_sub_margin_ccc_h"] = mc_ccc["mc_sub_margin_h"]
            out["mc_sub_margin_end_h"] = mc_end["mc_end_margin_h"]
            out["mc_sub_margin_enc_h"] = mc_enc["mc_enc_margin_h"]

        for c in THERMAL_SUB_FEATURES:
            if c not in out:
                out[c] = np.nan
        return out[THERMAL_SUB_FEATURES]


# ── Pilot-Level Table & Model (M5, ~173k results -> aggregated goal share) ───

def _harness_code(s: Any) -> float:
    if not isinstance(s, str) or not s.strip():
        return 0.0
    k = s.lower()
    if any(tok in k for tok in ("sub", "fairing", "xr", "xrated", "genierace", "enzo", "diva")):
        return 2.0
    return 1.0


def build_pilot_table(db: Path, tasks_df: pd.DataFrame) -> pd.DataFrame:
    """Build strictly pre-task pilot-result table aligned with `tasks_df` (~150k-173k rows)."""
    r = pd.read_parquet(
        db / "results.parquet",
        columns=["result_id", "task_key", "pilot_key", "status", "time_s", "distance_km"],
    )
    r = r[r["task_key"].isin(tasks_df["task_key"]) & ~r["status"].isin(["abs", "dnf", "dsq"])].copy()
    t_meta = tasks_df[["task_key", "event_key", "site_id", "date", "ss_km", "winner_kmh", "launched"]].copy()
    for opt in ("field_mean", "field_top10", "ccc_share"):
        t_meta[opt] = tasks_df[opt] if opt in tasks_df else np.nan
    r = r.merge(t_meta, on="task_key", how="inner")

    if (db / "ratings.parquet").exists():
        rat = pd.read_parquet(db / "ratings.parquet", columns=["task_key", "pilot_key", "rating_before", "tasks_before"])
        rat = rat.drop_duplicates(["task_key", "pilot_key"])
        r = r.merge(rat, on=["task_key", "pilot_key"], how="left")
    else:
        r["rating_before"] = np.nan
        r["tasks_before"] = 0.0
    r["rating_before"] = r["rating_before"].fillna(1500.0)
    r["tasks_before"] = r["tasks_before"].fillna(0.0)

    if (db / "result_gliders.parquet").exists():
        rg = pd.read_parquet(db / "result_gliders.parquet", columns=["result_id", "glider_class", "harness_key"])
        rg = rg.drop_duplicates("result_id")
        r = r.merge(rg, on="result_id", how="left")
    else:
        r["glider_class"] = None
        r["harness_key"] = ""

    r["in_goal"] = (r["status"] == "goal").astype("int8")
    r["pilot_kmh"] = np.where(
        (r["in_goal"] == 1) & (r["time_s"] > 0) & r["ss_km"].notna(),
        r["ss_km"] / (r["time_s"] / 3600.0),
        np.nan,
    )
    r.loc[(r["pilot_kmh"] < 5.0) | (r["pilot_kmh"] > 80.0), "pilot_kmh"] = np.nan
    r["_speed_ratio"] = np.where(r["pilot_kmh"].notna() & (r["winner_kmh"] > 0), r["pilot_kmh"] / r["winner_kmh"], np.nan)

    # Pre-task relative standing inside this task's entry list.
    task_mean_r = r.groupby("task_key")["rating_before"].transform("mean")
    r["rating_vs_field_mean"] = r["rating_before"] - r["field_mean"].fillna(task_mean_r)
    r["rating_vs_top10"] = r["rating_before"] - r["field_top10"].fillna(task_mean_r)
    r["rating_pct_in_field"] = r.groupby("task_key")["rating_before"].rank(pct=True)

    # Equipment features.
    r["glider_class_ord"] = r["glider_class"].map(GLIDER_ORD)
    r["is_ccc"] = (r["glider_class"] == "CCC").astype(float)
    r["is_en_d"] = (r["glider_class"] == "EN-D").astype(float)
    r["is_en_c_low"] = r["glider_class"].isin(["EN-C", "EN-B", "EN-A"]).astype(float)
    r["harness_code"] = r["harness_key"].map(_harness_code)
    r["glider_vs_field_ccc"] = r["is_ccc"] - r["ccc_share"].fillna(r.groupby("task_key")["is_ccc"].transform("mean"))

    # Strictly causal prior-competition pilot history (event_key never sees its own tasks).
    ev_start = r.groupby("event_key")["date"].transform("min")
    r["_ev_start"] = ev_start
    r = r.sort_values(["_ev_start", "event_key", "date", "task_key", "result_id"]).reset_index(drop=True)

    n_rows = len(r)
    prev_tasks = np.zeros(n_rows, dtype=np.float32)
    prev_goal_rate = np.full(n_rows, 0.5, dtype=np.float32)
    prev_speed_ratio = np.full(n_rows, np.nan, dtype=np.float32)
    prev_site_tasks = np.zeros(n_rows, dtype=np.float32)

    p_cnt: dict[str, int] = {}
    p_goal: dict[str, int] = {}
    p_sr_sum: dict[str, float] = {}
    p_sr_cnt: dict[str, int] = {}
    ps_cnt: dict[tuple[str, str], int] = {}

    # Process competition by competition so no task inside event_key leaks to another task in the same event.
    for _, ev_grp in r.groupby(["_ev_start", "event_key"], sort=False):
        idxs = ev_grp.index.to_numpy()
        pkeys = ev_grp["pilot_key"].to_numpy()
        sids = ev_grp["site_id"].to_numpy()
        for i_row, pk, sid in zip(idxs, pkeys, sids):
            if pd.isna(pk):
                continue
            pk_s = str(pk)
            n_p = p_cnt.get(pk_s, 0)
            g_p = p_goal.get(pk_s, 0)
            prev_tasks[i_row] = float(n_p)
            prev_goal_rate[i_row] = float((g_p + 2.5) / (n_p + 5.0))
            sc = p_sr_cnt.get(pk_s, 0)
            if sc > 0:
                prev_speed_ratio[i_row] = float(p_sr_sum[pk_s] / sc)
            prev_site_tasks[i_row] = float(ps_cnt.get((pk_s, str(sid)), 0))

        # Commit this competition's outcomes AFTER populating all tasks in the competition.
        for pk, sid, ig, sr in zip(pkeys, sids, ev_grp["in_goal"].to_numpy(), ev_grp["_speed_ratio"].to_numpy()):
            if pd.isna(pk):
                continue
            pk_s = str(pk)
            p_cnt[pk_s] = p_cnt.get(pk_s, 0) + 1
            p_goal[pk_s] = p_goal.get(pk_s, 0) + int(ig)
            if np.isfinite(sr):
                p_sr_sum[pk_s] = p_sr_sum.get(pk_s, 0.0) + float(sr)
                p_sr_cnt[pk_s] = p_sr_cnt.get(pk_s, 0) + 1
            key_ps = (pk_s, str(sid))
            ps_cnt[key_ps] = ps_cnt.get(key_ps, 0) + 1

    r["pilot_prev_tasks"] = prev_tasks
    r["pilot_prev_goal_rate"] = prev_goal_rate
    r["pilot_prev_speed_ratio"] = prev_speed_ratio
    r["pilot_prev_site_tasks"] = prev_site_tasks
    return r.drop(columns=["_ev_start", "_speed_ratio"])


def _attach_pilot_task_features(
    pilot_df: pd.DataFrame,
    task_df: pd.DataFrame,
    task_feat_cols: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Join task-level features onto pilot rows and compute pilot x weather interactions."""
    use_task = [c for c in task_feat_cols if c in task_df and c not in pilot_df.columns]
    extra_cols = [c for c in ("mc_sub_margin_ccc_h", "mc_sub_margin_end_h", "mc_sub_margin_enc_h",
                              "mc_rad_margin_lead_h", "mc_rad_margin_pack_h", "tail_lead_min_leg")
                  if c in task_df and c not in use_task and c not in pilot_df.columns]
    m = pilot_df.merge(task_df[["task_key", *use_task, *extra_cols]], on="task_key", how="inner")

    # Class-matched MacCready margin interaction.
    gc = m["glider_class"].fillna("")
    ccc_m = m["mc_sub_margin_ccc_h"] if "mc_sub_margin_ccc_h" in m else m.get("mc_rad_margin_lead_h", pd.Series(np.nan, index=m.index))
    end_m = m["mc_sub_margin_end_h"] if "mc_sub_margin_end_h" in m else ccc_m
    enc_m = m["mc_sub_margin_enc_h"] if "mc_sub_margin_enc_h" in m else m.get("mc_rad_margin_pack_h", ccc_m)
    m["pilot_mc_margin_h"] = np.where(gc == "CCC", ccc_m, np.where(gc == "EN-D", end_m, enc_m))

    vbar = gc.map(GLIDER_VBAR_MS).fillna(GLIDER_VBAR_MS["EN-D"]).to_numpy(dtype=np.float64)
    tail_min = m["tail_lead_min_leg"].to_numpy(dtype=np.float64) if "tail_lead_min_leg" in m else np.full(len(m), np.nan)
    m["pilot_headwind_penetration"] = vbar + tail_min

    feat_cols = list(dict.fromkeys(c for c in PILOT_FEATURES + task_feat_cols if c in m and m[c].notna().any()))
    return m, feat_cols


class PilotModelBundle:
    """Pilot-level goal probability (calibrated) + goal-speed regressor aggregated to tasks."""

    def __init__(self, seed: int = 7) -> None:
        self.seed = seed
        self.goal_model: Any = None
        self.goal_cal: IsotonicRegression | None = None
        self.speed_model: Any = None
        self.feat_cols: list[str] = []

    def fit(
        self,
        pilot_tr: pd.DataFrame,
        task_tr: pd.DataFrame,
        task_feat_cols: list[str],
        group_col: str = "event_key",
    ) -> "PilotModelBundle":
        m, feats = _attach_pilot_task_features(pilot_tr, task_tr, task_feat_cols)
        if not len(m) or not feats:
            return self
        self.feat_cols = feats
        w_goal = 1.0 / np.maximum(m["launched"].to_numpy(dtype=np.float64), 1.0)
        y_goal = m["in_goal"].to_numpy(dtype=np.int8)

        # In-fold probability calibration via inner 3-fold GroupKFold on training events.
        groups = m[group_col].astype(str).to_numpy()
        n_grp = len(np.unique(groups))
        if n_grp >= 3 and len(np.unique(y_goal)) == 2:
            oof_raw = np.full(len(m), np.nan, dtype=np.float64)
            gkf = GroupKFold(n_splits=min(3, n_grp))
            for itr, ite in gkf.split(m, groups=groups):
                if len(np.unique(y_goal[itr])) < 2:
                    continue
                clf_i = make_pilot_classifier(n_estimators=220, seed=self.seed)
                clf_i.fit(m.iloc[itr][feats], y_goal[itr], sample_weight=w_goal[itr])
                oof_raw[ite] = clf_i.predict_proba(m.iloc[ite][feats])[:, 1]
            ok = np.isfinite(oof_raw)
            if ok.sum() >= 50:
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(oof_raw[ok], y_goal[ok], sample_weight=w_goal[ok])
                self.goal_cal = iso

        if len(np.unique(y_goal)) == 2:
            self.goal_model = make_pilot_classifier(n_estimators=350, seed=self.seed)
            self.goal_model.fit(m[feats], y_goal, sample_weight=w_goal)

        # Pilot speed model on goal finishers (weighted equally per task).
        fin = m[m["pilot_kmh"].notna()].copy()
        if len(fin) >= 40:
            n_fin = fin.groupby("task_key")["pilot_kmh"].transform("size").to_numpy(dtype=np.float64)
            w_spd = 1.0 / np.maximum(n_fin, 1.0)
            self.speed_model = make_pilot_regressor(n_estimators=320, seed=self.seed)
            self.speed_model.fit(fin[feats], fin["pilot_kmh"].to_numpy(dtype=np.float64), sample_weight=w_spd)
        return self

    def predict(
        self,
        pilot_te: pd.DataFrame,
        task_te: pd.DataFrame,
        task_feat_cols: list[str],
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return `(task_agg_df indexed by task_key, pilot_level_pred_df)`."""
        m, _ = _attach_pilot_task_features(pilot_te, task_te, task_feat_cols)
        if not len(m) or not self.feat_cols:
            empty_t = pd.DataFrame(index=pd.Index(task_te["task_key"], name="task_key"))
            return empty_t, pd.DataFrame()
        X = pd.DataFrame({c: (m[c].to_numpy(dtype=np.float64) if c in m else np.nan) for c in self.feat_cols}, index=m.index)

        if self.goal_model is not None:
            p_raw = self.goal_model.predict_proba(X)[:, 1]
            p_cal = self.goal_cal.predict(p_raw) if self.goal_cal is not None else p_raw
            m["pred_p_goal_raw"] = p_raw
            m["pred_p_goal"] = np.clip(p_cal, 0.0, 1.0)
        else:
            m["pred_p_goal_raw"] = np.nan
            m["pred_p_goal"] = np.nan

        if self.speed_model is not None:
            m["pred_pilot_kmh"] = np.clip(self.speed_model.predict(X), 5.0, 80.0)
        else:
            m["pred_pilot_kmh"] = np.nan

        # Aggregate pilot-level predictions up to task level.
        by = m.groupby("task_key", sort=False)
        agg = pd.DataFrame({
            "pilot_goal_share": by["pred_p_goal"].mean(),
            "pilot_goal_share_raw": by["pred_p_goal_raw"].mean(),
            "pilot_median_kmh": by["pred_pilot_kmh"].median(),
        })
        # Winner speed proxy: mean predicted speed of top-3 highest-rated pilots in the field.
        top3 = m.sort_values(["task_key", "rating_before"], ascending=[True, False]).groupby("task_key", sort=False).head(3)
        agg["pilot_winner_kmh"] = top3.groupby("task_key")["pred_pilot_kmh"].mean()

        for cls, tag in (("CCC", "ccc"), ("EN-D", "end"), ("EN-C", "enc")):
            sub = m[m["glider_class"] == cls]
            if len(sub):
                agg[f"pilot_goal_share_{tag}"] = sub.groupby("task_key")["pred_p_goal"].mean()
            else:
                agg[f"pilot_goal_share_{tag}"] = np.nan

        keep_p = [
            "result_id", "task_key", "event_key", "site_id", "glider_class", "rating_before",
            "in_goal", "pilot_kmh", "pred_p_goal", "pred_p_goal_raw", "pred_pilot_kmh",
        ]
        return agg.reindex(task_te["task_key"]), m[[c for c in keep_p if c in m]].copy()


# ── Metrics & Paired Clustered Bootstrap ─────────────────────────────────────

def winner_time_err_min(df: pd.DataFrame, pred_kmh: np.ndarray) -> np.ndarray:
    err = np.full(len(df), np.nan, dtype=np.float64)
    ok = np.isfinite(pred_kmh) & df["winner_kmh"].notna().to_numpy()
    if ok.any():
        pk = np.maximum(5.0, pred_kmh[ok])
        err[ok] = np.abs(df["winner_h"].to_numpy(dtype=np.float64)[ok] - df["ss_km"].to_numpy(dtype=np.float64)[ok] / pk) * 60.0
    return err


def goal_share_err_pts(df: pd.DataFrame, pred_goal: np.ndarray) -> np.ndarray:
    err = np.full(len(df), np.nan, dtype=np.float64)
    ok = np.isfinite(pred_goal) & df["goal_share"].notna().to_numpy()
    if ok.any():
        err[ok] = np.abs(df["goal_share"].to_numpy(dtype=np.float64)[ok] - np.clip(pred_goal[ok], 0.0, 1.0)) * 100.0
    return err


def score_predictions(df: pd.DataFrame, pred_kmh: np.ndarray, pred_goal: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {}
    ok_k = np.isfinite(pred_kmh) & df["winner_kmh"].notna().to_numpy()
    if ok_k.any():
        yk = df["winner_kmh"].to_numpy(dtype=np.float64)[ok_k]
        pk = np.maximum(5.0, pred_kmh[ok_k])
        err_m = winner_time_err_min(df, pred_kmh)[ok_k]
        out.update({
            "n_time": int(ok_k.sum()),
            "winner_kmh_mae": round(float(np.mean(np.abs(yk - pk))), 3),
            "winner_time_mae_min": round(float(np.nanmean(err_m)), 2),
            "within_15_min": round(float(np.nanmean(err_m <= 15.0)), 4),
            "within_30_min": round(float(np.nanmean(err_m <= 30.0)), 4),
        })
    ok_g = np.isfinite(pred_goal) & df["goal_share"].notna().to_numpy()
    if ok_g.any():
        err_g = goal_share_err_pts(df, pred_goal)[ok_g]
        out.update({
            "n_goal": int(ok_g.sum()),
            "goal_share_mae_pts": round(float(np.mean(err_g)), 2),
            "within_10_pts": round(float(np.mean(err_g <= 10.0)), 4),
        })
    return out


def paired_bootstrap_diff(
    clusters: np.ndarray,
    err_a: np.ndarray,
    err_b: np.ndarray,
    n_iter: int = BOOTSTRAP_ITERS,
    seed: int = 42,
) -> dict[str, Any]:
    """Clustered paired bootstrap of mean(err_a - err_b). Negative = A improves over B."""
    ok = np.isfinite(err_a) & np.isfinite(err_b)
    if not ok.any():
        return {"tasks": 0}
    d = err_a[ok] - err_b[ok]
    uniq, inv = np.unique(np.asarray(clusters)[ok].astype(str), return_inverse=True)
    out: dict[str, Any] = {
        "tasks": int(ok.sum()),
        "clusters": int(len(uniq)),
        "mean_a": round(float(err_a[ok].mean()), 3),
        "mean_b": round(float(err_b[ok].mean()), 3),
        "mean_diff": round(float(d.mean()), 3),
    }
    if len(uniq) < 3:
        out.update({"ci95": [out["mean_diff"], out["mean_diff"]], "p_a_not_better": None, "significant": False})
        return out
    sum_d = np.bincount(inv, weights=d)
    cnt = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    mult = rng.multinomial(len(uniq), np.full(len(uniq), 1.0 / len(uniq)), size=n_iter).astype(float)
    stats = (mult @ sum_d) / np.maximum(mult @ cnt, 1.0)
    lo, hi = np.percentile(stats, [2.5, 97.5])
    out.update({
        "ci95": [round(float(lo), 3), round(float(hi), 3)],
        "p_a_not_better": round(float(np.mean(stats >= 0.0)), 4),
        "significant": bool(hi < 0.0 or lo > 0.0),
    })
    return out


def _fit_convex_weight(y: np.ndarray, p_task: np.ndarray, p_pilot: np.ndarray) -> float:
    """Grid-search optimal convex blend weight w in [0, 1] minimizing MAE(y, w*p_task + (1-w)*p_pilot)."""
    ok = np.isfinite(y) & np.isfinite(p_task) & np.isfinite(p_pilot)
    if ok.sum() < 15:
        return 0.5
    yt, a, b = y[ok], p_task[ok], p_pilot[ok]
    ws = np.linspace(0.0, 1.0, 21)
    errs = [float(np.mean(np.abs(yt - (w * a + (1.0 - w) * b)))) for w in ws]
    return float(ws[int(np.argmin(errs))])


# ── Cross-Validation Engine (Ablation Ladder M0..M5) ─────────────────────────

def _cv_task_only(
    t: pd.DataFrame,
    feats: list[str],
    split_col: str,
    n_splits: int = 5,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    """Run grouped CV for `winner_kmh` and `goal_share` on feature set `feats`."""
    use = [f for f in feats if f in t and t[f].notna().any()]
    n_grp = t[split_col].nunique()
    k = min(n_splits, n_grp)
    gkf = GroupKFold(n_splits=k)
    pred_k = np.full(len(t), np.nan, dtype=np.float64)
    pred_g = np.full(len(t), np.nan, dtype=np.float64)
    if not use or k < 2:
        return pred_k, pred_g

    for tr_i, te_i in gkf.split(t, groups=t[split_col]):
        tr = t.iloc[tr_i]
        te = t.iloc[te_i]
        tr_k = tr.dropna(subset=["winner_kmh"])
        if len(tr_k) >= 10:
            mk = make_task_regressor(seed=seed)
            mk.fit(tr_k[use], tr_k["winner_kmh"])
            pred_k[te_i] = mk.predict(te[use])
        tr_g = tr.dropna(subset=["goal_share"])
        if len(tr_g) >= 10:
            mg = make_task_regressor(seed=seed)
            mg.fit(tr_g[use], tr_g["goal_share"])
            pred_g[te_i] = np.clip(mg.predict(te[use]), 0.0, 1.0)
    return pred_k, pred_g


def run_ablation_cv(
    t_df: pd.DataFrame,
    v1_feats: list[str],
    lead_panel: pd.DataFrame,
    pack_panel: pd.DataFrame,
    task_window_h: dict[str, float],
    sub_data: dict[str, pd.DataFrame],
    pilot_df: pd.DataFrame,
    split_col: str,
    n_splits: int = 5,
    seed: int = 7,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame, dict[str, Any]]:
    """Run the complete M0..M5 ablation ladder strictly inside identical `split_col` folds."""
    base_v0 = [f for f in TB.FEATURES if f in t_df and t_df[f].notna().any()]
    base_strict = [f for f in STRICT_BASE_FEATURES if f in t_df and t_df[f].notna().any()]
    m2_feats = base_strict + [f for f in RW.HOURLY_CORE_FEATURES if f in t_df and t_df[f].notna().any()]
    m3_fc_feats = m2_feats + [f for f in RW.TERRAIN_ROUTE_FEATURES if f in t_df and t_df[f].notna().any()]
    m3_full_feats = m3_fc_feats + [f for f in RW.ERA5_LAND_FEATURES if f in t_df and t_df[f].notna().any()]

    preds: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    preds["M0_baseline"] = _cv_task_only(t_df, base_v0, split_col, n_splits=n_splits, seed=seed)
    preds["M1_task_v1_5h"] = _cv_task_only(t_df, base_v0 + v1_feats, split_col, n_splits=n_splits, seed=seed)
    preds["M2_hourly_1h"] = _cv_task_only(t_df, m2_feats, split_col, n_splits=n_splits, seed=seed)
    preds["M3_forecast_compat"] = _cv_task_only(t_df, m3_fc_feats, split_col, n_splits=n_splits, seed=seed)
    preds["M3_terrain_land"] = _cv_task_only(t_df, m3_full_feats, split_col, n_splits=n_splits, seed=seed)

    # M4 (in-fold thermal sub-models + MacCready) and M5 (pilot-level model + in-fold blend).
    n_grp = t_df[split_col].nunique()
    k = min(n_splits, n_grp)
    gkf = GroupKFold(n_splits=k)

    m4_k = np.full(len(t_df), np.nan, dtype=np.float64)
    m4_g = np.full(len(t_df), np.nan, dtype=np.float64)
    m4_fc_k = np.full(len(t_df), np.nan, dtype=np.float64)
    m4_fc_g = np.full(len(t_df), np.nan, dtype=np.float64)
    m5_k = np.full(len(t_df), np.nan, dtype=np.float64)
    m5_g = np.full(len(t_df), np.nan, dtype=np.float64)
    blend_k = np.full(len(t_df), np.nan, dtype=np.float64)
    blend_g = np.full(len(t_df), np.nan, dtype=np.float64)

    oof_sub_rows: list[pd.DataFrame] = []
    oof_pilot_rows: list[pd.DataFrame] = []
    blend_weights_g: list[float] = []
    blend_weights_k: list[float] = []

    has_pilots = len(pilot_df) > 0
    for fold_i, (tr_i, te_i) in enumerate(gkf.split(t_df, groups=t_df[split_col])):
        tr_base = t_df.iloc[tr_i].copy()
        te_base = t_df.iloc[te_i].copy()
        tr_groups = set(tr_base[split_col].unique())
        tr_keys = set(tr_base["task_key"])
        te_keys = set(te_base["task_key"])

        # Fit thermal sub-models strictly on training groups (zero leakage to test fold).
        sub_bundle = ThermalSubBundle(seed=seed + fold_i).fit(sub_data, group_col=split_col, train_groups=tr_groups)

        # Compute sub-model task features for test fold using the outer-train bundle.
        te_lead = lead_panel[lead_panel["task_key"].isin(te_keys)]
        te_pack = pack_panel[pack_panel["task_key"].isin(te_keys)]
        te_sub = sub_bundle.predict_task_features(te_lead, te_pack, pd.Index(te_base["task_key"]), task_window_h)
        oof_sub_rows.append(te_sub)

        # For training-fold tasks, produce strictly out-of-fold sub-model features via inner 3-fold CV
        # so the task model does not over-rely on in-sample overfit sub-model predictions.
        tr_sub_parts = []
        n_tr_g = tr_base[split_col].nunique()
        if n_tr_g >= 3 and any(len(v) > 0 for v in sub_data.values()):
            inner_gkf = GroupKFold(n_splits=min(3, n_tr_g))
            for itr, ite in inner_gkf.split(tr_base, groups=tr_base[split_col]):
                itr_groups = set(tr_base.iloc[itr][split_col].unique())
                ite_keys = set(tr_base.iloc[ite]["task_key"])
                ib = ThermalSubBundle(seed=seed + 100 + fold_i).fit(sub_data, group_col=split_col, train_groups=itr_groups)
                tr_sub_parts.append(
                    ib.predict_task_features(
                        lead_panel[lead_panel["task_key"].isin(ite_keys)],
                        pack_panel[pack_panel["task_key"].isin(ite_keys)],
                        pd.Index(tr_base.iloc[ite]["task_key"]),
                        task_window_h,
                    )
                )
            tr_sub = pd.concat(tr_sub_parts).reindex(tr_base["task_key"])
        else:
            tr_sub = sub_bundle.predict_task_features(
                lead_panel[lead_panel["task_key"].isin(tr_keys)],
                pack_panel[pack_panel["task_key"].isin(tr_keys)],
                pd.Index(tr_base["task_key"]),
                task_window_h,
            )

        tr_m4 = tr_base.merge(tr_sub, left_on="task_key", right_index=True, how="left")
        te_m4 = te_base.merge(te_sub, left_on="task_key", right_index=True, how="left")

        sub_cols_ok = [c for c in THERMAL_SUB_FEATURES if tr_m4[c].notna().any()]
        f_m4 = m3_full_feats + sub_cols_ok
        f_m4_fc = m3_fc_feats + sub_cols_ok

        # M4 (ERA5-full) and M4_forecast_compat (no ERA5-Land / sensible heat)
        for feats_use, arr_k, arr_g in ((f_m4, m4_k, m4_g), (f_m4_fc, m4_fc_k, m4_fc_g)):
            tr_k = tr_m4.dropna(subset=["winner_kmh"])
            if len(tr_k) >= 10:
                mk = make_task_regressor(seed=seed + fold_i)
                mk.fit(tr_k[feats_use], tr_k["winner_kmh"])
                arr_k[te_i] = mk.predict(te_m4[feats_use])
            tr_g = tr_m4.dropna(subset=["goal_share"])
            if len(tr_g) >= 10:
                mg = make_task_regressor(seed=seed + fold_i)
                mg.fit(tr_g[feats_use], tr_g["goal_share"])
                arr_g[te_i] = np.clip(mg.predict(te_m4[feats_use]), 0.0, 1.0)

        # M5: Pilot-level model + in-fold meta-blend weight.
        if has_pilots:
            p_tr = pilot_df[pilot_df["task_key"].isin(tr_keys)]
            p_te = pilot_df[pilot_df["task_key"].isin(te_keys)]
            if len(p_tr) >= 50 and len(p_te):
                pb = PilotModelBundle(seed=seed + fold_i).fit(p_tr, tr_m4, f_m4, group_col=split_col)
                t_agg, p_pred = pb.predict(p_te, te_m4, f_m4)
                oof_pilot_rows.append(p_pred)
                m5_g[te_i] = t_agg["pilot_goal_share"].to_numpy(dtype=np.float64)
                m5_k[te_i] = t_agg["pilot_winner_kmh"].to_numpy(dtype=np.float64)

                # Inner OOF convex blend weight fitted strictly inside tr_m4 (zero test leakage).
                w_g, w_k = 0.5, 0.75
                if n_tr_g >= 3:
                    in_gkf = GroupKFold(n_splits=min(3, n_tr_g))
                    in_tk = np.full(len(tr_m4), np.nan)
                    in_tg = np.full(len(tr_m4), np.nan)
                    in_pk = np.full(len(tr_m4), np.nan)
                    in_pg = np.full(len(tr_m4), np.nan)
                    for itr, ite in in_gkf.split(tr_m4, groups=tr_m4[split_col]):
                        itr_df = tr_m4.iloc[itr]
                        ite_df = tr_m4.iloc[ite]
                        ik = itr_df.dropna(subset=["winner_kmh"])
                        ig = itr_df.dropna(subset=["goal_share"])
                        if len(ik) >= 10:
                            m_ik = make_task_regressor(n_estimators=220, seed=seed)
                            m_ik.fit(ik[f_m4], ik["winner_kmh"])
                            in_tk[ite] = m_ik.predict(ite_df[f_m4])
                        if len(ig) >= 10:
                            m_ig = make_task_regressor(n_estimators=220, seed=seed)
                            m_ig.fit(ig[f_m4], ig["goal_share"])
                            in_tg[ite] = np.clip(m_ig.predict(ite_df[f_m4]), 0.0, 1.0)
                        ip_tr = p_tr[p_tr["task_key"].isin(set(itr_df["task_key"]))]
                        ip_te = p_tr[p_tr["task_key"].isin(set(ite_df["task_key"]))]
                        if len(ip_tr) >= 50 and len(ip_te):
                            ipb = PilotModelBundle(seed=seed).fit(ip_tr, itr_df, f_m4, group_col=split_col)
                            i_agg, _ = ipb.predict(ip_te, ite_df, f_m4)
                            in_pg[ite] = i_agg["pilot_goal_share"].to_numpy(dtype=np.float64)
                            in_pk[ite] = i_agg["pilot_winner_kmh"].to_numpy(dtype=np.float64)
                    w_g = _fit_convex_weight(tr_m4["goal_share"].to_numpy(float), in_tg, in_pg)
                    w_k = _fit_convex_weight(tr_m4["winner_kmh"].to_numpy(float), in_tk, in_pk)
                blend_weights_g.append(w_g)
                blend_weights_k.append(w_k)
                blend_g[te_i] = np.where(np.isfinite(m5_g[te_i]), w_g * m4_g[te_i] + (1.0 - w_g) * m5_g[te_i], m4_g[te_i])
                blend_k[te_i] = np.where(np.isfinite(m5_k[te_i]), w_k * m4_k[te_i] + (1.0 - w_k) * m5_k[te_i], m4_k[te_i])

    preds["M4_forecast_compat"] = (m4_fc_k, m4_fc_g)
    preds["M4_thermal_mc"] = (m4_k, m4_g)
    if has_pilots and np.isfinite(m5_g).any():
        preds["M5_pilot_agg"] = (m5_k, m5_g)
        preds["M5_blend"] = (blend_k, blend_g)

    oof_sub_df = pd.concat(oof_sub_rows).reindex(t_df["task_key"]) if oof_sub_rows else pd.DataFrame(index=t_df["task_key"])
    oof_pilot_df = pd.concat(oof_pilot_rows, ignore_index=True) if oof_pilot_rows else pd.DataFrame()
    meta = {
        "blend_weight_goal_task": round(float(np.median(blend_weights_g)), 3) if blend_weights_g else 1.0,
        "blend_weight_speed_task": round(float(np.median(blend_weights_k)), 3) if blend_weights_k else 1.0,
        "oof_sub_features": oof_sub_df,
    }
    return preds, oof_pilot_df, meta


def evaluate_stopped_model(
    db: Path,
    era_tables: dict[str, pd.DataFrame | None],
    include_land: bool = True,
    n_splits: int = 5,
    seed: int = 7,
) -> tuple[dict[str, Any], Any, list[str]]:
    """Auxiliary P(stopped) classifier on all non-training tasks using 1 h route weather."""
    t_all = build_all_tasks_for_stopped(db)
    if not len(t_all) or t_all["is_stopped"].nunique() < 2 or t_all["event_key"].nunique() < 3:
        return {"skipped": True}, None, []
    seg = RW.routes_segments(db, t_all)
    wx, _ = RW.extract_hourly_route_features(t_all, seg, era_tables, include_terrain=True, include_land=include_land)
    t_all = t_all.merge(wx, left_on="task_key", right_index=True, how="left")
    base = ["ss_km", "start_solar_h", "window_h", "day_length_h", "sun_decl_local", "abs_lat", "launched"]
    feats = [f for f in base + RW.HOURLY_CORE_FEATURES + RW.TERRAIN_ROUTE_FEATURES if f in t_all and t_all[f].notna().any()]
    y = t_all["is_stopped"].to_numpy(dtype=np.int8)
    oof = np.full(len(t_all), np.nan, dtype=np.float64)
    gkf = GroupKFold(n_splits=min(n_splits, t_all["event_key"].nunique()))
    for tr, te in gkf.split(t_all, groups=t_all["event_key"]):
        if len(np.unique(y[tr])) < 2:
            continue
        clf = make_binary_classifier(n_estimators=220, min_child=20, seed=seed)
        clf.fit(t_all.iloc[tr][feats], y[tr])
        oof[te] = clf.predict_proba(t_all.iloc[te][feats])[:, 1]
    ok = np.isfinite(oof)
    res: dict[str, Any] = {
        "tasks": int(len(t_all)),
        "stopped_count": int(y.sum()),
        "stopped_rate": round(float(y.mean()), 4),
    }
    if ok.any() and len(np.unique(y[ok])) == 2:
        p = np.clip(oof[ok], 1e-6, 1.0 - 1e-6)
        base_ll = float(log_loss(y[ok], np.full(ok.sum(), y[ok].mean())))
        ll = float(log_loss(y[ok], p))
        res.update({
            "auc": round(float(roc_auc_score(y[ok], p)), 4),
            "brier": round(float(brier_score_loss(y[ok], p)), 4),
            "log_loss_skill": round(1.0 - ll / max(base_ll, 1e-9), 4),
        })
    final_clf = make_binary_classifier(n_estimators=220, min_child=20, seed=seed)
    final_clf.fit(t_all[feats], y)
    return res, final_clf, feats


# ── Main CLI ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder")
    parser.add_argument("--out", type=Path, help="output folder (default DB/models/task_v2)")
    parser.add_argument("--folds", type=int, default=5, help="grouped CV folds (default 5)")
    parser.add_argument("--thermal-rows", type=int, default=DEFAULT_THERMAL_ROWS,
                        help=f"crossings sample for in-fold climb_prob sub-model (default {DEFAULT_THERMAL_ROWS}; 0 to skip)")
    parser.add_argument("--no-land", action="store_true", help="skip loading era5_land.parquet")
    parser.add_argument("--no-thermal-sub", action="store_true", help="skip all in-fold thermal sub-models")
    parser.add_argument("--no-stopped", action="store_true", help="skip auxiliary P(stopped) classifier")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    t0 = time.time()
    out = args.out or (args.db / "models" / "task_v2")
    models_dir = out / "models"
    out.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. Base tasks + routes
    t_df = build_base_task_table(args.db)
    seg_df = RW.routes_segments(args.db, t_df)
    log(f"Task v2: {len(t_df):,} tasks ({t_df['winner_kmh'].notna().sum():,} with winner's time), "
        f"{t_df['site_id'].nunique()} sites, {t_df['event_key'].nunique()} competitions")

    # 2. Load hourly ERA5 tables: keep single/levels/extra/static across all site chips so in-fold
    #    thermal sub-models have full weather coverage, while filtering large era5_land & cells to route only.
    chips_needed = set(seg_df["chip_id"].unique())
    cells_needed = set(seg_df["cell_id"].unique())
    wx_chips = None if not args.no_thermal_sub else chips_needed
    era_tables = RW.load_era5_hourly_tables(
        args.db, chips=wx_chips, cells_needed=cells_needed, include_land=not args.no_land, land_chips=chips_needed,
    )

    # 3. Extract task_v1 (5 h average) features on exact same tasks for apples-to-apples M1 bar.
    v1_wx = TM.weather(args.db, t_df)
    v1_ren = {c: f"v1_{c}" for c in TM.WEATHER if c in v1_wx}
    t_df = t_df.merge(v1_wx[list(v1_ren)].rename(columns=v1_ren), left_on="task_key", right_index=True, how="left")
    v1_feats = [f for f in v1_ren.values() if t_df[f].notna().any()]

    # 4. Extract hourly (1 h) along-route features + leader/pack segment panels.
    wx_v2, wx_diag = RW.extract_hourly_route_features(
        t_df, seg_df, era_tables, include_terrain=True, include_land=not args.no_land, return_panels=True,
    )
    t_df = t_df.merge(wx_v2, left_on="task_key", right_index=True, how="left")
    log(f"Hourly weather extracted: {wx_diag['n_chip_hours']:,} chip-hours, "
        f"era5_levels native 1h share = {wx_diag['levels_native_1h_share']:.1%}, "
        f"route coords = {t_df['has_route'].fillna(False).mean():.0%}")

    # 5. Load datasets for in-fold thermal sub-models + pilot-level table.
    if args.no_thermal_sub:
        sub_data: dict[str, pd.DataFrame] = {}
    else:
        sub_data = load_thermal_sub_datasets(args.db, era_tables, thermal_rows=args.thermal_rows, seed=args.seed)
        log("Thermal sub-model data: " + ", ".join(f"{k}={len(v):,}" for k, v in sub_data.items()) if sub_data else "Thermal sub-model data: none")

    pilot_df = build_pilot_table(args.db, t_df)
    log(f"Pilot-level table: {len(pilot_df):,} launched results across {pilot_df['task_key'].nunique():,} tasks")

    # 6. Run ablation ladder across both grouped CV splits (new competition & new site).
    ablation_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    all_metrics: dict[str, Any] = {
        "n_tasks": int(len(t_df)),
        "n_tasks_winner": int(t_df["winner_kmh"].notna().sum()),
        "n_sites": int(t_df["site_id"].nunique()),
        "n_events": int(t_df["event_key"].nunique()),
        "n_pilot_results": int(len(pilot_df)),
        "levels_native_1h_share": wx_diag["levels_native_1h_share"],
        "splits": {},
    }
    pred_out = t_df[["task_key", "event_key", "site_id", "date", "ss_km", "winner_h", "winner_kmh", "goal_share"]].copy()
    event_pilot_oof = pd.DataFrame()
    event_oof_sub = pd.DataFrame()
    event_meta: dict[str, Any] = {}

    for split_col, split_label in (("event_key", "new competition"), ("site_id", "new site")):
        log(f"\n=== Ablation CV ({args.folds}-fold grouped by {split_col}: {split_label}) ===")
        preds_s, pilot_oof, meta_s = run_ablation_cv(
            t_df, v1_feats, wx_diag["lead_panel"], wx_diag["pack_panel"], wx_diag["task_window_h"],
            sub_data, pilot_df, split_col=split_col, n_splits=args.folds, seed=args.seed,
        )
        split_metrics: dict[str, Any] = {"meta": {k: v for k, v in meta_s.items() if k != "oof_sub_features"}, "stages": {}}
        if split_col == "event_key":
            event_pilot_oof = pilot_oof
            event_oof_sub = meta_s["oof_sub_features"]
            event_meta = split_metrics["meta"]

        stage_order = list(preds_s.keys())
        v1_err_t = winner_time_err_min(t_df, preds_s["M1_task_v1_5h"][0])
        v1_err_g = goal_share_err_pts(t_df, preds_s["M1_task_v1_5h"][1])
        clusters = t_df[split_col].astype(str).to_numpy()

        for idx_s, sname in enumerate(stage_order):
            pk, pg = preds_s[sname]
            sc = score_predictions(t_df, pk, pg)
            split_metrics["stages"][sname] = sc
            ablation_rows.append({"split": split_label, "stage": sname, **sc})
            tag = "ev" if split_col == "event_key" else "site"
            if sname in ("M1_task_v1_5h", "M2_hourly_1h", "M4_forecast_compat", "M4_thermal_mc", "M5_pilot_agg", "M5_blend"):
                pred_out[f"{sname}_kmh_{tag}"] = pk
                pred_out[f"{sname}_goal_{tag}"] = pg

            log(
                f"  {sname:20s}  winner {sc.get('winner_time_mae_min', float('nan')):5.2f} min "
                f"(<=15m {sc.get('within_15_min', 0.0):5.1%}, <=30m {sc.get('within_30_min', 0.0):5.1%}, "
                f"{sc.get('winner_kmh_mae', float('nan')):.2f} km/h) | "
                f"goal {sc.get('goal_share_mae_pts', float('nan')):5.2f} pts "
                f"(<=10pts {sc.get('within_10_pts', 0.0):5.1%})"
            )

            # Paired bootstrap vs M1_task_v1_5h and vs previous stage.
            err_t = winner_time_err_min(t_df, pk)
            err_g = goal_share_err_pts(t_df, pg)
            if sname != "M1_task_v1_5h":
                bt = paired_bootstrap_diff(clusters, err_t, v1_err_t, n_iter=BOOTSTRAP_ITERS)
                bg = paired_bootstrap_diff(clusters, err_g, v1_err_g, n_iter=BOOTSTRAP_ITERS)
                boot_rows.append({
                    "split": split_label, "comparison": "vs_task_v1", "a": sname, "b": "M1_task_v1_5h",
                    "tasks": bt.get("tasks", 0), "clusters": bt.get("clusters", 0),
                    "time_diff_min": bt.get("mean_diff"), "time_ci95": bt.get("ci95"),
                    "time_significant": bt.get("significant"),
                    "goal_diff_pts": bg.get("mean_diff"), "goal_ci95": bg.get("ci95"),
                    "goal_significant": bg.get("significant"),
                })
            if idx_s >= 2:
                prev_name = stage_order[idx_s - 1]
                pt = paired_bootstrap_diff(clusters, err_t, winner_time_err_min(t_df, preds_s[prev_name][0]), n_iter=BOOTSTRAP_ITERS)
                pg_b = paired_bootstrap_diff(clusters, err_g, goal_share_err_pts(t_df, preds_s[prev_name][1]), n_iter=BOOTSTRAP_ITERS)
                boot_rows.append({
                    "split": split_label, "comparison": "step_gain", "a": sname, "b": prev_name,
                    "tasks": pt.get("tasks", 0), "clusters": pt.get("clusters", 0),
                    "time_diff_min": pt.get("mean_diff"), "time_ci95": pt.get("ci95"),
                    "time_significant": pt.get("significant"),
                    "goal_diff_pts": pg_b.get("mean_diff"), "goal_ci95": pg_b.get("ci95"),
                    "goal_significant": pg_b.get("significant"),
                })

        all_metrics["splits"][split_label] = split_metrics

    # 7. Auxiliary P(stopped) classifier.
    stopped_clf = None
    stopped_feats: list[str] = []
    if not args.no_stopped:
        st_res, stopped_clf, stopped_feats = evaluate_stopped_model(
            args.db, era_tables, include_land=not args.no_land, n_splits=args.folds, seed=args.seed,
        )
        all_metrics["stopped"] = st_res
        if not st_res.get("skipped"):
            log(f"\nAuxiliary P(stopped): {st_res['stopped_count']}/{st_res['tasks']} stopped "
                f"({st_res['stopped_rate']:.1%}), AUC {st_res.get('auc')}, Brier {st_res.get('brier')}")

    # 8. Fit final full-data models and export artifacts for Phase 2 API/forecast use.
    log("\nFitting final full-data Task v2 models...")
    full_sub = ThermalSubBundle(seed=args.seed).fit(sub_data)
    t_full = t_df.merge(event_oof_sub, left_on="task_key", right_index=True, how="left") if len(event_oof_sub) else t_df.copy()

    base_strict = [f for f in STRICT_BASE_FEATURES if f in t_full and t_full[f].notna().any()]
    fc_task_feats = base_strict + [
        f for f in RW.HOURLY_CORE_FEATURES + RW.TERRAIN_ROUTE_FEATURES + THERMAL_SUB_FEATURES
        if f in t_full and t_full[f].notna().any()
    ]
    full_task_feats = base_strict + [
        f for f in RW.HOURLY_CORE_FEATURES + RW.TERRAIN_ROUTE_FEATURES + RW.ERA5_LAND_FEATURES + THERMAL_SUB_FEATURES
        if f in t_full and t_full[f].notna().any()
    ]

    imp_frames = []
    tr_k = t_full.dropna(subset=["winner_kmh"])
    mk_full = make_task_regressor(seed=args.seed).fit(tr_k[full_task_feats], tr_k["winner_kmh"])
    _save_model_if_lgb(mk_full, models_dir / "winner_kmh.txt")
    imp_frames.append(_feature_importance(mk_full, full_task_feats, "winner_kmh"))

    tr_g = t_full.dropna(subset=["goal_share"])
    mg_full = make_task_regressor(seed=args.seed).fit(tr_g[full_task_feats], tr_g["goal_share"])
    _save_model_if_lgb(mg_full, models_dir / "goal_share.txt")
    imp_frames.append(_feature_importance(mg_full, full_task_feats, "goal_share"))

    # Forecast-ready models (strictly Open-Meteo compatible features).
    mk_fc = make_task_regressor(seed=args.seed).fit(tr_k[fc_task_feats], tr_k["winner_kmh"])
    _save_model_if_lgb(mk_fc, models_dir / "winner_kmh_forecast.txt")
    mg_fc = make_task_regressor(seed=args.seed).fit(tr_g[fc_task_feats], tr_g["goal_share"])
    _save_model_if_lgb(mg_fc, models_dir / "goal_share_forecast.txt")

    if full_sub.core60_model is not None:
        _save_model_if_lgb(full_sub.core60_model, models_dir / "sub_core60.txt")
        imp_frames.append(_feature_importance(full_sub.core60_model, full_sub.core60_feats, "sub_core60"))
    if full_sub.ceiling_model is not None:
        _save_model_if_lgb(full_sub.ceiling_model, models_dir / "sub_ceiling.txt")
        imp_frames.append(_feature_importance(full_sub.ceiling_model, full_sub.ceiling_feats, "sub_ceiling"))
    if full_sub.climb_model is not None:
        _save_model_if_lgb(full_sub.climb_model, models_dir / "sub_climb_prob.txt")

    full_pilot = PilotModelBundle(seed=args.seed).fit(pilot_df, t_full, full_task_feats, group_col="event_key") if len(pilot_df) >= 50 else None
    if full_pilot is not None:
        if full_pilot.goal_model is not None:
            _save_model_if_lgb(full_pilot.goal_model, models_dir / "pilot_goal.txt")
            imp_frames.append(_feature_importance(full_pilot.goal_model, full_pilot.feat_cols, "pilot_goal"))
        if full_pilot.speed_model is not None:
            _save_model_if_lgb(full_pilot.speed_model, models_dir / "pilot_speed.txt")
            imp_frames.append(_feature_importance(full_pilot.speed_model, full_pilot.feat_cols, "pilot_speed"))

    if stopped_clf is not None:
        _save_model_if_lgb(stopped_clf, models_dir / "stopped.txt")
        imp_frames.append(_feature_importance(stopped_clf, stopped_feats, "stopped"))

    # Write outputs.
    save_feat_cols = ["task_key"] + [c for c in full_task_feats if c in t_full]
    t_full[save_feat_cols].to_parquet(out / "features_task.parquet", index=False)
    pred_out.to_parquet(out / "predictions.parquet", index=False)
    if len(event_pilot_oof):
        event_pilot_oof.to_parquet(out / "pilot_predictions.parquet", index=False)
    pd.DataFrame(ablation_rows).to_csv(out / "ablation.csv", index=False)
    pd.DataFrame(boot_rows).to_csv(out / "paired_bootstrap.csv", index=False)
    if imp_frames:
        imp_all = pd.concat(imp_frames, ignore_index=True)
        imp_all.to_csv(out / "importance.csv", index=False)
        log("\nTop features by target:")
        for tgt in ("winner_kmh", "goal_share", "pilot_goal"):
            sub_i = imp_all[imp_all["target"] == tgt].head(10)
            if len(sub_i):
                log(f"  {tgt:12s}: " + ", ".join(f"{r.feature} {r.share:.0%}" for r in sub_i.itertuples()))

    all_metrics["paired_bootstrap"] = boot_rows
    all_metrics["runtime_s"] = round(time.time() - t0, 1)
    (out / "metrics.json").write_text(json.dumps(all_metrics, indent=2, default=str), encoding="utf-8")
    (models_dir / "meta.json").write_text(json.dumps({
        "full_task_features": full_task_feats,
        "forecast_task_features": fc_task_feats,
        "pilot_features": full_pilot.feat_cols if full_pilot is not None else [],
        "sub_core60_features": full_sub.core60_feats,
        "sub_ceiling_features": full_sub.ceiling_feats,
        "sub_climb_features": full_sub.climb_feats,
        "stopped_features": stopped_feats,
        "blend_weights": event_meta,
    }, indent=2), encoding="utf-8")
    log(f"\nSaved Task v2 outputs to {out} in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
