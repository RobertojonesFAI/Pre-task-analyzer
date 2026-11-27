"""Automated tests for Task Model v2, hourly route weather, ARCO-ERA5 1 h resume,
anti-leakage invariants, chip-hour Quantile Mapping, and forecast verification.

Runs completely locally without Google Drive using synthetic mini-databases.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from pta import grid, route_weather as RW, terrain  # noqa: E402
import era5_arco  # noqa: E402
import forecast_multivar_eval as FME  # noqa: E402
import task_model_v2 as TM2  # noqa: E402


# ── 1. Space-Time Route Geometry & Out-and-Return Wind Shift ─────────────────

def test_out_and_return_wind_shift_space_time_coupling():
    """An out-and-return task where the wind flips between outbound (h0-h1) and return (h2-h3)
    must register tailwind on BOTH legs (unlike a 5 h average that cancels out)."""
    # Launch at (45.0, 6.0), turnpoint ~35 km north at (45.315, 6.0), return to launch (70 km total).
    seg = RW.sample_route_segments([45.0, 45.315, 45.0], [6.0, 6.0, 6.0], step_km=5.0)
    assert len(seg) >= 14
    assert seg["leg_idx"].nunique() == 2
    assert seg["cum_km"].is_monotonic_increasing
    seg.insert(0, "task_key", "t1")
    seg["has_route"] = True

    tasks = pd.DataFrame([{
        "task_key": "t1", "event_key": "e1", "site_id": "s1",
        "date": pd.Timestamp("2024-07-01"), "lat": 45.0, "lon": 6.0,
        "utc_offset_h": 2.0, "start_time": "14:00:00", "deadline": "18:00:00",
        "ss_km": 70.0,
    }])
    chips = sorted(seg["chip_id"].unique())
    hours = pd.date_range("2024-07-01 12:00", "2024-07-01 16:00", freq="h")

    # Wind blows North (+8 m/s) at 12:00-13:00 UTC (outbound leg), then South (-8 m/s) at 14:00-16:00 UTC (return leg).
    rows = []
    for cid in chips:
        for h in hours:
            v_wind = 8.0 if h.hour <= 13 else -8.0
            rows.append({
                "chip_id": cid, "time": h,
                "blh_m": 1800.0, "cloud_base_m": 2200.0, "cape": 200.0, "cin": 10.0,
                "sensible_heat_wm2": 220.0, "latent_heat_wm2": 80.0, "sunshine_wm2": 700.0,
                "cloud_total": 0.2, "cloud_low": 0.1, "rain_mmh": 0.0, "t2m_c": 24.0, "dewpoint_c": 10.0,
                "u10": 0.0, "v10": v_wind * 0.6, "u100": 0.0, "v100": v_wind, "gust10": abs(v_wind) + 2.0,
            })
    single = pd.DataFrame(rows)
    lv_rows = []
    for cid in chips:
        for h in hours:
            v_wind = 8.0 if h.hour <= 13 else -8.0
            lv_rows.append({
                "chip_id": cid, "time": h,
                "u_925": 0.0, "v_925": v_wind, "z_925": 800.0, "t_925": 20.0,
                "u_850": 0.0, "v_850": v_wind, "z_850": 1500.0, "t_850": 15.0, "rh_850": 50.0,
                "u_700": 0.0, "v_700": v_wind, "z_700": 3100.0, "t_700": 4.0, "rh_700": 40.0,
                "u_600": 0.0, "v_600": v_wind, "z_600": 4300.0, "t_600": -4.0,
                "u_500": 0.0, "v_500": v_wind, "z_500": 5600.0, "t_500": -13.0,
            })
    tables = {
        "single": single,
        "levels": pd.DataFrame(lv_rows),
        "extra": None,
        "static": pd.DataFrame({"chip_id": chips, "ground_m": 400.0}),
        "elev": pd.DataFrame({"chip_id": chips, "route_elev": 400.0}),
        "cells": None,
        "land": None,
    }
    wx, diag = RW.extract_hourly_route_features(tasks, seg, tables, include_terrain=False, include_land=False)
    assert diag["levels_native_1h_share"] == 1.0
    # Leader flies at 28 km/h: outbound ~0-1.25 h (12-13 UTC, +8 m/s North -> +8 m/s tailwind),
    # return ~1.25-2.5 h (2/8 segments round to 13 UTC [-8 m/s], 6/8 round to 14 UTC [+8 m/s] -> +4.0 m/s exact).
    assert wx.loc["t1", "tail_lead_mean"] == pytest.approx(6.0, abs=0.2)
    assert wx.loc["t1", "tail_lead_last_leg"] == pytest.approx(4.0, abs=0.2)

    # Now flip to constant North wind (+8 m/s all day): outbound is +8 m/s tailwind, return leg is -8 m/s headwind.
    tables["single"]["v100"] = 8.0
    for p in (925, 850, 700, 600, 500):
        tables["levels"][f"v_{p}"] = 8.0
    wx2, _ = RW.extract_hourly_route_features(tasks, seg, tables, include_terrain=False, include_land=False)
    assert wx2.loc["t1", "tail_lead_last_leg"] < -7.0
    assert wx2.loc["t1", "tail_lead_min_leg"] < -7.0
    assert wx2.loc["t1", "headwind_share_lead"] == pytest.approx(0.5, abs=0.05)


# ── 2. Vectorized Vertical Wind Interpolation ────────────────────────────────

def test_vertical_wind_interpolation_matches_np_interp():
    rng = np.random.default_rng(123)
    n = 120
    g = rng.uniform(100.0, 1800.0, size=n)
    df = pd.DataFrame({
        "ground_m": g,
        "u10": rng.normal(0, 3, n), "v10": rng.normal(0, 3, n),
        "u100": rng.normal(1, 4, n), "v100": rng.normal(1, 4, n),
        "z_925": rng.uniform(600, 900, n), "u_925": rng.normal(2, 4, n), "v_925": rng.normal(2, 4, n),
        "z_850": rng.uniform(1350, 1650, n), "u_850": rng.normal(3, 5, n), "v_850": rng.normal(3, 5, n),
        "z_700": rng.uniform(2900, 3200, n), "u_700": rng.normal(5, 6, n), "v_700": rng.normal(5, 6, n),
        "z_600": rng.uniform(4100, 4400, n), "u_600": rng.normal(7, 6, n), "v_600": rng.normal(7, 6, n),
        "z_500": rng.uniform(5400, 5800, n), "u_500": rng.normal(9, 7, n), "v_500": rng.normal(9, 7, n),
    })
    # Inject NaNs and out-of-range targets to test every edge case.
    df.loc[5, "u_850"] = np.nan
    df.loc[10, ["u10", "u100", "u_925", "u_850", "u_700", "u_600"]] = np.nan
    target = rng.uniform(50.0, 6200.0, size=n)
    target[20] = np.nan

    mu, mv = RW.interp_wind_to_height(df, target)

    # Naive row-by-row reference matching wind_check.py.
    ref_u, ref_v = np.full(n, np.nan), np.full(n, np.nan)
    for i in range(n):
        if not np.isfinite(target[i]):
            continue
        hs = [g[i] + 10.0, g[i] + 100.0]
        us = [df.at[i, "u10"], df.at[i, "u100"]]
        vs = [df.at[i, "v10"], df.at[i, "v100"]]
        for p in RW.PRESSURE_LEVELS:
            zp = df.at[i, f"z_{p}"]
            if np.isfinite(zp) and zp > g[i] + 150.0:
                hs.append(zp)
                us.append(df.at[i, f"u_{p}"])
                vs.append(df.at[i, f"v_{p}"])
        hs_a, us_a, vs_a = np.array(hs), np.array(us), np.array(vs)
        ok = np.isfinite(hs_a) & np.isfinite(us_a) & np.isfinite(vs_a)
        if ok.sum() < 2:
            continue
        order = np.argsort(hs_a[ok])
        ref_u[i] = np.interp(target[i], hs_a[ok][order], us_a[ok][order])
        ref_v[i] = np.interp(target[i], hs_a[ok][order], vs_a[ok][order])

    np.testing.assert_allclose(mu, ref_u, rtol=1e-10, atol=1e-10, equal_nan=True)
    np.testing.assert_allclose(mv, ref_v, rtol=1e-10, atol=1e-10, equal_nan=True)


# ── 3. Exact 1 h Levels Alignment + 3 h Linear Interpolation Fallback ────────

def test_align_levels_hourly_exact_and_3h_fallback():
    need = pd.DataFrame({
        "chip_id": [100] * 8,
        "time": list(pd.date_range("2024-07-01 09:00", "2024-07-01 15:00", freq="h")) + [pd.Timestamp("2024-07-02 00:00")],
    })
    # Case A: 3-hourly source (09:00, 12:00, 15:00) + next day 09:00 (gap = 18 h > 3.5 h).
    lv3 = pd.DataFrame({
        "chip_id": [100, 100, 100, 100],
        "time": pd.to_datetime(["2024-07-01 09:00", "2024-07-01 12:00", "2024-07-01 15:00", "2024-07-02 09:00"]),
        "u_700": [0.0, 6.0, 12.0, 30.0],
        "t_700": [10.0, 13.0, 16.0, 20.0],
    })
    aligned, share = RW.align_levels_hourly(need, lv3)
    assert share == pytest.approx(3.0 / 8.0)
    vals = aligned.set_index("time")["u_700"]
    assert vals.loc[pd.Timestamp("2024-07-01 10:00")] == pytest.approx(2.0)
    assert vals.loc[pd.Timestamp("2024-07-01 11:00")] == pytest.approx(4.0)
    assert vals.loc[pd.Timestamp("2024-07-01 14:00")] == pytest.approx(10.0)
    # Cross-day gap (15:00 -> next day 09:00 = 18 h) must NOT be interpolated.
    assert np.isnan(vals.loc[pd.Timestamp("2024-07-02 00:00")])

    # Case B: Complete 1 h source -> 100% native.
    lv1 = pd.DataFrame({
        "chip_id": [100] * 8,
        "time": need["time"],
        "u_700": np.arange(8, dtype=float),
    })
    aligned1, share1 = RW.align_levels_hourly(need, lv1)
    assert share1 == 1.0
    np.testing.assert_allclose(aligned1["u_700"].to_numpy(), np.arange(8, dtype=float))


# ── 4. MacCready Kinematics Physics ──────────────────────────────────────────

def test_maccready_kinematics_physics():
    climbs = np.array([0.8, 1.5, 3.0])
    ccc = RW.maccready_air_xc_kmh(climbs, "CCC")
    end = RW.maccready_air_xc_kmh(climbs, "EN-D")
    enc = RW.maccready_air_xc_kmh(climbs, "EN-C")
    assert np.all(np.diff(ccc) > 0)
    assert np.all(ccc > end) and np.all(end > enc)

    # Strong headwind makes weak EN-C unflyable while strong tailwind boosts speed.
    v_hw, unfly_hw = RW.maccready_segment_groundspeed_kmh(np.array([0.8]), np.array([-8.0]), np.array([0.0]), "EN-C")
    v_tw, unfly_tw = RW.maccready_segment_groundspeed_kmh(np.array([1.5]), np.array([5.0]), np.array([0.0]), "CCC")
    assert bool(unfly_hw[0]) is True
    assert bool(unfly_tw[0]) is False
    assert v_tw[0] > 40.0


# ── 5. Hourly Chip-Level QM Monotonicity & Wind Direction Preservation ───────

def test_hourly_qm_reduces_bias_and_preserves_wind_direction():
    rng = np.random.default_rng(7)
    n = 300
    times = pd.date_range("2024-06-01 00:00", periods=n, freq="h")
    chips = np.full(n, 42, dtype=np.int64)
    era_blh = rng.uniform(500.0, 2500.0, n)
    ang = rng.uniform(-np.pi, np.pi, n)
    era_spd = rng.uniform(1.0, 12.0, n)
    era_u, era_v = era_spd * np.cos(ang), era_spd * np.sin(ang)

    # Forecast has +400 m BLH bias and 1.35x wind speed bias, same direction.
    fc_blh = era_blh + 400.0
    fc_u, fc_v = (era_spd * 1.35) * np.cos(ang), (era_spd * 1.35) * np.sin(ang)

    era_tbls = {
        "single": pd.DataFrame({"chip_id": chips, "time": times, "blh_m": era_blh, "u100": era_u, "v100": era_v}),
        "levels": pd.DataFrame({"chip_id": chips, "time": times, "u_700": era_u, "v_700": era_v}),
        "extra": None,
    }
    fc_tbls = {
        "single": pd.DataFrame({"chip_id": chips, "time": times, "blh_m": fc_blh, "u100": fc_u, "v100": fc_v}),
        "levels": pd.DataFrame({"chip_id": chips, "time3": times, "u_700": fc_u, "v_700": fc_v}),
        "extra": None,
    }
    keys = pd.DataFrame({"chip_id": chips, "time": times})
    qmaps = RW.fit_hourly_qmaps(fc_tbls, era_tbls, keys, min_pairs=30)
    assert "single.blh_m" in qmaps
    assert "single.wind100" in qmaps
    assert "levels.wind_700" in qmaps

    mapped = RW.apply_hourly_qmaps(fc_tbls, qmaps, include_wind=True)
    blh_bias_before = abs(float(np.mean(fc_blh - era_blh)))
    blh_bias_after = abs(float(np.mean(mapped["single"]["blh_m"] - era_blh)))
    assert blh_bias_after < 0.1 * blh_bias_before

    spd_after = np.hypot(mapped["single"]["u100"], mapped["single"]["v100"])
    assert abs(float(np.mean(spd_after - era_spd))) < 0.2

    # Wind direction must be preserved exactly.
    ang_before = np.arctan2(fc_v, fc_u)
    ang_after = np.arctan2(mapped["single"]["v100"].to_numpy(), mapped["single"]["u100"].to_numpy())
    ang_diff = np.abs((ang_after - ang_before + np.pi) % (2 * np.pi) - np.pi)
    assert float(np.max(ang_diff)) < 1e-6


# ── 6. ARCO-ERA5 1 h Incremental Missing & Chunking ──────────────────────────

def test_era5_arco_1h_incremental_and_chunk_checkpoint(tmp_path: Path):
    month_path = tmp_path / "2024-07.parquet"
    t3 = pd.to_datetime(["2024-07-01 06:00", "2024-07-01 09:00", "2024-07-01 12:00"])
    existing = pd.DataFrame({"chip_id": [10, 10, 10], "time": t3, "u_700": [1.0, 2.0, 3.0]})
    existing.to_parquet(month_path, index=False)

    wanted = {pd.Timestamp(t): np.array([10]) for t in pd.date_range("2024-07-01 06:00", "2024-07-01 12:00", freq="h")}
    miss = era5_arco.missing(month_path, wanted)
    assert sorted(miss.keys()) == list(pd.to_datetime([
        "2024-07-01 07:00", "2024-07-01 08:00", "2024-07-01 10:00", "2024-07-01 11:00",
    ]))

    chunks = era5_arco._hour_chunks(miss, chunk_hours=2)
    assert [len(c) for c in chunks] == [2, 2]
    for ch in chunks:
        new_df = pd.DataFrame({"chip_id": [10] * len(ch), "time": list(ch.keys()), "u_700": [5.0] * len(ch)})
        era5_arco._merge_save_month(month_path, new_df)
    assert era5_arco.missing(month_path, wanted) == {}
    final = pd.read_parquet(month_path)
    assert len(final) == 7


# ── 7. Synthetic Mini-DB Builder & Full End-to-End + Anti-Leakage Tests ──────

def _build_synthetic_db(db: Path, perturb_event: str | None = None) -> None:
    """Create a small, realistic self-consistent competition database for end-to-end testing."""
    rng = np.random.default_rng(42)
    db.mkdir(parents=True, exist_ok=True)
    (db / "weather" / "hist_forecast").mkdir(parents=True, exist_ok=True)
    (db / "weather" / "hist_forecast_thermo").mkdir(parents=True, exist_ok=True)
    (db / "truth").mkdir(parents=True, exist_ok=True)

    sites_spec = [
        ("s1", "SiteA", "FR", 45.0, 6.0),
        ("s2", "SiteB", "ES", 40.5, -5.0),
        ("s3", "SiteC", "CO", 4.5, -76.0),
        ("s4", "SiteD", "US", 43.5, -114.5),
    ]
    sites_df = pd.DataFrame([
        {"site_id": s, "name": n, "country": c, "lat": la, "lon": lo}
        for s, n, c, la, lo in sites_spec
    ])
    sites_df.to_parquet(db / "sites.parquet", index=False)

    # 8 competitions (2 per site) across 2021-2024, 3 tasks per competition = 24 tasks (+ 4 stopped tasks).
    ev_rows, ev_site_rows, task_rows, tp_rows, res_rows, rat_rows, gl_rows, tf_rows = [], [], [], [], [], [], [], []
    sc_rows = []
    pilots = [f"p{i:02d}" for i in range(25)]
    pilot_base_r = {p: 1350.0 + i * 25.0 for i, p in enumerate(pilots)}
    pilot_class = {p: ("CCC" if i >= 16 else "EN-D" if i >= 8 else "EN-C") for i, p in enumerate(pilots)}
    pilot_harness = {p: ("submarine" if i >= 18 else "genierace5" if i >= 10 else "") for i, p in enumerate(pilots)}

    res_id = 1
    all_chips = set()
    all_cells = set()
    task_weather_truth = {}

    for ev_idx in range(8):
        s_id, _, _, s_lat, s_lon = sites_spec[ev_idx % 4]
        ev_key = f"ev:{ev_idx}"
        year = 2021 if ev_idx < 2 else 2022 + (ev_idx - 2) // 2
        base_date = pd.Timestamp(f"{year}-06-{1 + ev_idx * 2:02d}")
        ev_rows.append({"event_key": ev_key, "name": f"Comp {ev_idx}", "country": "XX"})
        ev_site_rows.append({"event_key": ev_key, "site_id": s_id})

        for day_i in range(4):  # 3 valid + 1 stopped on day 3
            tk = f"{ev_key}-t{day_i}"
            t_date = base_date + pd.Timedelta(days=day_i)
            is_stopped = day_i == 3
            ss_km = float(rng.uniform(45.0, 85.0))
            lat_tp = s_lat + ss_km / 222.6
            pts_la = [s_lat, lat_tp, s_lat]
            pts_lo = [s_lon, s_lon + 0.15, s_lon]
            seg = RW.sample_route_segments(pts_la, pts_lo, step_km=5.0)
            all_chips.update(seg["chip_id"].unique().tolist())
            all_cells.update(seg["cell_id"].unique().tolist())
            for c_id in seg["chip_id"].unique():
                la_c, lo_c = grid.chip_centre(int(c_id))
                sc_rows.append({"site_id": s_id, "chip_id": int(c_id), "era5_lat": la_c, "era5_lon": lo_c, "in_area": True})

            for seq_i, (la_p, lo_p) in enumerate(zip(pts_la, pts_lo)):
                tp_rows.append({"task_key": tk, "seq": seq_i, "lat": la_p, "lon": lo_p, "radius_m": 1000.0, "type": ""})

            blh_day = float(rng.uniform(900.0, 2600.0))
            tail_day = float(rng.uniform(-4.0, 4.0))
            rain_day = 2.5 if is_stopped else (0.0 if rng.random() > 0.2 else 0.5)
            task_weather_truth[tk] = (t_date, s_id, ev_key, seg["chip_id"].unique().tolist(), blh_day, tail_day, rain_day, s_lon)

            true_win_kmh = float(np.clip(20.0 + 0.005 * blh_day + 1.2 * tail_day + rng.normal(0, 1.0), 15.0, 48.0))
            if perturb_event is not None and ev_key == perturb_event:
                true_win_kmh = 49.0  # extreme perturbation on held-out event

            task_rows.append({
                "task_key": tk, "event_key": ev_key, "date": t_date.strftime("%Y-%m-%d"),
                "name": f"Task {day_i}", "task_type": "race", "training": False, "stopped": is_stopped,
                "distance_km": ss_km + 5.0, "ss_distance_km": ss_km,
                "start_time": "12:00:00", "deadline": "16:00:00", "utc_offset_h": round(s_lon / 15.0),
                "turnpoints": json.dumps([[la_p, lo_p] for la_p, lo_p in zip(pts_la, pts_lo)]),
            })

            # 20 launched pilots per task
            chosen = pilots[:20]
            rs = np.array([pilot_base_r[p] for p in chosen])
            tf_rows.append({
                "task_key": tk, "n_rated": 20, "field_top10": float(np.sort(rs)[-10:].mean()),
                "field_top30": float(rs.mean()), "field_mean": float(rs.mean()),
            })
            for p_idx, pk in enumerate(chosen):
                r_p = pilot_base_r[pk]
                p_goal_prob = 1.0 / (1.0 + np.exp(-(-2.5 + 0.0018 * blh_day + 0.002 * (r_p - 1500.0) + 0.15 * tail_day - 1.5 * rain_day)))
                in_g = int(rng.random() < p_goal_prob)
                if perturb_event is not None and ev_key == perturb_event:
                    in_g = 1 - in_g  # flip every pilot's label on perturbed event
                if p_idx == 19 and not is_stopped:
                    in_g = 1  # ensure at least one finisher for winner time
                speed_p = true_win_kmh * (0.85 + 0.15 * (r_p - 1350.0) / 600.0)
                time_s = (ss_km / max(speed_p, 8.0)) * 3600.0 if in_g else np.nan
                res_rows.append({
                    "result_id": res_id, "task_key": tk, "pilot_key": pk,
                    "status": "goal" if in_g else "lo", "time_s": time_s,
                    "distance_km": ss_km if in_g else ss_km * 0.6,
                })
                rat_rows.append({
                    "task_key": tk, "pilot_key": pk, "rating_before": r_p, "tasks_before": ev_idx * 3,
                })
                gl_rows.append({
                    "result_id": res_id, "glider_class": pilot_class[pk], "harness_key": pilot_harness[pk],
                })
                res_id += 1

    pd.DataFrame(ev_rows).to_parquet(db / "events.parquet", index=False)
    pd.DataFrame(ev_site_rows).to_parquet(db / "event_sites.parquet", index=False)
    pd.DataFrame(task_rows).to_parquet(db / "tasks.parquet", index=False)
    pd.DataFrame(tp_rows).to_parquet(db / "turnpoints.parquet", index=False)
    pd.DataFrame(res_rows).to_parquet(db / "results.parquet", index=False)
    pd.DataFrame(rat_rows).to_parquet(db / "ratings.parquet", index=False)
    pd.DataFrame(gl_rows).to_parquet(db / "result_gliders.parquet", index=False)
    pd.DataFrame(tf_rows).to_parquet(db / "task_fields.parquet", index=False)
    pd.DataFrame(sc_rows).drop_duplicates(["site_id", "chip_id"]).to_parquet(db / "site_chips.parquet", index=False)

    # Cells table (1 km terrain + sun columns).
    cell_rows = []
    for cid in sorted(all_cells):
        la, lo = grid.cell_centre(int(cid))
        row = {
            "cell_id": int(cid), "chip_id": int(grid.chip_of_cell(int(cid))), "lat": float(la), "lon": float(lo),
            "elev_mean": 500.0, "elev_min": 420.0, "elev_max": 640.0, "elev_std": 55.0,
            "slope_mean": 14.0, "slope_p90": 26.0, "steep_share": 0.15,
            "facing_east": 0.1, "facing_north": -0.2, "curvature": -0.001,
            "ridge_share": 0.2, "valley_share": 0.15, "tpi_5km": 40.0, "tpi_20km": 90.0,
            "relief_5km": 450.0, "flat_share": 0.25,
        }
        for k in (1, 2, 3):
            for asp in terrain.ASPECTS:
                row[f"sun_{asp}_{k}"] = 0.75 / 24.0
        cell_rows.append(row)
    pd.DataFrame(cell_rows).to_parquet(db / "cells.parquet", index=False)

    # Weather tables (1 h single, 1 h levels, extra, static, land) + truth tables + 2 forecast models.
    s_rows, lv_rows, ex_rows, ld_rows = [], [], [], []
    cl_rows, ce_rows, cx_rows = [], [], []
    fc_w_rows = {"ecmwf_ifs": [], "gfs": []}
    fc_t_rows = {"ecmwf_ifs": [], "gfs": []}

    for tk, (t_date, s_id, ev_key, chips, blh_day, tail_day, rain_day, s_lon) in task_weather_truth.items():
        off = round(s_lon / 15.0)
        start_utc = t_date + pd.Timedelta(hours=12.0 - off)
        hrs = pd.date_range(start_utc.floor("h") - pd.Timedelta(hours=1), start_utc.floor("h") + pd.Timedelta(hours=5), freq="h")
        prev_hrs = [h - pd.Timedelta(days=1) for h in hrs]
        for cid in chips:
            la_c, lo_c = grid.chip_centre(int(cid))
            for h in list(hrs) + prev_hrs:
                rel = (h - start_utc).total_seconds() / 3600.0
                blh_h = max(300.0, blh_day * (0.85 + 0.08 * min(max(rel, 0.0), 3.0)))
                s_rows.append({
                    "chip_id": int(cid), "time": h, "blh_m": blh_h, "cloud_base_m": blh_h + 300.0,
                    "cape": 250.0 + 100.0 * rain_day, "cin": 15.0, "sensible_heat_wm2": 200.0,
                    "latent_heat_wm2": 70.0, "sunshine_wm2": 650.0, "cloud_total": 0.25 + 0.15 * rain_day,
                    "cloud_low": 0.15, "rain_mmh": rain_day if rel >= 2.0 else 0.0,
                    "t2m_c": 24.0, "dewpoint_c": 11.0,
                    "u10": 0.5, "v10": tail_day * 0.5, "u100": 0.8, "v100": tail_day * 0.8, "gust10": abs(tail_day) + 3.0,
                })
                lv_rows.append({
                    "chip_id": int(cid), "time": h,
                    "u_925": 0.8, "v_925": tail_day, "z_925": 820.0, "t_925": 20.0,
                    "u_850": 1.0, "v_850": tail_day, "z_850": 1520.0, "t_850": 14.5, "rh_850": 55.0,
                    "u_700": 1.2, "v_700": tail_day, "z_700": 3100.0, "t_700": 3.5, "rh_700": 45.0,
                    "u_600": 1.5, "v_600": tail_day, "z_600": 4320.0, "t_600": -4.5,
                    "u_500": 2.0, "v_500": tail_day, "z_500": 5650.0, "t_500": -13.5,
                })
                ex_rows.append({
                    "chip_id": int(cid), "time": h, "skin_temp_c": 31.0, "soil_moisture": 0.22,
                    "freezing_level_m": 3200.0, "albedo": 0.18, "snow_depth_m": 0.0, "conv_rain_mmh": 0.0,
                })
                ld_rows.append({
                    "chip_id": int(cid), "time": h, "lat": round(la_c, 1), "lon": round(lo_c, 1),
                    "sensible_heat_wm2": 215.0, "sunshine_wm2": 660.0, "skin_temp_c": 31.5,
                    "soil_moisture": 0.21, "albedo": 0.17, "snow_cover": 0.0,
                })
                if h in hrs and h.year >= 2022:
                    for mid, bias in (("ecmwf_ifs", 120.0), ("gfs", 320.0)):
                        fc_w_rows[mid].append({
                            "chip_id": int(cid), "hour": h,
                            "u10": 0.5, "v10": tail_day * 0.55, "u100": 0.8, "v100": tail_day * 0.85,
                            "u_925": 0.8, "v_925": tail_day * 1.05, "z_925": 820.0,
                            "u_850": 1.0, "v_850": tail_day * 1.1, "z_850": 1520.0,
                            "u_700": 1.2, "v_700": tail_day * 1.1, "z_700": 3100.0,
                            "u_600": 1.5, "v_600": tail_day * 1.1, "z_600": 4320.0,
                            "u_500": 2.0, "v_500": tail_day * 1.1, "z_500": 5650.0,
                        })
                        fc_t_rows[mid].append({
                            "chip_id": int(cid), "hour": h, "om_elev": 400.0,
                            "blh_native_m": blh_h + bias, "t2m_c": 24.5, "dewpoint_c": 11.0,
                            "sunshine_wm2": 670.0, "cloud_total": 0.25, "cloud_low": 0.15,
                            "rain_mmh": rain_day if rel >= 2.0 else 0.0, "cape": 260.0, "cin": 15.0,
                            "gust10": abs(tail_day) + 3.0, "skin_temp_c": 31.0, "soil_moisture": 0.22,
                            "freezing_level_m": 3200.0,
                            "t_925": 20.0, "z_925": 820.0, "t_850": 14.5, "z_850": 1520.0, "rh_850": 55.0,
                            "t_700": 3.5, "z_700": 3100.0, "rh_700": 45.0, "t_600": -4.5, "z_600": 4320.0,
                        })
            # Truth ceiling & climb groups for sub-model training.
            for h in hrs[1:4]:
                cl_rows.append({
                    "site_id": s_id, "chip_id": int(cid), "hour": h, "event_key": ev_key,
                    "core60_p75_ms": float(np.clip(0.8 + 0.0012 * blh_day + rng.normal(0, 0.2), 0.5, 5.0)),
                    "solar_hour": 13.0,
                })
                ce_rows.append({
                    "site_id": s_id, "chip_id": int(cid), "hour": h, "event_key": ev_key,
                    "ceiling_p90_asl": float(400.0 + blh_day * 0.88 + rng.normal(0, 60.0)),
                    "airspace_cap_flag": False, "solar_hour": 13.0,
                })

        # Synthetic crossings for sub_climb_prob.
        c_list = sorted(all_cells)[:4]
        for k_cx in range(15):
            cx_rows.append({
                "track_id": f"{tk}-tr{k_cx}", "task_key": tk, "cell_id": int(c_list[k_cx % len(c_list)]),
                "t_in": int((13.0 - off) * 3600.0), "duration_s": 60.0,
                "agl_in": float(rng.uniform(200.0, 900.0)),
                "climbing_s": 45.0 if rng.random() < 0.35 else 0.0,
            })

    pd.DataFrame(s_rows).drop_duplicates(["chip_id", "time"]).to_parquet(db / "weather" / "era5_single.parquet", index=False)
    pd.DataFrame(lv_rows).drop_duplicates(["chip_id", "time"]).to_parquet(db / "weather" / "era5_levels.parquet", index=False)
    pd.DataFrame(ex_rows).drop_duplicates(["chip_id", "time"]).to_parquet(db / "weather" / "era5_extra.parquet", index=False)
    pd.DataFrame(ld_rows).drop_duplicates(["lat", "lon", "time"]).to_parquet(db / "weather" / "era5_land.parquet", index=False)
    pd.DataFrame({"chip_id": sorted(all_chips), "ground_m": 400.0}).to_parquet(db / "weather" / "era5_static.parquet", index=False)
    pd.DataFrame(cl_rows).to_parquet(db / "truth" / "truth_climb_groups.parquet", index=False)
    pd.DataFrame(ce_rows).to_parquet(db / "truth" / "truth_ceiling.parquet", index=False)
    pd.DataFrame(cx_rows).to_parquet(db / "crossings.parquet", index=False)
    pd.DataFrame({"track_id": [r["track_id"] for r in cx_rows], "is_drive": False}).to_parquet(db / "flight_checks.parquet", index=False)

    for mid in ("ecmwf_ifs", "gfs"):
        pd.DataFrame(fc_w_rows[mid]).drop_duplicates(["chip_id", "hour"]).to_parquet(db / "weather" / "hist_forecast" / f"{mid}.parquet", index=False)
        pd.DataFrame(fc_t_rows[mid]).drop_duplicates(["chip_id", "hour"]).to_parquet(db / "weather" / "hist_forecast_thermo" / f"{mid}.parquet", index=False)


def test_strict_history_zero_same_event_leakage(tmp_path: Path):
    """Perturbing outcomes of a held-out competition must NOT change any pre-task feature
    (site_prev_*_strict or pilot_prev_*) for tasks in that competition."""
    db_a = tmp_path / "db_a"
    db_b = tmp_path / "db_b"
    _build_synthetic_db(db_a, perturb_event=None)
    _build_synthetic_db(db_b, perturb_event="ev:5")

    ta = TM2.build_base_task_table(db_a)
    tb = TM2.build_base_task_table(db_b)
    sa = ta[ta["event_key"] == "ev:5"].sort_values("task_key").reset_index(drop=True)
    sb = tb[tb["event_key"] == "ev:5"].sort_values("task_key").reset_index(drop=True)
    for col in ("site_prev_winner_kmh_strict", "site_prev_goal_share_strict", "site_prev_tasks_strict"):
        np.testing.assert_allclose(sa[col].to_numpy(float), sb[col].to_numpy(float), equal_nan=True)

    pa = TM2.build_pilot_table(db_a, ta)
    pb = TM2.build_pilot_table(db_b, tb)
    pa5 = pa[pa["event_key"] == "ev:5"].sort_values("result_id").reset_index(drop=True)
    pb5 = pb[pb["event_key"] == "ev:5"].sort_values("result_id").reset_index(drop=True)
    for col in ("pilot_prev_tasks", "pilot_prev_goal_rate", "pilot_prev_speed_ratio", "pilot_prev_site_tasks"):
        np.testing.assert_allclose(pa5[col].to_numpy(float), pb5[col].to_numpy(float), equal_nan=True)


def test_end_to_end_task_v2_and_forecast_v2(tmp_path: Path):
    """Run full CLI pipelines on the synthetic mini-DB and verify all artifacts and anti-leakage."""
    db = tmp_path / "db"
    _build_synthetic_db(db)

    # 1. Run task_model_v2.py CLI end-to-end
    res = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "task_model_v2.py"), str(db), "--folds", "4", "--thermal-rows", "500"],
        capture_output=True, text=True, check=False,
    )
    assert res.returncode == 0, f"task_model_v2 failed:\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"

    out = db / "models" / "task_v2"
    for fname in ("metrics.json", "ablation.csv", "paired_bootstrap.csv", "features_task.parquet",
                  "predictions.parquet", "pilot_predictions.parquet", "models/meta.json"):
        assert (out / fname).exists(), f"Missing output {fname}"

    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["levels_native_1h_share"] == 1.0
    stages = metrics["splits"]["new competition"]["stages"]
    for st in ("M0_baseline", "M1_task_v1_5h", "M2_hourly_1h", "M3_forecast_compat", "M3_terrain_land",
               "M4_thermal_mc", "M5_pilot_agg", "M5_blend"):
        assert st in stages, f"Missing ablation stage {st}"
        assert np.isfinite(stages[st]["winner_time_mae_min"])
        assert np.isfinite(stages[st]["goal_share_mae_pts"])
    assert "stopped" in metrics and not metrics["stopped"].get("skipped")

    # 2. Run forecast_multivar_eval.py --task-v2-only end-to-end
    res2 = subprocess.run(
        [
            sys.executable, str(ROOT / "scripts" / "forecast_multivar_eval.py"), str(db),
            "--models", "ecmwf_ifs,gfs", "--from-year", "2022", "--task-v2-only", "--thermal-rows", "500",
        ],
        capture_output=True, text=True, check=False,
    )
    assert res2.returncode == 0, f"forecast_multivar_eval --task-v2-only failed:\nSTDOUT:\n{res2.stdout}\nSTDERR:\n{res2.stderr}"

    fc_v2 = db / "models" / "forecast_multivar_eval" / "task_v2"
    for fname in ("task_v2_e2e_scores.csv", "task_v2_paired_bootstrap.csv",
                  "task_v2_blend_cv.json", "hourly_forecast_calibration.json"):
        assert (fc_v2 / fname).exists(), f"Missing forecast v2 output {fname}"

    cal = json.loads((fc_v2 / "hourly_forecast_calibration.json").read_text(encoding="utf-8"))
    assert set(cal["models"].keys()) == {"ecmwf_ifs", "gfs"}
    assert "hourly_quantile_maps" in cal["models"]["ecmwf_ifs"]
