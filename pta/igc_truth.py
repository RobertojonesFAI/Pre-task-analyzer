"""High-precision atmospheric sensor extraction from cleaned IGC flight tracks.

Turns cleaned competition paragliding fixes (5 s resampled or 1 Hz raw) into calibrated physical
measurements for verifying weather models and training next-generation thermal/task models:

  - Thermal wind via two independent kinematics estimators that are invariant to 5 s sampling and
    strong wind (wind speed > glider airspeed):
      1. Hodograph velocity-circle fit (Kasa initialization + Huber-weighted Gauss-Newton on
         consecutive velocity vectors), yielding (u, v), apparent circling airspeed, radial
         residual, and octant coverage (rejecting figure-8 ridge soaring).
      2. Air-relative integer-turn position displacement (exact multiples of 360 deg in unwrapped
         air-relative heading), eliminating partial-turn phase noise.
  - Vertical slicing of deep thermals so wind shear across height bands is preserved.
  - Peak sustained core climb rates (60 s and 30 s sliding net climb) separated from searching time.
  - Multi-heading glide wind cross-check via Huber regression controlled by glider class.
  - Airspace-cap detection on multi-pilot thermal ceilings.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

M_PER_DEG = 111320.0

# Quality thresholds
MAX_OFFSET_M = 150.0        # max landing-DEM altitude calibration offset accepted (m)
GOLD_DRIFT_M = 40.0         # max |takeoff_agl_raw - landing_agl_raw| for gold altitude tier (m)
MIN_DEM_COV = 0.90          # minimum fraction of flight fixes with DEM coverage

MIN_WIND_S = 60.0           # minimum duration of a thermal wind segment (s)
MIN_WIND_TURNS = 2.0        # minimum net full turns required for thermal wind
MAX_WIND_MS = 30.0          # maximum physically plausible wind speed (m/s)
MIN_TAS_MS = 4.5            # minimum apparent circling airspeed at 5 s chord (m/s)
MAX_TAS_MS = 16.0           # maximum apparent circling airspeed (m/s)
MAX_HODO_RESID_MS = 2.5     # max RMS radial residual around velocity circle (m/s)
MIN_OCTANTS = 6             # minimum populated azimuth octants (out of 8) to reject figure-8s
MAX_METHOD_DIFF_MS = 2.0    # max disagreement between hodograph and integer-turn wind (m/s)

SPLIT_GAIN_M = 500.0        # split thermals gaining at least this much vertically...
SPLIT_MIN_DUR_S = 140.0     # ...and lasting at least this long into vertical wind slices
SLICE_GAIN_M = 400.0        # target vertical gain per slice (m)
MAX_SLICES = 4

CORE60_S = 60.0
CORE30_S = 30.0
MAX_CORE60_MS = 10.0        # physically plausible sustained 60 s climb rate ceiling (m/s)
MAX_CORE30_MS = 12.0        # physically plausible sustained 30 s climb rate ceiling (m/s)
MIN_CEILING_GAIN_M = 150.0  # minimum gain for a thermal to represent the convective ceiling


def local_xy(
    lat: np.ndarray,
    lon: np.ndarray,
    lat0: float | None = None,
    lon0: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert (lat, lon) degrees to local east/north metres around (lat0, lon0)."""
    lat_a = np.asarray(lat, dtype=np.float64)
    lon_a = np.asarray(lon, dtype=np.float64)
    if lat0 is None:
        lat0 = float(np.mean(lat_a))
    if lon0 is None:
        lon0 = float(np.mean(lon_a))
    cos_lat = math.cos(math.radians(lat0))
    x = (lon_a - lon0) * M_PER_DEG * cos_lat
    y = (lat_a - lat0) * M_PER_DEG
    return x, y


def fit_hodograph_wind(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    max_gn: int = 5,
) -> dict[str, Any]:
    """Estimate wind (u, v) as the centre of the circling velocity hodograph.

    For a glider circling at airspeed V in steady wind (u, v), consecutive segment velocities
    (vx_i, vy_i) = (dx_i/dt_i, dy_i/dt_i) lie on a circle centred at (u, v) whose radius is
    V * sinc(omega * dt / 2). The centre (u, v) is mathematically invariant to the 5 s chord
    interval and remains well-defined even when wind speed exceeds glider airspeed.

    Returns dict with keys: u, v, tas_fit, resid_ms, octants, n_vel.
    """
    nan_out = {
        "u": np.nan, "v": np.nan, "tas_fit": np.nan,
        "resid_ms": np.nan, "octants": 0, "n_vel": 0,
    }
    t_a = np.asarray(t, dtype=np.float64)
    x_a = np.asarray(x, dtype=np.float64)
    y_a = np.asarray(y, dtype=np.float64)
    if len(t_a) < 7:
        return nan_out

    med_dt = float(np.median(np.diff(t_a)))
    w_step = max(1, int(round(5.0 / max(med_dt, 0.25))))
    if len(t_a) <= w_step + 5:
        return nan_out

    dt = t_a[w_step:] - t_a[:-w_step]
    ok = (dt > 0.0) & (dt <= 15.0)
    if ok.sum() < 6:
        return nan_out

    vx = (x_a[w_step:] - x_a[:-w_step])[ok] / dt[ok]
    vy = (y_a[w_step:] - y_a[:-w_step])[ok] / dt[ok]
    spd = np.hypot(vx, vy)
    keep = np.isfinite(spd) & (spd <= 50.0)
    vx, vy = vx[keep], vy[keep]
    n = len(vx)
    if n < 6:
        return nan_out

    # Initial estimate: centred Kasa algebraic circle fit.
    mx, my = float(np.median(vx)), float(np.median(vy))
    dx, dy = vx - mx, vy - my
    A = np.column_stack([2.0 * dx, 2.0 * dy, np.ones(n, dtype=np.float64)])
    b = dx * dx + dy * dy
    svals = np.linalg.svd(A, compute_uv=False)
    if len(svals) < 3 or svals[0] <= 0.0 or svals[-1] < 1e-5 * svals[0]:
        return nan_out
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    u = mx + float(coef[0])
    v = my + float(coef[1])
    r2 = float(coef[0] ** 2 + coef[1] ** 2 + coef[2])
    if not np.isfinite(r2) or r2 <= 0.25:
        return nan_out
    R = math.sqrt(r2)

    # Robust geometric circle refinement: Huber-weighted Gauss-Newton on radial residuals.
    for _ in range(max_gn):
        rx = vx - u
        ry = vy - v
        r_i = np.hypot(rx, ry)
        if (r_i < 1e-6).all():
            return nan_out
        e = r_i - R
        mad = float(np.median(np.abs(e)))
        scale = max(1.4826 * mad, 0.5)
        cutoff = 1.345 * scale
        abs_e = np.abs(e)
        w = np.where(abs_e <= cutoff, 1.0, cutoff / np.maximum(abs_e, 1e-9))

        safe_r = np.maximum(r_i, 1e-6)
        J = np.column_stack([-rx / safe_r, -ry / safe_r, -np.ones(n, dtype=np.float64)])
        sw = np.sqrt(w)
        Jw = J * sw[:, None]
        ew = e * sw
        # Small Levenberg-Marquardt damping keeps ill-posed partial arcs stable.
        JtJ = Jw.T @ Jw + 1e-4 * np.eye(3)
        Jte = Jw.T @ ew
        try:
            step = -np.linalg.solve(JtJ, Jte)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(step).all():
            break
        u += float(step[0])
        v += float(step[1])
        R = max(0.5, R + float(step[2]))
        if float(np.hypot(step[0], step[1])) < 1e-3 and abs(float(step[2])) < 1e-3:
            break

    if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(R)):
        return nan_out

    rx, ry = vx - u, vy - v
    r_i = np.hypot(rx, ry)
    resid = r_i - R
    resid_ms = float(np.sqrt(np.mean(resid * resid)))
    ang = np.mod(np.arctan2(rx, ry), 2.0 * math.pi)
    octants = int(len(np.unique(np.floor(ang / (math.pi / 4.0)).astype(np.int32) % 8)))
    return {
        "u": u,
        "v": v,
        "tas_fit": R,
        "resid_ms": resid_ms,
        "octants": octants,
        "n_vel": n,
    }


def integer_turn_wind(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    min_turns: float = MIN_WIND_TURNS,
) -> dict[str, Any]:
    """Estimate wind (u, v) from displacement over exact integer full turns.

    Uses air-relative velocity centred on the segment mean velocity so that heading unwraps
    monotonically by 360 deg per circle even when ambient wind speed exceeds glider airspeed.
    Eliminates fractional-turn phase error and automatically rejects figure-8 ridge climbs
    (whose net unwrapped heading change stays near 0 deg).
    """
    nan_out = {"u": np.nan, "v": np.nan, "net_turns": 0.0, "n_pairs": 0}
    t_a = np.asarray(t, dtype=np.float64)
    x_a = np.asarray(x, dtype=np.float64)
    y_a = np.asarray(y, dtype=np.float64)
    if len(t_a) < 8:
        return nan_out

    med_dt = float(np.median(np.diff(t_a)))
    w_step = max(1, int(round(5.0 / max(med_dt, 0.25))))
    if len(t_a) <= w_step + 6:
        return nan_out

    dt = t_a[w_step:] - t_a[:-w_step]
    ok = (dt > 0.0) & (dt <= 15.0)
    if ok.sum() < 7:
        return nan_out

    idx = np.flatnonzero(ok)
    tm = 0.5 * (t_a[idx] + t_a[idx + w_step])
    xm = 0.5 * (x_a[idx] + x_a[idx + w_step])
    ym = 0.5 * (y_a[idx] + y_a[idx + w_step])
    vx = (x_a[idx + w_step] - x_a[idx]) / dt[ok]
    vy = (y_a[idx + w_step] - y_a[idx]) / dt[ok]

    # Air-relative heading around mean velocity (independent of fit_hodograph_wind).
    u0, v0 = float(np.mean(vx)), float(np.mean(vy))
    rx, ry = vx - u0, vy - v0
    if float(np.median(np.hypot(rx, ry))) < 2.0:
        return nan_out
    psi = np.unwrap(np.arctan2(rx, ry))
    total_rad = float(psi[-1] - psi[0])
    net_turns = abs(total_rad) / (2.0 * math.pi)
    if net_turns < min_turns:
        return {"u": np.nan, "v": np.nan, "net_turns": net_turns, "n_pairs": 0}

    sgn = 1.0 if total_rad >= 0.0 else -1.0
    # Cumulative monotonic angle envelope in the dominant turn direction.
    ang = sgn * (psi - psi[0])
    ang_mono = np.maximum.accumulate(ang)
    # Require strictly increasing sample points for interpolation.
    uniq = np.concatenate([[True], np.diff(ang_mono) > 1e-6])
    if uniq.sum() < 6:
        return {"u": np.nan, "v": np.nan, "net_turns": net_turns, "n_pairs": 0}
    am, tu, xu, yu = ang_mono[uniq], tm[uniq], xm[uniq], ym[uniq]

    # Use the largest integer turn span <= net_turns (at least 2 full turns), evaluated at
    # evenly spaced start phases across the leftover fraction (and at least 1 full turn of
    # start phases when net_turns >= 3) to average out single-fix noise.
    span_turns = max(int(min_turns), int(math.floor(net_turns)))
    if net_turns >= span_turns + 0.25 or span_turns == int(min_turns):
        k_span = span_turns
    else:
        k_span = max(int(min_turns), span_turns - 1)
    span_rad = 2.0 * math.pi * k_span
    max_start = float(am[-1] - span_rad)
    if max_start < 0.0:
        return {"u": np.nan, "v": np.nan, "net_turns": net_turns, "n_pairs": 0}

    n_samples = max(1, min(24, int(round(max_start / (math.pi / 4.0))) + 1))
    starts = np.linspace(0.0, max_start, n_samples)
    ends = starts + span_rad

    t0 = np.interp(starts, am, tu)
    t1 = np.interp(ends, am, tu)
    x0 = np.interp(starts, am, xu)
    x1 = np.interp(ends, am, xu)
    y0 = np.interp(starts, am, yu)
    y1 = np.interp(ends, am, yu)

    d_t = t1 - t0
    valid = d_t >= max(30.0, 15.0 * k_span)
    if not valid.any():
        return {"u": np.nan, "v": np.nan, "net_turns": net_turns, "n_pairs": 0}

    us = (x1[valid] - x0[valid]) / d_t[valid]
    vs = (y1[valid] - y0[valid]) / d_t[valid]
    return {
        "u": float(np.median(us)),
        "v": float(np.median(vs)),
        "net_turns": net_turns,
        "n_pairs": int(valid.sum()),
    }


def peak_climb_rate(
    t: np.ndarray,
    alt: np.ndarray,
    window_s: float,
    max_plausible_ms: float = MAX_CORE30_MS,
) -> float:
    """Peak sustained net climb rate (m/s) over any exact `window_s` interval within a climb."""
    t_a = np.asarray(t, dtype=np.float64)
    z_a = np.asarray(alt, dtype=np.float64)
    if len(t_a) < 2 or (t_a[-1] - t_a[0]) < window_s:
        return float("nan")
    # Evaluate windows starting at every fix where t + window_s lies inside the segment.
    t_end = t_a + window_s
    ok = t_end <= t_a[-1] + 1e-6
    if not ok.any():
        return float("nan")
    z_start = z_a[ok]
    z_end = np.interp(t_end[ok], t_a, z_a)
    rates = (z_end - z_start) / window_s
    rates = rates[np.isfinite(rates) & (rates <= max_plausible_ms)]
    if not len(rates):
        return float("nan")
    return float(np.max(rates))


def _evaluate_wind_segment(
    t: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    alt_cal: np.ndarray,
) -> dict[str, Any]:
    """Compute old, hodograph, integer-turn and QC-gated clean wind for one circling segment."""
    dur = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    u_old = float((x[-1] - x[0]) / dur) if dur >= MIN_WIND_S else np.nan
    v_old = float((y[-1] - y[0]) / dur) if dur >= MIN_WIND_S else np.nan
    if np.isfinite(u_old) and math.hypot(u_old, v_old) > MAX_WIND_MS:
        u_old, v_old = np.nan, np.nan

    if dur < MIN_WIND_S or len(t) < 8:
        return {
            "u_old": u_old, "v_old": v_old,
            "u_hodo": np.nan, "v_hodo": np.nan,
            "u_int": np.nan, "v_int": np.nan,
            "u_clean": np.nan, "v_clean": np.nan,
            "tas_fit": np.nan, "hodo_resid_ms": np.nan,
            "octants": 0, "net_turns": 0.0, "method_diff_ms": np.nan,
            "wind_qc_ok": False,
            "wind_alt_cal": float(np.mean(alt_cal)) if len(alt_cal) else np.nan,
            "duration_s": dur,
        }

    hodo = fit_hodograph_wind(t, x, y)
    iturn = integer_turn_wind(t, x, y)

    uh, vh = hodo["u"], hodo["v"]
    ui, vi = iturn["u"], iturn["v"]
    both_ok = np.isfinite(uh) and np.isfinite(vh) and np.isfinite(ui) and np.isfinite(vi)
    diff = float(math.hypot(uh - ui, vh - vi)) if both_ok else np.nan

    spd_h = math.hypot(uh, vh) if np.isfinite(uh) and np.isfinite(vh) else np.nan
    spd_i = math.hypot(ui, vi) if np.isfinite(ui) and np.isfinite(vi) else np.nan
    qc_ok = bool(
        both_ok
        and hodo["octants"] >= MIN_OCTANTS
        and MIN_TAS_MS <= hodo["tas_fit"] <= MAX_TAS_MS
        and hodo["resid_ms"] <= MAX_HODO_RESID_MS
        and iturn["net_turns"] >= MIN_WIND_TURNS
        and diff <= MAX_METHOD_DIFF_MS
        and spd_h <= MAX_WIND_MS
        and spd_i <= MAX_WIND_MS
    )

    u_clean = 0.5 * (uh + ui) if qc_ok else np.nan
    v_clean = 0.5 * (vh + vi) if qc_ok else np.nan
    return {
        "u_old": u_old, "v_old": v_old,
        "u_hodo": uh, "v_hodo": vh,
        "u_int": ui, "v_int": vi,
        "u_clean": u_clean, "v_clean": v_clean,
        "tas_fit": hodo["tas_fit"],
        "hodo_resid_ms": hodo["resid_ms"],
        "octants": int(hodo["octants"]),
        "net_turns": float(iturn["net_turns"]),
        "method_diff_ms": diff,
        "wind_qc_ok": qc_ok,
        "wind_alt_cal": float(np.mean(alt_cal)),
        "duration_s": dur,
    }


def split_thermal_slices(
    t: np.ndarray,
    alt_cal: np.ndarray,
    split_gain_m: float = SPLIT_GAIN_M,
    slice_gain_m: float = SLICE_GAIN_M,
    min_dur_s: float = MIN_WIND_S,
) -> list[tuple[int, int]]:
    """Return index ranges [(a, b), ...] splitting a deep thermal vertically into height slices.

    Short thermals return a single slice [(0, len(t))]. Deep thermals (>= split_gain_m gain and
    >= SPLIT_MIN_DUR_S duration) are split across the ascending portion [0..argmax(alt)] into 2..4
    slices of ~slice_gain_m each so wind shear with altitude is resolved.
    """
    n = len(t)
    if n < 2:
        return [(0, n)]
    top_idx = int(np.argmax(alt_cal))
    gain = float(alt_cal[top_idx] - alt_cal[0])
    dur = float(t[top_idx] - t[0])
    if gain < split_gain_m or dur < SPLIT_MIN_DUR_S:
        return [(0, n)]

    n_slices = min(MAX_SLICES, max(2, int(round(gain / slice_gain_m))))
    # Also cap by duration so every slice has enough time for >= 2 full turns.
    n_slices = min(n_slices, int(dur // (min_dur_s + 10.0)))
    if n_slices < 2:
        return [(0, n)]

    edges_z = np.linspace(float(alt_cal[0]), float(alt_cal[top_idx]), n_slices + 1)
    cut_idx = [0]
    for z_edge in edges_z[1:-1]:
        hit = np.flatnonzero(alt_cal[: top_idx + 1] >= z_edge)
        pos = int(hit[0]) if len(hit) else cut_idx[-1]
        if pos > cut_idx[-1] and (t[pos] - t[cut_idx[-1]]) >= min_dur_s:
            cut_idx.append(pos)
    cut_idx.append(n)
    # Merge last slice into previous if too short.
    if len(cut_idx) >= 3 and (t[cut_idx[-1] - 1] - t[cut_idx[-2]]) < min_dur_s:
        cut_idx.pop(-2)
    return [(cut_idx[i], cut_idx[i + 1]) for i in range(len(cut_idx) - 1)]


def extract_thermal_sensors(
    t: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    alt: np.ndarray,
    agl_offset: float = 0.0,
    ground_base: float = np.nan,
    ground_top: float = np.nan,
    split_deep: bool = True,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Extract high-precision thermal summary + vertical wind observation slices from one thermal.

    Returns (thermal_summary, wind_slices).
    - `thermal_summary` is always 1 dict per thermal (core60/core30/mean climb, calibrated top &
      base ASL/AGL, full-thermal wind metrics).
    - `wind_slices` has 1 entry for normal thermals and 2..4 vertically split entries for deep
      thermals (>= 500 m gain) when `split_deep=True`.
    """
    t_a = np.asarray(t, dtype=np.float64)
    lat_a = np.asarray(lat, dtype=np.float64)
    lon_a = np.asarray(lon, dtype=np.float64)
    alt_cal = np.asarray(alt, dtype=np.float64) - float(agl_offset)

    order = np.argsort(t_a, kind="stable")
    t_a, lat_a, lon_a, alt_cal = t_a[order], lat_a[order], lon_a[order], alt_cal[order]
    uniq = np.concatenate([[True], np.diff(t_a) > 0.0]) if len(t_a) else np.empty(0, dtype=bool)
    t_a, lat_a, lon_a, alt_cal = t_a[uniq], lat_a[uniq], lon_a[uniq], alt_cal[uniq]

    n = len(t_a)
    if n < 3:
        return {}, []

    lat0, lon0 = float(np.mean(lat_a)), float(np.mean(lon_a))
    x, y = local_xy(lat_a, lon_a, lat0, lon0)

    top_i = int(np.argmax(alt_cal))
    alt_base_cal = float(alt_cal[0])
    alt_top_cal = float(alt_cal[top_i])
    gain_m = alt_top_cal - alt_base_cal
    dt_top = float(t_a[top_i] - t_a[0])
    dur_s = float(t_a[-1] - t_a[0])
    rate_mean = gain_m / dt_top if dt_top > 0 else np.nan
    rate_core60 = peak_climb_rate(t_a, alt_cal, CORE60_S, MAX_CORE60_MS)
    rate_core30 = peak_climb_rate(t_a, alt_cal, CORE30_S, MAX_CORE30_MS)

    full_wind = _evaluate_wind_segment(t_a, x, y, alt_cal)
    agl_base = alt_base_cal - ground_base if np.isfinite(ground_base) else np.nan
    g_top = ground_top if np.isfinite(ground_top) else ground_base
    agl_top = alt_top_cal - g_top if np.isfinite(g_top) else np.nan
    agl_mid = 0.5 * (agl_base + agl_top) if np.isfinite(agl_base) and np.isfinite(agl_top) else np.nan

    summary = {
        "t_start": int(t_a[0]),
        "t_end": int(t_a[-1]),
        "t_mid": float(0.5 * (t_a[0] + t_a[-1])),
        "duration_s": dur_s,
        "lat": lat0,
        "lon": lon0,
        "alt_base_cal": alt_base_cal,
        "alt_top_cal": alt_top_cal,
        "alt_mid_cal": float(0.5 * (alt_base_cal + alt_top_cal)),
        "agl_base_cal": agl_base,
        "agl_top_cal": agl_top,
        "agl_mid_cal": agl_mid,
        "gain_m": gain_m,
        "rate_mean_ms": rate_mean,
        "rate_core60_ms": rate_core60,
        "rate_core30_ms": rate_core30,
        **full_wind,
    }

    slices_idx = split_thermal_slices(t_a, alt_cal) if split_deep else [(0, n)]
    wind_slices: list[dict[str, Any]] = []
    if len(slices_idx) == 1:
        w = {
            "slice_idx": 0,
            "n_slices": 1,
            "t_mid": summary["t_mid"],
            "lat": lat0,
            "lon": lon0,
            "agl_cal": float(full_wind["wind_alt_cal"] - ground_base) if np.isfinite(ground_base) else np.nan,
            **full_wind,
        }
        wind_slices.append(w)
    else:
        for s_i, (a, b) in enumerate(slices_idx):
            if b - a < 6:
                continue
            w_seg = _evaluate_wind_segment(t_a[a:b], x[a:b], y[a:b], alt_cal[a:b])
            # Interpolate ground elevation linearly between base and top along thermal.
            frac = float((a + b) * 0.5 / max(n, 1))
            g_est = (
                ground_base + frac * (g_top - ground_base)
                if np.isfinite(ground_base) and np.isfinite(g_top)
                else ground_base
            )
            wind_slices.append({
                "slice_idx": s_i,
                "n_slices": len(slices_idx),
                "t_mid": float(0.5 * (t_a[a] + t_a[b - 1])),
                "lat": float(np.mean(lat_a[a:b])),
                "lon": float(np.mean(lon_a[a:b])),
                "agl_cal": float(w_seg["wind_alt_cal"] - g_est) if np.isfinite(g_est) else np.nan,
                **w_seg,
            })

    return summary, wind_slices


def angular_spread_deg(bearings_rad: np.ndarray) -> float:
    """Span (0..360 deg) of track bearings computed as 360 minus the largest empty gap."""
    b = np.sort(np.mod(np.asarray(bearings_rad, dtype=np.float64), 2.0 * math.pi))
    if len(b) < 2:
        return 0.0
    gaps = np.diff(np.concatenate([b, [b[0] + 2.0 * math.pi]]))
    return float(360.0 - math.degrees(float(np.max(gaps))))


def fit_glide_wind_huber(
    bearings_rad: np.ndarray,
    speeds_ms: np.ndarray,
    class_codes: np.ndarray,
    min_glides: int = 8,
    min_spread_deg: float = 90.0,
    max_cond: float = 15.0,
    max_irls: int = 8,
) -> dict[str, Any] | None:
    """Independent wind estimate from multi-heading glides controlled by glider class.

    Model: ground_speed_i = V_class(c_i) + u * sin(bearing_i) + v * cos(bearing_i)
    where bearing is track direction clockwise from north (radians). Solved via Huber IRLS
    so speedbar/lift-line outliers have bounded influence.
    """
    b = np.asarray(bearings_rad, dtype=np.float64)
    s = np.asarray(speeds_ms, dtype=np.float64)
    c = np.asarray(class_codes)
    ok = np.isfinite(b) & np.isfinite(s) & (s >= 3.0) & (s <= 26.0)
    b, s, c = b[ok], s[ok], c[ok]
    n = len(s)
    if n < min_glides:
        return None

    spread = angular_spread_deg(b)
    if spread < min_spread_deg:
        return None

    # One-hot intercepts for glider classes present with at least 2 glides (merge rare ones).
    uniq_c, counts = np.unique(c, return_counts=True)
    major = set(uniq_c[counts >= 2].tolist())
    if not major:
        c_grp = np.zeros(n, dtype=np.int32)
        labels = ["all"]
    else:
        c_str = np.array([str(x) if x in major else "_other" for x in c])
        labels = sorted(set(c_str.tolist()))
        lbl_idx = {k: i for i, k in enumerate(labels)}
        c_grp = np.array([lbl_idx[k] for k in c_str], dtype=np.int32)

    k_cls = len(labels)
    if n < k_cls + 4:
        return None

    X = np.zeros((n, 2 + k_cls), dtype=np.float64)
    X[:, 0] = np.sin(b)
    X[:, 1] = np.cos(b)
    for j in range(k_cls):
        X[c_grp == j, 2 + j] = 1.0

    # Check conditioning on the wind sub-problem (well-excited headings).
    svals = np.linalg.svd(X, compute_uv=False)
    if svals[-1] <= 0.0 or (svals[0] / svals[-1]) > max_cond:
        return None

    beta, *_ = np.linalg.lstsq(X, s, rcond=None)
    for _ in range(max_irls):
        res = s - X @ beta
        mad = float(np.median(np.abs(res)))
        scale = max(1.4826 * mad, 0.8)
        cutoff = 1.345 * scale
        abs_r = np.abs(res)
        w = np.where(abs_r <= cutoff, 1.0, cutoff / np.maximum(abs_r, 1e-9))
        sw = np.sqrt(w)
        beta_new, *_ = np.linalg.lstsq(X * sw[:, None], s * sw, rcond=None)
        if float(np.max(np.abs(beta_new - beta))) < 1e-3:
            beta = beta_new
            break
        beta = beta_new

    u, v = float(beta[0]), float(beta[1])
    if not (np.isfinite(u) and np.isfinite(v)) or math.hypot(u, v) > MAX_WIND_MS:
        return None

    v_classes = {lbl: float(beta[2 + j]) for j, lbl in enumerate(labels)}
    # Sanity check: fitted trim/cruise airspeeds must be physically plausible for paragliders.
    if any(not (6.0 <= va <= 18.0) for va in v_classes.values()):
        return None

    res = s - X @ beta
    return {
        "glide_u": u,
        "glide_v": v,
        "glide_speed_ms": float(math.hypot(u, v)),
        "n_glides": int(n),
        "spread_deg": round(spread, 1),
        "resid_ms": float(np.sqrt(np.mean(res * res))),
        "v_air_by_class": v_classes,
    }


def detect_airspace_cap(
    pilot_tops_asl: np.ndarray,
    min_pilots: int = 5,
    tight_iqr_m: float = 25.0,
    round_tol_m: float = 30.0,
) -> bool:
    """Flag a chip-hour ceiling likely constrained by an airspace cap rather than convection.

    A hard airspace ceiling stops every pilot within ~25 m below a round aviation altitude
    (multiples of 500 ft = 152.4 m or multiples of 100 m) with almost no upper tail above p95.
    """
    vals = np.asarray(pilot_tops_asl, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if len(vals) < min_pilots:
        return False
    p75, p95, pmax = np.percentile(vals, [75, 95, 100])
    if p95 < 1000.0:
        return False
    if (p95 - p75) > tight_iqr_m or (pmax - p95) > tight_iqr_m:
        return False
    # Check proximity to standard 500 ft (152.4 m) flight-level / airspace steps or 100 m steps.
    rem_500ft = min(p95 % 152.4, 152.4 - (p95 % 152.4))
    rem_100m = min(p95 % 100.0, 100.0 - (p95 % 100.0))
    return bool(rem_500ft <= round_tol_m or rem_100m <= 20.0)
