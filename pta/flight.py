"""Split one track into its flight, climbs (thermals) and glides.

Works on the fixes from pta.igc.fixes:

1. Clean: drop repeated times, fixes that imply impossible speeds (GPS spikes), and altitude
   glitches (fixes far from the local median altitude or implying impossible climb rates).
2. Resample to an even time step (1 s for 1 Hz loggers; coarser for live-tracking archives) so
   speeds, climb rates and turn rates are computed over fixed time windows.
3. Takeoff and landing: the track is cut at every stop (standing still for a minute at a steady
   altitude, with real fixes, not a logger gap) and at logger gaps over half an hour. The flight is
   the stretch that circles the most; the others are the drive up, waiting at launch or the
   retrieve drive that live-tracking archives often include. A flight ends where the final glide
   stops sinking at walking pace, if that comes before the next stop.
4. Climbs: periods of sustained circling (turning faster than CIRCLING_DEG_S for at least
   MIN_CLIMB_S and at least one full turn). A climb is a thermal when it gains height at
   THERMAL_RATE m/s or more on average.
5. Glides: the straight parts between climbs.

Heights are GPS altitudes above sea level; heights above ground come later from the terrain model.
"""

import numpy as np

EARTH_R = 6371000.0
MOVING_MS = 4.0          # faster than this is flying (or running to launch)
MOVING_FOR_S = 30        # for at least this long
CIRCLING_DEG_S = 8.0     # smoothed turn rate that counts as circling (a full turn in 45 s)
MIN_CLIMB_S = 30         # shortest circling period kept
MERGE_GAP_S = 15         # circling periods closer than this are one climb
MIN_TURN_DEG = 300       # a climb needs (about) one full turn
THERMAL_RATE = 0.5       # m/s average gain for a climb to count as a thermal
SPIKE_MS = 80.0          # horizontal speed between fixes above this is a GPS error
ALT_SPIKE_M = 80.0       # a fix this far from the median altitude of the surrounding 31 fixes is an error
MAX_VZ_MS = 20.0         # climb or sink between fixes faster than this is an altitude error
STOPPED_MS = 1.5         # slower than this is standing still
STOPPED_FOR_S = 60       # standing still this long is a stop (before takeoff or after landing)
STOPPED_ALT_M = 100      # (with the altitude within this range, 5th to 95th percentile)
STOPPED_FIXED = 0.2      # (and real fixes over at least this share of it)
MAX_GAP_S = 1800         # a logger gap this long splits the track
DRIVE_UP_M = 100         # a stretch with no circling that ends this much higher than it began is a drive
FLAT_FOR_S = 90          # after the last climb, altitude this steady for this long means on the ground:
FLAT_M = 30              # a glider in its final glide loses well over FLAT_M in FLAT_FOR_S
MAX_THERMAL_RATE = 8.0   # average climbs faster than this are altitude errors, not thermals
STORE_EVERY_S = 5        # time step of the fixes kept for later steps


def _xy(lat, lon, lat0, lon0):
    """Local east/north metres around (lat0, lon0)."""
    k = np.pi / 180
    return (lon - lon0) * k * EARTH_R * np.cos(lat0 * k), (lat - lat0) * k * EARTH_R


def _dist(lat1, lon1, lat2, lon2):
    k = np.pi / 180
    dlat, dlon = (lat2 - lat1) * k, (lon2 - lon1) * k
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1 * k) * np.cos(lat2 * k) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R * np.arcsin(np.sqrt(np.minimum(a, 1)))


def _runs(mask):
    """(start, end) index pairs of consecutive True values, end exclusive."""
    if not mask.any():
        return []
    m = np.concatenate([[False], mask, [False]]).astype(np.int8)
    d = np.diff(m)
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1)))


def _smooth(x, n):
    """Centred moving average over n samples (n >= 1)."""
    if n <= 1 or len(x) < n:
        return x.astype(float)
    c = np.cumsum(np.concatenate([[0.0], x]))
    out = (c[n:] - c[:-n]) / n
    pad = n // 2
    return np.concatenate([np.full(pad, out[0]), out, np.full(len(x) - len(out) - pad, out[-1])])


def clean(fx):
    """Fixes in time order with repeated times and GPS spikes removed. Uses GPS altitude, or
    pressure altitude when the logger recorded no GPS altitude."""
    t, lat, lon = fx["t"].astype(float), fx["lat"], fx["lon"]
    alt = fx["alt_gps"].astype(float)
    if len(alt) and (alt == 0).mean() > 0.5 and (fx["alt_baro"] != 0).mean() > 0.5:
        alt = fx["alt_baro"].astype(float)
    order = np.argsort(t, kind="stable")
    t, lat, lon, alt = t[order], lat[order], lon[order], alt[order]
    keep = np.concatenate([[True], np.diff(t) > 0])
    t, lat, lon, alt = t[keep], lat[keep], lon[keep], alt[keep]
    for _ in range(5):  # drop fixes that jump away and back
        if len(t) < 3:
            break
        fast = _dist(lat[:-1], lon[:-1], lat[1:], lon[1:]) / np.diff(t) > SPIKE_MS
        bad = np.zeros(len(t), bool)
        bad[1:-1] = fast[:-1] & fast[1:]
        if not bad.any():
            break
        t, lat, lon, alt = t[~bad], lat[~bad], lon[~bad], alt[~bad]
    if len(alt) >= 31:  # altitude spikes and glitches (some phone loggers jump by hundreds of metres)
        med = np.median(np.lib.stride_tricks.sliding_window_view(np.pad(alt, 15, mode="edge"), 31), axis=1)
        good = np.abs(alt - med) <= ALT_SPIKE_M
        t, lat, lon, alt = t[good], lat[good], lon[good], alt[good]
    for _ in range(5):
        if len(t) < 3:
            break
        steep = np.abs(np.diff(alt)) / np.diff(t) > MAX_VZ_MS
        bad = np.zeros(len(t), bool)
        bad[1:-1] = steep[:-1] & steep[1:]
        if not bad.any():
            break
        t, lat, lon, alt = t[~bad], lat[~bad], lon[~bad], alt[~bad]
    return t, lat, lon, alt


def analyse(fx):
    """(flight dict, list of climb dicts, list of glide dicts, dict of fixes every STORE_EVERY_S s).
    Returns (None, [], [], None) when there is no flight in the track."""
    t, lat, lon, alt = clean(fx)
    if len(t) < 20:
        return None, [], [], None
    step = float(max(1, round(np.median(np.diff(t)))))
    grid = np.arange(t[0], t[-1] + step / 2, step)
    lat_g, lon_g, alt_g = (np.interp(grid, t, v) for v in (lat, lon, alt))
    x, y = _xy(lat_g, lon_g, lat_g[0], lon_g[0])

    w = max(1, int(round(5 / step)))  # 5 s baseline for speed and heading
    dx, dy = np.zeros_like(x), np.zeros_like(y)
    dx[w:], dy[w:] = x[w:] - x[:-w], y[w:] - y[:-w]
    speed = np.hypot(dx, dy) / (w * step)
    heading = np.unwrap(np.arctan2(dx, dy))
    turn = np.zeros_like(heading)
    turn[1:] = np.degrees(np.diff(heading)) / step
    turn = _smooth(turn, max(1, int(round(10 / step))))
    vz = np.zeros_like(alt_g)
    wv = max(1, int(round(10 / step)))
    vz[wv:-wv] = (alt_g[2 * wv:] - alt_g[:-2 * wv]) / (2 * wv * step) if len(alt_g) > 2 * wv else 0

    # Stops: standing still for a minute, and not climbing or sinking (a frozen GPS position can fake
    # standing still in the air). Ground altitude from live trackers wanders by tens of metres over a
    # long stop, so the test uses the 5-95 % range. Interpolated stretches across logger gaps are not
    # evidence of a stop: a stop needs real fixes over STOPPED_FIXED of its length. Gaps longer than
    # MAX_GAP_S split the track as well. The flight is the stretch between stops that climbs the most;
    # the others are the drive up, waiting and walking at launch, or the retrieve drive.
    fixed = np.zeros(len(grid), bool)
    fixed[np.clip(np.round((t - grid[0]) / step).astype(int), 0, len(grid) - 1)] = True
    still = _smooth(speed, max(1, int(round(10 / step)))) < STOPPED_MS
    stops = []
    for a, b in _runs(still):
        if ((b - a) * step >= STOPPED_FOR_S and fixed[a:b].mean() >= STOPPED_FIXED
                and np.subtract(*np.percentile(alt_g[a:b], [95, 5])) <= STOPPED_ALT_M):
            stops.append((a, b))
    for a, b in _runs(~fixed):
        if (b - a) * step >= MAX_GAP_S:
            stops.append((a, b))
    merged_stops = []
    for a, b in sorted(stops):
        if merged_stops and a <= merged_stops[-1][1]:
            merged_stops[-1] = (merged_stops[-1][0], max(b, merged_stops[-1][1]))
        else:
            merged_stops.append((a, b))
    stops = merged_stops
    bounds = [0] + [x for st in stops for x in st] + [len(grid)]
    # Which stretch is the flight: gliders circle and cars do not, so the score is the time spent
    # circling at flying speed, plus the height range (a sled ride with no thermals still drops a
    # long way). A stretch that ends well above where it started without circling is a drive up.
    circling_any = (np.abs(turn) >= CIRCLING_DEG_S) & (speed > MOVING_MS)
    smooth_alt = _smooth(alt_g, max(1, int(round(10 / step))))
    best, best_score = None, -1.0
    for a, b in zip(bounds[0::2], bounds[1::2]):
        moving = [r for r in _runs(speed[a:b] > MOVING_MS) if (r[1] - r[0]) * step >= MOVING_FOR_S]
        if not moving:
            continue
        s0, s1 = a + moving[0][0], a + moving[-1][1] - 1
        circ_s = float(circling_any[s0:s1 + 1].sum() * step)
        if circ_s < MIN_CLIMB_S and smooth_alt[s1] - smooth_alt[s0] > DRIVE_UP_M:
            continue
        score = circ_s + float(np.ptp(smooth_alt[s0:s1 + 1])) / 10
        if score > best_score:
            best, best_score = (s0, s1), score
    if best is None:
        return None, [], [], None
    i0, i1 = best

    circling = np.zeros(len(grid), bool)
    circling[i0:i1 + 1] = np.abs(turn[i0:i1 + 1]) >= CIRCLING_DEG_S
    # A flight glides down after its last thermal; when the altitude then changes by less than FLAT_M
    # over FLAT_FOR_S at walking pace the pilot is on the ground (some loggers have no clean stop
    # between landing and the retrieve drive). A glider gliding through lift moves much faster.
    rising = np.flatnonzero(circling[i0:i1 + 1] & (vz[i0:i1 + 1] > THERMAL_RATE))
    last_climb = i0 + int(rising[-1]) if len(rising) else i0
    n_flat = max(2, int(round(FLAT_FOR_S / step)))
    if i1 - last_climb > n_flat:
        seg = alt_g[last_climb:i1 + 1]
        change = np.abs(seg[n_flat:] - seg[:-n_flat])
        pace = _dist(lat_g[last_climb:i1 + 1 - n_flat], lon_g[last_climb:i1 + 1 - n_flat],
                     lat_g[last_climb + n_flat:i1 + 1], lon_g[last_climb + n_flat:i1 + 1]) / (n_flat * step)
        flat = np.flatnonzero((change <= FLAT_M) & (pace < MOVING_MS))
        if len(flat):
            i1 = last_climb + int(flat[0])
    circling[i1 + 1:] = False
    runs = _runs(circling)
    merged = []
    for a, b in runs:
        if merged and (a - merged[-1][1]) * step <= MERGE_GAP_S:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))

    climbs = []
    for a, b in merged:
        dur = (b - a) * step
        turned = abs(np.degrees(heading[b - 1] - heading[a]))
        if dur < MIN_CLIMB_S or turned < MIN_TURN_DEG:
            continue
        seg = slice(a, b)
        top = a + int(np.argmax(alt_g[seg]))
        gain = alt_g[top] - alt_g[a]
        rate = gain / max((top - a) * step, step)
        climbs.append({
            "t_start": grid[a], "t_end": grid[b - 1], "duration_s": dur,
            "lat": float(np.mean(lat_g[seg])), "lon": float(np.mean(lon_g[seg])),
            "alt_base": float(alt_g[a]), "alt_top": float(alt_g[top]), "gain_m": float(gain),
            "rate_ms": float(rate), "net_rate_ms": float((alt_g[b - 1] - alt_g[a]) / max(dur, step)),
            "turns": float(turned / 360), "direction": "right" if heading[b - 1] > heading[a] else "left",
            "is_thermal": bool(THERMAL_RATE <= rate <= MAX_THERMAL_RATE and gain > 0),
            "_a": a, "_b": b,
        })

    glides = []
    edges = [i0] + [x for c in climbs for x in (c["_a"], c["_b"] - 1)] + [i1]
    for a, b in zip(edges[0::2], edges[1::2]):
        if b - a < 2:
            continue
        dist = float(_dist(lat_g[a], lon_g[a], lat_g[b], lon_g[b]))
        loss = float(alt_g[a] - alt_g[b])
        dur = (b - a) * step
        glides.append({
            "t_start": grid[a], "t_end": grid[b], "duration_s": dur,
            "lat_start": float(lat_g[a]), "lon_start": float(lon_g[a]), "alt_start": float(alt_g[a]),
            "lat_end": float(lat_g[b]), "lon_end": float(lon_g[b]), "alt_end": float(alt_g[b]),
            "distance_m": dist, "alt_loss_m": loss, "speed_kmh": dist / dur * 3.6,
            "glide_ratio": dist / loss if loss > 1 else None,
        })
    for c in climbs:
        del c["_a"], c["_b"]

    thermals = [c for c in climbs if c["is_thermal"]]
    away = _dist(lat_g[i0], lon_g[i0], lat_g[i0:i1 + 1], lon_g[i0:i1 + 1])
    flight_s = (i1 - i0) * step
    flight = {
        "step_s": step, "t_takeoff": grid[i0], "t_landing": grid[i1], "duration_s": flight_s,
        "lat_takeoff": float(lat_g[i0]), "lon_takeoff": float(lon_g[i0]), "alt_takeoff": float(alt_g[i0]),
        "lat_landing": float(lat_g[i1]), "lon_landing": float(lon_g[i1]), "alt_landing": float(alt_g[i1]),
        "alt_max": float(alt_g[i0:i1 + 1].max()), "max_dist_from_takeoff_m": float(away.max()),
        "n_climbs": len(climbs), "n_thermals": len(thermals),
        "thermal_gain_m": float(sum(c["gain_m"] for c in thermals)),
        "mean_thermal_rate_ms": float(np.average([c["rate_ms"] for c in thermals],
                                                 weights=[c["duration_s"] for c in thermals])) if thermals else None,
        "circling_share": float(sum(c["duration_s"] for c in climbs) / flight_s) if flight_s else None,
        "quality": "high" if step <= 2 else "medium" if step <= 5 else "low",
    }

    keep = np.arange(i0, i1 + 1, max(1, int(round(STORE_EVERY_S / step))))
    kept = {"t": grid[keep].astype(np.int32), "lat": lat_g[keep].astype(np.float32),
            "lon": lon_g[keep].astype(np.float32), "alt": alt_g[keep].astype(np.int16),
            "vz": np.round(vz[keep], 1).astype(np.float32), "circling": circling[keep]}
    return flight, climbs, glides, kept
