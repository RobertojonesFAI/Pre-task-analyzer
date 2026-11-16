"""Read IGC tracklogs: header fields, a summary of the fixes, and the fixes themselves."""

import hashlib
import re

import numpy as np

HEADERS = {  # three-letter IGC header codes kept in the catalog
    "DTE": "date", "PLT": "pilot", "GTY": "glider", "GID": "glider_id", "CID": "comp_id",
    "CCL": "comp_class", "FTY": "logger",
}


def _lines(data):
    return data.replace(b"\r", b"").split(b"\n")


def _date(value):
    """IGC date ('150826', 'DATE:150826,01') -> 'YYYY-MM-DD' (years 2000-2099)."""
    m = re.search(r"(\d{2})(\d{2})(\d{2})", value)
    if not m:
        return None
    d, mo, y = m.groups()
    if not (1 <= int(mo) <= 12 and 1 <= int(d) <= 31):
        return None
    return f"20{y}-{mo}-{d}"


def header(lines):
    out = {}
    for line in lines:
        if line[:1] != b"H":
            if line[:1] == b"B":
                break
            continue
        text = line.decode("latin-1").strip()
        code = text[2:5]
        if code in HEADERS and HEADERS[code] not in out:
            value = text.split(":", 1)[1].strip() if ":" in text else text[5:].strip()
            out[HEADERS[code]] = _date(value) if code == "DTE" else value
    return out


def _fix(line):
    """(seconds of day, lat, lon, gps alt) from one B record, or None if it is malformed."""
    try:
        t = int(line[1:3]) * 3600 + int(line[3:5]) * 60 + int(line[5:7])
        lat = int(line[7:9]) + int(line[9:14]) / 60000.0
        if line[14:15] == b"S":
            lat = -lat
        lon = int(line[15:18]) + int(line[18:23]) / 60000.0
        if line[23:24] == b"W":
            lon = -lon
        alt = int(line[30:35])
        return t, lat, lon, alt
    except ValueError:
        return None


def summary(data):
    """Header fields and a fix summary of one IGC file (bytes).

    sha1 hashes the B records only, so the same flight saved by two servers with different
    headers is still recognised as a duplicate.
    """
    lines = _lines(data)
    out = header(lines)
    b = [line for line in lines if line[:1] == b"B" and len(line) >= 35]
    out["n_fixes"] = len(b)
    out["sha1"] = hashlib.sha1(b"\n".join(b)).hexdigest() if b else None
    first = next((f for f in map(_fix, b[:20]) if f), None)
    last = next((f for f in map(_fix, reversed(b[-20:])) if f), None)
    if first and last:
        duration = (last[0] - first[0]) % 86400
        out.update(t_first=first[0], t_last=last[0], duration_s=duration,
                   interval_s=round(duration / (len(b) - 1), 2) if len(b) > 1 else None,
                   lat_first=round(first[1], 6), lon_first=round(first[2], 6), alt_first=first[3],
                   lat_last=round(last[1], 6), lon_last=round(last[2], 6), alt_last=last[3])
    return out


def fixes(data):
    """All fixes of an IGC file as numpy arrays, parsed in one vectorised pass:
    t (seconds; rolls over past 24 h for flights across midnight UTC), lat, lon, alt_gps,
    alt_baro (metres). Malformed records are dropped."""
    lines = [line[:35] for line in _lines(data) if line[:1] == b"B" and len(line) >= 35]
    empty = {k: np.empty(0) for k in ("t", "lat", "lon", "alt_gps", "alt_baro")}
    if not lines:
        return empty
    a = np.frombuffer(b"".join(lines), dtype=np.uint8).reshape(-1, 35)
    d = a.astype(np.int64) - 48

    def num(i, j):
        out = np.zeros(len(a), np.int64)
        for k in range(i, j):
            out = out * 10 + d[:, k]
        return out

    def digits(i, j):
        return ((d[:, i:j] >= 0) & (d[:, i:j] <= 9)).all(axis=1)

    def alt(i):  # five characters, optionally a leading minus sign
        neg = a[:, i] == ord("-")
        body = np.where(neg, num(i + 1, i + 5), num(i, i + 5))
        ok = digits(i + 1, i + 5) & (neg | digits(i, i + 1))
        return np.where(neg, -body, body), ok

    ok = digits(1, 14) & digits(15, 23) & np.isin(a[:, 14], [ord("N"), ord("S")]) & np.isin(a[:, 23], [ord("E"), ord("W")])
    baro, ok_b = alt(25)
    gps, ok_g = alt(30)
    t = num(1, 3) * 3600 + num(3, 5) * 60 + num(5, 7)
    lat = num(7, 9) + num(9, 14) / 60000.0
    lon = num(15, 18) + num(18, 23) / 60000.0
    lat = np.where(a[:, 14] == ord("S"), -lat, lat)
    lon = np.where(a[:, 23] == ord("W"), -lon, lon)
    keep = ok & ok_g & (np.abs(lat) <= 90) & (np.abs(lon) <= 180)
    t, lat, lon = t[keep], lat[keep], lon[keep]
    gps, baro = np.where(ok_g, gps, 0)[keep], np.where(ok_b, baro, 0)[keep]
    if len(t) > 1:  # flights past midnight UTC
        t = t + 86400 * np.concatenate([[0], np.cumsum(np.diff(t) < -43200)])
    return {"t": t, "lat": lat, "lon": lon, "alt_gps": gps, "alt_baro": baro}
