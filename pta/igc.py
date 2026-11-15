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
    """All valid fixes as a dict of numpy arrays: t (seconds since the first fix's midnight, rolls
    over past 24 h), lat, lon, alt_gps, alt_baro (metres)."""
    lines = [line for line in _lines(data) if line[:1] == b"B" and len(line) >= 35]
    n = len(lines)
    t = np.empty(n, np.int32)
    lat = np.empty(n, np.float64)
    lon = np.empty(n, np.float64)
    gps = np.empty(n, np.int32)
    baro = np.empty(n, np.int32)
    k = 0
    for line in lines:
        try:
            tt = int(line[1:3]) * 3600 + int(line[3:5]) * 60 + int(line[5:7])
            la = int(line[7:9]) + int(line[9:14]) / 60000.0
            lo = int(line[15:18]) + int(line[18:23]) / 60000.0
            t[k] = tt
            lat[k] = -la if line[14:15] == b"S" else la
            lon[k] = -lo if line[23:24] == b"W" else lo
            baro[k] = int(line[25:30])
            gps[k] = int(line[30:35])
            k += 1
        except ValueError:
            continue
    t, lat, lon, gps, baro = t[:k], lat[:k], lon[:k], gps[:k], baro[:k]
    if k > 1:  # flights past midnight UTC
        t = t + 86400 * np.concatenate([[0], np.cumsum(np.diff(t) < -43200)])
    return {"t": t, "lat": lat, "lon": lon, "alt_gps": gps, "alt_baro": baro}
