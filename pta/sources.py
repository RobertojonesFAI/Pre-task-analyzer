"""Turn the pages saved by the three download scripts into common event, task and result rows.

Every parser returns plain dicts with the same keys whatever the source:

    event   event_key, source, source_id, name, discipline, country, place, lat, lon,
            start_date, end_date, fai_category, utc_offset_h
    task    task_key, event_key, source, source_task_id, name, date, task_type, distance_km,
            ss_distance_km, start_time, deadline, utc_offset_h, n_present, n_launched, n_ess,
            n_goal, n_landed_out, day_quality, stopped, training, turnpoints (list of dicts:
            seq, name, type, lat, lon, radius_m, alt_m)
    result  task_key, comp_id, name, nat, sex, civl_id, glider, glider_cert, status, rank,
            ss_time, ess_time, time_s, distance_km, score, track_file

Times (start_time, deadline, ss_time, ess_time) are local "HH:MM:SS" strings as published;
utc_offset_h converts them to UTC when it is known.
"""

import json
import re
from pathlib import Path

from .util import hms, js_value, norm_name, pair_table, read_tables, strip_tags, to_float, to_int

CERT = re.compile(r"^(EN[- ]?[A-D]|CCC|LTF ?[A-D1-2/]+|DHV ?[1-3/]+|[A-D])$", re.I)
TASK_TYPES = {"RACE": "race", "ELAPSED-TIME": "elapsed", "ELAPSED_TIME": "elapsed", "ELAPSED": "elapsed",
              "OPEN-DISTANCE": "open_distance"}


STATUS = {"goal": "goal", "ess": "ess", "lo": "lo", "landed out": "lo", "mindist": "lo", "min dist": "lo",
          "dnf": "dnf", "did not fly": "dnf", "abs": "abs", "absent": "abs", "dsq": "dsq", "nyp": "nyp"}


def _status(value):
    """Result status in common words: goal, ess, lo (landed out), dnf, abs, dsq, nyp."""
    s = strip_tags(value).lower()
    return STATUS.get(s, s or None)


def _cert(value):
    """Normalised certification ('en c' -> 'EN-C', 'ccc' -> 'CCC'), or None."""
    s = strip_tags(value).upper().replace(" ", "")
    if not s or not CERT.match(s):
        return None
    if s in ("A", "B", "C", "D"):
        return "EN-" + s
    if s.startswith("EN"):
        return "EN-" + s[-1]
    return s


def _offset(local, utc):
    """UTC offset in hours from a local and a UTC clock time (seconds of day)."""
    if local is None or utc is None:
        return None
    h = round((local - utc) / 1800) / 2  # nearest half hour
    return h - 24 if h > 14 else h + 24 if h < -12 else h


def _read(path):
    p = Path(path)
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


# ── Flymaster ───────────────────────────────────────────────────────────────

def flymaster_task(group, task, folder, catalog_row):
    """Task and result rows of one Flymaster task from its saved results page and .xctsk."""
    folder = Path(folder)
    page = _read(folder / f"{task}_results.html")
    tables = read_tables(page)
    info = pair_table(tables)
    xctsk = {}
    try:
        xctsk = json.loads(_read(folder / f"{task}.xctsk") or "{}")
    except json.JSONDecodeError:
        pass

    turnpoints = []
    for i, tp in enumerate(xctsk.get("turnpoints", [])):
        w = tp.get("waypoint", {})
        turnpoints.append({"seq": i, "name": w.get("name"), "type": (tp.get("type") or "").upper(),
                           "lat": w.get("lat"), "lon": w.get("lon"), "radius_m": to_float(tp.get("radius")),
                           "alt_m": to_float(w.get("altSmoothed"))})
    if turnpoints and not any(t["type"] == "GOAL" for t in turnpoints):
        turnpoints[-1]["type"] = turnpoints[-1]["type"] or "GOAL"

    start_local = hms(info.get("Start time"))
    gates = (xctsk.get("sss") or {}).get("timeGates") or []
    start_utc = hms(gates[0]) if gates else None
    task_key = f"fm:{task}"
    sss_type = ((xctsk.get("sss") or {}).get("type") or "RACE").upper()
    row = {
        "task_key": task_key, "event_key": f"fm:{group}", "source": "flymaster", "source_task_id": str(task),
        "name": info.get("Name") or catalog_row.get("name"), "date": info.get("Date") or catalog_row.get("date"),
        "task_type": TASK_TYPES.get(sss_type, sss_type.lower()),
        "distance_km": to_float(info.get("Task distance")) or to_float(catalog_row.get("task_km")),
        "ss_distance_km": to_float(info.get("SS→ESS dist")) or to_float(catalog_row.get("ss_ess_km")),
        "start_time": strip_tags(info.get("Start time")) or None, "deadline": None,
        "utc_offset_h": _offset(start_local, start_utc),
        "n_present": to_int(info.get("Pilots present")), "n_launched": to_int(info.get("Launched")),
        "n_ess": to_int(info.get("ESS")), "n_goal": to_int(info.get("Goal")),
        "n_landed_out": to_int(info.get("Landed out")), "day_quality": to_float(info.get("Day quality")),
        "stopped": "stop" in str(catalog_row.get("name", "")).lower(),
        "training": "training" in str(catalog_row.get("status", "")).lower(),
        "turnpoints": turnpoints, "results_found": False,
    }

    results = []
    res = next((t for t in tables if "Glider" in t.columns and "Name" in t.columns), None)
    if res is not None:
        sex_col = next((c for c in res.columns if str(c).startswith("Unnamed")), None)
        for _, r in res.iterrows():
            name = r.get("Name")
            if not isinstance(name, str) or not name.strip():
                continue
            results.append({
                "task_key": task_key, "comp_id": str(to_int(r.get("Id"))) if to_int(r.get("Id")) is not None else None,
                "name": name.strip(), "nat": str(r.get("Nation", "")).upper() or None,
                "sex": r.get(sex_col) if sex_col else None, "civl_id": to_int(r.get("CIVL id")),
                "glider": r.get("Glider") if isinstance(r.get("Glider"), str) else None, "glider_cert": None,
                "status": _status(r.get("Flight")), "rank": to_int(r.get("#")),
                "ss_time": strip_tags(r.get("SS")) or None, "ess_time": strip_tags(r.get("ESS")) or None,
                "time_s": hms(r.get("Time [h:m:s]")), "distance_km": to_float(r.get("Dist [km]")),
                "score": to_float(r.get("Total")), "track_file": None,
            })
        row["results_found"] = bool(results)
    return row, results


def flymaster_event(group, catalog_rows, first_task):
    tp = (first_task or {}).get("turnpoints") or [{}]
    dates = sorted(r["date"] for r in catalog_rows if r.get("date"))
    c = catalog_rows[0]
    return {"event_key": f"fm:{group}", "source": "flymaster", "source_id": str(group), "name": c.get("title"),
            "discipline": c.get("discipline") or "PG", "country": None, "place": None,
            "lat": tp[0].get("lat"), "lon": tp[0].get("lon"),
            "start_date": dates[0] if dates else None, "end_date": dates[-1] if dates else None,
            "fai_category": None, "utc_offset_h": (first_task or {}).get("utc_offset_h")}


def flymaster_track_id(member):
    """Competition id of a Flymaster track from its file name ('378_George_Milheim.igc' -> '378')."""
    m = re.match(r"^(?:.*/)?(\d+)_", member)
    return m.group(1) if m else None


# ── AirScore ────────────────────────────────────────────────────────────────

def _airscore_radius(value):
    s = strip_tags(value).lower()
    v = to_float(s)
    return v * 1000 if v is not None and "km" in s else v


def airscore_task(comp_id, task, folder):
    page = _read(Path(folder) / f"{task}_results.html")
    data = js_value(page, "populate_task(") or {}
    info, stats = data.get("info") or {}, data.get("stats") or {}
    route = data.get("route") or []
    if isinstance(route, str):
        try:
            route = json.loads(route)
        except json.JSONDecodeError:
            route = []
    kinds = {"launch": "TAKEOFF", "ss": "SSS", "es": "ESS", "goal": "GOAL"}
    turnpoints = [{"seq": i, "name": w.get("name"), "type": kinds.get(str(w.get("type", "")).lower(), ""),
                   "lat": to_float(w.get("lat")), "lon": to_float(w.get("lon")),
                   "radius_m": _airscore_radius(w.get("radius")), "alt_m": to_float(w.get("altitude"))}
                  for i, w in enumerate(route)]
    offset = info.get("time_offset")
    task_key = f"as:{task}"
    row = {
        "task_key": task_key, "event_key": f"as:{comp_id}", "source": "airscore", "source_task_id": str(task),
        "name": info.get("task_name"), "date": info.get("date"),
        "task_type": str(info.get("task_type") or "race").lower(),
        "distance_km": to_float(info.get("opt_dist")) or to_float(info.get("distance")),
        "ss_distance_km": to_float(info.get("SS_distance")),
        "start_time": info.get("start_time") or None, "deadline": info.get("task_deadline") or None,
        "utc_offset_h": hms(offset) / 3600 if hms(offset) is not None else None,
        "n_present": to_int(stats.get("pilots_present")), "n_launched": to_int(stats.get("pilots_launched")),
        "n_ess": to_int(stats.get("pilots_ess")), "n_goal": to_int(stats.get("pilots_goal")),
        "n_landed_out": None, "day_quality": to_float(stats.get("day_quality")),
        "stopped": bool(info.get("stopped_time")), "training": str(info.get("training")) == "1",
        "turnpoints": turnpoints, "results_found": False,
    }
    if row["n_launched"] is not None and row["n_ess"] is not None:
        row["n_landed_out"] = row["n_launched"] - row["n_ess"]

    results = []
    for r in data.get("results") or []:
        certs = [_cert(v) for v in [r.get("glider_cert"), *(r.get("custom") or {}).values()]]
        results.append({
            "task_key": task_key, "comp_id": str(r.get("ID")) if r.get("ID") else None,
            "name": strip_tags(r.get("name")), "nat": (r.get("nat") or "").upper() or None, "sex": r.get("sex"),
            "civl_id": to_int(r.get("civl_id")), "glider": strip_tags(r.get("glider")) or None,
            "glider_cert": next((c for c in certs if c), None),
            "status": _status(r.get("result_type")), "rank": to_int(r.get("rank")),
            "ss_time": strip_tags(r.get("SSS_time")) or None,
            "ess_time": None if "<del>" in str(r.get("ESS_time")) else strip_tags(r.get("ESS_time")) or None,
            "time_s": None if "<del>" in str(r.get("ss_time")) else hms(r.get("ss_time")),
            "distance_km": to_float(r.get("distance")), "score": to_float(r.get("score")),
            "track_file": r.get("track_file") or None,
        })
    row["results_found"] = bool(results)
    return row, results


def airscore_event(comp, first_task):
    tp = (first_task or {}).get("turnpoints") or [{}]
    return {"event_key": f"as:{comp['comp_id']}", "source": "airscore", "source_id": str(comp["comp_id"]),
            "name": strip_tags(comp.get("comp_name")), "discipline": "PG", "country": None,
            "place": comp.get("comp_site"), "lat": tp[0].get("lat"), "lon": tp[0].get("lon"),
            "start_date": comp.get("date_from"), "end_date": comp.get("date_to"),
            "fai_category": comp.get("sanction"), "utc_offset_h": (first_task or {}).get("utc_offset_h")}


# ── Airtribune ──────────────────────────────────────────────────────────────

def airtribune_events(events_page):
    """{event id: event dict from Airtribune's list of past events}."""
    lists = js_value(events_page, "window.ATDATA.eventLists") or {}
    return {e["id"]: e for e in (lists.get("past-events") or {}).get("content", [])}


def airtribune_event(e, first_task=None):
    tp = (first_task or {}).get("turnpoints") or [{}]
    offset = e.get("timeoffset")
    return {"event_key": f"at:{e['id']}", "source": "airtribune", "source_id": str(e["id"]),
            "name": (e.get("title") or "").strip(), "discipline": "PG",
            "country": ((e.get("country") or {}).get("ioc_code")), "place": e.get("place"),
            "lat": e.get("lat") if e.get("lat") is not None else tp[0].get("lat"),
            "lon": e.get("lon") if e.get("lon") is not None else tp[0].get("lon"),
            "start_date": e.get("start_date"), "end_date": e.get("end_date"),
            "fai_category": f"FAI {e['fai_category']}" if e.get("fai_category") in (1, 2) else None,
            "utc_offset_h": offset / 3600 if isinstance(offset, (int, float)) else None}


def airtribune_pilots(results_page):
    """{normalised name: registration} from window.ATDATA.pilots on an event's results page."""
    data = js_value(results_page, "window.ATDATA.pilots") or {}
    out = {}
    for p in data.get("pilots") or []:
        country = p.get("country") or {}
        model = " ".join(x for x in [p.get("glider_manufacturer"), p.get("glider_model")] if x)
        out[norm_name(p.get("name"))] = {
            "civl_id": to_int(p.get("civl_id_official") or p.get("civl_id")),
            "nat": country.get("ioc_code"), "sex": p.get("gender"), "glider": model or None,
            "glider_cert": _cert(p.get("glider_certification")),
        }
    return out


def airtribune_task_links(results_page, task):
    """Category names that have a day result page for a task, e.g. ['overall', 'sport']."""
    return list(dict.fromkeys(re.findall(rf"/results/task{task}/day/([\w-]+)", results_page)))


def airtribune_task(event_id, task, folder, pilots, zip_url=""):
    page = _read(Path(folder) / f"{task}_results.html")
    tables = read_tables(page)
    params = pair_table(tables)
    turnpoints = []
    tp_table = next((t for t in tables if "Coordinates" in t.columns), None)
    if tp_table is not None:
        for i, r in tp_table.reset_index(drop=True).iterrows():
            m = re.search(r"Lat:\s*([-\d.]+)\s*Lon:\s*([-\d.]+)", str(r.get("Coordinates")))
            no = str(r.get("No", ""))
            kind = "SSS" if "SS" in no else "ESS" if "ES" in no else ""
            turnpoints.append({"seq": i, "name": r.get("Id"), "type": kind,
                               "lat": float(m.group(1)) if m else None, "lon": float(m.group(2)) if m else None,
                               "radius_m": to_float(r.get("Radius")), "alt_m": to_float(r.get("Altitude"))})
        if turnpoints:
            turnpoints[0]["type"] = turnpoints[0]["type"] or "TAKEOFF"
            turnpoints[-1]["type"] = "GOAL"

    first_start = str(params.get("first_start_time") or "")
    date = first_start[:10] if re.match(r"\d{4}-\d{2}-\d{2}", first_start) else None
    if not date:  # from the archive's file name: igc7706_2026-06-22.zip or 08-01-2018-live.zip
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})\.zip", zip_url) or re.search(r"(\d{2})-(\d{2})-(\d{4})-live", zip_url)
        if m:
            date = "-".join(m.groups()) if len(m.group(1)) == 4 else f"{m.group(3)}-{m.group(2)}-{m.group(1)}"
    offset = re.search(r"([+-]\d{2}):(\d{2})$", first_start)
    task_distance = to_float(params.get("task_distance"))
    task_key = f"at:{task}"
    utc_offset = None
    if offset:
        sign = -1 if offset.group(1).startswith("-") else 1
        utc_offset = int(offset.group(1)) + sign * int(offset.group(2)) / 60
    row = {
        "task_key": task_key, "event_key": f"at:{event_id}", "source": "airtribune", "source_task_id": str(task),
        "name": None, "date": date, "task_type": "race",
        "distance_km": task_distance, "ss_distance_km": to_float(params.get("ss_distance")),
        "start_time": first_start[11:19] or None, "deadline": None,
        "utc_offset_h": utc_offset,
        "n_present": to_int(params.get("no_of_pilots_present")), "n_launched": to_int(params.get("no_of_pilots_flying")),
        "n_ess": to_int(params.get("no_of_pilots_reaching_es")), "n_goal": to_int(params.get("no_of_pilots_reaching_goal")),
        "n_landed_out": to_int(params.get("no_of_pilots_lo")), "day_quality": to_float(params.get("day_quality")),
        "stopped": (to_int(params.get("no_of_pilots_landed_before_stop")) or 0) > 0, "training": False,
        "turnpoints": turnpoints, "results_found": False,
    }

    results = []
    res = next((t for t in tables if "Glider" in t.columns and "Name" in t.columns), None)
    if res is not None:
        sex_col = next((c for c in res.columns if str(c).startswith("Unnamed")), None)
        col = lambda *names: next((c for c in res.columns if any(str(c).startswith(n) for n in names)), None)  # noqa: E731
        c_time, c_dist, c_es = col("Time"), col("Distance"), col("ES")
        for _, r in res.iterrows():
            name = r.get("Name")
            if not isinstance(name, str) or not name.strip():
                continue
            reg = pilots.get(norm_name(name), {})
            distance = to_float(r.get(c_dist)) if c_dist else None
            ess = strip_tags(r.get(c_es)) if c_es else ""
            goal = bool(ess) and task_distance is not None and distance is not None and distance >= 0.999 * task_distance
            results.append({
                "task_key": task_key, "comp_id": str(to_int(r.get("Id"))) if to_int(r.get("Id")) is not None else None,
                "name": name.strip(), "nat": str(r.get("Nat", "")).upper() or reg.get("nat"),
                "sex": (r.get(sex_col) if sex_col else None) or reg.get("sex"), "civl_id": reg.get("civl_id"),
                "glider": reg.get("glider") or (r.get("Glider") if isinstance(r.get("Glider"), str) else None),
                "glider_cert": reg.get("glider_cert"),
                "status": "goal" if goal else "ess" if ess else "lo" if distance else "dnf", "rank": to_int(r.get("#")),
                "ss_time": strip_tags(r.get("SS")) or None, "ess_time": ess or None,
                "time_s": hms(r.get(c_time)) if c_time else None, "distance_km": distance,
                "score": to_float(r.get("Total")), "track_file": None,
            })
        row["results_found"] = bool(results)
    return row, results


def airtribune_track_id(member):
    """Competition id of an Airtribune track from its file name
    ('Bill Hughes.20220626-191500.941.igc' -> '941')."""
    m = re.search(r"\.(\d+)\.igc$", member, re.I)
    return m.group(1) if m else None
