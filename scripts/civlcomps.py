#!/usr/bin/env python3
"""Catalog and download the paragliding XC competitions published on CIVL's event portal (civlcomps.org).

Used with the permission of CIVL.

Organisers who publish their results on civlcomps.org upload the FS results of every task and,
since 2024, a zip of the task's IGC files. For every past paragliding XC event from --from-year on,
this script reads the event's results page and saves into OUT/<event_id>/:

    results.html               the event's results page (one row per task)
    participants.html          the registered pilots: CIVL ID, competition number, glider
    <task>.zip                 the task's IGC files (only tasks that have them)
    <task>_results.html        the task's FS results: pilots, turnpoints with coordinates, task
                               statistics and GAP parameters (the overall category when there is one)

where <task> is the task's position on the results page (1, 2, ...). It writes
OUT/civlcomps_catalog.csv with one row per task that has an IGC zip. Safe to stop and rerun: files
already on disk are skipped (event pages are fetched again while the event is recent, in case
results were added late), and a file the server keeps refusing is noted in OUT/failed.txt and tried
again on the next run. One request at a time with a pause between them.

    python3 scripts/civlcomps.py /path/to/civlcomps_igc
    python3 scripts/civlcomps.py /path/to/civlcomps_igc --from-year 2015 --all-tasks   # results without tracks too
"""

import argparse
import csv
import datetime as dt
import json
import re
import shutil
import ssl
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

import lxml.html

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_flymaster import USER_AGENT, fetch, log_failure, zip_ok  # noqa: E402  same downloader, same rules

BASE = "https://civlcomps.org"
COLUMNS = ["event_id", "slug", "title", "country", "place", "lat", "lon", "start_date", "end_date",
           "fai_category", "task", "task_name", "date", "distance_km", "task_type", "zip_url",
           "results_url", "results_category", "replay_url"]
RECENT_DAYS = 60   # event pages younger than this are fetched again on every run (late uploads)


def get_json(url, context, retries=3):
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={
                "User-Agent": USER_AGENT, "X-Requested-With": "XMLHttpRequest", "Accept": "application/json"})
            with urllib.request.urlopen(request, timeout=120, context=context) as response:
                return json.load(response)
        except Exception as e:  # noqa: BLE001  network or decoding error: wait and retry
            error = e
            time.sleep(10 * attempt)
    raise RuntimeError(f"failed after {retries} tries: {url} ({error})")


def events_of_year(year, context, pause):
    """Every paragliding XC event of one year, from the JSON the events calendar loads page by page."""
    events, page = [], 0
    while True:
        query = urllib.parse.urlencode({
            "search[mode]": "list", "search[pg]": "paragliding-xc", "search[dates]": "custom",
            "search[dates_range]": f"{year}-01-01 - {year}-12-31", "search[page]": page})
        data = get_json(f"{BASE}/events/paragliding-xc?{query}", context)
        batch = data.get("events") or []
        events += batch
        if not batch or len(events) >= int(data.get("totalCount") or 0) or page >= int(data.get("pagesCount") or 0):
            return events
        page += 1
        time.sleep(pause)


def _text(el):
    return re.sub(r"\s+", " ", el.text_content().replace("\xa0", " ")).strip() if el is not None else ""


def task_rows(page):
    """The tasks listed on an event's results page, in order: date, name, distance, type, IGC zip,
    replay and the results links ({category: url}) of each."""
    if "task-list" not in page:
        return []
    doc = lxml.html.fromstring(page)
    tasks = []
    for tr in doc.xpath('//table[contains(@class, "task-list")]//tr[td]'):
        cells = tr.xpath("./td")
        head = _text(cells[0].xpath('.//div[contains(@class, "date-item")]')[0]) if cells[0].xpath(
            './/div[contains(@class, "date-item")]') else ""
        m = re.match(r"(\d{4}-\d{2}-\d{2})\s*(.*)", head)
        items = [_text(d) for d in cells[0].xpath('.//div[contains(@class, "task-item")]')]
        dist = next((re.match(r"([\d.]+)\s*km\s*(.*)", i) for i in items if re.match(r"[\d.]+\s*km", i)), None)
        igc = next((a.get("href") for a in cells[0].xpath(".//a[@href]") if _text(a).upper() == "IGC"), None)
        replay = next((a.get("href") for a in cells[0].xpath(".//a[@href]") if _text(a).lower() == "replay"), None)
        results = {}
        if len(cells) > 1:  # the "Task results" cell (the next cell holds competition totals)
            for a in cells[1].xpath('.//a[contains(@class, "jsResultsLink")]'):
                results.setdefault(_text(a), a.get("href"))
        tasks.append({
            "date": m.group(1) if m else None, "task_name": (m.group(2) if m else head) or None,
            "distance_km": float(dist.group(1)) if dist else None, "task_type": dist.group(2) if dist else None,
            "zip_url": igc, "replay_url": replay, "results": results,
        })
    return tasks


def results_choice(results):
    """(category, url) of the task results to keep: the overall category when there is one."""
    if not results:
        return None, None
    for name, url in results.items():
        if "overall" in name.lower() and "female" not in name.lower() and "women" not in name.lower():
            return name, url
    for name, url in results.items():
        if any(w in name.lower() for w in ("open", "general", "all")):
            return name, url
    return next(iter(results.items()))


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="folder for the downloads")
    parser.add_argument("--from-year", type=int, default=2023, help="first year to read (IGC uploads began in 2024)")
    parser.add_argument("--all-tasks", action="store_true",
                        help="also catalog and fetch the results of tasks without an IGC zip (for pilot ratings)")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between requests (default 2)")
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="stop when free space drops below this")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    today = dt.date.today()

    events = []
    for year in range(args.from_year, today.year + 1):
        batch = events_of_year(year, context, args.pause)
        print(f"{year}: {len(batch)} paragliding XC events", flush=True)
        events += batch
        time.sleep(args.pause)
    seen, unique = set(), []
    for e in events:
        end = dt.datetime.strptime(e["end"], "%Y/%m/%d").date() if e.get("end") else None
        if e["id"] not in seen and end and end < today and "cancel" not in e.get("eventTitle", "").lower():
            seen.add(e["id"])
            unique.append((e, end))
    print(f"{len(unique)} finished events", flush=True)

    rows = []
    for n, (event, end) in enumerate(unique, 1):
        folder = args.out / str(event["id"])
        folder.mkdir(exist_ok=True)
        dest = folder / "results.html"
        link = event["eventLink"].rstrip("/")
        stale = dest.exists() and (today - end).days < RECENT_DAYS
        if not dest.exists() or stale:
            try:
                fetch(f"{link}/results", dest, context)
            except RuntimeError as e:
                log_failure(args.out, e)
            time.sleep(args.pause)
        page = dest.read_text(encoding="utf-8", errors="replace") if dest.exists() else ""
        for i, task in enumerate(task_rows(page), 1):
            if not task["zip_url"] and not args.all_tasks:
                continue
            category, url = results_choice(task["results"])
            rows.append({
                "event_id": event["id"], "slug": link.rsplit("/", 1)[-1], "title": event.get("eventTitle", "").strip(),
                "country": (event.get("countryIso3") or "").upper(), "place": event.get("cityTitle"),
                "lat": event.get("latitude"), "lon": event.get("longitude"),
                "start_date": event["start"].replace("/", "-"), "end_date": event["end"].replace("/", "-"),
                "fai_category": event.get("faiCategory"), "task": i, "task_name": task["task_name"],
                "date": task["date"], "distance_km": task["distance_km"], "task_type": task["task_type"],
                "zip_url": task["zip_url"], "results_url": url, "results_category": category,
                "replay_url": task["replay_url"],
            })
        if n % 50 == 0:
            print(f"... read {n}/{len(unique)} events, {sum(1 for r in rows if r['zip_url'])} tasks with IGC", flush=True)

    with open(args.out / "civlcomps_catalog.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    with_igc = [r for r in rows if r["zip_url"]]
    print(f"{len(with_igc)} tasks with IGC in {len({r['event_id'] for r in with_igc})} events"
          + (f" ({len(rows)} tasks in all)" if args.all_tasks else "") + "; catalog saved", flush=True)

    total = 0
    events_done = set()
    for i, row in enumerate(rows, 1):
        folder = args.out / str(row["event_id"])
        targets = []
        if row["event_id"] not in events_done:
            targets.append((f"{BASE}/event/{row['slug']}/participants", folder / "participants.html"))
            events_done.add(row["event_id"])
        if row["zip_url"]:
            targets.append((row["zip_url"], folder / f"{row['task']}.zip"))
        if row["results_url"]:
            targets.append((row["results_url"], folder / f"{row['task']}_results.html"))
        for url, dest in targets:
            if dest.exists() and (dest.suffix != ".zip" or zip_ok(dest)):
                continue
            free_gb = shutil.disk_usage(args.out).free / 1e9
            if free_gb < args.min_free_gb:
                sys.exit(f"Stopping: only {free_gb:.1f} GB free on the destination disk.")
            try:
                size = fetch(url, dest, context)
            except RuntimeError as e:
                log_failure(args.out, e)
                continue
            if dest.suffix == ".zip" and dest.exists() and not zip_ok(dest):
                print(f"    {dest.name}: empty or not a valid zip, removed")
                dest.unlink()
                size = 0
            total += size
            time.sleep(args.pause)
        print(f"[{i}/{len(rows)}] {row['date']} {row['title'][:50]} task {row['task']}  "
              f"total {total / 1e9:.2f} GB", flush=True)
    print(f"Done: {len(rows)} tasks, {total / 1e9:.2f} GB new in {args.out}")


if __name__ == "__main__":
    main()
