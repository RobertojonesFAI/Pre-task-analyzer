#!/usr/bin/env python3
"""Catalog and download the paragliding competitions published on Airtribune (airtribune.com).

Used with the permission of the Airtribune administrators.

The list of past events is embedded in https://airtribune.com/events/past. For every
paragliding event this script reads its results page, finds the track archive of each task,
and saves into OUT/<event_id>/:

    results.html               the event's results page
    <task>.zip                 the task's track archive: the scored IGC package when there is one
                               (media/task/archive/...), otherwise the live-tracking archive
                               (media/task/live_archive/...)
    <task>_results.html        the task's results page (/results/task<task>/day/overall)

and writes OUT/airtribune_catalog.csv with one row per task. Safe to stop and rerun: files already
on disk are skipped, and a file the server keeps refusing is noted in OUT/failed.txt and tried
again on the next run. One request at a time with a pause between them.

    python3 scripts/airtribune.py /path/to/airtribune_igc
"""

import argparse
import csv
import json
import re
import shutil
import ssl
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_flymaster import fetch, log_failure, zip_ok  # noqa: E402  same downloader, same rules

BASE = "https://airtribune.com"
PARAGLIDING = 0  # Airtribune's "sport" code; 2 is hang gliding
ZIP_RE = re.compile(r"https?://[^\"' ]+?/media/task/(archive|live_archive)/\d{4}/\d{2}/(\d+)/[^\"' ]+?\.zip")
COLUMNS = ["event_id", "slug", "title", "country", "place", "lat", "lon", "start_date", "end_date",
           "fai_category", "task", "kind", "zip_url"]


def event_list(page):
    """The past events embedded in the events page as window.ATDATA.eventLists = {...}."""
    start = page.index("window.ATDATA.eventLists")
    start = page.index("{", start)
    lists, _ = json.JSONDecoder().raw_decode(page[start:])
    return lists["past-events"]["content"]


def read(url, dest, out, context):
    """Text of url, saved at dest (downloaded once); empty if it cannot be fetched."""
    if not dest.exists():
        try:
            fetch(url, dest, context)
        except RuntimeError as e:
            log_failure(out, e)
    return dest.read_text(encoding="utf-8", errors="replace") if dest.exists() else ""


def task_archives(page):
    """{task id: (kind, url)}, preferring the scored IGC package over the live archive."""
    found = {}
    for match in ZIP_RE.finditer(page):
        kind, task = match.group(1), int(match.group(2))
        if task not in found or (kind == "archive" and found[task][0] != "archive"):
            found[task] = (kind, match.group(0))
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="folder for the downloads")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between requests (default 2)")
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="stop when free space drops below this")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()

    events_page = args.out / "events_past.html"
    fetch(f"{BASE}/events/past", events_page, context)  # refreshed every run
    events = [e for e in event_list(events_page.read_text(encoding="utf-8")) if e.get("sport") == PARAGLIDING]
    events.sort(key=lambda e: e.get("start_date") or "", reverse=True)
    print(f"{len(events)} past paragliding events on Airtribune", flush=True)

    catalog_file = args.out / "airtribune_catalog.csv"
    rows = []
    for n, event in enumerate(events, 1):
        folder = args.out / str(event["id"])
        folder.mkdir(exist_ok=True)
        new = not (folder / "results.html").exists()
        page = read(f"{BASE}{event['url']}results", folder / "results.html", args.out, context)
        for task, (kind, url) in sorted(task_archives(page).items()):
            rows.append({"event_id": event["id"], "slug": event["url"].strip("/"), "title": event["title"].strip(),
                         "country": (event.get("country") or {}).get("ioc_code", ""), "place": event.get("place"),
                         "lat": event.get("lat"), "lon": event.get("lon"), "start_date": event.get("start_date"),
                         "end_date": event.get("end_date"), "fai_category": event.get("fai_category"),
                         "task": task, "kind": kind, "zip_url": url})
        if new:
            time.sleep(args.pause)
        if n % 50 == 0:
            print(f"... read {n}/{len(events)} events, {len(rows)} tasks with a track archive", flush=True)

    with open(catalog_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} tasks with a track archive in {len({r['event_id'] for r in rows})} events; "
          f"catalog saved", flush=True)

    total = 0
    for i, row in enumerate(rows, 1):
        folder = args.out / str(row["event_id"])
        targets = [(row["zip_url"], folder / f"{row['task']}.zip"),
                   (f"{BASE}/{row['slug']}/results/task{row['task']}/day/overall", folder / f"{row['task']}_results.html")]
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
        print(f"[{i}/{len(rows)}] {row['start_date']} {row['title'][:50]} task {row['task']} ({row['kind']})  "
              f"total {total / 1e9:.2f} GB", flush=True)
    print(f"Done: {len(rows)} tasks, {total / 1e9:.2f} GB new in {args.out}")


if __name__ == "__main__":
    main()
