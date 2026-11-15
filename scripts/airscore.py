#!/usr/bin/env python3
"""Catalog and download the paragliding tasks published on FAI AirScore (airscore.fai.org).

AirScore lists every competition at /_get_all_comps. For every finished paragliding
competition this script reads its page, finds its tasks, and saves into OUT/<comp_id>/:

    <task>.zip           IGC package of the task (/download/igc_zip/<task>)
    <task>_results.html  task results page (/task_result/<task>), which embeds the scored
                         results and the task definition

and writes OUT/airscore_catalog.csv with one row per task. Safe to stop and rerun: files
already on disk are skipped, and a file the server keeps refusing is noted in
OUT/failed.txt and tried again on the next run. One request at a time with a pause between them.

    python3 scripts/airscore.py /path/to/airscore_igc
"""

import argparse
import csv
import html
import json
import re
import shutil
import ssl
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from download_flymaster import fetch, log_failure, zip_ok  # noqa: E402  same downloader, same rules

BASE = "https://airscore.fai.org"
USER_AGENT = "pre-task-analyzer research downloader (academic, non-commercial)"
COLUMNS = ["comp_id", "task", "comp_name", "site", "date_from", "date_to", "sanction", "comp_type"]


def text(fragment):
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", str(fragment))).split())


def get_text(url, dest, context):
    """Download url to dest (reusing the downloader) and return its text."""
    if not dest.exists():
        try:
            fetch(url, dest, context)
        except RuntimeError as e:
            log_failure(dest.parent.parent, e)
    return dest.read_text(encoding="utf-8", errors="replace") if dest.exists() else ""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path, help="folder for the downloads")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between requests (default 2)")
    parser.add_argument("--min-free-gb", type=float, default=2.0, help="stop when free space drops below this")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()

    comps_file = args.out / "airscore_comps.json"
    fetch(f"{BASE}/_get_all_comps", comps_file, context)  # refreshed every run
    comps = json.loads(comps_file.read_text())
    comps = comps.get("data", comps)
    comps = [c for c in comps if "PG.png" in str(c.get("comp_class")) and c.get("status") == "Finished"]
    print(f"{len(comps)} finished paragliding competitions on AirScore")

    rows = []
    for comp in comps:
        comp_id = comp["comp_id"]
        folder = args.out / str(comp_id)
        folder.mkdir(exist_ok=True)
        page = get_text(f"{BASE}/competition/{comp_id}", folder / "competition.html", context)
        tasks = sorted({int(t) for t in re.findall(r"/download/igc_zip/(\d+)", page)})
        for task in tasks:
            rows.append({"comp_id": comp_id, "task": task, "comp_name": text(comp.get("comp_name")),
                         "site": comp.get("comp_site"), "date_from": comp.get("date_from"),
                         "date_to": comp.get("date_to"), "sanction": comp.get("sanction"),
                         "comp_type": comp.get("comp_type")})
        time.sleep(args.pause)

    with open(args.out / "airscore_catalog.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"{len(rows)} tasks with an IGC package; catalog saved")

    total = 0
    for i, row in enumerate(rows, 1):
        folder = args.out / str(row["comp_id"])
        targets = [(f"{BASE}/download/igc_zip/{row['task']}", folder / f"{row['task']}.zip"),
                   (f"{BASE}/task_result/{row['task']}", folder / f"{row['task']}_results.html")]
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
            if dest.suffix == ".zip" and size and not zip_ok(dest):
                print(f"    {dest.name}: not a valid zip, removed")
                dest.unlink()
                size = 0
            total += size
            time.sleep(args.pause)
        print(f"[{i}/{len(rows)}] comp {row['comp_id']} task {row['task']}  {row['comp_name'][:50]}  "
              f"total {total / 1e9:.2f} GB", flush=True)
    print(f"Done: {len(rows)} tasks, {total / 1e9:.2f} GB new in {args.out}")


if __name__ == "__main__":
    main()
