#!/usr/bin/env python3
"""Download Flymaster competition tasks: IGC packages, task definitions and results pages.

Reads the catalog made by the group scan (one row per task: group, task, title, date) and,
for every task, saves into OUT/<group>/:

    <task>.zip           IGC package of the task (every pilot's track)
    <task>.xctsk         task definition (turnpoints, cylinders, start, goal)
    <task>_results.html  official task results page (ranks, goal, times)

Safe to stop and rerun: files already on disk are skipped. One request at a time with a pause
between them, so the servers are not loaded. Stops before the disk fills up.

Uses only the Python standard library.

    python3 scripts/download_flymaster.py data/flymaster_catalog.csv /Volumes/Data/flymaster
"""

import argparse
import csv
import shutil
import ssl
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://s3.eu-west-1.amazonaws.com/flights.flymaster.net/task-igc-packages/{group}/{task}/{task}.zip"
XCTSK_URL = "https://lt.flymaster.net/task.php?task_id={task}&format=xctsk"
RESULTS_URL = "https://lt.flymaster.net/bsTaskResultsFiltered.php?idgroup={group}&taskId={task}"
USER_AGENT = "pre-task-analyzer research downloader (academic, non-commercial)"


def fetch(url, dest, context, retries=3):
    """Download url to dest through a temporary file. Returns bytes written, 0 for a 404."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=120, context=context) as response, open(tmp, "wb") as f:
                shutil.copyfileobj(response, f, 1 << 20)
            tmp.replace(dest)
            return dest.stat().st_size
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                tmp.unlink(missing_ok=True)
                return 0
            error = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = e
        tmp.unlink(missing_ok=True)
        wait = 10 * attempt
        print(f"    retry {attempt}/{retries} in {wait} s ({error})", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"failed after {retries} tries: {url}")


def zip_ok(path):
    try:
        with zipfile.ZipFile(path) as z:
            return len(z.namelist()) > 0
    except zipfile.BadZipFile:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("catalog", type=Path, help="CSV with columns group, task (from the group scan)")
    parser.add_argument("out", type=Path, help="folder for the downloads (an external drive is best)")
    parser.add_argument("--pause", type=float, default=2.0, help="seconds between requests (default 2)")
    parser.add_argument("--min-free-gb", type=float, default=5.0, help="stop when free space drops below this")
    parser.add_argument("--only", choices=["zip", "xctsk", "results"], help="download one kind of file only")
    args = parser.parse_args()

    rows = list(csv.DictReader(open(args.catalog, newline="", encoding="utf-8")))
    args.out.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    kinds = [args.only] if args.only else ["zip", "xctsk", "results"]
    total = 0
    started = time.time()

    for i, row in enumerate(rows, 1):
        group, task = row["group"], row["task"]
        folder = args.out / group
        folder.mkdir(exist_ok=True)
        targets = {
            "zip": (ZIP_URL, folder / f"{task}.zip"),
            "xctsk": (XCTSK_URL, folder / f"{task}.xctsk"),
            "results": (RESULTS_URL, folder / f"{task}_results.html"),
        }
        for kind in kinds:
            url_pattern, dest = targets[kind]
            if dest.exists() and (kind != "zip" or zip_ok(dest)):
                continue
            free_gb = shutil.disk_usage(args.out).free / 1e9
            if free_gb < args.min_free_gb:
                print(f"Stopping: only {free_gb:.1f} GB free on the destination disk.")
                sys.exit(1)
            size = fetch(url_pattern.format(group=group, task=task), dest, context)
            if kind == "zip" and size and not zip_ok(dest):
                print(f"    {dest.name}: not a valid zip, removed")
                dest.unlink()
                size = 0
            total += size
            time.sleep(args.pause)
        elapsed = (time.time() - started) / 60
        print(f"[{i}/{len(rows)}] group {group} task {task}  {row.get('title', '')[:50]}  "
              f"total {total / 1e9:.2f} GB, {elapsed:.0f} min", flush=True)

    print(f"Done: {len(rows)} tasks, {total / 1e9:.2f} GB new in {args.out}")


if __name__ == "__main__":
    try:
        main()
    except ssl.SSLError as e:
        sys.exit(f"SSL error ({e}). On a python.org install of Python, run "
                 f"'Install Certificates.command' from its Applications folder once, then retry.")
