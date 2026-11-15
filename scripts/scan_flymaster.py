#!/usr/bin/env python3
"""Build the catalog of Flymaster competition tasks that have a public IGC package.

Flymaster's public results pages (lt.flymaster.net/bsCompetitionFrontPage.php?idgroup=N) number
competition groups from 1 upward; most numbers are tracker fleets or tests with no tasks. This
script visits every group id (newest first), keeps the tasks that have an IGC package, and writes
one row per task:

    group, task, title, discipline, code, date, name, status, task_km, ss_ess_km, zip

Safe to stop and rerun: progress is saved next to the catalog, and the scan resumes from the
last group it finished. One request at a time with a pause between them.

    python3 scripts/scan_flymaster.py data/flymaster_catalog.csv
"""

import argparse
import csv
import html
import json
import re
import ssl
import time
import urllib.error
import urllib.request
from pathlib import Path

GROUP_URL = "https://lt.flymaster.net/bsCompetitionFrontPage.php?idgroup={group}"
USER_AGENT = "pre-task-analyzer research catalog (academic, non-commercial)"
COLUMNS = ["group", "task", "title", "discipline", "code", "date", "name", "status",
           "task_km", "ss_ess_km", "zip"]


def text(fragment):
    """Visible text of an HTML fragment, whitespace collapsed."""
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", fragment)).split())


def parse_group(group, page):
    """One dict per task card on a group's front page."""
    title = text((re.search(r"<h1>(.*?)</h1>", page, re.S) or [None, ""])[1])
    meta = text((re.search(r'class="meta">(.*?)</div>', page, re.S) or [None, ""])[1])
    parts = [p.strip() for p in meta.split("·")]
    discipline = parts[0] if parts else ""
    code = parts[1] if len(parts) > 1 else ""
    rows = []
    for card in page.split('class="task-card"')[1:]:
        task = re.search(r"taskId=(\d+)", card)
        if not task:
            continue
        date = re.search(r'class="task-date">\s*(\d{4}-\d{2}-\d{2})', card)
        name_html = (re.search(r'class="task-name">(.*?)</div>', card, re.S) or [None, ""])[1]
        badges = [text(b) for b in re.findall(r'<span class="label[^"]*">(.*?)</span>', name_html, re.S)]
        name = text(re.sub(r"<span.*?</span>", " ", name_html, flags=re.S))
        task_km = re.search(r"Task ([\d.]+) km", card)
        ss_ess = re.search(r"SS\S*ESS ([\d.]+) km", card)
        zip_link = re.search(r"task-igc-packages/\d+/\d+/\d+\.zip", card)
        rows.append({
            "group": group,
            "task": int(task.group(1)),
            "title": title,
            "discipline": discipline,
            "code": code,
            "date": date.group(1) if date else "",
            "name": name,
            "status": "; ".join(badges),
            "task_km": float(task_km.group(1)) if task_km else "",
            "ss_ess_km": float(ss_ess.group(1)) if ss_ess else "",
            "zip": 1 if zip_link else 0,
        })
    return rows


def get(url, context, retries=3):
    """Page text, or None for a 404."""
    for attempt in range(1, retries + 1):
        try:
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(request, timeout=60, context=context) as response:
                return response.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            error = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            error = e
        print(f"    retry {attempt}/{retries} ({error})", flush=True)
        time.sleep(10 * attempt)
    raise RuntimeError(f"failed after {retries} tries: {url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("catalog", type=Path, help="CSV to write (rows are appended as groups are found)")
    parser.add_argument("--first", type=int, default=8400, help="highest group id to try (default 8400)")
    parser.add_argument("--last", type=int, default=1, help="lowest group id to try (default 1)")
    parser.add_argument("--pause", type=float, default=0.5, help="seconds between requests (default 0.5)")
    args = parser.parse_args()

    state_file = args.catalog.with_suffix(".state.json")
    state = json.loads(state_file.read_text()) if state_file.exists() else {"next": args.first}
    new_file = not args.catalog.exists()
    args.catalog.parent.mkdir(parents=True, exist_ok=True)
    context = ssl.create_default_context()
    started = time.time()

    with open(args.catalog, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        group = state["next"]
        while group >= args.last:
            page = get(GROUP_URL.format(group=group), context)
            rows = parse_group(group, page) if page else []
            if rows:
                writer.writerows(rows)
                f.flush()
                print(f"group {group}: {rows[0]['title'][:60]} ({rows[0]['discipline']}), "
                      f"{len(rows)} tasks, {sum(r['zip'] for r in rows)} with IGC", flush=True)
            group -= 1
            state["next"] = group
            state_file.write_text(json.dumps(state))
            if group % 500 == 0:
                print(f"... down to group {group}, {(time.time() - started) / 60:.0f} min", flush=True)
            time.sleep(args.pause)
    print("Scan finished.")


if __name__ == "__main__":
    main()
