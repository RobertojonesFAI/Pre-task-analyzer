# Competition database

Parquet tables in `database/` inside the project's Google Drive folder, built by `notebooks/04_build_catalog.ipynb` (`scripts/build_catalog.py`) from the downloads of notebooks 01–03. They can be read with pandas or queried with SQL through DuckDB, with no database server:

```python
import duckdb
db = duckdb.connect()
db.sql("SELECT * FROM read_parquet('database/tasks.parquet') LIMIT 5").df()
```

Keys are prefixed by source: `fm:` Flymaster, `as:` AirScore, `at:` Airtribune. Times published by the scoring sites (`start_time`, `ss_time`, `ess_time`, `deadline`) are local `HH:MM:SS`; `utc_offset_h` converts them to UTC when it is known. IGC fix times are UTC.

The tables hold pilot names because they are needed to link tracks, results and ratings. They stay in the private Drive folder; anything published from them uses codes and aggregates only.

## events

One row per competition.

| Column | Meaning |
|---|---|
| `event_key` | `fm:<group>`, `as:<comp_id>` or `at:<event_id>` |
| `source`, `source_id` | scoring site and its id |
| `name`, `discipline` | competition name; `PG` for paragliding |
| `country`, `place`, `lat`, `lon` | IOC country code and place when published; otherwise lat/lon of the first task's takeoff |
| `start_date`, `end_date` | dates of the competition |
| `fai_category` | `FAI 1`, `FAI 2` or empty |
| `utc_offset_h` | local time minus UTC, in hours |

## tasks

One row per task with a track package.

| Column | Meaning |
|---|---|
| `task_key`, `event_key` | task and its competition |
| `source_task_id`, `name`, `date` | id on the scoring site, task name, flying date |
| `date_from_tracks` | the date was taken from the tracks' IGC headers because the results gave none |
| `task_type` | `race`, `elapsed` or `open_distance` |
| `distance_km`, `ss_distance_km` | optimised task distance and start-to-ESS distance |
| `start_time`, `deadline`, `utc_offset_h` | first start gate and task deadline (local) |
| `n_present`, `n_launched`, `n_ess`, `n_goal`, `n_landed_out` | pilot counts from the official results |
| `day_quality` | GAP day quality (0–1) |
| `stopped`, `training` | stopped task, training day |
| `turnpoints` | the route as JSON (same content as the `turnpoints` table); some older Airtribune reports give turnpoint names and radii but no coordinates |
| `results_found` | whether official results could be read. Many Airtribune events of 2015–2020 published track archives but no per-task results |
| `track_kind` | Airtribune only: `archive` (scored IGC package) or `live_archive` (live-tracking archive) |
| `n_tracks` | unique tracks of the task |

## turnpoints

| Column | Meaning |
|---|---|
| `task_key`, `seq` | task and position in the route (0 = takeoff) |
| `name`, `type` | waypoint name; `TAKEOFF`, `SSS`, `ESS`, `GOAL` or empty for an ordinary turnpoint |
| `lat`, `lon`, `alt_m`, `radius_m` | cylinder centre, waypoint altitude and cylinder radius |

## results

One row per pilot in a task's official results.

| Column | Meaning |
|---|---|
| `result_id`, `task_key` | result and its task |
| `comp_id` | pilot's number in the competition |
| `name`, `name_norm`, `nat`, `sex` | pilot as published, normalised name, IOC nation, sex |
| `civl_id` | CIVL ID when published (Flymaster, AirScore, Airtribune registrations) |
| `pilot_key` | `civl:<id>`, or `name:<normalised name>:<nation>` when there is no CIVL ID |
| `glider`, `glider_cert` | declared glider and its class (`EN-A`…`EN-D`, `CCC`) when published |
| `status` | `goal`, `ess`, `lo` (landed out), `dnf`, `abs`, `dsq` |
| `rank`, `score` | task rank and points |
| `ss_time`, `ess_time`, `time_s` | start and ESS times (local) and speed-section time in seconds |
| `distance_km` | scored distance |
| `track_file` | AirScore only: the track's file name |

## pilots

One row per `pilot_key`: `civl_id`, latest `name`, most frequent `nat` and `sex`, and `n_results`.

## tracks

One row per IGC file in a task package.

| Column | Meaning |
|---|---|
| `track_id`, `task_key`, `source` | track, its task and source |
| `zip_path`, `member` | package on Drive and file name inside it |
| `date`, `pilot`, `glider`, `glider_id`, `comp_id`, `comp_class`, `logger` | IGC header fields (often empty) |
| `n_fixes`, `interval_s` | number of fixes and mean seconds between them |
| `t_first`, `t_last`, `duration_s` | first and last fix (seconds of day, UTC) and duration |
| `lat_first`, `lon_first`, `alt_first`, `lat_last`, `lon_last`, `alt_last` | first and last fix |
| `sha1` | hash of the fixes; equal for the same flight saved by two servers |
| `result_id`, `pilot_key`, `match` | the pilot's result and how it was found: `file` (AirScore file name), `comp_id` (competition number in the file name or IGC header), `name`, `fuzzy_name` |
| `duplicate_of` | when the same fixes appear in another file, the `track_id` of the copy kept (preferring one linked to a result, then AirScore, Flymaster, Airtribune); empty for unique tracks |
| `error` | why a file could not be read, if it could not |

Use `WHERE duplicate_of IS NULL` to count each flight once.
