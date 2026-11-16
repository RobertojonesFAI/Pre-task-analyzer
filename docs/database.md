# Competition database

Parquet tables in `database/` inside the project's Google Drive folder, built by `notebooks/04_build_catalog.ipynb` (`scripts/build_catalog.py`) from the downloads of notebooks 01–03, and by `notebooks/05_build_flights.ipynb` (`scripts/build_flights.py`) from the tracks. They can be read with pandas or queried with SQL through DuckDB, with no database server:

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

## flights

One row per unique track (`duplicate_of IS NULL`), from `pta/flight.py`. Times are seconds since 00:00 UTC of the task's date (above 86,400 for flights that end after midnight UTC); heights are GPS altitudes in metres above sea level.

| Column | Meaning |
|---|---|
| `track_id`, `task_key`, `source`, `result_id`, `pilot_key` | the track and what it is linked to |
| `error` | why no flight was found (`no flight found`, an unreadable file or package); the other columns are then empty |
| `step_s`, `quality` | time step the track was resampled to (the logger's usual interval) and `high` (≤ 2 s), `medium` (≤ 5 s) or `low` |
| `t_takeoff`, `lat_takeoff`, `lon_takeoff`, `alt_takeoff` | takeoff |
| `t_landing`, `lat_landing`, `lon_landing`, `alt_landing` | landing |
| `duration_s`, `alt_max`, `max_dist_from_takeoff_m` | flight time, highest point, farthest point from takeoff (straight line) |
| `n_climbs`, `n_thermals`, `thermal_gain_m`, `mean_thermal_rate_ms` | circling climbs, the ones that are thermals, height gained in thermals and their climb rate weighted by time |
| `circling_share` | share of the flight spent circling |

How the flight is found: the track is cut at every stop (standing still for a minute at a steady altitude, with real fixes rather than a logger gap) and at logger gaps over half an hour, and the flight is the stretch that circles the most. That leaves out the drive up, the wait at launch and the retrieve drive that live-tracking archives often include. The flight ends where the final glide stops sinking at walking pace, or at the next stop.

## climbs

One row per period of sustained circling (at least 30 s and one full turn).

| Column | Meaning |
|---|---|
| `track_id`, `task_key`, `seq` | flight and order within it |
| `t_start`, `t_end`, `duration_s` | when (seconds UTC, as in `flights`) |
| `lat`, `lon` | mean position |
| `alt_base`, `alt_top`, `gain_m` | height entering the climb, highest point, gain |
| `rate_ms`, `net_rate_ms` | climb rate from entry to the top, and from entry to exit |
| `turns`, `direction` | number of turns, `left` or `right` |
| `is_thermal` | gains 0.5–8 m/s on average (weaker circling is searching; faster is an altitude error) |

## glides

One row per stretch between climbs (including takeoff to the first climb and the last climb to landing).

| Column | Meaning |
|---|---|
| `track_id`, `task_key`, `seq` | flight and order within it |
| `t_start`, `t_end`, `duration_s` | when |
| `lat_start`, `lon_start`, `alt_start`, `lat_end`, `lon_end`, `alt_end` | start and end |
| `distance_m`, `alt_loss_m`, `speed_kmh`, `glide_ratio` | straight-line distance, height lost, ground speed, distance over height lost (empty if no height was lost) |

## fixes

`fixes/<task_key with : as _>.parquet`: the fixes of every flight of a task, every 5 s from takeoff to landing, after cleaning. Columns `track_id`, `t` (seconds UTC), `lat`, `lon`, `alt` (m), `vz` (m/s, over 20 s) and `circling`. These are the input for the cell crossings and landout labels of the next steps.
