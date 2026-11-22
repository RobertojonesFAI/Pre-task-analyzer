# Competition database

Parquet tables in `database/` inside the project's Google Drive folder, built by `notebooks/04_build_catalog.ipynb` (`scripts/build_catalog.py`) from the downloads of notebooks 01–03, by `notebooks/05_build_flights.ipynb` (`scripts/build_flights.py`) from the tracks, by `notebooks/06_sites_terrain.ipynb` (`scripts/build_sites.py`, `scripts/build_terrain.py`) from the flights and the Copernicus GLO-30 elevation model, and by `notebooks/07_weather_era5.ipynb` and `07b`–`07d` (`scripts/era5.py`, `scripts/era5_arco.py`) from ERA5. They can be read with pandas or queried with SQL through DuckDB, with no database server:

```python
import duckdb
db = duckdb.connect()
db.sql("SELECT * FROM read_parquet('database/tasks.parquet') LIMIT 5").df()
```

Keys are prefixed by source: `fm:` Flymaster, `as:` AirScore, `at:` Airtribune, `cc:` civlcomps.org (CIVL's event portal; task keys `cc:<event>-<n>`). Times published by the scoring sites (`start_time`, `ss_time`, `ess_time`, `deadline`) are local `HH:MM:SS`; `utc_offset_h` converts them to UTC when it is known. IGC fix times are UTC.

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
| `civl_id` | CIVL ID when published (Flymaster, AirScore, Airtribune and civlcomps.org registrations) |
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
| `same_flight_as` | when this flight is another file's flight (the same competition published by two sources, a live-tracker file on one and the pilot's instrument on the other, or a pilot's second instrument), the `track_id` of the copy kept; empty otherwise. Found by tracks that take off and land together and stay within 100 m of each other (median) over most of their common time |
| `shared_path` | the track follows the same path as other pilots' tracks (several pilots driven down from launch in one car are recorded as one "flight"); checked against the terrain before use |

Use `WHERE error IS NULL AND same_flight_as IS NULL` to count each flight once.

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

## The grid

Every spatial table uses the same global grid (`pta/grid.py`). A **chip** is the 0.25° box around one ERA5 grid point (centre at multiples of 0.25°), so a chip's weather is one ERA5 value. Each chip holds 25 x 25 **cells** of 0.01° (1.1 km north-south; 0.79 km east-west at 45°). Ids are computed from a latitude and longitude, with no lookup:

```python
from pta import grid
grid.cell_id(lat, lon)     # row * 36000 + col, rows from 90.125° S, columns from 180.125° W
grid.chip_id(lat, lon)     # row * 1440 + col
grid.chip_of_cell(cell_id)
```

## sites

Competitions grouped by launch: the competition with the most flights founds a site at its launch (the median takeoff of its flights), and every competition launching within 20 km of it belongs to that site, so neighbouring sites in a mountain range stay apart. A competition's area is the box around its turnpoints (cylinders up to 10 km) and the 1st-99th percentiles of its climbs, takeoffs and landings, plus about 5 km; a site's area is the union of its competitions' areas.

| Column | Meaning |
|---|---|
| `site_id` | `s` + the chip id of the main launch |
| `name`, `country` | most common place name of its competitions (or the latest competition's name) |
| `lat`, `lon`, `south`, `north`, `west`, `east` | main launch and the site's box |
| `n_events`, `n_flights` | competitions and analysed flights |
| `n_chips`, `n_chips_flown`, `n_chips_outside_area` | chips of the site, chips its flights crossed, and how many of those lie outside the competitions' areas |
| `first_date`, `last_date`, `events` | first and last competition, and their names |

`event_sites` links each competition (`event_key`) to its `site_id`; competitions with no turnpoint coordinates and no flights are left out. `site_chips` lists every chip of each site with its ERA5 point (`era5_lat`, `era5_lon`): the chips of the competitions' areas (`in_area`) plus every chip the site's flights crossed for at least a minute in all, with the fixes (`n_fixes`, every 5 s) and flights (`n_flights`) in it, so every flown fix has terrain and weather under it. A chip can belong to two neighbouring sites.

## cells

Terrain of every 1 km cell of the site chips, from the Copernicus GLO-30 DEM (30 m; tiles kept in `dem/glo30/` in the project folder). Each cell summarises its 36 x 36 DEM pixels (`pta/terrain.py`).

| Column | Meaning |
|---|---|
| `cell_id`, `chip_id`, `lat`, `lon` | cell, its chip and its centre |
| `elev_mean`, `elev_min`, `elev_max`, `elev_std` | elevation (m above sea level) |
| `slope_mean`, `slope_p90`, `steep_share` | slope in degrees, and the share of the cell steeper than 25° |
| `facing_east`, `facing_north` | mean horizontal part of the ground's normal: which way the cell faces overall (−0.34 north = a south-facing 20° slope) |
| `curvature` | mean Laplacian of the (smoothed) surface: positive in valleys and bowls, negative on ridges |
| `ridge_share`, `valley_share` | share of pixels more than 15 m above or below their surroundings within about 300 m |
| `tpi_5km`, `tpi_20km` | the cell's mean elevation minus the mean of the cells within about 5 km and 20 km: positive on ridges and peaks, negative in valleys |
| `relief_5km` | highest minus lowest point within about 5 km |
| `flat_share`, `sun_<aspect>_<class>` | share of the cell that is flat (under 5°), and the share facing each of 8 directions (`n`, `ne`, …, `nw`) in each slope class (1: 5–15°, 2: 15–30°, 3: over 30°). `pta.terrain.sun_on_cell` turns these into direct sunlight on the ground for any sun position (`pta.terrain.sun_position`) |
| `nodata_share` | share of the cell with no DEM tile (open sea), set to sea level |

## weather

Hourly ERA5 and ERA5-Land in `database/weather/`, for every site on its competition days and the day before each (`scripts/era5.py`). Times are UTC. Raw downloads stay in `weather/raw/`. Single levels can also come from ARCO-ERA5 on Google Cloud Storage (`scripts/era5_arco.py`, notebook 07b), saved by month in `weather/arco/single/` and merged into `era5_single`; those cover 5 to 21 h local solar time of each site's days, the CDS downloads all 24 hours.

**`era5_single`** and **`era5_levels`**: one row per chip and hour; a chip's weather is the ERA5 grid point at its centre (`pta.grid.chip_centre`).

| Column | Meaning |
|---|---|
| `blh_m`, `cloud_base_m` | boundary-layer height and cloud base, metres above the ground |
| `cape`, `cin`, `k_index` | convective available energy and inhibition (J/kg), K index |
| `sensible_heat_wm2`, `latent_heat_wm2` | heat fluxes over the hour, W/m², positive from the ground into the air |
| `sunshine_wm2` | solar radiation reaching the ground over the hour, W/m² |
| `cloud_total`, `cloud_low`, `cloud_mid`, `cloud_high` | cloud cover, 0–1 |
| `t2m_c`, `dewpoint_c`, `pressure_hpa`, `rain_mmh` | 2 m temperature and dew point, surface pressure, rain |
| `u10`, `v10`, `u100`, `v100`, `gust10` | wind at 10 and 100 m (m/s, towards east and north) and gusts |
| `u_<hPa>`, `v_<hPa>`, `t_<hPa>`, `rh_<hPa>`, `z_<hPa>` | wind, temperature (°C), relative humidity (%) and height (m) at 925, 850, 700, 600 and 500 hPa |

**`era5_extra`** (ARCO-ERA5 only, `--product extra`): one row per chip and hour: `skin_temp_c`, `soil_moisture` (top layer, m³/m³), `albedo`, `snow_depth_m` (water equivalent), `water_vapour_kgm2` (whole column), `freezing_level_m` (height of 0 °C above the ground), `conv_rain_mmh`, `friction_velocity` (m/s).

**`era5_static`** (ARCO-ERA5, `--product static`): one row per chip: `ground_m` (ERA5's own smoothed ground height, which `blh_m` and `cloud_base_m` are measured from), `land_share`, `veg_high`, `veg_low` (cover 0–1), `veg_high_type`, `veg_low_type`, `soil_type` (ECMWF codes), `relief_std_m` (sub-grid relief), `lake_share`.

Pressure levels from ARCO-ERA5 (notebook 07c) are read every 3rd UTC hour (those from the CDS are hourly) and add `q_<hPa>`, specific humidity in g/kg; ARCO does not store relative humidity, so `rh_<hPa>` is computed from it and the temperature with ERA5's own saturation formula (over water above 0 °C, over ice below −23 °C, mixed in between).

ERA5-Land can also come from Google Earth Engine (`scripts/era5_land_gee.py`, notebook 07e), saved by month in `weather/gee/land/` (hours 5 to 21 local solar time, fluxes already hourly) and merged with the CDS files into `era5_land`.

**`era5_land`**: one row per ERA5-Land point (0.1°, `lat`, `lon`, with its `chip_id`) and hour: `sensible_heat_wm2`, `sunshine_wm2` (hourly, from ERA5-Land's daily accumulations), `skin_temp_c`, `soil_moisture` (top layer, m³/m³), `albedo`, `snow_cover` (0–1).

**`hist_forecast/<model>.parquet`** (notebook 14, `scripts/forecast_wind_weights.py`): resumable cache of hourly archived forecast wind profiles from Open-Meteo's Historical Forecast API (`chip_id`, `hour`, `u10`, `v10`, `u100`, `v100`, and `u_<hPa>`, `v_<hPa>`, `z_<hPa>` at 925, 850, 700, 600 and 500 hPa), alongside `<model>_fetched.json` tracking completed chip-date windows.

## flight_checks, crossings, thermals, landings

Built by `notebooks/08_crossings_labels.ipynb` (`scripts/build_crossings.py`) from the 5 s fixes and the GLO-30 DEM, for every flight that is not a copy of another file's flight. Heights above ground (`agl`) are GPS altitude minus the DEM (GLO-30 is a surface model, so over forest they are low by the canopy height), calibrated per flight on its landing: the median height over the last 10 s, when the pilot is on the ground, is the logger's altitude error (many loggers record height above the WGS84 ellipsoid instead of sea level, tens of metres apart depending on the region) and is subtracted from the whole flight when it is within ±150 m (`flight_checks.agl_offset`; `landings.agl_raw` keeps the uncalibrated value). Times are seconds since 00:00 UTC of the IGC date, as in `flights`.

**`flight_checks`**: `agl_median`, `agl_p10`, `near_ground_share` (fixes under 50 m above ground), `dem_coverage`, and `is_drive`: at least 60 % of the fixes on the ground and circling less than 5 % of the time (cars do not circle), a "flight" that is really a drive (for example several pilots driven down from launch, see `flights.shared_path`). Drives are left out of the other tables' labels.

**`crossings`**: one row per flight and run of fixes in one 1 km cell: `cell_id`, `t_in`, `t_out`, `duration_s`, `agl_in`, `agl_out`, `agl_min`, `circling_s`, `climbing_s` (circling at 0.5 m/s or more), `best_vz`, `mean_vz_climbing`. A cell crossed without climbing is not proof of no thermal (a pilot high above may not have searched); the height above ground in the cell is kept for that reason.

**`thermals`**: one row per thermal (`climbs` with `is_thermal`): position, `cell_id`, `alt_base`, `alt_top`, `agl_base`, `agl_top`, `rate_ms`, `turns`; `wind_u`, `wind_v` (m/s) and `wind_alt`: the wind the pilot measured, from the drift of at least two full turns (the glider's own speed cancels over whole turns; drifts over 30 m/s or climbs shorter than a minute are left empty); `trigger_lat`, `trigger_lon`, `trigger_cell`: the thermal traced back upwind from its base, for the time the air took to rise from the ground at the measured climb rate (at most 30 minutes).

**`landings`**: one row per flight: `t`, `lat`, `lon`, `cell_id` and `agl` of the last fix of the flight, `result_id` (join `results` for goal, ess or landed out) and `is_drive`.

## Gliders and pilot ratings

Built by `notebooks/10_task_basics.ipynb` from `results`.

**`result_gliders`**: one row per result: `glider_key` (the model with brand, size and spelling removed: "OZONE Enzo 3 MS" → `enzo3`), `glider_class` (`CCC`, `EN-D`, `EN-C`, `EN-B`, `EN-A`) and `class_source`: `declared` (the class most results with that key publish), `curated` (a short list of well-known competition families, used only when no result declares one) or `unknown`; `harness_key` is the harness when the pilot wrote it after the wing ("Ozone Enzo 3 / Ozone Submarine" → `submarine`), empty otherwise, since a pod's drag matters at competition speeds. **`gliders`** has one row per key with its most common spelling and number of results.

**`ratings`**: one row per pilot and task: `rating_before` and `tasks_before` (an Elo rating over head-to-head results within each task, in date order, recorded before the task so a model of the task only sees the past), `place` and `n_pilots`. **`pilot_ratings`**: each pilot's latest `rating`, `tasks` and `last_date`. **`task_fields`**: per task, the mean rating before the task of its top 10 (`field_top10`), top 30 (`field_top30`) and whole field (`field_mean`), and how many pilots had flown a rated task before (`n_rated`).

## models

`models/thermal_v0/` (notebook 09): `metrics.json` (scores per feature group on sites left out, calibration, climb-rate model, ceiling check), `importance.csv`, `model.txt` (LightGBM), `oof.parquet` (each crossing's out-of-fold prediction), `cells.parquet` (per cell: crossings, observed and predicted climb share, mean climb rate), `ceiling.parquet` (90th percentile of thermal tops per chip and hour with ERA5's boundary layer and cloud base).

`models/task_v0/` (notebook 10): `metrics.json`, `predictions.parquet` (out-of-fold winner's speed and goal share, by competition and by site left out), `importance.csv`.

`models/wind_check/` (notebook 11): `groups.parquet`, one row per site, chip, hour and height band (`near the ground` below 400 m above the ground, `middle`, `free air` above 1000 m or above the chip's highest ground) with at least 3 thermals from 2 flights: the median wind pilots measured (`pilot_u`, `pilot_v`) and ERA5's wind interpolated to the same heights (`model_u`, `model_v`: 10 m, 100 m and the pressure levels, levels under ERA5's ground skipped), speeds, directions the wind comes from, `vector_error` and `dir_error` (where the model wind is at least 3 m/s); `by_site.csv`; `summary.json`.

`models/task_v1/` (notebook 13): `metrics.json` (with and without weather, competitions and sites left out: speed error, winner's time error in minutes and share within 15 and 30 min, goal share error and share within 10 points), `predictions.parquet` (out-of-fold, competitions left out: `pred_*` with weather, `base_*` without), `features.parquet` (the weather of each task: route-mean values over the task window), `importance.csv`.

`models/forecast_wind_weights/` (notebook 14): `matched_groups.parquet` (`wind_check/groups.parquet` joined with each forecast model's vertically interpolated wind `<model>_u`, `<model>_v` and vector error `<model>_err`), `model_scores.csv` (per-model accuracy by height band paired with ERA5 on identical rows), `by_site.csv` (per-site breakdown on `free air` + `middle`), `cv_comparison.json` (5-fold competition-grouped CV comparing single-model, equal-weight mean, median and global/regional/site-shrunk weighted blends with 95% event-clustered bootstrap CIs, plus the common global intersection), and `forecast_weights.json` (also copied to `web_data/forecast_weights.json` on complete runs).
