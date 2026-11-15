# Pre-task Analyzer

ML model for analyzing paragliding competition tasks. CS 457 Intro to AI final project, Boise State University, Fall 2026. Roberto Jones.

Before a task is set, the task committee enters the task they have in mind (turnpoints, cylinders, start, goal), the day's forecast and the pilots entered. The analyzer returns:

- the expected share of pilots in goal (with a range), the winner's time and a difficulty index;
- the likely route over the predicted thermals, and the legs where many pilots land out on days like this;
- two or three alternative tasks that land in the committee's target (for example 40–70% in goal), each with its chance of success.

The Meet Director always makes the decision; the tool only advises.

## How it works

| Part | What it does |
|---|---|
| Chips | A grid aligned with the weather model (ERA5, about 27 km), with 1 km cells inside for terrain, sunlight on slopes, climbs and landouts, by hour and type of day |
| Pilot profiles | Speed relative to the winner, goal rate, consistency and style from each pilot's past tracks, plus a results-based rating; only flights before the task are used |
| Task model | Predicts goal share, winner time and difficulty from the route, the weather, the chips it crosses and the strength of the field |
| Route planner | A* over the predicted thermals, with wind and speed-to-fly |
| Alternatives | Builds candidate tasks from the competition's waypoints under the committee's rules and keeps the ones in the target |

The general thermal and landout model, its labels, features and evaluation are described in [docs/model_design.md](docs/model_design.md).

Data: competition IGC tracks (Flymaster, AirScore and Airtribune), task definitions and official results; ERA5 and HRRR weather; Copernicus GLO-30 terrain.

## Repo layout

```
notebooks/   step-by-step notebooks (run in Colab, in order)
scripts/     command-line steps
pta/         shared code: IGC reader and results parsers
docs/        design notes (model_design.md: the general model; database.md: the tables)
data/        small tables only (catalogs); tracks live in Google Drive
```

## Steps

| # | File | What it does | Output |
|---|------|--------------|--------|
| 1 | `notebooks/01_download_flymaster.ipynb` | Lists every Flymaster task with a public IGC package (`scripts/scan_flymaster.py`), then downloads each paragliding task into the project's Drive folder: IGC package, task definition and results page (`scripts/download_flymaster.py`) | `flymaster_igc/flymaster_catalog.csv` and `flymaster_igc/<group>/` in Drive |
| 2 | `notebooks/02_download_airscore.ipynb` | Downloads every finished paragliding competition on FAI AirScore (`scripts/airscore.py`): IGC package and results page per task | `airscore_igc/airscore_catalog.csv` and `airscore_igc/<comp_id>/` in Drive |
| 3 | `notebooks/03_download_airtribune.ipynb` | Downloads every past paragliding event on Airtribune, 2012 to today (`scripts/airtribune.py`), with the administrators' permission: track archive and results page per task | `airtribune_igc/airtribune_catalog.csv` and `airtribune_igc/<event_id>/` in Drive |
| 4 | `notebooks/04_build_catalog.ipynb` | Fetches the Airtribune task results still missing, then reads every task package and results page into the database tables (`scripts/build_catalog.py`, `pta/`) | `database/*.parquet` in Drive, described in [docs/database.md](docs/database.md) |

The tracks are too big for git. They live in the project's Google Drive folder.
