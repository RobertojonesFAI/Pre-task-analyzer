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

Data: competition IGC tracks (Flymaster first, then Airtribune and AirScore), task definitions and official results; ERA5 and HRRR weather; Copernicus GLO-30 terrain.

## Repo layout

```
notebooks/   step-by-step notebooks (run in Colab, in order)
scripts/     command-line steps
data/        small tables only (catalogs); tracks live in Google Drive
```

## Steps

| # | File | What it does | Output |
|---|------|--------------|--------|
| 1 | `notebooks/01_download_flymaster.ipynb` | Downloads every task in `data/flymaster_catalog.csv` into the project's Drive folder: IGC package, task definition and results page | `flymaster_igc/<group>/` in Drive |

The tracks are too big for git. They live in the project's Google Drive folder.
