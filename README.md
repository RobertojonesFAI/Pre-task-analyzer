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

Data: competition IGC tracks (Flymaster, AirScore, Airtribune and civlcomps.org), task definitions and official results; ERA5 and HRRR weather; Copernicus GLO-30 terrain.

## Repo layout

```
notebooks/   step-by-step notebooks (run in Colab, in order)
scripts/     command-line steps
pta/         shared code: IGC reader, results parsers, flight analysis, the global grid and terrain
docs/        design notes (model_design.md: the general model; database.md: the tables; deployment.md: how it will be served)
data/        small tables only (catalogs); tracks live in Google Drive
```

## Steps

| # | File | What it does | Output |
|---|------|--------------|--------|
| 1 | `notebooks/01_download_flymaster.ipynb` | Lists every Flymaster task with a public IGC package (`scripts/scan_flymaster.py`), then downloads each paragliding task into the project's Drive folder: IGC package, task definition and results page (`scripts/download_flymaster.py`) | `flymaster_igc/flymaster_catalog.csv` and `flymaster_igc/<group>/` in Drive |
| 2 | `notebooks/02_download_airscore.ipynb` | Downloads every finished paragliding competition on FAI AirScore (`scripts/airscore.py`): IGC package and results page per task | `airscore_igc/airscore_catalog.csv` and `airscore_igc/<comp_id>/` in Drive |
| 3 | `notebooks/03_download_airtribune.ipynb` | Downloads every past paragliding event on Airtribune, 2012 to today (`scripts/airtribune.py`), with the administrators' permission: track archive and results page per task | `airtribune_igc/airtribune_catalog.csv` and `airtribune_igc/<event_id>/` in Drive |
| 3b | `notebooks/03b_download_civlcomps.ipynb` | Downloads the paragliding XC events published on CIVL's event portal civlcomps.org (`scripts/civlcomps.py`), with CIVL's permission: IGC zip, FS results page and participants per task (IGC uploads began in 2024) | `civlcomps_igc/civlcomps_catalog.csv` and `civlcomps_igc/<event_id>/` in Drive |
| 4 | `notebooks/04_build_catalog.ipynb` | Fetches the Airtribune task results still missing, then reads every task package and results page into the database tables (`scripts/build_catalog.py`, `pta/`) | `database/*.parquet` in Drive, described in [docs/database.md](docs/database.md) |
| 5 | `notebooks/05_build_flights.ipynb` | Analyses every unique track (`scripts/build_flights.py`, `pta/flight.py`): cleans GPS and altitude errors, separates the flight from the drive up and the retrieve, and splits it into climbs and glides | `database/flights`, `climbs`, `glides` and `fixes/` (5 s fixes per task) |
| 6 | `notebooks/06_sites_terrain.ipynb` | Groups the competitions into flying sites (`scripts/build_sites.py`), downloads the Copernicus GLO-30 DEM around them and describes the terrain of every 1 km cell (`scripts/build_terrain.py`, `pta/terrain.py`, `pta/grid.py`) | `database/sites`, `event_sites`, `site_chips`, `cells`; `dem/glo30/` tiles |
| 7 | `notebooks/07_weather_era5.ipynb` | Hourly ERA5 (single and pressure levels) and ERA5-Land for every site on its competition days, from the Copernicus Climate Data Store (`scripts/era5.py`; needs a free CDS key), and a first check of ERA5's boundary-layer height against real thermal tops | `database/weather/era5_single`, `era5_levels`, `era5_land` |

The tracks are too big for git. They live in the project's Google Drive folder.

## Next steps

In order of priority for the end of the semester (December 2026):

1. **Weather base (07, written):** ERA5 and ERA5-Land for every site, hourly, on the days of each competition (Copernicus Climate Data Store).
2. **Energy on the slopes:** sunlight on each cell's slopes with shading by neighbouring mountains, clouds, albedo and land cover, and the trigger point of each thermal traced back upwind.
   **Wind over the terrain (2b):** WindNinja (US Forest Service) run per site for a table of inflow directions and speeds, giving terrain-scale wind (speed-up over ridges, channelling in valleys, lee sides) for every hour by interpolating with the ERA5 wind.
3. **Validation against the pilots:** weather sources compared with the ceilings of the real thermals and with radiosonde soundings (IGRA2); terrain winds compared with the wind pilots measured (the drift of their circles in thermals).
4. **HRRR** as a regional layer for the United States.
5. **CERRA** as a regional layer for Europe.

Extension: a CFD surrogate trained on OpenFOAM simulations of competition terrain (wind at flight heights, which WindNinja does not give), kept if it beats WindNinja against the winds pilots measured.

Then the cell crossings and labels, the general thermal and landout model, the task model and the route planner (see [docs/model_design.md](docs/model_design.md)).
