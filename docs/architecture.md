# Architecture

How the parts of the project fit together: where the data comes from, how the database and the models are built, and how the API and the web page use them.

**Interactive graph:** [pta.xcmeteo.com/graph/](https://pta.xcmeteo.com/graph/) maps the code and the docs as a graph (about 1,000 functions, files, tables and concepts, grouped into communities). Click a node to see its connections, or search for a function or a table.

## Overview

```mermaid
flowchart LR
    subgraph sources["Data sources"]
        comps["Competition IGC tracks,<br/>tasks and results<br/>(Flymaster, AirScore,<br/>Airtribune, civlcomps.org)"]
        era5["ERA5 and ERA5-Land<br/>(CDS, ARCO-ERA5,<br/>Earth Engine)"]
        dem["Copernicus GLO-30 DEM"]
        om["Open-Meteo forecasts<br/>(live and archived)"]
    end

    subgraph db["Database (Parquet, Google Drive)"]
        catalog["Catalog: events, tasks,<br/>results, pilots, tracks"]
        flights["Flights, climbs, glides,<br/>5 s fixes (05)"]
        terrain["Sites, chips and<br/>1 km cells (06)"]
        weather["Weather tables (07)"]
        labels["Crossings, thermals,<br/>landings (08)"]
        truth["IGC truth tables (15)"]
    end

    subgraph models["Models"]
        thermal["Thermal model v0 (09)"]
        basics["Gliders, Elo ratings,<br/>task baseline (10)"]
        wind["Wind check (11)"]
        task["Task model v1 (13)"]
        fcw["Forecast weights and<br/>verification (14, 16)"]
    end

    subgraph serve["Serving"]
        api["Prediction API<br/>(FastAPI, api.xcmeteo.com)"]
        web["Web page and task planner<br/>(Astro + MapLibre,<br/>pta.xcmeteo.com)"]
    end

    stats["results/: statistics<br/>and coverage maps (12)"]

    comps --> catalog --> flights --> labels
    dem --> terrain --> labels
    era5 --> weather
    flights --> truth
    labels --> thermal
    terrain --> thermal
    weather --> thermal
    catalog --> basics --> task
    weather --> task
    labels --> wind
    weather --> wind
    om --> fcw
    truth --> fcw
    thermal -->|web data| web
    task -->|task_model.json| web
    task --> api
    fcw --> api
    om -->|live forecast| api
    web -->|POST /predict| api
    db --> stats
```

Numbers in brackets are the notebooks in `notebooks/` that build each part; the README's [Steps](../README.md#steps) table lists them one by one.

## The shared code

```mermaid
flowchart TB
    igc["pta/igc.py<br/>IGC reader"]
    flight["pta/flight.py<br/>flight, climbs, glides"]
    grid["pta/grid.py<br/>global grid: chips and cells"]
    terr["pta/terrain.py<br/>DEM, cell terrain, sun on slopes"]
    truthp["pta/igc_truth.py<br/>thermal winds, glide winds, ceilings"]

    igc -.->|fixes| flight
    grid --> terr

    bf["scripts/build_flights.py"] --> igc
    bf --> flight
    bt["scripts/build_terrain.py"] --> terr
    bc["scripts/build_crossings.py"] --> terr
    tm["scripts/thermal_model.py"] --> terr
    bit["scripts/build_igc_truth.py"] --> truthp
    bit --> flight
    bit --> terr
    era["scripts/era5*.py, task_model.py,<br/>forecast_*.py, wind_check.py"] --> grid
    fc["api/forecast.py"] --> grid
```

Arrows are imports (`terrain.py` imports `grid.py`, so the scripts that use the terrain also use the grid); the dotted arrow is data: `pta/flight.py` works on the fixes `pta/igc.py` reads.

- **The grid** (`pta/grid.py`) is the key every table shares: 0.25° chips aligned with ERA5, and 1 km cells inside them. Weather joins on `chip_id`, terrain and crossings on `cell_id` ([database.md](database.md)).
- **Sun on the slopes** (`pta.terrain.sun_on_cell`) turns each cell's share of slopes and aspects (stored in the `cells` table) into direct sunlight for any sun position. The thermal model uses it for its `sun_direct` and `sun_energy` features. Shading by neighbouring mountains, clouds and albedo are the planned extension ([model_design.md](model_design.md)).
- **The web page** (`web/src/scripts/`) runs the task model in the browser from `task_model.json` and asks the API only to re-plan a task with a new route or a live forecast.
