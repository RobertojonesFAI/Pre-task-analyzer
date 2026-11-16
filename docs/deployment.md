# Deployment plan

How the analyzer will be served once the models exist. Nothing here is built yet.

## Pieces

| Piece | Where | What it does |
|---|---|---|
| Web app (`web/`) | Cloudflare Workers (Astro + MapLibre) | Task editor (turnpoints, cylinders, `.xctsk` import), thermal and landout layers by hour, suggested route, alternatives side by side |
| API (`api/`) | Home server (Dell OptiPlex 7040: Intel i5-6500T, 4 cores, 16 GB RAM, SSD, Windows 11 Pro) in Docker (Docker Desktop on WSL2), reached through a Cloudflare Tunnel | Evaluates a task with the same Python code and models used in training: goal share with a range, winner time, difficulty, hardest legs (SHAP), A* route, search for alternative tasks |
| Fallback API | Cloudflare Containers, same Docker image, asleep unless needed | The Worker sends requests here when the tunnel does not answer (power or internet cut at home) |
| Worker in front | Cloudflare Workers | Routes `/api` to the tunnel or the fallback, caches evaluated tasks |
| Storage | R2: models, precomputed maps per site and hour, terrain tiles. D1: sites, competition waypoints, saved tasks | |
| Daily forecast | Cron on the home server | Downloads the forecast for each site (HRRR in the US, ECMWF or GFS elsewhere), runs the thermal model, writes the maps to R2 before the task committee meets |
| Access | Cloudflare Access | Only task committees and organisers allowed at first |

The tunnel is an outgoing connection from the server, so no router ports are opened and the home IP is not exposed. The API returns small JSON responses; the large maps are served from R2, so home upload speed does not matter.

## Cost

Workers Paid plan ($5 a month, needed for Containers) plus R2 storage beyond the free tier. The fallback container is billed only while it runs.

## Forecast versus reanalysis

The models are trained on ERA5 reanalysis but will be run on forecasts, which carry errors. Goal shares need to be calibrated (or at least checked) against archived forecasts so the tool is not overconfident on forecast days.

## Privacy

Only maps, models and aggregate numbers are published. Pilot names and individual tracks stay in the private Drive folder.
