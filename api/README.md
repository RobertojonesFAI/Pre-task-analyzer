# API

A small FastAPI service that evaluates a task drawn on the web page with either:

- **Past competition day (ERA5):** same day, same pilots, a new route and start time, reading ERA5
  along the route exactly as `scripts/task_model.py` does for training.
- **Live weather forecast (Open-Meteo):** future dates (today through +6 days) using one or more
  forecast models (`ecmwf_ifs`, `ecmwf_aifs`, `gfs`, `icon`, `gem`, plus regional models where
  covered: `rrfs`, `hrdps`, `icon_d2`, `icon_eu`, `arome`), fetched in parallel per model with only
  the variables each model publishes, returning each model's prediction and a median/mean blend with
  min–max range (`api/forecast.py`, 1 h per-chip cache). Pilots and ratings are taken from the
  selected past competition day as a template.

The web page works without it (the task planner still imports and exports tasks); with it, the
planner predicts the tasks set on it.

Endpoints:
- `GET /health`: loaded tables, sites and forecast model count.
- `GET /forecast/models?site=<site_id>&date=<YYYY-MM-DD>`: forecast models covering the site, with
  resolution, boundary-layer source (`native` or `parcel`) and whether each covers the chosen date.
- `POST /predict`: takes `{"site", "template" (the day's task_key), "route": [[lat, lon, radius_m], ...],
  "ss_km", "start_solar_h", "window_h", "weather": {"source": "era5" | "forecast", "date", "models", "weights"}}`.
  The page sends the route's optimized speed-section distance (`ss_km`), measured from the cylinders
  as the official one is.

To check live Open-Meteo model coverage at sample sites:

```powershell
python -m api.forecast --probe
```

It runs on the OptiPlex (see [docs/deployment.md](../docs/deployment.md)) and is published through a
Cloudflare Tunnel on the project's domain.

## 1. Data

The API needs the weather tables, the cells and the web data (about 1–2 GB). A local copy is faster
and does not depend on Google Drive being online:

```powershell
$src = "G:\My Drive\IGCS\database"
robocopy "$src\weather" C:\pta\database\weather era5_single.parquet era5_levels.parquet era5_extra.parquet era5_static.parquet
robocopy "$src" C:\pta\database cells.parquet
robocopy "$src\web_data" C:\pta\database\web_data /E
```

Copy them again after notebooks 07–13 change them.

## 2. Run it

```powershell
cd $HOME\Documents\Pre-task-analyzer
py -m pip install -r api\requirements.txt
powershell -ExecutionPolicy Bypass -File api\start.ps1
```

Loading takes about a minute; then http://localhost:8000/health answers with what it loaded.

## 3. Publish it with a Cloudflare Tunnel

With a domain on Cloudflare (`example.com` below), once:

```powershell
winget install Cloudflare.cloudflared
cloudflared tunnel login                       # opens the browser: choose the domain
cloudflared tunnel create pta-api              # prints the tunnel id and writes its credentials file
cloudflared tunnel route dns pta-api api.example.com
```

`%USERPROFILE%\.cloudflared\config.yml`:

```yaml
tunnel: pta-api
credentials-file: C:\Users\<you>\.cloudflared\<tunnel id>.json
ingress:
  - hostname: api.example.com
    service: http://localhost:8000
  - service: http_status:404
```

Then, in a second terminal, `cloudflared tunnel run pta-api`, and https://api.example.com/health
answers from anywhere. To start both by themselves when the computer starts, add `api\start.ps1` and
`cloudflared tunnel run pta-api` to the Task Scheduler ("At startup"), or install cloudflared as a
Windows service (`cloudflared service install`, with the config in the system profile).

## 4. Point the web page at it

The page calls `https://api.xcmeteo.com` unless told otherwise, so a plain build is enough:

```powershell
cd web
npm run build
npx wrangler pages deploy dist --project-name pre-task-analyzer
```

To build it against another API, set `$env:PUBLIC_API_URL = "https://api.example.com"` before
`npm run build` (an empty value builds a page that works without the API). Without rebuilding, open
the page with `?api=https://api.example.com` in its address.

`PTA_ORIGINS` (comma-separated) limits which sites may call the API; by default any.
