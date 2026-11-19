# API

A small FastAPI service that re-plans a past task with a new route: same day, same pilots, the
route a committee draws on the web page. It reads ERA5 along the new route in the task's window,
exactly as `scripts/task_model.py` does for training, and returns the task model's winner's time and
goal share. The web page works without it (the task planner still imports and exports tasks); with it,
the planner predicts the tasks set on it.

`POST /predict` takes `{"site", "template" (the day's task_key), "route": [[lat, lon, radius_m], ...],
"ss_km", "start_solar_h"}`. The page sends the route's optimized speed-section distance (`ss_km`), measured
from the cylinders as the official one is, and a start time only when it differs from the day's.

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

Build the page with the API's address:

```powershell
cd web
$env:PUBLIC_API_URL = "https://api.example.com"
npm run build
npx wrangler pages deploy dist --project-name pre-task-analyzer
```

Or, without rebuilding, open the page with `?api=https://api.example.com` in its address.

`PTA_ORIGINS` (comma-separated) limits which sites may call the API; by default any.
