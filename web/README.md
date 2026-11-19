# Web page

A static site (Astro + MapLibre GL) with a map of the competition sites and, per site:

- the 1 km cells pilots crossed, coloured by how often they found a climb there (observed) or by what the thermal model predicts; hovering a cell shows its crossings, both shares and the mean climb rate;
- the site's past tasks: the route on the map, the weather along it, and the winner's time and goal share the task model predicted without having seen that competition, next to what happened;
- "what if": change a task's distance or start time and see the prediction move. The task model runs in the browser (`task_model.json`, LightGBM trees as nested arrays), so the site needs no server.

The page reads its data at runtime from `public/data/`, written by `scripts/export_web.py` (last step of notebook 09) and `scripts/task_model.py` (notebook 13), both into `database/web_data/` in Drive. That folder is not in git. It holds places and aggregates only, never pilot names or tracks.

The basemap is OpenFreeMap's Positron style (free, no key). If it cannot load, the data still shows on a plain background.

## Run it locally

```bash
cd web
npm install
# copy the exported data, e.g. from Google Drive for desktop on Windows:
#   robocopy "G:\My Drive\IGCS\database\web_data" public\data /E
npm run dev          # http://localhost:4321
```

## Publish on Cloudflare Pages

```bash
npm run build                                   # writes dist/ (with public/data copied in)
npx wrangler pages deploy dist --project-name pre-task-analyzer
```

The first `wrangler` run opens the browser to sign in to Cloudflare and creates the project; later runs only upload. A site can be opened directly on a site with its id in the address: `/#s785575`.
