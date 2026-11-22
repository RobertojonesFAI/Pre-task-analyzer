# Web page

A static site (Astro + MapLibre GL) with a map of the competition sites and, per site:

- where pilots circled in thermals, at about 200 m: every thermal of the competition flights snapped to a square mosaic grid of ~200 m cells with 7 discrete terracotta steps (relative to the site's busiest cells), subtle per-step borders and a high-contrast outline with each cell's thermals, flights, mean climb and median top on hover (`scripts/export_hotspots.py`);
- the relief: a hillshade, and a 3D view (the "3D" button; tilt and turn with the right mouse button or two fingers) with the thermals, the cells and the task draped on the terrain. Elevations from the free [Terrain Tiles](https://registry.opendata.aws/terrain-tiles/) on AWS Open Data;
- the 1 km cells pilots crossed, drawn in the same discrete mosaic grid style, coloured by how often they found a climb there (observed) or by what the thermal model predicts; hovering a cell highlights it and shows its crossings, both shares and the mean climb rate;
- the site's past tasks: the route on the map, the weather along it, and the winner's time and goal share the task model predicted without having seen that competition, next to what happened;
- "what if": change a task's distance or start time and see the prediction move. The task model runs in the browser (`task_model.json`, LightGBM trees as nested arrays), so the site needs no server;
- the task planner (below).

## Task planner

In a task card, "Plan a task for this day" opens an editor for a competition task on that day:

- turnpoints with a name, a type (takeoff, start of speed section, turnpoint, end of speed section, goal) and a cylinder radius; a start gate, a deadline and a goal cylinder or line. Start from the task as it was set, or click the map: on a waypoint the site's competitions used (the dots) or anywhere. Turnpoints can be dragged, reordered and inserted;
- the optimized task and speed-section distances of FAI Sporting Code S7F §7.2, on the WGS84 ellipsoid (the shortest route from launch touching every cylinder in order, from three starting configurations). On the 22 reference tasks of [pyxctsk](https://github.com/simonsteiner/pyxctsk) it agrees with that library to within a metre;
- import: XCTrack and SeeYou Navigator tasks (`.xctsk`, `.cup`, `.wpt`, GPX, and dense QR codes decoded locally with ZXing-C++ WebAssembly via `Scan QR` live auto-detecting camera, `Upload QR image` screenshot/photo upload, pasting a screenshot with `Cmd/Ctrl+V`, dragging and dropping an image/file onto the planner, or pasting `XCTSK:`/`XCTSKZ:` text). Waypoint files add their waypoints to the map to build a task from;
- export: an XCTrack QR code, `.xctsk`, SeeYou `.cup` (waypoints and task), Flymaster and CompeGPS `.wpt`, GPX;
- with the API (`api/`), the task model's winner's time and goal share for the task, using either live multi-model forecasts from Open-Meteo (select any individual global/regional model or a Median/Mean blend with a min–max range bar and per-model comparison table) or a past competition day's ERA5 weather. In ERA5 mode, the task as it was set predicts what the card shows; changes add what the model sees.

The formats and the distances are in `src/scripts/xctask.ts` (no DOM, so it can be tested in Node); the editor is `src/scripts/planner.ts`. The waypoints and each task's settings come from `scripts/export_waypoints.py`:

```bash
python scripts/export_waypoints.py --db "G:/My Drive/IGCS/database" --web web/public/data
```

It reads `tasks/` in the web data, so run it after copying the exported data, and again whenever notebook 13 rewrites the tasks.

The thermal hotspots come from `scripts/export_hotspots.py` (a cell is published when thermals from at least two flights went up there; places and aggregates only):

```bash
python scripts/export_hotspots.py --db "G:/My Drive/IGCS/database" --web web/public/data
```

The page reads its data at runtime from `public/data/`, written by `scripts/export_web.py` (last step of notebook 09) and `scripts/task_model.py` (notebook 13), both into `database/web_data/` in Drive. That folder is not in git. It holds places and aggregates only, never pilot names or tracks.

The basemap is OpenFreeMap's Positron style (free, no key). If it cannot load, the data still shows on a plain background.

## Run it locally

```bash
cd web
npm install
# copy the exported data, e.g. from Google Drive for desktop on Windows:
#   robocopy "G:\My Drive\IGCS\database\web_data" public\data /E
python ../scripts/export_waypoints.py --db "G:/My Drive/IGCS/database" --web public/data
python ../scripts/export_hotspots.py --db "G:/My Drive/IGCS/database" --web public/data
npm run dev          # http://localhost:4321
```

When `public/data/` is not present locally, `npm run dev` automatically proxies `/data` from `https://pta.xcmeteo.com` (override with `DATA_ORIGIN=...`, or set `DATA_ORIGIN=` to disable), so UI changes can be previewed with real site data before deploying. Production builds (`npm run build`) are unaffected.

## Publish on Cloudflare Pages

```bash
npm run build                                   # writes dist/ (with public/data copied in)
npx wrangler pages deploy dist --project-name pre-task-analyzer
```

The first `wrangler` run opens the browser to sign in to Cloudflare and creates the project; later runs only upload. A site can be opened directly on a site with its id in the address: `/#s785575`.
