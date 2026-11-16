# Model design: a general thermal and landout model

The core model predicts, for any 1 km cell and hour, what happens to a pilot who crosses it: the chance of finding a climb, how strong and how high it goes, the chance of landing out, and the chance of leaving the cell in a better position. It has to work at sites it has never seen, so it learns physics, not places: not "pilots land in cell 1234 at Valle de Bravo", but "when the top of the boundary layer sits just above the terrain and the wind comes over the ridge from behind, pilots land".

## 1. Unit of training: one pilot crossing one cell

The world is split into chips aligned with the ERA5 grid (0.25°, about 27 km), each divided into 1 km cells. Every time a track passes through a cell is one example:

| Part | Content |
|---|---|
| Where | terrain, land cover and weather of that cell at that hour |
| How the pilot arrived | height above ground on entry, time, glider class, pilot rating and profile |
| What happened (labels) | climbed or not (with climb rate and ceiling); landed in the cell or not; left the cell higher or lower than it entered ("success") |

Asking what happened to each pilot who crossed, instead of whether anyone climbed in the cell, removes the traffic bias found in the preliminary project (busy cells looked more thermic only because more pilots passed).

The landout label is conditional on arrival height: the model predicts P(land out | arriving at this cell at h metres above ground). Crossing at 2,000 m and at 200 m are different questions, and the task model uses the heights pilots are expected to reach each leg at.

### Where a thermal starts

The wind carries a thermal downwind as it rises, so where a pilot circles is not where the thermal left the ground. Each climb is traced back upwind (wind between the ground and the climb's height, over the time the air took to rise) to estimate its trigger point, and the terrain and energy features of that point are used alongside those of the cell where the pilot climbed.

### Labels from a track

- **Climb**: circling with at least 0.5 m/s mean climb for 30 s or more; rate, base and top are recorded.
- **Landout**: the track ends away from goal and away from the launch, with ground speed near zero and altitude close to the terrain (DEM). The landing cell gets the label.
- **Success**: the pilot leaves the cell with at least the height they entered with, or reaches the next turnpoint.

## 2. Features that transfer between sites

No latitude, longitude or site name as features. Everything is a physical or relative quantity.

**Terrain** (Copernicus GLO-30, global, 30 m)
- elevation, slope, aspect, curvature
- topographic position index (ridge, slope or valley floor), local relief, distance to the nearest ridge, sky-view factor
- sunlight on the slope at that hour (W/m²) and since sunrise, from the solar model of the launch-forecast site

**Surface** (global)
- land cover class: forest, grass, rock, water, urban (ESA WorldCover, 10 m)
- monthly NDVI (how green or dry the vegetation is), albedo, soil moisture
- distance to lakes and reservoirs

**Weather at that hour** (ERA5 for training; HRRR where available; the day's forecast at prediction time)
- boundary-layer height (BLH), CAPE, CIN, lapse rates, surface sensible heat flux
- cloud cover, cloud base (LCL), humidity, precipitation
- wind at several heights and its shear

**Energy reaching the ground** (the sun-on-slope model of `pta/terrain.py`, extended)
- direct sunlight on each cell's slopes for the sun's position at that hour (its share of slopes and aspects is stored per cell), shaded by neighbouring mountains (horizon computed from the DEM)
- times the surface solar radiation ERA5 says actually arrived (clouds) and (1 − albedo) (MODIS 500 m, ERA5-Land), accumulated since sunrise: an energy index per cell and hour
- land cover (ESA WorldCover 10 m) and snow cover, which decide how much of that energy heats the air

**The day as it develops**
- weather at every hour of the task window, not a daily summary: boundary-layer growth, CAPE and CIN, mid-level humidity and cloud building through the afternoon (overdevelopment), temperatures on pressure levels (inversions that do not break), and the wind at several heights and its timing against each leg of the route

**Derived features**
- **Usable height = (terrain elevation + BLH) − cell elevation**: the real working height. High terrain under a low boundary layer leaves little usable height, and the model learns the landout risk from it directly.
- **Wind against the slope**: wind component perpendicular to the slope; positive is upwind (lift), negative is lee side (sink and rotor).
- cloud base above the cell, sun accumulated on the slope since sunrise

**Time, independent of hemisphere**
- hours since sunrise and sun elevation instead of clock time
- day length and solar declination instead of month, so a January day in Mexico and a July day in Washington compare by their physics and not by the calendar

**Pilot and context**
- arrival height above ground, glider class, results-based rating and track-based profile (only flights before the task)
- distance from launch, time since the start, share of the field still flying

### Weather sources: a global base and regional layers

The general model is trained on one weather source everywhere, so it learns physics rather than which model a region uses: **ERA5** (0.25°, hourly, 1940 to present: boundary-layer height, CAPE, CIN, heat fluxes, cloud, winds and temperatures on pressure levels) plus **ERA5-Land** (0.1°, hourly: surface heat fluxes and soil moisture). Many competition sites (Colombia, Brazil, Mexico, Turkey, Kazakhstan, Korea, Nepal) have no regional model with a public archive, so only a global base works everywhere.

Where a high-resolution archive exists, a second, regional model is trained on top of the base and compared with it, the way a task committee learns which forecast model works best at its site:

| Region | Source | Resolution | Archive |
|---|---|---|---|
| United States | HRRR | 3 km, hourly | 2014 to present (AWS) |
| Europe | CERRA (Copernicus regional reanalysis) | 5.5 km, 3-hourly analyses with hourly short forecasts | 1984 to present, updated monthly |
| Australia, New Zealand | BARRA2 | 12 km and 4.4 km | 1979 to present (NCI, CC BY 4.0) |
| India | IMDAA | 12 km, hourly | 1979–2018 only |

A regional model is kept only where it beats the base on held-out sites and days. Mexico and Colombia have no public regional archive (the national WRF runs publish images only), so the base serves there; in Europe, COSMO-REA6 (6 km) could be a second regional source if its period covers enough competitions.

**Soundings.** The radiosonde archive (NOAA IGRA2, global, including stations in Mexico and Colombia) has two soundings a day at few stations, too sparse as an input, but it measures the bias of each weather source in lapse rates and inversions by region, and corrects it.

**Checking weather sources against the pilots.** The climbs table holds the real top of every thermal (about 1.9 million thermals before the civlcomps.org tracks). Comparing each source's boundary-layer height with the ceilings pilots actually reached, by site and time of day, measures directly which source describes soaring conditions best, before any model is trained.

## 3. Models

- **Main**: gradient-boosted trees (LightGBM or XGBoost). They handle hundreds of features, interactions such as "low usable height × lee side", and missing values. SHAP values explain each prediction ("this cell is risky mainly because of low usable height and lee-side wind"), which is what the Meet Director sees.
- **Comparison**: a multilayer perceptron.
- **Stretch**: a U-Net that sees a whole chip as an image, because a cell's surroundings matter too.

## 4. Proving it works where it has not been trained

- **Leave-one-site-out**: train without one site and test on it; repeat for every site. This is the main result.
- **Bar to beat**: in the preliminary project a model trained at one site scored AUC 0.52–0.55 at another, close to chance; a model with terrain and time of day alone reached 0.65 on held-out days.
- **Baselines**: a season-and-hour climatology per cell (thermal.kk7.ch style) and a terrain-plus-time model.
- **Calibration**: when the model says 30%, about 30% of those pilots should land out.
- **Balance**: examples are weighted by site so sites with many tracks do not dominate.

### Days that change

Overdevelopment, inversions that do not break and wind that arrives mid-task make the worst errors. The hourly weather features above let the model see most of them coming, and stopped tasks (`tasks.stopped`) are a label of their own: the tool also gives the chance that a task will be stopped. What a forecast cannot say for sure (when the day will overdevelop) comes from forecast ensembles (ECMWF ENS, GEFS): when their members disagree, the predicted ranges widen, and the committee sees forecast uncertainty separately from the model's own.

### Forecast, not reanalysis, at prediction time

Training uses ERA5, the best estimate of what the weather actually was. On a task day the model gets a forecast, which carries its own errors, so a model calibrated on ERA5 would be overconfident. Before the goal shares are shown to a committee they are checked, and recalibrated if needed, on past competition days run with the forecast that was available that morning (archived HRRR in the US, ECMWF open data elsewhere).

## 5. Where it can be trusted

Terrain, land cover and weather inputs are global, so the model runs anywhere. It is reliable where conditions resemble its training data. Each cell gets a distance to the training data in feature space, and cells far from anything seen are shown as **low confidence** instead of a made-up number.

## 6. Output for a task day

A grid over the competition area, hour by hour:

- chance of finding a climb, expected climb rate and ceiling
- chance of landing out for a given arrival height
- chance of success when crossing

The route planner searches this grid for the fastest route, and the task model turns route, grid and field strength into goal share, winner time and difficulty for the proposed task and its alternatives.
