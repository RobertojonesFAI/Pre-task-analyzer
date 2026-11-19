// Relief on the map: a hillshade in 2D, and a 3D view of the terrain (tilted, rotatable) with the
// thermals, the cells and the task draped on it. Elevations from the free Terrain Tiles on AWS Open
// Data (Mapzen's terrarium tiles, built from SRTM and other sources; about 30 m in the mountains).

import * as maplibregl from "maplibre-gl";

const TILES = "https://s3.amazonaws.com/elevation-tiles-prod/terrarium/{z}/{x}/{y}.png";
const ATTRIBUTION = '<a href="https://registry.opendata.aws/terrain-tiles/" target="_blank" rel="noopener">Terrain Tiles</a> (Mapzen, AWS Open Data)';
const EXAGGERATION = 1.3;

const dem = (attribution?: string): maplibregl.RasterDEMSourceSpecification => ({
  type: "raster-dem", tiles: [TILES], tileSize: 256, maxzoom: 14, encoding: "terrarium", ...(attribution ? { attribution } : {}),
});

/** Adds the hillshade under the basemap's labels, and the 3D button. */
export function addRelief(map: maplibregl.Map) {
  // separate sources for the shading and the 3D surface, as MapLibre recommends
  map.addSource("dem-shade", dem(ATTRIBUTION));
  map.addSource("dem", dem());
  const labels = map.getStyle().layers?.find((l) => l.type === "symbol")?.id;
  map.addLayer({
    id: "hillshade", type: "hillshade", source: "dem-shade",
    paint: { "hillshade-exaggeration": 0.4, "hillshade-shadow-color": "#4b4a45", "hillshade-highlight-color": "#ffffff",
      "hillshade-accent-color": "#6b6a64" },
  }, labels);
  map.addControl(new ThreeD(), "top-right");
}

class ThreeD implements maplibregl.IControl {
  private map: maplibregl.Map | null = null;
  private box = document.createElement("div");
  private button = document.createElement("button");
  private on = false;

  onAdd(map: maplibregl.Map) {
    this.map = map;
    this.box.className = "maplibregl-ctrl maplibregl-ctrl-group";
    this.button.type = "button";
    this.button.className = "ctrl-3d";
    this.button.textContent = "3D";
    this.button.title = "Show the terrain in 3D (drag with the right button, or two fingers, to tilt and turn)";
    this.button.setAttribute("aria-pressed", "false");
    this.button.addEventListener("click", () => this.toggle());
    this.box.appendChild(this.button);
    return this.box;
  }

  onRemove() {
    this.box.remove();
    this.map = null;
  }

  toggle() {
    const map = this.map;
    if (!map) return;
    this.on = !this.on;
    this.button.setAttribute("aria-pressed", String(this.on));
    this.button.classList.toggle("on", this.on);
    if (this.on) {
      map.setTerrain({ source: "dem", exaggeration: EXAGGERATION });
      map.setSky({ "sky-color": "#bcd4ec", "horizon-color": "#eef2f6", "fog-color": "#f3f2ef", "sky-horizon-blend": 0.5,
        "horizon-fog-blend": 0.6, "fog-ground-blend": 0.3, "atmosphere-blend": 0 });
      map.easeTo({ pitch: 62, duration: 900 });
    } else {
      map.setTerrain(null);
      map.easeTo({ pitch: 0, bearing: 0, duration: 700 });
    }
  }
}
