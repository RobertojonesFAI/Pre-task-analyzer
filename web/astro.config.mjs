import { defineConfig } from "astro/config";

// Static site: the map loads its data from /data at runtime (scripts/export_web.py writes it),
// so new data needs no rebuild.
export default defineConfig({
  output: "static",
});
