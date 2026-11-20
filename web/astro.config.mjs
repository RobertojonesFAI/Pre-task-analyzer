import { existsSync } from "node:fs";
import { defineConfig } from "astro/config";

const dataOrigin =
  process.env.DATA_ORIGIN ??
  (!existsSync(new URL("./public/data", import.meta.url)) ? "https://pta.xcmeteo.com" : "");

// Static site: the map loads its data from /data at runtime (scripts/export_web.py writes it),
// so new data needs no rebuild. In dev, when public/data is absent, proxy /data from the live site.
export default defineConfig({
  output: "static",
  vite: dataOrigin
    ? { server: { proxy: { "/data": { target: dataOrigin, changeOrigin: true } } } }
    : {},
});
