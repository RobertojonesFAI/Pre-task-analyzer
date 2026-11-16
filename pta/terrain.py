"""Terrain from the Copernicus GLO-30 DEM: tiles, a window reader, features per 1 km cell, and the
sun on the slopes of a cell.

GLO-30 is a global 30 m (1 arc-second) surface model in 1° x 1° tiles, free from the Copernicus
programme and mirrored on AWS Open Data (no account needed). Tiles at sea have no file. Above 50°
latitude tiles are stored with wider pixels east-west; the reader samples every tile onto the same
1 arc-second grid, so cells hold 36 x 36 pixels everywhere.
"""

import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from scipy import ndimage

from pta import grid

URL = "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
PX_PER_DEG = 3600
PX_PER_CELL = round(grid.CELL_DEG * PX_PER_DEG)          # 36
M_PER_DEG = 111320.0
HALO_CELLS = 10                                           # cells around a chip read for neighbourhood features
SLOPE_CLASSES = [(5, 15, 10.0), (15, 30, 22.0), (30, 90, 38.0)]   # (from, to, representative slope) in degrees
ASPECTS = ["n", "ne", "e", "se", "s", "sw", "w", "nw"]
RIDGE_M = 15.0                                            # a pixel this far above (below) its 300 m surroundings is ridge (valley)


def tile_name(lat_floor, lon_floor):
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def tiles_for_box(south, north, west, east):
    return [(la, lo) for la in range(math.floor(south), math.floor(north - 1e-9) + 1)
            for lo in range(math.floor(west), math.floor(east - 1e-9) + 1)]


def chip_window(chip, halo_cells=HALO_CELLS):
    """(south, north, west, east) of a chip with a halo of cells around it."""
    s, n, w, e = grid.chip_bounds(chip)
    h = halo_cells * grid.CELL_DEG
    return s - h, n + h, w - h, e + h


class DEM:
    """Reads elevation windows from a folder of GLO-30 tiles, keeping the last few tiles in memory."""

    def __init__(self, folder, keep=6):
        self.folder, self.keep, self.cache = Path(folder), keep, OrderedDict()

    def tile(self, lat_floor, lon_floor):
        key = (lat_floor, lon_floor)
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        path = self.folder / f"{tile_name(lat_floor, lon_floor)}.tif"
        value = None
        if path.exists():
            import rasterio  # imported here so the rest of the module works without it
            with rasterio.open(path) as src:
                z = src.read(1).astype(np.float32)
                if src.nodata is not None:
                    z[z == src.nodata] = np.nan
                value = (z, src.transform)
        self.cache[key] = value
        if len(self.cache) > self.keep:
            self.cache.popitem(last=False)
        return value

    def window(self, south, north, west, east):
        """Elevation on the 1 arc-second grid whose pixel edges fall on whole arc-seconds; rows run
        north to south. Returns (z, lat of each row, lon of each column); NaN where there is no tile."""
        r0, r1 = round(south * PX_PER_DEG), round(north * PX_PER_DEG)
        c0, c1 = round(west * PX_PER_DEG), round(east * PX_PER_DEG)
        lat = (r1 - np.arange(r1 - r0) - 0.5) / PX_PER_DEG
        lon = (c0 + np.arange(c1 - c0) + 0.5) / PX_PER_DEG
        z = np.full((len(lat), len(lon)), np.nan, np.float32)
        lat_f, lon_f = np.floor(lat).astype(int), np.floor(lon).astype(int)
        for la in np.unique(lat_f):
            rows = np.flatnonzero(lat_f == la)
            for lo in np.unique(lon_f):
                t = self.tile(int(la), int(lo))
                if t is None:
                    continue
                tz, tr = t
                cols = np.flatnonzero(lon_f == lo)
                src_r = np.clip(np.floor((tr.f - lat[rows]) / -tr.e).astype(int), 0, tz.shape[0] - 1)
                src_c = np.clip(np.floor((lon[cols] - tr.c) / tr.a).astype(int), 0, tz.shape[1] - 1)
                z[np.ix_(rows, cols)] = tz[np.ix_(src_r, src_c)]
        return z, lat, lon


    def sample(self, lat, lon):
        """Ground elevation (m) at points, from the nearest DEM pixel; NaN where there is no tile."""
        lat, lon = np.asarray(lat, float), np.asarray(lon, float)
        out = np.full(len(lat), np.nan, np.float32)
        ok = np.isfinite(lat) & np.isfinite(lon)
        keys = np.floor(lat[ok]).astype(int) * 1000 + np.floor(lon[ok]).astype(int)
        idx = np.flatnonzero(ok)
        for key in np.unique(keys):
            la, lo = (key + 500) // 1000, key - ((key + 500) // 1000) * 1000
            t = self.tile(int(la), int(lo))
            if t is None:
                continue
            z, tr = t
            sel = idx[keys == key]
            r = np.clip(np.floor((tr.f - lat[sel]) / -tr.e).astype(int), 0, z.shape[0] - 1)
            c = np.clip(np.floor((lon[sel] - tr.c) / tr.a).astype(int), 0, z.shape[1] - 1)
            out[sel] = z[r, c]
        return out


def _blocks(a, n=PX_PER_CELL):
    """(rows/n, cols/n, n*n) view of an array split into n x n blocks."""
    r, c = a.shape[0] // n, a.shape[1] // n
    return a[:r * n, :c * n].reshape(r, n, c, n).swapaxes(1, 2).reshape(r, c, n * n)


def chip_cells(chip, dem):
    """Terrain features of the 25 x 25 cells of one chip, as a dict of 1-D arrays (one per cell)."""
    s, n, w, e = chip_window(chip)
    z, lat, lon = dem.window(s, n, w, e)
    nodata = np.isnan(z)
    if nodata.all():
        return None
    z = np.where(nodata, 0.0, z)  # sea or missing tile: sea level, flagged in nodata_share

    dy = M_PER_DEG / PX_PER_DEG
    dx = dy * np.cos(np.radians(lat))[:, None]
    gy_rows, gx = np.gradient(z)
    g_north = -gy_rows / dy                   # rows run north to south
    g_east = gx / dx
    grad = np.hypot(g_east, g_north)
    slope = np.degrees(np.arctan(grad))
    norm = np.sqrt(grad ** 2 + 1)
    n_east, n_north = -g_east / norm, -g_north / norm    # horizontal part of the unit normal (the way the slope faces)
    aspect = np.degrees(np.arctan2(n_east, n_north)) % 360
    smooth = ndimage.uniform_filter(z, 3)
    lap = (ndimage.laplace(smooth) / dy ** 2)  # > 0 concave (valley), < 0 convex (ridge); dx≈dy is close enough here
    tpi_px = z - ndimage.uniform_filter(z, 21)  # about 300 m around

    B = _blocks
    zb = B(z)
    out = {
        "elev_mean": zb.mean(-1), "elev_min": zb.min(-1), "elev_max": zb.max(-1), "elev_std": zb.std(-1),
        "slope_mean": B(slope).mean(-1), "slope_p90": np.percentile(B(slope), 90, axis=-1),
        "steep_share": (B(slope) > 25).mean(-1),
        "facing_east": B(n_east).mean(-1), "facing_north": B(n_north).mean(-1),
        "curvature": B(lap).mean(-1),
        "ridge_share": (B(tpi_px) > RIDGE_M).mean(-1), "valley_share": (B(tpi_px) < -RIDGE_M).mean(-1),
        "nodata_share": B(nodata).mean(-1),
        "flat_share": (B(slope) < SLOPE_CLASSES[0][0]).mean(-1),
    }
    sector = (((aspect + 22.5) % 360) // 45).astype(np.int8)
    sb = B(sector)
    for k, (lo, hi, _) in enumerate(SLOPE_CLASSES):
        in_class = B((slope >= lo) & (slope < hi))
        for a, name in enumerate(ASPECTS):
            out[f"sun_{name}_{k + 1}"] = (in_class & (sb == a)).mean(-1)

    # Neighbourhood features on the cell grid (the halo makes them seamless across chips)
    em = out["elev_mean"]
    out["tpi_5km"] = em - ndimage.uniform_filter(em, 5, mode="nearest")
    out["tpi_20km"] = em - ndimage.uniform_filter(em, 21, mode="nearest")
    out["relief_5km"] = (ndimage.maximum_filter(out["elev_max"], 5, mode="nearest")
                         - ndimage.minimum_filter(out["elev_min"], 5, mode="nearest"))

    h, m = HALO_CELLS, grid.CELLS_PER_CHIP
    out = {k: np.asarray(v[h:h + m, h:h + m], np.float32).ravel() for k, v in out.items()}
    # cell ids: rows of the window run north to south
    s0, _, w0, _ = grid.chip_bounds(chip)
    row0, col0 = grid.cell_rc(s0 + grid.CELL_DEG / 2, w0 + grid.CELL_DEG / 2)
    rows = row0 + (m - 1 - np.arange(m))
    cols = col0 + np.arange(m)
    out["cell_id"] = (rows[:, None] * grid.CELL_COLS + cols[None, :]).ravel()
    return out


# ── Sun ──────────────────────────────────────────────────────────────────────

def sun_position(lat, lon, when):
    """Sun azimuth (degrees from north, clockwise) and elevation (degrees) at UTC times `when`
    (numpy datetime64 or pandas timestamps). NOAA's approximation, good to a fraction of a degree."""
    t = np.asarray(when, dtype="datetime64[s]").astype(np.int64)
    days = t / 86400.0 - 10957.5          # days since 2000-01-01 12:00 UTC
    g = np.radians((357.529 + 0.98560028 * days) % 360)
    q = (280.459 + 0.98564736 * days) % 360
    lam = np.radians(q + 1.915 * np.sin(g) + 0.020 * np.sin(2 * g))
    eps = np.radians(23.439 - 0.00000036 * days)
    decl = np.arcsin(np.sin(eps) * np.sin(lam))
    ra = np.degrees(np.arctan2(np.cos(eps) * np.sin(lam), np.cos(lam))) % 360
    gmst = (280.46061837 + 360.98564736629 * days) % 360
    ha = np.radians((gmst + np.asarray(lon) - ra + 540) % 360 - 180)
    la = np.radians(np.asarray(lat))
    elev = np.arcsin(np.sin(la) * np.sin(decl) + np.cos(la) * np.cos(decl) * np.cos(ha))
    az = np.degrees(np.arctan2(-np.sin(ha), np.tan(decl) * np.cos(la) - np.sin(la) * np.cos(ha))) % 360
    return az, np.degrees(elev)


def sun_on_cell(cells, azimuth, elevation):
    """Direct sunlight on a cell's ground relative to a flat surface facing the sun (0-1), from the
    cell's slope and aspect shares (the sun_* and flat_share columns). Shadows cast by
    neighbouring mountains are not included."""
    el = np.radians(np.maximum(np.asarray(elevation, float), 0))
    az = np.radians(np.asarray(azimuth, float))
    total = np.asarray(cells["flat_share"], float) * np.sin(el)
    for k, (_, _, rep) in enumerate(SLOPE_CLASSES):
        sl = math.radians(rep)
        for a, name in enumerate(ASPECTS):
            cos_i = math.cos(sl) * np.sin(el) + math.sin(sl) * np.cos(el) * np.cos(az - math.radians(45 * a))
            total = total + np.asarray(cells[f"sun_{name}_{k + 1}"], float) * np.maximum(cos_i, 0)
    return np.where(np.asarray(elevation) > 0, total, 0.0)
