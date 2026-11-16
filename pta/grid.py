"""The global grid shared by every step: chips aligned with ERA5 and 1 km cells inside them.

ERA5 values belong to the 0.25° box centred on each grid point, so a chip is the box around one
ERA5 point: centre at multiples of 0.25°, edges at multiples of 0.25° plus 0.125°. Each chip is
split into 25 x 25 cells of 0.01° (1.1 km north-south; east-west shrinks with latitude, 0.79 km at
45°). Ids are integers that any step can compute from a latitude and longitude, with no lookup:

    chip_id = chip_row * 1440 + chip_col          (1440 chips around the world)
    cell_id = cell_row * 36000 + cell_col         (36000 cells around the world)

Rows count north from 90.125° S and columns east from 180.125° W.
"""

import numpy as np

CHIP_DEG = 0.25
CELLS_PER_CHIP = 25
CELL_DEG = CHIP_DEG / CELLS_PER_CHIP
LAT0, LON0 = -90.0 - CHIP_DEG / 2, -180.0 - CHIP_DEG / 2
CHIP_COLS, CELL_COLS = 1440, 1440 * CELLS_PER_CHIP


def cell_rc(lat, lon):
    """(row, col) of the cell holding each point (arrays or scalars)."""
    lon = (np.asarray(lon, float) - LON0) % 360.0 + LON0  # wrap to [-180.125, 179.875)
    row = np.floor((np.asarray(lat, float) - LAT0) / CELL_DEG + 1e-9).astype(np.int64)
    col = np.floor((lon - LON0) / CELL_DEG + 1e-9).astype(np.int64)
    return row, col


def cell_id(lat, lon):
    row, col = cell_rc(lat, lon)
    return row * CELL_COLS + col


def chip_id(lat, lon):
    row, col = cell_rc(lat, lon)
    return (row // CELLS_PER_CHIP) * CHIP_COLS + col // CELLS_PER_CHIP


def cell_centre(cell):
    cell = np.asarray(cell, np.int64)
    row, col = cell // CELL_COLS, cell % CELL_COLS
    return LAT0 + (row + 0.5) * CELL_DEG, LON0 + (col + 0.5) * CELL_DEG


def chip_of_cell(cell):
    cell = np.asarray(cell, np.int64)
    row, col = cell // CELL_COLS, cell % CELL_COLS
    return (row // CELLS_PER_CHIP) * CHIP_COLS + col // CELLS_PER_CHIP


def chip_bounds(chip):
    """(south, north, west, east) edges of a chip."""
    row, col = divmod(int(chip), CHIP_COLS)
    south, west = LAT0 + row * CHIP_DEG, LON0 + col * CHIP_DEG
    return south, south + CHIP_DEG, west, west + CHIP_DEG


def chip_centre(chip):
    """The ERA5 grid point of a chip (latitude, longitude)."""
    s, n, w, e = chip_bounds(chip)
    return (s + n) / 2, (w + e) / 2


def chips_in_box(south, north, west, east):
    """Every chip that overlaps a latitude/longitude box."""
    r0, c0 = cell_rc(south, west)
    r1, c1 = cell_rc(north, east)
    rows = range(int(r0) // CELLS_PER_CHIP, int(r1) // CELLS_PER_CHIP + 1)
    cols = range(int(c0) // CELLS_PER_CHIP, int(c1) // CELLS_PER_CHIP + 1)
    return [r * CHIP_COLS + c for r in rows for c in cols]
