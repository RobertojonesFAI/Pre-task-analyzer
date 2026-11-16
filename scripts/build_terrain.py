#!/usr/bin/env python3
"""Terrain features of every 1 km cell of every site, from the Copernicus GLO-30 DEM.

Downloads the 1° DEM tiles the sites need (about 25-40 MB each, from the AWS Open Data mirror,
no account) to DEM_DIR, then computes the features of each chip's 625 cells (pta/terrain.py) and
writes DB/cells.parquet. With --local the tiles are copied to a local disk first and read from
there. Finished chips are saved in batches (DB/_terrain_parts), so a second run only does what is
missing.

    python3 scripts/build_terrain.py /content/drive/MyDrive/IGCS/database --dem /content/drive/MyDrive/IGCS/dem/glo30 \
        --local /content/dem_glo30
"""

import argparse
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pta import grid, terrain  # noqa: E402


def download(tile, folder, missing):
    """Download one tile. Returns 'ok', 'cached', 'sea' (no tile: open water) or an error."""
    name = terrain.tile_name(*tile)
    dest = folder / f"{name}.tif"
    if dest.exists():
        return tile, "cached"
    if name in missing:
        return tile, "sea"
    for attempt in range(3):
        try:
            with requests.get(terrain.URL.format(name=name), stream=True, timeout=120) as r:
                if r.status_code in (403, 404):  # the bucket has no file for tiles at sea
                    return tile, "sea"
                r.raise_for_status()
                tmp = dest.with_suffix(".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.rename(dest)
                return tile, "ok"
        except requests.RequestException as e:
            error = f"{type(e).__name__}: {e}"
            time.sleep(5 * (attempt + 1))
    return tile, error


def process(chips, dem_dir):
    """Features of a run of neighbouring chips (one process keeps their tiles in memory).
    Returns (chips, cells) for the main process to write: chips all at sea give no cells."""
    dem = terrain.DEM(dem_dir)
    frames = []
    for chip in chips:
        out = terrain.chip_cells(chip, dem)
        if out:
            df = pd.DataFrame(out)
            lat, lon = grid.cell_centre(df["cell_id"].to_numpy())
            df.insert(1, "chip_id", chip)
            df.insert(2, "lat", lat)
            df.insert(3, "lon", lon)
            frames.append(df)
    return chips, (pd.concat(frames, ignore_index=True) if frames else None)


def write_batch(parts, frames, chips):
    """One part file for many chips (a Drive mount copes badly with thousands of small files), then
    the chips it holds are added to done_chips.txt. A lost connection stops the run cleanly."""
    try:
        name = f"batch_{int(time.time() * 1000)}"
        if frames:
            tmp = parts / f"{name}.tmp"
            pd.concat(frames, ignore_index=True).to_parquet(tmp, index=False)
            tmp.rename(parts / f"{name}.parquet")
        with open(parts / "done_chips.txt", "a") as f:
            f.write("".join(f"{c}\n" for c in chips))
    except OSError as e:
        sys.exit(f"\nCould not write to {parts} ({e}).\nIf Google Drive disconnected, run the Drive cell "
                 "again (it remounts Drive) and then this cell: finished chips are kept.")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("db", type=Path, help="database folder (with site_chips.parquet from build_sites.py)")
    parser.add_argument("--dem", type=Path, required=True, help="folder for the GLO-30 tiles")
    parser.add_argument("--workers", type=int, default=2, help="processes computing features (default 2)")
    parser.add_argument("--downloads", type=int, default=8, help="tiles downloaded at once (default 8)")
    parser.add_argument("--redo", action="store_true", help="recompute every chip (tiles are kept)")
    parser.add_argument("--local", type=Path, help="local folder to copy the tiles to before computing "
                        "(much faster and gentler than reading them from a Drive mount)")
    args = parser.parse_args()
    args.dem.mkdir(parents=True, exist_ok=True)
    parts = args.db / "_terrain_parts"
    if args.redo:
        shutil.rmtree(parts, ignore_errors=True)
    parts.mkdir(parents=True, exist_ok=True)

    chips = sorted(pd.read_parquet(args.db / "site_chips.parquet")["chip_id"].unique().tolist())
    tiles = sorted({t for c in chips for t in terrain.tiles_for_box(*terrain.chip_window(c))})
    missing_file = args.dem / "_sea_tiles.txt"
    missing = set(missing_file.read_text().split()) if missing_file.exists() else set()

    print(f"DEM: {len(tiles)} tiles for {len(chips):,} chips", flush=True)
    counts, started = {}, time.time()
    with ThreadPoolExecutor(args.downloads) as pool:
        futures = [pool.submit(download, t, args.dem, missing) for t in tiles]
        for n, f in enumerate(as_completed(futures), 1):
            tile, status = f.result()
            kind = status if status in ("ok", "cached", "sea") else "failed"
            counts[kind] = counts.get(kind, 0) + 1
            if kind == "sea":
                missing.add(terrain.tile_name(*tile))
            elif kind == "failed":
                print(f"    {terrain.tile_name(*tile)}: {status}", flush=True)
            if n % 25 == 0 or n == len(tiles):
                print(f"... {n}/{len(tiles)} tiles {counts}, {(time.time() - started) / 60:.0f} min", flush=True)
    missing_file.write_text("\n".join(sorted(missing)))
    if counts.get("failed"):
        print("Some tiles failed to download; run again to retry them. Their chips are skipped this time.")

    failed = {t for t in tiles if not (args.dem / f"{terrain.tile_name(*t)}.tif").exists()
              and terrain.tile_name(*t) not in missing}
    done_file = parts / "done_chips.txt"
    have = set(int(x) for x in done_file.read_text().split()) if done_file.exists() else set()
    have |= {int(p.stem) for p in parts.glob("*.parquet") if p.stem.isdigit()}  # one file per chip (older runs)
    waiting = [c for c in chips if c not in have and failed & set(terrain.tiles_for_box(*terrain.chip_window(c)))]
    todo = [c for c in chips if c not in have and c not in set(waiting)]
    print(f"\nCells: {len(chips) - len(todo) - len(waiting):,} chips done before, {len(todo):,} to compute"
          + (f", {len(waiting):,} waiting for tiles that failed" if waiting else ""), flush=True)

    dem_dir = args.dem
    if args.local and todo:
        args.local.mkdir(parents=True, exist_ok=True)
        need = sorted({t for c in todo for t in terrain.tiles_for_box(*terrain.chip_window(c))})
        copied = 0
        for t in need:
            src = args.dem / f"{terrain.tile_name(*t)}.tif"
            dst = args.local / src.name
            if src.exists() and (not dst.exists() or dst.stat().st_size != src.stat().st_size):
                shutil.copyfile(src, dst)
                copied += 1
        print(f"Copied {copied} tiles to {args.local} ({len(need)} needed)", flush=True)
        dem_dir = args.local

    runs = [todo[i:i + 20] for i in range(0, len(todo), 20)]  # neighbouring chips share tiles
    started, done, frames, batch = time.time(), 0, [], []
    with ProcessPoolExecutor(args.workers) as pool:
        futures = [pool.submit(process, r, dem_dir) for r in runs]
        for n, f in enumerate(as_completed(futures), 1):
            run_chips, cells = f.result()
            batch += run_chips
            if cells is not None:
                frames.append(cells)
            done += len(run_chips)
            if len(batch) >= 300 or n == len(runs):
                write_batch(parts, frames, batch)
                frames, batch = [], []
            if n % 10 == 0 or n == len(runs):
                print(f"... {done:,}/{len(todo):,} chips, {(time.time() - started) / 60:.0f} min", flush=True)

    files = sorted(parts.glob("batch_*.parquet")) + [p for p in parts.glob("*.parquet") if p.stem.isdigit()]
    with ThreadPoolExecutor(16) as pool:
        frames = [f for f in pool.map(pd.read_parquet, files) if len(f)]
    cells = pd.concat(frames, ignore_index=True).drop_duplicates("cell_id", keep="last")
    cells = cells[cells["chip_id"].isin(chips)]
    cells.to_parquet(args.db / "cells.parquet", index=False)
    print(f"\ncells.parquet: {len(cells):,} cells in {cells['chip_id'].nunique():,} chips "
          f"({len(chips) - cells['chip_id'].nunique():,} chips all sea or still missing)")
    print(cells[["elev_mean", "slope_mean", "relief_5km", "tpi_5km", "nodata_share"]].describe()
          .round(1).loc[["mean", "min", "50%", "max"]].to_string())


if __name__ == "__main__":
    main()
