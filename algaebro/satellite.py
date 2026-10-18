"""Sentinel-2 access (Microsoft Planetary Computer, free, no account) and the chlorophyll index.

The index is NDCI (Normalized Difference Chlorophyll Index):

    NDCI = (B05 - B04) / (B05 + B04)        B04 = red 665 nm, B05 = red edge 705 nm

Higher NDCI means more chlorophyll-a in the surface water. It is a PROXY for algae of all
kinds (not only cyanobacteria) and it is NOT a toxin or safety measurement.

Everything that touches the network lives in a few small functions (open_catalog,
search_items, load_item). The math (ndci, zone_stats, dn_offset) is plain numpy and is
unit-tested without any network.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from . import config


class GridMismatch(RuntimeError):
    """A scene did not come back on the same pixel grid as the saved zones."""


# --------------------------------------------------------------------------- math
def dn_offset(item) -> int:
    """Offset to ADD to raw Sentinel-2 digital numbers before dividing by 10000.

    Scenes processed with baseline 04.00 or newer (from 25 Jan 2022) carry a +1000 shift.
    Band ratios like NDCI are sensitive to it, so we undo it per scene.
    """
    baseline = item.properties.get("s2:processing_baseline")
    if baseline is not None:
        try:
            return -1000 if float(str(baseline)) >= 4.0 else 0
        except ValueError:
            pass
    dt = item.datetime
    if dt is not None and dt.replace(tzinfo=None) >= datetime(2022, 1, 25):
        return -1000
    return 0


def to_reflectance(dn: np.ndarray, offset: int) -> np.ndarray:
    """Raw DN -> surface reflectance. DN 0 is 'no data' and becomes NaN."""
    dn = dn.astype("float32")
    refl = (dn + offset) / 10000.0
    refl[dn == 0] = np.nan
    refl[refl <= 0.001] = np.nan  # non-physical or essentially black: unusable for a ratio
    return refl


def ndci(b04: np.ndarray, b05: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        v = (b05 - b04) / (b05 + b04)
    v = v.astype("float32")
    v[~np.isfinite(v)] = np.nan
    return v


def bad_pixel_fraction(scl: np.ndarray) -> float:
    return float(np.isin(scl, config.SCL_BAD).mean())


def zone_stats(
    index: np.ndarray,
    valid: np.ndarray,
    labels: np.ndarray,
    n_zones: int,
    min_pixels: int = config.MIN_ZONE_PIXELS,
    min_frac: float = config.MIN_VALID_FRAC,
) -> List[dict]:
    """Median index per zone, only for zones with enough clean water pixels."""
    rows = []
    for z in range(1, n_zones + 1):
        zmask = labels == z
        total = int(zmask.sum())
        if total == 0:
            continue
        m = zmask & valid & np.isfinite(index)
        n = int(m.sum())
        if n < min_pixels or n / total < min_frac:
            continue
        vals = index[m]
        rows.append(
            {
                "zone": z,
                "ndci": float(np.median(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p75": float(np.percentile(vals, 75)),
                "n_valid": n,
                "frac_valid": round(n / total, 3),
            }
        )
    return rows


# --------------------------------------------------------------------------- network
def open_catalog():
    import planetary_computer
    from pystac_client import Client

    return Client.open(config.STAC_URL, modifier=planetary_computer.sign_inplace)


def search_items(catalog, start: str, end: str, bbox=None, max_cloud: Optional[float] = None):
    """Sentinel-2 L2A scenes over the study box, restricted to the bloom-season months."""
    bbox = bbox or config.BBOX
    search = catalog.search(
        collections=[config.COLLECTION],
        bbox=list(bbox),
        datetime=f"{start}/{end}",
        query={"eo:cloud_cover": {"lt": max_cloud if max_cloud is not None else config.MAX_ITEM_CLOUD}},
    )
    items = [i for i in search.item_collection() if i.datetime and i.datetime.month in config.SEASON_MONTHS]
    items.sort(key=lambda i: i.datetime)
    return items


def scl_to_uint8(values: np.ndarray) -> np.ndarray:
    """Scene-classification layer -> uint8, turning no-data (NaN) into 0 instead of an undefined cast."""
    v = np.nan_to_num(np.asarray(values, dtype="float64"), nan=0.0, posinf=0.0, neginf=0.0)
    return np.clip(v, 0, 255).astype("uint8")


def load_item(item, bbox=None, scl_only: bool = False) -> Dict[str, np.ndarray]:
    """Read B04, B05 and the scene classification for one scene on the shared 20 m grid.

    With scl_only=True only the (small) classification layer is read: enough to see if a scene is
    too cloudy before paying to download the two reflectance bands.
    """
    from odc.stac import load as odc_load

    bbox = bbox or config.BBOX
    bands = ["SCL"] if scl_only else ["B04", "B05", "SCL"]
    per_band = {"B04": "average", "B05": "nearest", "SCL": "nearest"}
    kwargs = dict(bands=bands, bbox=bbox, crs=config.CRS, resolution=config.RESOLUTION_M)
    try:
        ds = odc_load([item], resampling={b: per_band[b] for b in bands}, **kwargs)
    except Exception:
        ds = odc_load([item], resampling="nearest", **kwargs)
    out = {"scl": scl_to_uint8(ds["SCL"].isel(time=0).values), "x": ds["x"].values, "y": ds["y"].values}
    if not scl_only:
        off = dn_offset(item)
        out["b04"] = to_reflectance(ds["B04"].isel(time=0).values, off)
        out["b05"] = to_reflectance(ds["B05"].isel(time=0).values, off)
    return out


def check_grid(arrs: Dict[str, np.ndarray], grid_x: np.ndarray, grid_y: np.ndarray) -> None:
    if arrs["scl"].shape != (len(grid_y), len(grid_x)):
        raise GridMismatch(f"scene shape {arrs['scl'].shape} != saved grid {(len(grid_y), len(grid_x))}")
    if not (np.allclose(arrs["x"], grid_x, atol=1.0) and np.allclose(arrs["y"], grid_y, atol=1.0)):
        raise GridMismatch("scene pixel coordinates differ from the saved grid")


def process_arrays(arrs: Dict[str, np.ndarray], labels: np.ndarray, n_zones: int) -> dict:
    """Turn one scene's arrays into per-zone readings (pure numpy, easy to test)."""
    bad = bad_pixel_fraction(arrs["scl"])
    if bad > config.MAX_BAD_PIXEL_FRAC:
        return {"status": "skipped_cloudy", "bad_frac": round(bad, 3), "rows": []}
    index = ndci(arrs["b04"], arrs["b05"])
    valid = (arrs["scl"] == config.SCL_WATER) & np.isfinite(arrs["b04"]) & np.isfinite(arrs["b05"])
    rows = zone_stats(index, valid, labels, n_zones)
    return {"status": "ok" if rows else "skipped_no_clean_water", "bad_frac": round(bad, 3), "rows": rows}


# --------------------------------------------------------------------------- resumable fetch
def _read_cache(path=None) -> Dict[str, dict]:
    path = path or config.NDCI_CACHE
    done: Dict[str, dict] = {}
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                rec = json.loads(line)
                done[rec["id"]] = rec
    return done


def fetch_all(
    start: str,
    end: str,
    labels: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    n_zones: int,
    limit: Optional[int] = None,
    log: Callable[[str], None] = print,
    site: str = config.DEFAULT_SITE,
) -> pd.DataFrame:
    """Download every season scene in [start, end] over one site, compute zone readings, cache each scene.

    Safe to stop and re-run: finished scenes are skipped. Failed scenes are retried next time.
    `labels` carries the site's global zone numbers, so `n_zones` is the highest number on it.
    """
    bbox = config.SITE_BY_KEY[site]["bbox"]
    cache = config.site_cache_path(site)
    cache.parent.mkdir(parents=True, exist_ok=True)
    catalog = open_catalog()
    items = search_items(catalog, start, end, bbox=bbox)
    log(f"Found {len(items)} Sentinel-2 scenes in season between {start} and {end}.")
    if limit:
        items = items[:limit]
        log(f"(limited to the first {limit} for a quick test)")
    done = _read_cache(cache)
    todo = [i for i in items if i.id not in done]
    log(f"{len(items) - len(todo)} already cached, {len(todo)} to process.")

    failures = 0
    with cache.open("a") as fh:
        for k, item in enumerate(todo, 1):
            try:
                quick = load_item(item, bbox=bbox, scl_only=True)
                check_grid(quick, grid_x, grid_y)
                bad = bad_pixel_fraction(quick["scl"])
                if bad > config.MAX_BAD_PIXEL_FRAC:
                    res = {"status": "skipped_cloudy", "bad_frac": round(bad, 3), "rows": []}
                else:
                    arrs = load_item(item, bbox=bbox)
                    check_grid(arrs, grid_x, grid_y)
                    res = process_arrays(arrs, labels, n_zones)
            except GridMismatch as e:
                log(f"  [{k}/{len(todo)}] {item.datetime:%Y-%m-%d} grid mismatch, skipped: {e}")
                failures += 1
                continue
            except Exception as e:  # network hiccup, corrupt tile, etc.: retry on next run
                log(f"  [{k}/{len(todo)}] {item.datetime:%Y-%m-%d} FAILED ({type(e).__name__}: {str(e)[:120]}), will retry next run")
                failures += 1
                continue
            rec = {"id": item.id, "date": item.datetime.strftime("%Y-%m-%d"), **res}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            if k % 10 == 0 or k == len(todo):
                log(f"  [{k}/{len(todo)}] {rec['date']} {rec['status']} ({len(rec['rows'])} zones)")
    if failures:
        log(f"{failures} scenes failed; re-run the same command to retry them.")
    return build_ndci_table()


def build_ndci_table() -> pd.DataFrame:
    """Flatten every site's cache into data/ndci.csv: one row per (zone, date), with the site it belongs to."""
    rows = []
    for s in config.SITES:
        for rec in _read_cache(config.site_cache_path(s["key"])).values():
            for r in rec.get("rows", []):
                rows.append({"date": rec["date"], "zone_id": f"Z{int(r['zone']):02d}", "site": s["key"], "item_id": rec["id"],
                             **{k: r[k] for k in ("ndci", "p25", "p75", "n_valid", "frac_valid")}})
    df = pd.DataFrame(rows)
    if df.empty:
        df = pd.DataFrame(columns=["date", "zone_id", "site", "item_id", "ndci", "p25", "p75", "n_valid", "frac_valid"])
    else:
        # two scenes on the same day (overlapping tiles): keep the one with more clean pixels
        df = df.sort_values("n_valid", ascending=False).drop_duplicates(["date", "zone_id"]).sort_values(["date", "zone_id"])
    df.to_csv(config.NDCI_CSV, index=False)
    return df


# --------------------------------------------------------------------------- water frequency (for zones)
def water_frequency(items: Iterable, log: Callable[[str], None] = print, bbox=None):
    """Share of clear observations in which each pixel was classified as water."""
    water_count = None
    seen_count = None
    x = y = None
    used = 0
    for item in items:
        try:
            arrs = load_item(item, bbox=bbox, scl_only=True)
        except Exception as e:
            log(f"  skipped {item.id}: {type(e).__name__}: {str(e)[:100]}")
            continue
        scl = arrs["scl"]
        if water_count is None:
            water_count = np.zeros(scl.shape, dtype="float32")
            seen_count = np.zeros(scl.shape, dtype="float32")
            x, y = arrs["x"], arrs["y"]
        elif scl.shape != water_count.shape:
            continue
        observable = ~np.isin(scl, config.SCL_BAD) & (scl != 0)
        seen_count += observable
        water_count += (scl == config.SCL_WATER) & observable
        used += 1
        log(f"  scene {used}: {item.datetime:%Y-%m-%d}")
    if water_count is None or used == 0:
        raise RuntimeError("Could not read any scene to find the river. Run scripts/00_check_apis.py.")
    with np.errstate(divide="ignore", invalid="ignore"):
        freq = np.where(seen_count > 0, water_count / seen_count, 0.0)
    return freq.astype("float32"), x, y, used
