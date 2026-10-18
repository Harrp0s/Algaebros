"""Per-pixel chlorophyll maps (20 m), for the real heat gradient. Optional: needs a one-time download.

    python scripts/07_fetch_pixels.py --years 2025          # about 10-15 minutes for one season

Each clear scene is saved as a small file in data/pixels/. Re-running skips scenes already saved.
The dashboard uses these when they exist and falls back to a smooth gradient between areas when they do not.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from . import config, satellite

PIXEL_DIR = config.DATA_DIR / "pixels"


def pixel_map(arrs: dict, labels: np.ndarray) -> np.ndarray:
    """NDCI at every zone pixel that is clear open water in this scene; NaN everywhere else (float16 to stay small)."""
    index = satellite.ndci(arrs["b04"], arrs["b05"])
    valid = (arrs["scl"] == config.SCL_WATER) & np.isfinite(arrs["b04"]) & np.isfinite(arrs["b05"]) & (labels > 0)
    return np.where(valid, index, np.nan).astype("float16")


def coverage(arr: np.ndarray, labels: np.ndarray) -> float:
    n = int((labels > 0).sum())
    return float(np.isfinite(arr[labels > 0]).sum()) / n if n else 0.0


def save_scene(date: str, item_id: str, arr: np.ndarray, directory: Path = None) -> Path:
    d = Path(directory or PIXEL_DIR)
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"{date}__{item_id}.npy"
    np.save(p, arr.astype("float16"))
    return p


def list_scenes(directory: Path = None) -> List[Tuple[str, Path]]:
    d = Path(directory or PIXEL_DIR)
    if not d.exists():
        return []
    return sorted((p.name.split("__")[0], p) for p in d.glob("*__*.npy"))


def best_scene_before(as_of: str, labels: np.ndarray, min_coverage: float = 0.4, max_age_days: int = 45,
                      directory: Path = None) -> Optional[Tuple[str, np.ndarray, float]]:
    """The freshest usable scene that a person could have had on `as_of`.

    Same no-peeking rule as the forecast: a scene is only usable OBS_LATENCY_DAYS after it was taken.
    """
    cutoff = np.datetime64(as_of) - np.timedelta64(config.OBS_LATENCY_DAYS, "D")
    oldest = np.datetime64(as_of) - np.timedelta64(max_age_days, "D")
    best = None
    for date, path in reversed(list_scenes(directory)):
        d = np.datetime64(date)
        if d > cutoff:
            continue
        if d < oldest:
            break
        arr = np.load(path).astype("float32")
        if arr.shape != labels.shape:
            continue
        cov = coverage(arr, labels)
        if cov >= min_coverage:
            return date, arr, cov
        if best is None or cov > best[2]:
            best = (date, arr, cov)
    return None


def fetch_season(years: List[int], labels: np.ndarray, grid_x: np.ndarray, grid_y: np.ndarray,
                 limit: Optional[int] = None, log: Callable[[str], None] = print, directory: Path = None) -> int:
    """Download and save per-pixel maps for the bloom-season scenes of the given years. Returns scenes saved."""
    catalog = satellite.open_catalog()
    items = []
    for y in years:
        items += satellite.search_items(catalog, f"{y}-05-01", f"{y}-10-31")
    items.sort(key=lambda i: i.datetime)
    if limit:
        items = items[:limit]
    have = {p.name.split("__")[1][:-4] for _, p in list_scenes(directory)}
    todo = [i for i in items if i.id not in have]
    log(f"{len(items)} scenes in season, {len(items) - len(todo)} already saved, {len(todo)} to download.")
    saved = failures = 0
    for k, item in enumerate(todo, 1):
        date = item.datetime.strftime("%Y-%m-%d")
        try:
            quick = satellite.load_item(item, scl_only=True)
            satellite.check_grid(quick, grid_x, grid_y)
            if satellite.bad_pixel_fraction(quick["scl"]) > config.MAX_BAD_PIXEL_FRAC:
                continue
            arrs = satellite.load_item(item)
            satellite.check_grid(arrs, grid_x, grid_y)
            arr = pixel_map(arrs, labels)
            if coverage(arr, labels) >= 0.2:
                save_scene(date, item.id, arr, directory)
                saved += 1
        except satellite.GridMismatch as e:
            log(f"  {date} grid mismatch, skipped: {e}")
            failures += 1
        except Exception as e:
            log(f"  {date} FAILED ({type(e).__name__}: {str(e)[:100]}), will retry next run")
            failures += 1
        if k % 10 == 0 or k == len(todo):
            log(f"  [{k}/{len(todo)}] saved {saved} so far")
    if failures:
        log(f"{failures} scenes failed; run the same command again to retry them.")
    return saved
