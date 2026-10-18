"""Split the river into zones (reaches) using the satellite's own view of where the water is.

We do NOT hand-place zone coordinates. Instead:
  1. count how often each 20 m pixel is classified as water on clear days,
  2. keep pixels that are water almost every time, shrink away from the banks (mixed pixels),
  3. drop tiny ponds, then cluster the remaining pixels into N zones, numbered west to east.

Because bridges are land pixels, the river appears as several pieces; that is fine.
"""
from __future__ import annotations

import json
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage
from sklearn.cluster import KMeans

from . import config, geo

MIN_WATER_FREQ = 0.6
MIN_COMPONENT_PIXELS = 150  # 150 pixels * 400 m^2 = 0.06 km^2; smaller bits (ponds) are ignored


def build_zone_labels(
    freq: np.ndarray,
    n_zones: int = config.N_ZONES,
    erode: int = config.ERODE_PIXELS,
    min_freq: float = MIN_WATER_FREQ,
    min_component: int = MIN_COMPONENT_PIXELS,
    first: int = 1,
) -> np.ndarray:
    """Return an int array: 0 = not a zone, first..first+n-1 = zone number (west to east)."""
    water = freq >= min_freq
    core = ndimage.binary_erosion(water, iterations=erode) if erode > 0 else water
    comp, n = ndimage.label(core)
    if n == 0:
        raise RuntimeError("No water found. Check config.BBOX and that scenes loaded correctly.")
    sizes = ndimage.sum(core, comp, index=np.arange(1, n + 1))
    keep_ids = [i + 1 for i, s in enumerate(sizes) if s >= min_component]
    if not keep_ids:
        raise RuntimeError(
            f"Water was found but every piece is smaller than {min_component} pixels after trimming the banks. "
            "Try ERODE_PIXELS = 0 in config.py or widen BBOX."
        )
    mask = np.isin(comp, keep_ids)
    rows, cols = np.nonzero(mask)
    n_zones = min(n_zones, max(1, len(rows) // 40))
    coords = np.column_stack([cols, rows]).astype("float64")
    km = KMeans(n_clusters=n_zones, n_init=10, random_state=0).fit(coords)
    # number zones west -> east by cluster centre
    order = np.argsort(km.cluster_centers_[:, 0])
    rank = np.empty(n_zones, dtype=int)
    rank[order] = np.arange(first, first + n_zones)
    labels = np.zeros(freq.shape, dtype="int16")
    labels[rows, cols] = rank[km.labels_]
    return labels


def zone_records(labels: np.ndarray, x: np.ndarray, y: np.ndarray, names: Optional[Dict[str, str]] = None,
                 max_points: int = 300, site_of: Optional[Dict[int, str]] = None) -> List[dict]:
    """Describe each zone: centre, area, and a sample of its water pixels (lon/lat) for maps and 'near me'."""
    names = names or {}
    site_of = site_of or {}
    rng = np.random.default_rng(0)
    out = []
    for z in range(1, int(labels.max()) + 1):
        rr, cc = np.nonzero(labels == z)
        if len(rr) == 0:
            continue
        xs, ys = x[cc], y[rr]
        lon_c, lat_c = geo.utm_to_lonlat(float(xs.mean()), float(ys.mean()))
        pick = rng.choice(len(rr), size=min(max_points, len(rr)), replace=False)
        lons, lats = geo.utm_to_lonlat(xs[pick], ys[pick])
        zid = f"Z{z:02d}"
        site = site_of.get(z, config.DEFAULT_SITE)
        out.append(
            {
                "id": zid,
                "number": z,
                "site": site,
                "site_name": config.SITE_BY_KEY.get(site, {}).get("name", site),
                "name": names.get(zid, f"Zone {z}"),
                "n_pixels": int(len(rr)),
                "area_km2": round(len(rr) * (config.RESOLUTION_M ** 2) / 1e6, 3),
                "lat": round(lat_c, 5),
                "lon": round(lon_c, 5),
                "points": [[round(float(a), 5), round(float(b), 5)] for a, b in zip(lons, lats)],  # [lon, lat]
            }
        )
    return out


def load_names() -> Dict[str, str]:
    if config.ZONE_NAMES_JSON.exists():
        try:
            return json.loads(config.ZONE_NAMES_JSON.read_text())
        except Exception:
            return {}
    return {}


def save_zones(labels: np.ndarray, x: np.ndarray, y: np.ndarray, freq: Optional[np.ndarray] = None) -> List[dict]:
    np.savez_compressed(config.ZONES_GRID, labels=labels, x=x, y=y, freq=freq if freq is not None else np.zeros(labels.shape, "float32"))
    recs = zone_records(labels, x, y, load_names())
    config.ZONES_JSON.write_text(json.dumps({"bbox": config.BBOX, "zones": recs}, indent=1))
    return recs


def load_grid():
    g = np.load(config.ZONES_GRID)
    return g["labels"], g["x"], g["y"]


def load_zone_records() -> List[dict]:
    data = json.loads(config.ZONES_JSON.read_text())
    names = load_names()
    for z in data["zones"]:
        z["name"] = names.get(z["id"], z.get("name", f"Zone {z['number']}"))
        z.setdefault("site", config.DEFAULT_SITE)
        z.setdefault("site_name", config.SITE_BY_KEY.get(z["site"], {}).get("name", z["site"]))
    return data["zones"]


# ---------------------------------------------------------------------- several water bodies
def save_site_grid(key: str, labels: np.ndarray, x: np.ndarray, y: np.ndarray, freq: Optional[np.ndarray] = None) -> None:
    config.site_dir(key).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(config.site_grid_path(key), labels=labels, x=x, y=y,
                        freq=freq if freq is not None else np.zeros(labels.shape, "float32"))


def load_site_grid(key: str):
    g = np.load(config.site_grid_path(key))
    return g["labels"], g["x"], g["y"], g["freq"]


def built_sites() -> List[str]:
    return [s["key"] for s in config.SITES if config.site_grid_path(s["key"]).exists()]


def migrate_single_site() -> bool:
    """Older builds kept only the Lower Charles grid in data/zones_grid.npz: file it as the 'charles' site."""
    if config.site_grid_path(config.DEFAULT_SITE).exists() or not config.ZONES_GRID.exists():
        return False
    g = np.load(config.ZONES_GRID)
    save_site_grid(config.DEFAULT_SITE, g["labels"], g["x"], g["y"], g["freq"] if "freq" in g else None)
    return True


def next_zone_number(exclude: str) -> int:
    """First free zone number for a site being (re)built: after every other built site's zones."""
    top = 0
    for key in built_sites():
        if key != exclude:
            top = max(top, int(load_site_grid(key)[0].max()))
    return top + 1


def mosaic(grids: List[tuple], res: float = config.RESOLUTION_M):
    """Paste several site grids (labels, x, y, freq) into one grid covering them all.

    Every site is read on the same 20 m UTM grid, so each one is an exact window of the union.
    Where two windows overlap, the site listed first keeps the pixel.
    """
    xs = np.concatenate([g[1] for g in grids])
    ys = np.concatenate([g[2] for g in grids])
    x = np.arange(xs.min(), xs.max() + res / 2, res)
    y = np.arange(ys.max(), ys.min() - res / 2, -res)          # north-up, like the scenes
    labels = np.zeros((len(y), len(x)), dtype="int16")
    freq = np.zeros((len(y), len(x)), dtype="float32")
    for lab, gx, gy, gf in grids:
        c0 = int(round((gx[0] - x[0]) / res))
        r0 = int(round((y[0] - gy[0]) / res))
        if not (np.allclose(x[c0:c0 + len(gx)], gx, atol=1.0) and np.allclose(y[r0:r0 + len(gy)], gy, atol=1.0)):
            raise ValueError("site grids are not on the same 20 m grid")
        win = labels[r0:r0 + len(gy), c0:c0 + len(gx)]
        take = (win == 0) & (lab > 0)
        win[take] = lab[take]
        fwin = freq[r0:r0 + len(gy), c0:c0 + len(gx)]
        np.maximum(fwin, gf, out=fwin)
    return labels, x, y, freq


def build_union() -> List[dict]:
    """Rebuild data/zones_grid.npz and data/zones.json from every site built so far."""
    keys = built_sites()
    if not keys:
        raise RuntimeError("No site grids yet. Run scripts/01_build_zones.py.")
    grids, site_of, bboxes = [], {}, {}
    for key in keys:
        lab, gx, gy, gf = load_site_grid(key)
        grids.append((lab, gx, gy, gf))
        for z in np.unique(lab[lab > 0]):
            site_of[int(z)] = key
        bboxes[key] = config.SITE_BY_KEY[key]["bbox"]
    labels, x, y, freq = mosaic(grids)
    np.savez_compressed(config.ZONES_GRID, labels=labels, x=x, y=y, freq=freq)
    recs = zone_records(labels, x, y, load_names(), site_of=site_of)
    lons = [b[0] for b in bboxes.values()] + [b[2] for b in bboxes.values()]
    lats = [b[1] for b in bboxes.values()] + [b[3] for b in bboxes.values()]
    config.ZONES_JSON.write_text(json.dumps({"bbox": [min(lons), min(lats), max(lons), max(lats)],
                                             "sites": [{"key": k, "name": config.SITE_BY_KEY[k]["name"], "bbox": bboxes[k]} for k in keys],
                                             "zones": recs}, indent=1))
    return recs


def site_of_zone_ids() -> Dict[str, str]:
    """{'Z01': 'charles', ...} from zones.json (every zone is 'charles' in older single-site data)."""
    try:
        return {z["id"]: z.get("site", config.DEFAULT_SITE) for z in json.loads(config.ZONES_JSON.read_text())["zones"]}
    except (OSError, ValueError, KeyError):
        return {}


def save_preview(labels: np.ndarray, freq: np.ndarray, recs: List[dict], x: np.ndarray, y: np.ndarray,
                 path=None, title: Optional[str] = None) -> None:
    """A picture to check by eye that the zones really sit on the river."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 6))
    extent = [x[0], x[-1], y[-1], y[0]] if y[0] > y[-1] else [x[0], x[-1], y[0], y[-1]]
    ax.imshow(freq, cmap="Blues", extent=extent, origin="upper" if y[0] > y[-1] else "lower", vmin=0, vmax=1)
    shown = np.ma.masked_equal(labels, 0)
    ax.imshow(shown, cmap="tab20", alpha=0.75, extent=extent, origin="upper" if y[0] > y[-1] else "lower",
              vmin=1, vmax=max(10, int(labels.max())))
    for r in recs:
        ux, uy = geo.lonlat_to_utm(r["lon"], r["lat"])
        ax.text(ux, uy, str(r["number"]), ha="center", va="center", fontsize=11, weight="bold", color="black",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.8))
    ax.set_title(title or "Algaebro zones on the water mask (light blue = how often a pixel is water). Check they sit on the river.")
    ax.set_xlabel("UTM easting (m)")
    ax.set_ylabel("UTM northing (m)")
    fig.tight_layout()
    fig.savefig(path or config.ZONES_PREVIEW, dpi=130)
    plt.close(fig)
