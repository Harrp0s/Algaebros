"""The heat overlay: a smooth colour gradient along the river, drawn as one transparent PNG over the map.

Two sources, and the dashboard always says which one it is showing:

  pixels        Real satellite pixels (20 m), from a saved clear scene. Needs the one-time download
                (python scripts/07_fetch_pixels.py). Cloud-covered pixels stay transparent.
  interpolated  The satellite gives one reading per area (zone). Without saved pixels we blend between neighbouring
                areas (inverse-distance weighting) and smooth the result. This is an INTERPOLATION between measured
                areas, not a pixel-by-pixel measurement.

Both go through the same renderer, so switching sources never changes the look.
"""
from __future__ import annotations

import io
import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

from . import geo

PX_M = 8.0              # metres per output pixel
SOURCE_PIXEL_M = 20.0   # one satellite pixel is about this wide; each is drawn at its real footprint
CORE_SIGMA_M = 34.0     # how crisply the colour hugs the water (wider than the 20 m pixels, so there is no dotting)
GLOW_SIGMA_M = 95.0     # the soft halo around it
VALUE_SIGMA_M = 75.0    # smoothing of the colour values themselves (this is what makes it a gradient)
MAX_SIDE = 1100

MINT = (112, 220, 190)
YELLOW = (246, 214, 74)
ORANGE = (242, 140, 40)
RED = (214, 48, 31)


# ---------------------------------------------------------------------- colours
def color_stops(thresholds: dict) -> List[Tuple[float, Tuple[int, int, int]]]:
    """Colour positions follow the thresholds, so 'the flag starts here' is always the orange stop."""
    hi, el = float(thresholds["high"]), float(thresholds["elevated"])
    return [(el - 0.16, MINT), (el, YELLOW), (hi, ORANGE), (hi + 0.14, RED)]


def legend(thresholds: dict) -> dict:
    return {"stops": [{"value": round(v, 4), "color": "#%02x%02x%02x" % c} for v, c in color_stops(thresholds)],
            "elevated": float(thresholds["elevated"]), "high": float(thresholds["high"])}


def colorize(values: np.ndarray, thresholds: dict) -> np.ndarray:
    """values (h, w) -> rgb (h, w, 3) uint8 along the piecewise-linear ramp."""
    stops = color_stops(thresholds)
    xs = np.array([s[0] for s in stops], dtype=np.float32)
    out = np.zeros(values.shape + (3,), dtype=np.float32)
    for ch in range(3):
        out[..., ch] = np.interp(values, xs, np.array([s[1][ch] for s in stops], dtype=np.float32))
    return out.astype(np.uint8)


# ---------------------------------------------------------------------- image maths (numpy only)
def _gauss_kernel(sigma_px: float) -> np.ndarray:
    r = max(1, int(math.ceil(3 * sigma_px)))
    x = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2 * sigma_px * sigma_px))
    return k / k.sum()


def gaussian_blur(a: np.ndarray, sigma_px: float) -> np.ndarray:
    k = _gauss_kernel(sigma_px)
    r = len(k) // 2
    out = a.astype(np.float32)
    for axis in (0, 1):
        pad = [(0, 0), (0, 0)]
        pad[axis] = (r, r)
        padded = np.pad(out, pad, mode="constant")
        acc = np.zeros_like(out)
        for i, w in enumerate(k):
            sl = [slice(None), slice(None)]
            sl[axis] = slice(i, i + out.shape[axis])
            acc += w * padded[tuple(sl)]
        out = acc
    return out


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Grow a 0/1 mask by `radius` cells in every direction."""
    if radius <= 0:
        return mask
    h, w = mask.shape
    out = mask.copy()
    padded = np.pad(mask, radius)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            np.maximum(out, padded[dy:dy + h, dx:dx + w], out=out)
    return out


def smoothstep(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    t = np.clip((x - lo) / (hi - lo), 0.0, 1.0)
    return t * t * (3 - 2 * t)


# ---------------------------------------------------------------------- where the pixels are
_pixel_cache: Dict[int, dict] = {}


def pixel_coords(labels: np.ndarray, x: np.ndarray, y: np.ndarray) -> dict:
    """Longitude and latitude of every labelled water pixel in the zone grid (cached)."""
    key = id(labels)
    hit = _pixel_cache.get(key)
    if hit is not None and hit["shape"] == labels.shape:
        return hit
    rr, cc = np.nonzero(labels > 0)
    lon, lat = geo.utm_to_lonlat(np.asarray(x)[cc], np.asarray(y)[rr])
    out = {"shape": labels.shape, "rr": rr, "cc": cc, "lon": np.asarray(lon, dtype=np.float64),
           "lat": np.asarray(lat, dtype=np.float64), "zone": labels[rr, cc].astype(int)}
    _pixel_cache.clear()
    _pixel_cache[key] = out
    return out


def interpolate_values(fc: dict, lon: np.ndarray, lat: np.ndarray, zone_number: np.ndarray, centers: List[dict],
                       power: float = 2.0) -> np.ndarray:
    """Per-pixel value as an inverse-distance blend of the area readings; NaN where the area has no reading.

    centers: [{'number', 'lat', 'lon'[, 'site']}, ...] for the areas, in any order. When centres carry a 'site',
    a pixel only blends the areas of its own water body (a lake never borrows the river's colour).
    """
    sites = {c["number"]: c.get("site") for c in centers}
    if len({v for v in sites.values()}) > 1:
        out = np.full(lon.shape, np.nan, dtype=np.float64)
        pix_site = np.array([sites.get(int(n)) for n in zone_number], dtype=object)
        for site in set(sites.values()):
            m = pix_site == site
            if m.any():
                out[m] = interpolate_values(fc, lon[m], lat[m], zone_number[m],
                                            [{k: v for k, v in c.items() if k != "site"} for c in centers if c.get("site") == site], power)
        return out
    reading = {z["number"]: z["last_obs"] for z in fc["zones"] if z.get("last_obs") is not None}
    cs = [c for c in centers if c["number"] in reading]
    out = np.full(lon.shape, np.nan, dtype=np.float64)
    if not cs or lon.size == 0:
        return out
    c_lat = np.array([c["lat"] for c in cs])
    c_lon = np.array([c["lon"] for c in cs])
    c_val = np.array([reading[c["number"]] for c in cs])
    lat0 = float(np.mean(lat))
    dx = (lon[:, None] - c_lon[None, :]) * 111320.0 * math.cos(math.radians(lat0))
    dy = (lat[:, None] - c_lat[None, :]) * 110574.0
    d = np.sqrt(dx * dx + dy * dy) + 40.0       # +40 m keeps a pixel from locking onto one centre
    w = 1.0 / d ** power
    val = (w * c_val[None, :]).sum(axis=1) / w.sum(axis=1)
    has = np.isin(zone_number, list(reading))   # areas with no reading stay transparent: the honest picture
    out[has] = val[has]
    return out


# ---------------------------------------------------------------------- rendering
def render(points: np.ndarray, bbox: Tuple[float, float, float, float], thresholds: dict, px_m: float = PX_M) -> Tuple[bytes, dict]:
    """points: (n, 3) lon, lat, value  ->  (png bytes, meta with the georeferenced corners)."""
    lon0, lat0, lon1, lat1 = bbox
    mid = (lat0 + lat1) / 2
    # several water bodies make a big box: coarsen the output pixel so the image stays within MAX_SIDE
    span = max((lon1 - lon0) * 111320.0 * math.cos(math.radians(mid)), (lat1 - lat0) * 110574.0)
    px_m = max(px_m, span / MAX_SIDE)
    w = int(min(MAX_SIDE, max(8, math.ceil((lon1 - lon0) * 111320.0 * math.cos(math.radians(mid)) / px_m))))
    h = int(min(MAX_SIDE, max(8, math.ceil((lat1 - lat0) * 110574.0 / px_m))))
    meta = {"bounds": [lon0, lat0, lon1, lat1], "width": w, "height": h,
            "coordinates": [[lon0, lat1], [lon1, lat1], [lon1, lat0], [lon0, lat0]],   # NW, NE, SE, SW (MapLibre order)
            "legend": legend(thresholds)}
    if len(points) == 0:
        buf = io.BytesIO()
        Image.new("RGBA", (w, h), (0, 0, 0, 0)).save(buf, "PNG")
        return buf.getvalue(), meta

    ix = np.clip(np.round((points[:, 0] - lon0) / (lon1 - lon0) * (w - 1)).astype(int), 0, w - 1)
    iy = np.clip(np.round((lat1 - points[:, 1]) / (lat1 - lat0) * (h - 1)).astype(int), 0, h - 1)
    num = np.zeros((h, w), dtype=np.float32)
    den = np.zeros((h, w), dtype=np.float32)
    mask = np.zeros((h, w), dtype=np.float32)
    np.add.at(num, (iy, ix), points[:, 2].astype(np.float32))
    np.add.at(den, (iy, ix), 1.0)
    mask[den > 0] = 1.0
    mask = dilate(mask, int(round(SOURCE_PIXEL_M / px_m / 2)))     # each satellite pixel at its real footprint

    s_val = VALUE_SIGMA_M / px_m
    val = gaussian_blur(num, s_val) / np.maximum(gaussian_blur(den, s_val), 1e-6)
    core = smoothstep(gaussian_blur(mask, CORE_SIGMA_M / px_m), 0.18, 0.62)
    glow = 0.42 * smoothstep(gaussian_blur(mask, GLOW_SIGMA_M / px_m), 0.03, 0.30)
    alpha = np.clip(np.maximum(core * 0.92, glow), 0, 1)

    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[..., :3] = colorize(val, thresholds)
    rgba[..., 3] = (alpha * 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=True)
    return buf.getvalue(), meta


def bbox_of(lon: np.ndarray, lat: np.ndarray, pad_m: float = 250.0) -> Tuple[float, float, float, float]:
    mid = float(np.mean(lat))
    dlat = pad_m / 110574.0
    dlon = pad_m / (111320.0 * math.cos(math.radians(mid)))
    return (float(lon.min() - dlon), float(lat.min() - dlat), float(lon.max() + dlon), float(lat.max() + dlat))


def heat_overlay(fc: dict, zones: List[dict], labels: Optional[np.ndarray], x: Optional[np.ndarray], y: Optional[np.ndarray],
                 prefer: str = "auto", scene_loader: Optional[Callable] = None) -> Tuple[bytes, dict]:
    """The overlay for the forecast being shown: (png bytes, meta).

    prefer: 'auto' (real pixels if a usable saved scene exists, else interpolated), 'pixels', or 'interpolated'.
    scene_loader(as_of, labels) -> (date, array, coverage) or None. Defaults to pixels.best_scene_before.
    """
    thresholds = fc["thresholds"]
    if labels is not None and x is not None and y is not None:
        pc = pixel_coords(labels, x, y)
        lon, lat, zone = pc["lon"], pc["lat"], pc["zone"]
    else:   # no pixel grid: fall back to the sampled points stored with each zone
        pts = [(p[0], p[1], z["number"]) for z in zones for p in z["points"]]
        arr = np.array(pts, dtype=np.float64).reshape(-1, 3)
        lon, lat, zone = arr[:, 0], arr[:, 1], arr[:, 2].astype(int)
        pc = None
    bbox = bbox_of(lon, lat)

    scene = None
    if pc is not None and prefer in ("auto", "pixels"):
        if scene_loader is None:
            from . import pixels
            scene_loader = lambda as_of, lab: pixels.best_scene_before(as_of, lab)    # noqa: E731
        try:
            scene = scene_loader(fc["as_of"], labels)
        except Exception:
            scene = None

    if scene is not None:
        date, arr2d, cov = scene
        vals = arr2d[pc["rr"], pc["cc"]].astype(np.float64)
        ok = np.isfinite(vals)
        points = np.column_stack([lon[ok], lat[ok], vals[ok]])
        extra = {"source": "pixels", "scene_date": str(date), "coverage": round(float(cov), 2),
                 "note": f"Real satellite pixels from {date}. Pixels hidden by cloud are left transparent."}
    else:
        centers = [{"number": z["number"], "lat": z["lat"], "lon": z["lon"], "site": z.get("site", "charles")} for z in zones]
        vals = interpolate_values(fc, lon, lat, zone, centers)
        ok = np.isfinite(vals)
        points = np.column_stack([lon[ok], lat[ok], vals[ok]])
        why = ("" if prefer != "pixels" else " (no saved satellite pixels for this date)")
        extra = {"source": "interpolated", "scene_date": None, "coverage": None,
                 "note": "Smooth blend between the measured areas, not a pixel-by-pixel measurement" + why +
                         ". Run scripts/07_fetch_pixels.py for the real pixels."}
    png, meta = render(points, bbox, thresholds)
    meta.update(extra)
    return png, meta


def medoid(zone: dict) -> Tuple[float, float]:
    """The zone's own water pixel closest to its centre: a point certainly on the river (for markers). Returns (lat, lon)."""
    best = min(zone["points"], key=lambda p: geo.haversine_m(zone["lat"], zone["lon"], p[1], p[0]))
    return best[1], best[0]
