"""Settings for Algaebro. Change values here, not deep in the code.

Environment variables (see .env.example):
  ALGAEBRO_DATA_DIR  where generated data, model and database live (default ./data)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("ALGAEBRO_DATA_DIR", ROOT / "data")).resolve()
DATA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- study area
# (lon_min, lat_min, lon_max, lat_max) around the Lower Charles River Basin.
# These numbers are from memory: LOOK AT data/zones_preview.png after step 1 and
# adjust if the river is cut off or the wrong water body is included.
BBOX = (-71.140, 42.340, -71.068, 42.380)
# Weather grids are ~10 km, so one point stands in for the whole basin.
WEATHER_POINT = (42.357, -71.100)  # (lat, lon)

# Every water body we watch. Each site has its own box, its own satellite grid and its own cache, and its zones
# are numbered after the previous site's (the Lower Charles keeps Z01-Z10). The extra boxes come from
# OpenStreetMap outlines plus a margin; check data/sites/<key>/zones_preview.png after building them.
# All sites lie within ~12 km of WEATHER_POINT, i.e. within about one weather-model grid cell, so they share it.
#   erode / min_component / min_freq: narrow water (the Upper Charles is 30-60 m wide) cannot lose a
#   pixel to the banks, so it keeps edge pixels; those mix in some land, which the dashboard says.
SITES = [
    {"key": "charles", "name": "Lower Charles River", "bbox": BBOX, "n_zones": 10,
     "erode": 1, "min_component": 150, "min_freq": 0.6, "kind": "river"},
    {"key": "upper_charles", "name": "Charles above the Basin (Watertown)", "bbox": (-71.245, 42.350, -71.1405, 42.3785),
     "n_zones": 2, "erode": 0, "min_component": 100, "min_freq": 0.5, "kind": "river",
     "low_confidence": True},   # 30-60 m wide: edge pixels mix in land, so readings are noisy; shown, never alerted
    {"key": "mystic_lakes", "name": "Mystic Lakes", "bbox": (-71.160, 42.418, -71.137, 42.450),
     "n_zones": 3, "erode": 1, "min_component": 150, "min_freq": 0.6, "kind": "lake"},
    {"key": "fresh_spy", "name": "Fresh Pond and Spy Pond", "bbox": (-71.168, 42.3785, -71.1405, 42.416),
     "n_zones": 2, "erode": 1, "min_component": 150, "min_freq": 0.6, "kind": "lake",
     "drinking_water": True},   # Fresh Pond is Cambridge's drinking-water reservoir
]
SITE_BY_KEY = {s["key"]: s for s in SITES}
DEFAULT_SITE = "charles"
TIMEZONE = "America/New_York"

# ---------------------------------------------------------------- satellite
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION = "sentinel-2-l2a"
CRS = "EPSG:32619"          # UTM zone 19N (Boston)
RESOLUTION_M = 20           # B05 (red edge) is natively 20 m
START_DATE = "2019-05-01"   # Sentinel-2 L2A coverage is dependable from 2019
SEASON_MONTHS = (5, 6, 7, 8, 9, 10)   # May to October
MAX_ITEM_CLOUD = 70         # skip scenes the catalogue says are >70% cloudy
MAX_BAD_PIXEL_FRAC = 0.35   # skip if >35% of the study box is cloud/shadow/snow
SCL_WATER = 6               # Sentinel-2 scene classification code for water
SCL_BAD = (1, 3, 8, 9, 10, 11)  # saturated, cloud shadow, cloud (med/high), cirrus, snow
N_ZONES = 10
ERODE_PIXELS = 1            # drop water pixels touching land (mixed pixels)
MIN_ZONE_PIXELS = 20        # observation needs at least this many valid pixels
MIN_VALID_FRAC = 0.30       # ...and at least this share of the zone's pixels

# ---------------------------------------------------------------- model
HORIZON_DAYS = 7            # forecast this many days ahead
MAX_OBS_AGE_DAYS = 21       # last satellite reading must be at most this old
OBS_LATENCY_DAYS = 2        # a scene is only usable this many days after it was taken (processing delay)
TIER_ELEVATED_Q = 0.70      # "elevated" = top 30% of historical zone readings
TIER_HIGH_Q = 0.90          # "high" = top 10%
HEAT_WAVE_TMAX_C = 32.0     # ~90 F; 3+ days in a row is the common heat wave definition
HEAT_WAVE_DAYS = 3

# ---------------------------------------------------------------- alerts
ALERT_RADIUS_M = 1500       # legacy zone-follow radius
PROXIMITY_RADIUS_M = 400    # a person is "near" a flagged area when their location is within this distance of its water
DEMO_COOLDOWN_S = 45        # replay/demo alerts for the same person and area repeat no sooner than this
ALERT_COOLDOWN_H = 48       # don't re-alert the same person about the same zone sooner
QUIET_START_H = 21          # no alerts from 9 pm...
QUIET_END_H = 8             # ...until 8 am local time
OFFICIAL_INFO = "crwa.org"  # VERIFY before demoing; where people check official advisories
POLL_SECONDS = 3
ALERT_CHECK_MINUTES = 30

# ---------------------------------------------------------------- files
ZONES_GRID = DATA_DIR / "zones_grid.npz"
ZONES_JSON = DATA_DIR / "zones.json"
ZONE_NAMES_JSON = DATA_DIR / "zone_names.json"   # optional: {"Z03": "Near the BU Bridge"}
ZONES_PREVIEW = DATA_DIR / "zones_preview.png"
NDCI_CSV = DATA_DIR / "ndci.csv"
NDCI_CACHE = DATA_DIR / "ndci_cache.jsonl"
WEATHER_CSV = DATA_DIR / "weather.csv"
MODEL_PATH = DATA_DIR / "model.joblib"
METRICS_JSON = DATA_DIR / "metrics.json"
FORECAST_JSON = DATA_DIR / "forecast.json"
REPLAY_JSON = DATA_DIR / "replay.json"
DB_PATH = DATA_DIR / "algaebro.db"
PHOTO_DIR = DATA_DIR / "photos"
MANIFEST_JSON = DATA_DIR / "manifest.json"
SITES_DIR = DATA_DIR / "sites"


def site_dir(key: str) -> Path:
    return SITES_DIR / key


def site_grid_path(key: str) -> Path:
    return site_dir(key) / "zones_grid.npz"


def site_cache_path(key: str) -> Path:
    """The Lower Charles keeps the original cache file, so nothing already downloaded is lost."""
    return NDCI_CACHE if key == DEFAULT_SITE else site_dir(key) / "ndci_cache.jsonl"


def data_source() -> str:
    """'real' or 'synthetic'. Synthetic data only exists to test the code."""
    try:
        return json.loads(MANIFEST_JSON.read_text()).get("source", "unknown")
    except Exception:
        return "unknown"


def is_synthetic() -> bool:
    return data_source() == "synthetic"

