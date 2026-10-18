"""Small geography helpers: distance, projection, and pulling a location out of text."""
from __future__ import annotations

import math
import re
from typing import Optional, Tuple

from pyproj import Transformer

from . import config

_to_ll = Transformer.from_crs(config.CRS, "EPSG:4326", always_xy=True)
_to_utm = Transformer.from_crs("EPSG:4326", config.CRS, always_xy=True)


def utm_to_lonlat(x, y):
    return _to_ll.transform(x, y)


def lonlat_to_utm(lon, lat):
    return _to_utm.transform(lon, lat)


def haversine_m(lat1, lon1, lat2, lon2) -> float:
    r = 6371000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


_NUM = r"(-?\d{1,3}\.\d{2,})"
_PATTERNS = [
    re.compile(r"[?&](?:ll|q|sll|daddr|saddr|coordinate)=" + _NUM + r"\s*,\s*" + _NUM),  # Apple/Google map links
    re.compile(r"@" + _NUM + r"\s*,\s*" + _NUM),                                           # Google Maps @lat,lon
    re.compile(r"!3d" + _NUM + r"!4d" + _NUM),                                              # Google place links
    re.compile(_NUM + r"\s*[, ]\s*" + _NUM),                                                # plain "42.355, -71.110"
]


def parse_location(text: str) -> Optional[Tuple[float, float]]:
    """Find a (lat, lon) in free text or in an Apple/Google Maps link. None if absent."""
    if not text:
        return None
    for pat in _PATTERNS:
        m = pat.search(text)
        if m:
            lat, lon = float(m.group(1)), float(m.group(2))
            if -90 <= lat <= 90 and -180 <= lon <= 180:
                return lat, lon
    return None
