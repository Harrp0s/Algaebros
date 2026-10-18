"""Suggest human place names for the zones from OpenStreetMap (Overpass API, free, no key).

These are SUGGESTIONS. A person looks at the street map and confirms them, because a wrong place name
in an alert is worse than 'Zone 7'. The suggestion script never overwrites a name you set yourself.
"""
from __future__ import annotations

import math
import re
from typing import Dict, List, Optional

from . import config, geo

OVERPASS_URL = "https://overpass-api.de/api/interpreter"
# Penalty (metres) added to a landmark's distance so that a more useful name can beat a slightly nearer, less useful one.
# People recognise "Harvard Bridge" and "the Esplanade"; they do not recognise "Storrow Drive" as a place on the river.
PENALTY_NAMED_BRIDGE = 0
PENALTY_PARK_OR_BOATING = 60
PENALTY_ROAD_ON_A_BRIDGE = 250     # OSM tags many roads as bridges: "River Street", "Memorial Drive"
PENALTY_OVERPASS_OR_TRACK = 400
PENALTY_OTHER = 200
KIND_PENALTY_M = {"bridge": PENALTY_NAMED_BRIDGE, "park": PENALTY_PARK_OR_BOATING, "boating": PENALTY_PARK_OR_BOATING,
                  "road": PENALTY_ROAD_ON_A_BRIDGE, "landmark": PENALTY_OTHER}
_NOT_A_RIVER_LANDMARK = re.compile(r"overpass|underpass|running track", re.I)
_IS_A_BRIDGE_NAME = re.compile(r"bridge", re.I)   # also matches "Drawbridge"


def build_query(bbox=None, margin: float = 0.004) -> str:
    lon0, lat0, lon1, lat1 = bbox or config.BBOX
    box = f"({lat0 - margin},{lon0 - margin},{lat1 + margin},{lon1 + margin})"
    return (
        "[out:json][timeout:60];("
        f'way["bridge"]["name"]{box};'
        f'way["man_made"="bridge"]["name"]{box};'
        f'nwr["leisure"="park"]["name"]{box};'
        f'nwr["leisure"="marina"]["name"]{box};'
        f'nwr["sport"="rowing"]["name"]{box};'
        f'nwr["amenity"="boat_rental"]["name"]{box};'
        ");out center tags;"
    )


def _kind_and_penalty(tags: dict, name: str):
    if tags.get("bridge") or tags.get("man_made") == "bridge":
        if _NOT_A_RIVER_LANDMARK.search(name):
            return "road", PENALTY_OVERPASS_OR_TRACK
        if _IS_A_BRIDGE_NAME.search(name):
            return "bridge", PENALTY_NAMED_BRIDGE
        return "road", PENALTY_ROAD_ON_A_BRIDGE
    if tags.get("leisure") == "park":
        return "park", PENALTY_PARK_OR_BOATING
    if tags.get("leisure") == "marina" or tags.get("sport") == "rowing" or tags.get("amenity") == "boat_rental":
        return "boating", PENALTY_PARK_OR_BOATING
    return "landmark", PENALTY_OTHER


def parse_features(payload: dict) -> List[dict]:
    """Overpass JSON -> [{name, kind, lat, lon}]. Drops anything without a name or a position."""
    out, seen = [], set()
    for el in payload.get("elements", []):
        tags = el.get("tags") or {}
        name = (tags.get("bridge:name") or tags.get("name") or "").strip()
        pos = el.get("center") or ({"lat": el["lat"], "lon": el["lon"]} if "lat" in el and "lon" in el else None)
        if not name or not pos:
            continue
        key = (name.lower(), round(pos["lat"], 3), round(pos["lon"], 3))
        if key in seen:
            continue
        seen.add(key)
        kind, penalty = _kind_and_penalty(tags, name)
        out.append({"name": name, "kind": kind, "penalty": penalty, "lat": pos["lat"], "lon": pos["lon"]})
    return out


def _compass(from_lat: float, from_lon: float, to_lat: float, to_lon: float) -> str:
    dy = to_lat - from_lat
    dx = (to_lon - from_lon) * math.cos(math.radians(from_lat))
    if abs(dx) >= abs(dy):
        return "east" if dx > 0 else "west"
    return "north" if dy > 0 else "south"


def suggest(zones: List[dict], features: List[dict], max_m: float = 700.0) -> Dict[str, dict]:
    """zone id -> {'name', 'distance_m', 'kind', 'alternatives'}; name is None when nothing is close enough."""
    picks: Dict[str, dict] = {}
    for z in zones:
        cands = []
        for f in features:
            d = min(geo.haversine_m(f["lat"], f["lon"], p[1], p[0]) for p in z["points"])
            if d <= max_m:
                cands.append((d + f.get("penalty", KIND_PENALTY_M.get(f["kind"], PENALTY_OTHER)), d, f))
        cands.sort(key=lambda t: t[0])
        if not cands:
            picks[z["id"]] = {"name": None, "distance_m": None, "kind": None, "alternatives": [], "_feature": None}
            continue
        _, d, f = cands[0]
        alts, seen_names = [], {f["name"].lower()}
        for c in cands[1:]:                       # nearest first, each name once
            if c[2]["name"].lower() not in seen_names:
                seen_names.add(c[2]["name"].lower())
                alts.append(f"{c[2]['name']} ({round(c[1])} m)")
            if len(alts) == 3:
                break
        picks[z["id"]] = {"name": f["name"], "distance_m": round(d), "kind": f["kind"], "_feature": f, "alternatives": alts}

    # Two zones named after the same place: say which side of it each one is on.
    by_name: Dict[str, List[str]] = {}
    for zid, p in picks.items():
        if p["name"]:
            by_name.setdefault(p["name"], []).append(zid)
    zone_by_id = {z["id"]: z for z in zones}
    for name, zids in by_name.items():
        if len(zids) < 2:
            continue
        used: Dict[str, int] = {}
        for zid in zids:
            z, f = zone_by_id[zid], picks[zid]["_feature"]
            side = _compass(f["lat"], f["lon"], z["lat"], z["lon"])
            used[side] = used.get(side, 0) + 1
            suffix = f"{side} side" if used[side] == 1 else f"{side} side, {used[side]}"
            picks[zid]["name"] = f"{name} ({suffix})"
    for p in picks.values():
        p.pop("_feature", None)
    return picks
