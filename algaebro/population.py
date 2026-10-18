"""Who lives near each zone: US Census population by census tract, for the cleanup planner.

Two public files, no key needed (scripts/08_fetch_population.py downloads them once):
  * Census Gazetteer 2022, Massachusetts tracts: each tract's land area and an internal point (lat/lon);
  * American Community Survey 2022 5-year, table B01003 (total population), every tract in the US.

A tract counts as "near" a zone when its internal point is within RADIUS_M of the zone's water. Whole tracts are
counted (a Boston tract is roughly 0.5-2 km across), so this is a neighbourhood-scale estimate, not a head count.
"""
from __future__ import annotations

import io
import json
import math
from typing import Dict, List

import numpy as np

from . import config

GAZETTEER_URL = "https://www2.census.gov/geo/docs/maps-data/data/gazetteer/2022_Gazetteer/2022_gaz_tracts_25.txt"
ACS_URL = "https://www2.census.gov/programs-surveys/acs/summary_file/2022/table-based-SF/data/5YRData/acsdt5y2022-b01003.dat"
SOURCE = "US Census Bureau: 2022 Gazetteer tracts (MA) and ACS 2022 5-year table B01003 (total population)"
POPULATION_JSON = config.DATA_DIR / "population.json"
RADIUS_M = 1000


def parse_gazetteer(text: str) -> List[dict]:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    head = [h.strip() for h in lines[0].split("\t")]
    out = []
    for ln in lines[1:]:
        row = dict(zip(head, [c.strip() for c in ln.split("\t")]))
        try:
            out.append({"geoid": row["GEOID"], "lat": float(row["INTPTLAT"]), "lon": float(row["INTPTLONG"]),
                        "land_km2": float(row["ALAND"]) / 1e6})
        except (KeyError, ValueError):
            continue
    return out


def parse_acs(stream: io.TextIOBase, state_fips: str = "25") -> Dict[str, int]:
    """GEO_ID|B01003_E001|... -> {tract GEOID: population}, tracts of one state only (the file covers the US)."""
    prefix = "1400000US" + state_fips
    out = {}
    for ln in stream:
        if ln.startswith(prefix):
            parts = ln.split("|")
            try:
                out[parts[0][len("1400000US"):]] = int(parts[1])
            except (IndexError, ValueError):
                continue
    return out


def _dist_m(lat1, lon1, lat2, lon2):
    k = math.cos(math.radians(42.4))
    return np.hypot((np.asarray(lon2) - lon1) * 111320.0 * k, (np.asarray(lat2) - lat1) * 110574.0)


def exposure(zones: List[dict], tracts: List[dict], pop: Dict[str, int], radius_m: float = RADIUS_M) -> dict:
    """Per zone and per site: people in tracts whose internal point lies within radius_m of the zone's water."""
    t_lat = np.array([t["lat"] for t in tracts])
    t_lon = np.array([t["lon"] for t in tracts])
    near_by_zone, out_z = {}, {}
    for z in zones:
        pts = z.get("points") or [[z["lon"], z["lat"]]]
        dmin = np.full(len(tracts), np.inf)
        for lon, lat in pts[::3]:
            dmin = np.minimum(dmin, _dist_m(lat, lon, t_lat, t_lon))
        idx = [i for i in np.nonzero(dmin <= radius_m)[0] if tracts[i]["geoid"] in pop]
        near_by_zone[z["id"]] = set(idx)
        people = sum(pop[tracts[i]["geoid"]] for i in idx)
        land = sum(tracts[i]["land_km2"] for i in idx)
        out_z[z["id"]] = {"people": int(people), "tracts": len(idx), "density_per_km2": round(people / land) if land else None}
    out_s = {}
    for site in {z.get("site", config.DEFAULT_SITE) for z in zones}:
        idx = set().union(*[near_by_zone[z["id"]] for z in zones if z.get("site", config.DEFAULT_SITE) == site])
        people = sum(pop[tracts[i]["geoid"]] for i in idx)
        land = sum(tracts[i]["land_km2"] for i in idx)
        out_s[site] = {"people": int(people), "tracts": len(idx), "density_per_km2": round(people / land) if land else None}
    return {"source": SOURCE, "radius_m": radius_m, "zones": out_z, "sites": out_s,
            "note": f"People living in census tracts whose centre is within {radius_m / 1000:g} km of the water (whole tracts counted)."}


def load() -> dict:
    try:
        return json.loads(POPULATION_JSON.read_text())
    except (OSError, ValueError):
        return {}
