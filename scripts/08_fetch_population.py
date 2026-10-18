"""STEP 8 (optional, for the cleanup planner). How many people live near each zone, from the US Census.

    python scripts/08_fetch_population.py

Downloads two public Census files (no key; about 18 MB, read as it streams) and writes data/population.json.
Re-run after adding a site with scripts/01_build_zones.py.
"""
import io
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algaebro import population, zones

UA = {"User-Agent": "algaebro (hackathon project; satellite algal-bloom watch)"}


def get(url: str, timeout: int = 180):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout)


if __name__ == "__main__":
    print("Downloading the Massachusetts tract list (Census Gazetteer 2022)...")
    tracts = population.parse_gazetteer(get(population.GAZETTEER_URL).read().decode("utf-8", "replace"))
    print(f"  {len(tracts)} tracts")
    print("Downloading tract populations (ACS 2022 5-year, table B01003; ~18 MB)...")
    with get(population.ACS_URL) as resp:
        pop = population.parse_acs(io.TextIOWrapper(resp, encoding="utf-8", errors="replace"))
    print(f"  {len(pop)} Massachusetts tracts with a population")
    zs = zones.load_zone_records()
    out = population.exposure(zs, tracts, pop)
    population.POPULATION_JSON.write_text(json.dumps(out, indent=1))
    for site, s in out["sites"].items():
        print(f"  {site:15s} ~{s['people']:,} people within 1 km ({s['tracts']} tracts, {s['density_per_km2']}/km2)")
    print(f"Wrote {population.POPULATION_JSON}")
