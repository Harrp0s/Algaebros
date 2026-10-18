"""Suggest place names for the zones from OpenStreetMap. YOU confirm them.

    python scripts/suggest_zone_names.py            # print suggestions (changes nothing)
    python scripts/suggest_zone_names.py --write    # save them to data/zone_names.json (keeps names you already set)

Then open data/zone_names.json in Notepad and fix anything that looks wrong using the street map
(the dashboard shows the zone number when you hover a zone). Names appear in alerts as "near <name>",
so plain place names work best: "Harvard Bridge", "the Esplanade".
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import requests

from algaebro import config, naming, zones

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--overwrite", action="store_true", help="also replace names you already set")
    ap.add_argument("--max-m", type=float, default=700.0, help="farthest landmark to consider, in metres")
    args = ap.parse_args()

    recs = zones.load_zone_records()
    payload, last = None, None
    for attempt in range(3):
        try:
            r = requests.post(naming.OVERPASS_URL, data={"data": naming.build_query()}, timeout=90,
                              headers={"User-Agent": "algaebro-hackathon/1.0 (naming river zones for a student project)"})
            if r.status_code == 200:
                payload = r.json()
                break
            last = f"HTTP {r.status_code}"
        except (requests.RequestException, ValueError) as e:
            last = f"{type(e).__name__}: {e}"
        time.sleep(4 * (attempt + 1))
    if payload is None:
        sys.exit(f"Could not reach OpenStreetMap ({last}). Try again in a minute, or name the zones by hand in {config.ZONE_NAMES_JSON}.")

    feats = naming.parse_features(payload)
    print(f"{len(feats)} named places found near the river.\n")
    picks = naming.suggest(recs, feats, max_m=args.max_m)
    for z in recs:
        p = picks[z["id"]]
        label = f"{p['name']}  ({p['distance_m']} m, {p['kind']})" if p["name"] else "(nothing close; name it yourself)"
        print(f"  {z['id']}  zone {z['number']:>2}: {label}")
        if p["alternatives"]:
            print(f"        other options: {'; '.join(p['alternatives'])}")

    current = json.loads(config.ZONE_NAMES_JSON.read_text()) if config.ZONE_NAMES_JSON.exists() else {}
    merged = dict(current)
    for z in recs:
        name = picks[z["id"]]["name"]
        if name and (args.overwrite or z["id"] not in current):
            merged[z["id"]] = name
    if args.write:
        config.ZONE_NAMES_JSON.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        print(f"\nSaved {config.ZONE_NAMES_JSON}. OPEN IT and fix anything that looks wrong, then restart the server.")
    else:
        print("\nNothing saved. Run again with --write to save these, then edit the file.")
