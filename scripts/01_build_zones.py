"""STEP 1. Find each water body from satellite water observations and split it into numbered zones.

    python scripts/01_build_zones.py                       # build every site that has no grid yet
    python scripts/01_build_zones.py --site mystic_lakes   # (re)build one site
    python scripts/01_build_zones.py --list                # show the sites (config.SITES)

Writes data/sites/<site>/zones_grid.npz and zones_preview.png per site, then the combined
data/zones_grid.npz, data/zones.json and data/zones_preview.png.
OPEN EACH PREVIEW PICTURE and check the numbered zones sit on the water. If not, edit that
site's bbox in config.SITES and run this step again for it.

The Lower Charles (zones Z01-Z10) is never rebuilt unless you ask for it with --site charles,
so its zone numbers and downloaded history stay valid.
"""
import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from algaebro import config, satellite, zones


def build_site(site: dict, n_scenes: int) -> None:
    key = site["key"]
    first = zones.next_zone_number(exclude=key)
    print(f"\n== {site['name']} ({key}): zones from Z{first:02d}")
    cat = satellite.open_catalog()
    items = satellite.search_items(cat, config.START_DATE, date.today().isoformat(), bbox=site["bbox"], max_cloud=10)
    print(f"{len(items)} very clear season scenes available")
    if not items:
        sys.exit("No clear scenes found. Run scripts/00_check_apis.py.")
    pick = [items[i] for i in np.linspace(0, len(items) - 1, min(n_scenes, len(items))).astype(int)]
    freq, x, y, used = satellite.water_frequency(pick, bbox=site["bbox"])
    print(f"water frequency built from {used} scenes; grid {freq.shape}")
    labels = zones.build_zone_labels(freq, n_zones=site["n_zones"], erode=site["erode"], min_freq=site["min_freq"],
                                     min_component=site["min_component"], first=first)
    zones.save_site_grid(key, labels, x, y, freq)
    recs = zones.zone_records(labels, x, y, zones.load_names(), site_of={int(z): key for z in np.unique(labels[labels > 0])})
    zones.save_preview(labels, freq, recs, x, y, path=config.site_dir(key) / "zones_preview.png",
                       title=f"{site['name']}: zones on the water mask. Check they sit on the water.")
    for r in recs:
        print(f"  {r['id']}  {r['area_km2']:.2f} km2  centre {r['lat']:.4f}, {r['lon']:.4f}")
    print(f"LOOK AT {config.site_dir(key) / 'zones_preview.png'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--site", help="build only this site (a key from config.SITES)")
    ap.add_argument("--scenes", type=int, default=16, help="how many clear scenes to use to find the water")
    ap.add_argument("--list", action="store_true")
    args = ap.parse_args()

    if args.list:
        for s in config.SITES:
            print(f"{s['key']:15s} {s['name']:40s} built={config.site_grid_path(s['key']).exists()}")
        sys.exit(0)
    if zones.migrate_single_site():
        print("Filed the existing Lower Charles grid as site 'charles' (zone numbers unchanged).")
    if args.site:
        if args.site not in config.SITE_BY_KEY:
            sys.exit(f"Unknown site {args.site!r}. Choose from: {', '.join(config.SITE_BY_KEY)}")
        todo = [config.SITE_BY_KEY[args.site]]
    else:
        todo = [s for s in config.SITES if not config.site_grid_path(s["key"]).exists()]
    for site in todo:
        build_site(site, args.scenes)

    recs = zones.build_union()
    labels, x, y = zones.load_grid()
    zones.save_preview(labels, np.load(config.ZONES_GRID)["freq"], recs, x, y)
    if not config.MANIFEST_JSON.exists():
        config.MANIFEST_JSON.write_text(json.dumps({"source": "real", "built": date.today().isoformat()}))
    print(f"\n{len(recs)} zones in total across {len(zones.built_sites())} sites. Combined preview: {config.ZONES_PREVIEW}")
    print(f"To name zones, edit {config.ZONE_NAMES_JSON} (or run scripts/suggest_zone_names.py --write).")
