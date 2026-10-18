"""STEP 2. Download Sentinel-2 history and turn it into a chlorophyll-index time series per zone, site by site.

    python scripts/02_fetch_satellite.py --limit 5               # quick test first!
    python scripts/02_fetch_satellite.py                         # the full history of every built site (can take a while)
    python scripts/02_fetch_satellite.py --site mystic_lakes     # one site only

Safe to stop (Ctrl+C) and re-run: finished scenes are cached per site
(data/ndci_cache.jsonl for the Lower Charles, data/sites/<site>/ndci_cache.jsonl for the others).
Writes data/ndci.csv with every site.
"""
import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algaebro import config, satellite, zones

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default=config.START_DATE)
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--limit", type=int, help="only process the first N scenes per site (for a quick test)")
    ap.add_argument("--site", help="only this site (a key from config.SITES)")
    args = ap.parse_args()

    zones.migrate_single_site()
    built = zones.built_sites()
    if not built:
        sys.exit("Run scripts/01_build_zones.py first.")
    keys = [args.site] if args.site else built
    for key in keys:
        if key not in built:
            sys.exit(f"Site {key!r} has no grid yet. Run: python scripts/01_build_zones.py --site {key}")
        labels, x, y, _ = zones.load_site_grid(key)
        print(f"\n== {config.SITE_BY_KEY[key]['name']} ({key})")
        satellite.fetch_all(args.start, args.end, labels, x, y, int(labels.max()), limit=args.limit, site=key)

    df = satellite.build_ndci_table()
    print(f"\n{len(df)} zone readings from {df['date'].nunique() if len(df) else 0} dates.")
    if len(df):
        print(df.groupby("site")["date"].nunique().rename("dates").to_string())
        print(df.assign(year=df["date"].str[:4]).groupby("year")["date"].nunique().to_string())
        print(f"NDCI range {df['ndci'].min():+.3f} to {df['ndci'].max():+.3f}, median {df['ndci'].median():+.3f}")
    print(f"Wrote {config.NDCI_CSV}")
