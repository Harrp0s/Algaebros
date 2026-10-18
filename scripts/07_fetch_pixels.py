"""STEP 7 (optional). Download per-pixel satellite maps for the real heat gradient.

    python scripts/07_fetch_pixels.py --years 2025           # one season, about 10-15 minutes
    python scripts/07_fetch_pixels.py --years 2025 2026
    python scripts/07_fetch_pixels.py --years 2025 --limit 5  # quick test

Safe to stop (Ctrl+C) and re-run: scenes already saved are skipped. Saves small files in data/pixels/.
Without this the dashboard still works: it shows a smooth gradient between the area readings, labelled as interpolated.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algaebro import config, pixels, zones

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2025])
    ap.add_argument("--limit", type=int, help="only the first N scenes (quick test)")
    args = ap.parse_args()
    if not config.ZONES_GRID.exists():
        sys.exit("Run scripts/01_build_zones.py first.")
    labels, x, y = zones.load_grid()
    n = pixels.fetch_season(args.years, labels, x, y, limit=args.limit)
    have = pixels.list_scenes()
    print(f"\nSaved {n} new scenes. {len(have)} in total in {pixels.PIXEL_DIR}")
    print("Restart the dashboard to use them.")
