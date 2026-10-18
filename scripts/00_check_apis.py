"""STEP 0. Check that your computer can reach every service BEFORE running the long steps.

    python scripts/00_check_apis.py            # weather + satellite
    python scripts/00_check_apis.py --photon   # also check your Photon key (needs .env)

Prints PASS/FAIL for each check with what to try next. Nothing is written to disk.
"""
import argparse
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from algaebro import config, satellite, weather
from algaebro.run_bot import load_dotenv


def step(name, fn):
    print(f"\n--- {name}")
    try:
        fn()
        print(f"PASS  {name}")
        return True
    except Exception as e:
        print(f"FAIL  {name}: {type(e).__name__}: {str(e)[:400]}")
        if os.environ.get("ALGAEBRO_DEBUG"):
            traceback.print_exc()
        return False


def check_weather():
    arch = weather.fetch_archive("2025-07-01", "2025-07-03")
    print(arch.to_string(index=False))
    assert len(arch) == 3 and arch["tmean"].notna().all(), "archive returned missing values"
    rec = weather.fetch_recent(past_days=3, forecast_days=3)
    print(f"recent+forecast API returned {len(rec)} days, last date {rec['date'].max().date()}")


def check_satellite():
    cat = satellite.open_catalog()
    print("opened the Planetary Computer catalogue")
    items = satellite.search_items(cat, "2025-07-01", "2025-08-31", max_cloud=30)
    assert items, "no Sentinel-2 scenes found over the study box (check config.BBOX)"
    item = items[0]
    print(f"{len(items)} clear-ish scenes; reading {item.id} from {item.datetime:%Y-%m-%d}; "
          f"processing baseline {item.properties.get('s2:processing_baseline')}, DN offset {satellite.dn_offset(item)}")
    arrs = satellite.load_item(item)
    scl = arrs["scl"]
    water = scl == config.SCL_WATER
    idx = satellite.ndci(arrs["b04"], arrs["b05"])
    vals = idx[water & np.isfinite(idx)]
    print(f"grid {scl.shape} at {config.RESOLUTION_M} m; water pixels {int(water.sum())}; "
          f"bad-pixel share {satellite.bad_pixel_fraction(scl):.2f}")
    assert water.sum() > 50, "almost no water pixels: the study box may not cover the river. Check config.BBOX."
    print(f"median NDCI over water in this scene: {np.median(vals):+.3f} (typical inland water: about -0.2 to +0.2)")


def check_photon():
    load_dotenv(config.ROOT / ".env")
    from photon_api import Photon
    from photon_api.rpc_generated import ListMessagesInput

    key, proj = os.environ.get("PHOTON_API_KEY"), os.environ.get("PHOTON_PROJECT_ID")
    assert key and proj, "put PHOTON_API_KEY and PHOTON_PROJECT_ID in .env (see .env.example)"
    with Photon(headers={"Authorization": f"Bearer {key}"}) as p:
        page = p.projects.list_messages(
            ListMessagesInput.model_validate({"path": {"id": proj}, "query": {"direction": "received", "pageSize": 1, "platform": "imessage"}})
        )
        print(f"Photon accepted the key; {len(page.items)} recent message(s) visible")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--photon", action="store_true")
    args = ap.parse_args()
    results = [step("Open-Meteo weather", check_weather), step("Sentinel-2 satellite (Planetary Computer)", check_satellite)]
    if args.photon:
        results.append(step("Photon 2.0 key", check_photon))
    print("\n" + ("All checks passed. Continue with step 1." if all(results) else
          "Some checks failed. Fix those before continuing (set ALGAEBRO_DEBUG=1 for full error traces)."))
    sys.exit(0 if all(results) else 1)
