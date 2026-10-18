"""STEP 5. Make a forecast.

    python scripts/05_forecast.py                       # live: as of today
    python scripts/05_forecast.py --as-of 2026-07-20    # REPLAY a past date (uses only what was known then)
    python scripts/05_forecast.py --replay-year 2026    # weekly replay for the whole season (feeds the map)
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from algaebro import forecast

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of", type=forecast.parse_as_of)
    ap.add_argument("--replay-year", type=int)
    ap.add_argument("--no-weather-refresh", action="store_true")
    args = ap.parse_args()

    if args.replay_year:
        rp = forecast.make_replay(args.replay_year)
        print(f"Replay {args.replay_year}: {len(rp['frames'])} weekly forecasts saved to data/replay.json. Now run scripts/06_make_map.py")
        sys.exit(0)

    fc = forecast.get_forecast(args.as_of, refresh_weather=not args.no_weather_refresh) if not args.as_of else forecast.make_forecast(args.as_of)
    print(f"As of {fc['as_of']}{' (REPLAY)' if fc['replay'] else ''}, forecasting {fc['target_date']}. "
          f"Method: {fc['method']}. Data: {fc['data_source']}. In season: {fc['in_season']}")
    print(f"Weather: {fc['weather']}")
    rows = []
    for z in fc["zones"]:
        tr = z.get("track") or {}
        rows.append({"zone": z["number"], "tier": z["tier"], "outlook": z.get("ndci_pred"),
                     "past_next_HIGH": None if tr.get("next_high") is None else f"{tr['next_high']:.0%}",
                     "last_obs": z.get("last_obs"), "age_d": z.get("obs_age_days"),
                     "satellite_saw": (z.get("verification") or {}).get("observed_tier"), "why_unknown": z.get("reason")})
    print(pd.DataFrame(rows).to_string(index=False))
