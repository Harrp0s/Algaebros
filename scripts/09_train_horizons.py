"""Train the 1-7 day forecasts: Daytona searches every model idea x setting for every day ahead, in parallel.

    python scripts/09_train_horizons.py            # Daytona if DAYTONA_API_KEY is set, else this computer (slow)
    python scripts/09_train_horizons.py --local    # force this computer (about 25 minutes)
    python scripts/09_train_horizons.py --small    # 2 ideas, 1 setting: a quick check

Writes data/horizon_models.joblib and data/horizons.json. Restart the server (or press "Train 1-7 day forecasts"
in the Forecast Lab, which does the same and reloads by itself) to see the ML forecast on the map.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algaebro import config, horizons  # noqa: E402
from algaebro.run_bot import load_dotenv  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--local", action="store_true", help="run on this computer, not Daytona")
    ap.add_argument("--small", action="store_true", help="a quick check: 2 ideas, 1 setting")
    a = ap.parse_args()
    load_dotenv(config.ROOT / ".env")
    job = {"log": [], "done": 0, "total": 0}
    seen = [0]

    def show():
        for line in job["log"][seen[0]:]:
            print(line, flush=True)
        seen[0] = len(job["log"])

    import threading
    out = {}
    kw = {"ideas": ["hgb_tuned", "no_weather"], "settings": ["a"]} if a.small else {}
    t = threading.Thread(target=lambda: out.update(horizons.run_training(job, prefer_daytona=not a.local, **kw)))
    t0 = time.time()
    t.start()
    while t.is_alive():
        t.join(2)
        show()
    show()
    if not out:
        sys.exit("training failed: see the messages above")
    s = out["summary"]
    print(f"\n{s['headline']}\nRan {out['n']} experiments on {out['runner']} in {time.time() - t0:.0f}s.")
    for d in s["days"]:
        print(f"  {d['days']} day(s): rule {d['rule_f1'] or 0:.2f}  best ML {d['ml_f1'] or 0:.2f} ({d['ml_name']}, "
              f"{d['ml_setting_name']})  -> {d['verdict']}")


if __name__ == "__main__":
    main()
