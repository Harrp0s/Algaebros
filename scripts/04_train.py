"""STEP 4. Build the training table, evaluate honestly, and save the model.

    python scripts/04_train.py [--model hgb|xgb] [--horizon 7]

Prints how the model did on years it never saw, next to two simple baselines.
If the model does not beat 'persistence' the forecast automatically uses persistence instead.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from algaebro import config, features, model, weather

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="hgb", choices=["hgb", "xgb"])
    ap.add_argument("--horizon", type=int, default=config.HORIZON_DAYS)
    args = ap.parse_args()
    config.HORIZON_DAYS = args.horizon

    print(f"DATA SOURCE: {config.data_source().upper()}")
    if config.is_synthetic():
        print("*** SYNTHETIC DATA: results below only prove the code runs. They say nothing about the Charles. ***")
    obs = features.prepare_obs(pd.read_csv(config.NDCI_CSV))
    wfeat = features.weather_features(weather.load())
    table = features.build_training_table(obs, wfeat, horizon=args.horizon)
    print(f"{len(obs)} satellite readings -> {len(table)} usable training rows "
          f"({table['year'].nunique()} years: {[int(y) for y in sorted(table['year'].unique())]})")
    if len(table) < 60:
        sys.exit("Too few training rows. Fetch more satellite history (step 2) or loosen MIN_VALID_FRAC in config.py.")

    bundle = model.train_and_save(table, args.model)
    rep = bundle["report"]
    print(f"\nMETHOD USED FOR FORECASTS: {bundle['method'].upper()}\n  {bundle['method_reason']}")
    if rep:
        print(f"\nOut-of-sample test years: {rep['test_years']}  ({rep['n_test_rows']} readings, {rep['n_actual_high']} actually 'high')")
        print(f"CAUTION: those {rep['n_actual_high']} HIGH readings fall on only {rep.get('n_actual_high_dates', '?')} dates (a whole basin lights up at once), "
              "so the rates below are rough. Differences smaller than ~0.15 in F1 are within noise.")
        rows = []
        for name, r in rep["by_method"].items():
            rows.append({"method": name, "MAE": round(r["mae"], 4), "RMSE": round(r["rmse"], 4),
                         "high: precision": None if r["precision"] is None else round(r["precision"], 2),
                         "high: recall": None if r["recall"] is None else round(r["recall"], 2),
                         "F1": None if r["f1"] is None else round(r["f1"], 2),
                         "AUC": None if r["auc"] is None else round(r["auc"], 2)})
        print(pd.DataFrame(rows).to_string(index=False))
        if rep["skill_vs_persistence"] is not None:
            print(f"\nAverage-error skill vs persistence: {rep['skill_vs_persistence'] * 100:+.1f}%. Average error is NOT what the alerts depend on: "
                  "a model can have low average error and still almost never reach 'high'. Compare the F1 and recall columns.")
        print("\nPer test year (MAE):")
        print(pd.DataFrame(rep["by_year"]).T.round(4).to_string())
    rec = bundle.get("tier_record")
    if rec:
        print(f"\nTRACK RECORD of the method in use ({bundle['method']}), a week after each tier, in years it never saw:")
        for t in ("high", "elevated", "low"):
            r = rec[t]
            if r["n"]:
                print(f"  tier {t:8s} n={r['n']:4d}   next reading HIGH {r['next_high']*100:4.0f}%   at least elevated {r['next_elevated_or_high']*100:4.0f}%")
    print(f"\nThresholds from all training data: elevated >= {bundle['thr_elev']:+.4f}, high >= {bundle['thr_high']:+.4f}")
    print(f"Saved {config.MODEL_PATH} and {config.METRICS_JSON}")
