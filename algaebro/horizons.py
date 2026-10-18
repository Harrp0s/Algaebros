"""1-7 day forecasts: Daytona searches for the best model for EACH day ahead, then the winners are used.

How it works (the "Train 1-7 day forecasts" button in the Forecast Lab, or scripts/09_train_horizons.py):
  1. SEARCH   for every day ahead (1..7), every model idea x every model setting, plus the simple rule, is tested
              the honest way (train on earlier years only) on every season except the latest. One Daytona sandbox each,
              all in parallel (about 120 sandboxes).
  2. CONFIRM  for every day ahead, the best model and the simple rule are tested on the held-out latest season.
  3. REFIT    each day's best model is refitted on all the data on this computer (seconds) and saved.

The ML forecast is ALWAYS shown on the map when asked for (the "ML model" toggle), with its measured score next to it.
Alerts only switch to the ML for a given day ahead if it was clearly better on earlier seasons AND held up, clearly, on
the held-out season. Otherwise alerts keep the simple rule (latest reading carried forward).

Only regression ideas are candidates: they predict a reading, so the map can be coloured from them.
"""
from __future__ import annotations

import json
import time
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from . import config, features, labcore

HORIZONS = list(range(1, 8))
ML_IDEAS = ["hgb_tuned", "delta", "no_weather", "neighbors", "memory"]
SETTINGS = list(labcore.SETTINGS)          # a, b, c
SETTING_NAMES = {"a": "standard trees", "b": "shallow trees", "c": "deeper trees, slower learning"}
HIGH_Q = 0.9


def models_path():
    return config.DATA_DIR / "horizon_models.joblib"


def summary_path():
    return config.DATA_DIR / "horizons.json"


def _name(method: str) -> str:
    from .lab import METHOD_NAMES
    return METHOD_NAMES.get(method, method)


# ---------------------------------------------------------------------- experiment grids
def _spec(method: str, h: int, setting: Optional[str], holdout: Optional[int], only_year: Optional[int] = None,
          stage: str = "screen") -> Dict:
    sid = f"d{h}-{method}" + (f"-{setting}" if setting else "") + (f"-y{only_year}" if only_year else "")
    label = f"{_name(method)}" + (f" ({SETTING_NAMES[setting]})" if setting else "") + f", {h} day{'s' if h > 1 else ''} ahead"
    if only_year:
        label += f" (confirm on {only_year})"
    return {"id": sid, "method": method, "horizon": h, "high_q": HIGH_Q, "label": label, "stage": stage,
            "holdout": holdout, "only_year": only_year, "setting": setting}


def search_specs(holdout: Optional[int], horizons: List[int] = None, ideas: List[str] = None,
                 settings: List[str] = None) -> List[Dict]:
    out = []
    for h in horizons or HORIZONS:
        out.append(_spec("persistence", h, None, holdout))
        out += [_spec(m, h, st, holdout) for m in (ideas or ML_IDEAS) for st in (settings or SETTINGS)]
    return out


def best_per_day(results: List[Dict]) -> Dict[int, Dict]:
    """The best ML spec for each day ahead, judged on the screening seasons (F1, then ranking AP)."""
    out = {}
    for h in sorted({r["horizon"] for r in results}):
        rs = [r for r in results if r.get("ok") and r["horizon"] == h and r.get("stage", "screen") == "screen"
              and r["method"] != "persistence"]
        if rs:
            out[h] = max(rs, key=lambda r: ((r.get("f1") or 0), (r.get("ap") or 0)))
    return out


def confirm_specs(results: List[Dict], holdout: Optional[int]) -> List[Dict]:
    if holdout is None:
        return []
    out = []
    for h, best in best_per_day(results).items():
        out += [_spec(best["method"], h, best.get("setting"), None, only_year=holdout, stage="confirm"),
                _spec("persistence", h, None, None, only_year=holdout, stage="confirm")]
    return out


def summarise(results: List[Dict], holdout: Optional[int]) -> Dict:
    """Per day ahead: the rule's score, the best model's score, whether it held up, and what the alerts use."""
    best = best_per_day(results)
    days = []
    for h in sorted({r["horizon"] for r in results if r.get("ok")}):
        rule = next((r for r in results if r.get("ok") and r["horizon"] == h and r["method"] == "persistence"
                     and r.get("stage", "screen") == "screen"), None)
        b = best.get(h)
        conf_ml = conf_rule = None
        if b is not None:
            conf_ml = next((r for r in results if r.get("ok") and r.get("stage") == "confirm" and r["horizon"] == h
                            and r["method"] == b["method"]), None)
            conf_rule = next((r for r in results if r.get("ok") and r.get("stage") == "confirm" and r["horizon"] == h
                              and r["method"] == "persistence"), None)
        rule_f1 = (rule or {}).get("f1")
        clear_screen = bool(b and rule and b.get("f1_range") and b["f1_range"][0] > (rule_f1 or 0))
        clear_holdout = bool(conf_ml and conf_rule and conf_ml.get("f1_range")
                             and conf_ml["f1_range"][0] > (conf_rule.get("f1") or 0))
        use_ml = clear_screen and clear_holdout
        if b is None:
            verdict = "no model could be tested"
        elif use_ml:
            verdict = "the model is clearly better and held up on the unseen season: alerts use it"
        elif (b.get("f1") or 0) > (rule_f1 or 0):
            verdict = "the model scored a bit higher, but within noise: alerts keep the simple rule"
        else:
            verdict = "the simple rule did better: alerts keep it"
        tested = sum(1 for r in results if r.get("ok") and r["horizon"] == h and r.get("stage", "screen") == "screen")
        days.append({
            "days": h, "rule_f1": rule_f1, "rule_f1_range": (rule or {}).get("f1_range"),
            "rule_recall": (rule or {}).get("recall"), "rule_precision": (rule or {}).get("precision"),
            "ml_method": b["method"] if b else None, "ml_name": _name(b["method"]) if b else None,
            "ml_setting": b.get("setting") if b else None,
            "ml_setting_name": SETTING_NAMES.get(b.get("setting")) if b else None,
            "ml_f1": b.get("f1") if b else None, "ml_f1_range": b.get("f1_range") if b else None,
            "ml_recall": b.get("recall") if b else None, "ml_precision": b.get("precision") if b else None,
            "ml_mae": b.get("mae") if b else None, "rule_mae": (rule or {}).get("mae"),
            "holdout_ml_f1": (conf_ml or {}).get("f1"), "holdout_rule_f1": (conf_rule or {}).get("f1"),
            "use_ml": use_ml, "verdict": verdict, "tested": tested,
        })
    wins = [d["days"] for d in days if d["use_ml"]]
    if wins:
        headline = (f"The ML model clearly beat the simple rule at {', '.join(map(str, wins))} day(s) ahead and held up "
                    f"on the unseen season, so alerts use it there. Everywhere else alerts keep the latest reading.")
    elif days:
        headline = ("No model clearly beat simply carrying the latest reading forward at any day ahead, so alerts keep the "
                    "simple rule. The ML forecast is still shown (toggle 'ML model') so you can compare.")
    else:
        headline = "No experiment produced a result."
    return {"headline": headline, "days": days, "holdout": holdout}


# ---------------------------------------------------------------------- refit the winners
def fit_final(method: str, setting: Optional[str], table: pd.DataFrame, high_q: float = HIGH_Q) -> Dict:
    """Fit on ALL data. The alert cutoff is tuned year-by-year inside the data, as in the lab."""
    params = labcore.SETTINGS.get(setting or "a")
    thr_high = float(np.quantile(table["y"], high_q))
    thr_elev = float(np.quantile(table["y"], config.TIER_ELEVATED_Q))
    scorer = labcore.fit_scorer(method, table, thr_high, params)
    cut = labcore._tuned_cutoff(method, table, high_q, default=thr_high, params=params)
    return {"method": method, "setting": setting, "scorer": scorer, "cut": float(cut), "thr_high": thr_high,
            "thr_elev": thr_elev, "extra": method in labcore.NEEDS_EXTRA, "rows": int(len(table))}


def refit(summary: Dict, ndci_csv=None, weather_csv=None, log=lambda m: None) -> Dict[int, Dict]:
    models = {}
    for d in summary["days"]:
        if not d["ml_method"]:
            continue
        t0 = time.time()
        table = labcore.load_table(str(ndci_csv or config.NDCI_CSV), str(weather_csv or config.WEATHER_CSV), d["days"],
                                   extra=d["ml_method"] in labcore.NEEDS_EXTRA)
        m = fit_final(d["ml_method"], d["ml_setting"], table)
        m.update({"use_ml": d["use_ml"], "ml_f1": d["ml_f1"], "rule_f1": d["rule_f1"], "name": d["ml_name"]})
        models[d["days"]] = m
        log(f"refitted the {d['days']}-day model ({d['ml_name']}) on {m['rows']} rows in {time.time() - t0:.1f}s")
    return models


def save(models: Dict[int, Dict], payload: Dict) -> None:
    joblib.dump(models, models_path())
    summary_path().write_text(json.dumps(payload, default=str))


def load_models() -> Dict[int, Dict]:
    try:
        return joblib.load(models_path())
    except Exception:       # missing, or written by another library version: no ML forecast until retrained
        return {}


def latest() -> Dict:
    try:
        return json.loads(summary_path().read_text())
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------- use the models for one forecast
def predict(m: Dict, rows: List[Dict], ids: List[str], obs: pd.DataFrame, zones: List[dict], as_of: pd.Timestamp,
            target: pd.Timestamp) -> Dict[str, Dict]:
    """ML forecast per zone: the predicted reading and its tier (tier uses the model's own tuned alert cutoff)."""
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    df["zone_id"] = ids
    if m["extra"]:
        df["target_date"] = pd.Timestamp(target)
        df["T_obs"] = pd.Timestamp(as_of) - pd.Timedelta(days=config.OBS_LATENCY_DAYS)
        df = features.add_extra_features(df, obs)
    vals = np.asarray(m["scorer"](df), dtype=float)
    out = {}
    for zid, v in zip(ids, vals):
        tier = "high" if v >= m["cut"] else ("elevated" if v >= min(m["thr_elev"], m["cut"]) else "low")
        out[zid] = {"value": round(float(v), 4), "tier": tier,
                    # shifted so the model's own alert cutoff lands on the map's flag line (colour only)
                    "display": round(float(v + (m["thr_high"] - m["cut"])), 4)}
    return out


# ---------------------------------------------------------------------- the job (background, polled by the dashboard)
def run_training(job: Dict, prefer_daytona: bool = True, runner=None, horizons: List[int] = None,
                 ideas: List[str] = None, settings: List[str] = None, data_source: Optional[str] = None) -> Dict:
    from . import lab

    holdout = lab.holdout_year()
    specs = search_specs(holdout, horizons, ideas, settings)
    job["total"] = len(specs)
    offset = {"n": 0}

    def progress(done: int, total: int, label: str) -> None:
        job["done"] = offset["n"] + done
        job["log"].append(f"finished {label} ({offset['n'] + done}/{job['total']})")

    notes = []
    if runner is None:
        if prefer_daytona and lab.DaytonaRunner.available():
            runner = lab.DaytonaRunner()
        else:
            if prefer_daytona:
                notes.append("Daytona is not set up, so the search ran on this computer.")
            runner = lab.LocalRunner()
    job["runner"] = runner.label
    job["log"].append(f"searching {len(specs)} model x setting x day combinations in {runner.label}")
    results = runner.run(specs, progress)
    if isinstance(runner, lab.DaytonaRunner) and not any(r.get("ok") for r in results):
        notes.append("Daytona produced no results, so the search ran on this computer instead.")
        runner = lab.LocalRunner()
        job["runner"] = runner.label
        job["done"] = 0
        results = runner.run(specs, progress)
    more = confirm_specs(results, holdout)
    if more:
        offset["n"] = len(specs)
        job["total"] = len(specs) + len(more)
        job["log"].append(f"confirming each day's best model on the unseen {holdout} season ({len(more)} more)")
        results = results + runner.run(more, progress)
    summary = summarise(results, holdout)
    job["log"].append("refitting each day's winner on all the data (this computer, a few seconds each)")
    models = refit(summary, log=lambda m: job["log"].append(m))
    payload = {"runner": runner.label, "notes": notes, "n": len(specs) + len(more), "holdout": holdout,
               "data_source": data_source or config.data_source(), "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "summary": summary,
               "results": sorted(results, key=lambda r: (r.get("horizon", 0), r.get("id", "")))}
    save(models, payload)
    return payload
