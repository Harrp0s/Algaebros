"""One forecasting experiment, run the honest way. Used by the Forecast Lab.

An experiment is a small dict, for example {"id": "h7-delta", "horizon": 7, "method": "delta", "high_q": 0.9}.
It answers: "if we had used THIS idea to flag areas `horizon` days ahead, how many real flagged
readings would it have caught, and how many false alarms would it have caused, in years it never saw?"

This file only needs pandas, numpy and scikit-learn plus features.py and config.py, so the same code runs
on your laptop or inside a Daytona sandbox. It never touches the network, the database, or phone numbers.

Rules that keep it honest (the same as the main model):
  * expanding window by year: to test year Y we train only on years before Y;
  * the 'flagged' threshold and any alert cutoff are chosen from earlier years only;
  * a 95% confidence range comes from resampling whole DATES, because a whole river lights up at once;
  * testing many ideas makes it likely one wins by luck, so the latest year can be HELD OUT: ideas are screened
    on earlier years ("holdout": Y skips test year Y and later) and only the winners are then confirmed on Y
    ("only_year": Y tests just that year, training on everything before it).

The ideas (METHODS):
  persistence   the latest reading carried forward (what the alerts use today)
  climatology   the usual value for this zone in this month
  hgb           the trained model's recipe: gradient boosting on all 19 inputs, flags when its prediction is 'high'
  hgb_tuned     the same model with an alert cutoff tuned on earlier years
  logit         a logistic model on a few inputs, tuned cutoff
  delta         predict the CHANGE from the last reading instead of the level (the plain model hugs the average)
  classifier    predict "will it be high?" directly, giving the rare high readings more weight
  no_weather    hgb_tuned without any weather input: does weather help at all?
  neighbors     hgb_tuned plus the neighbouring (and upstream) zones' latest readings
  memory        hgb_tuned plus a longer memory (30-day mean, last change, last season's peak)
  importance    not a forecast: how much catching high readings suffers when each input is scrambled
"""
from __future__ import annotations

import json
import math
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import features

METHODS = ("persistence", "climatology", "hgb", "hgb_tuned", "logit", "delta", "classifier", "no_weather", "neighbors",
           "memory", "importance")
LOGIT_FEATURES = ["last_ndci", "last3_mean", "basin_last", "tmean_7", "calm_7", "dry_days", "doy_sin", "doy_cos"]
FEATURE_SETS = {
    "hgb": features.FEATURES,
    "hgb_tuned": features.FEATURES,
    "delta": features.FEATURES,
    "classifier": features.FEATURES,
    "no_weather": [f for f in features.FEATURES if f not in features.WEATHER_COLS],
    "neighbors": features.FEATURES + features.NEIGHBOR_COLS,
    "memory": features.FEATURES + features.MEMORY_COLS,
}
TUNED = ("logit", "hgb_tuned", "delta", "classifier", "no_weather", "neighbors", "memory")
NEEDS_EXTRA = ("neighbors", "memory")
MIN_TRAIN_ROWS = 150
MIN_TEST_ROWS = 20

# Plain-language names for every input, for the importance chart.
FEATURE_LABELS = {
    "last_ndci": "Latest reading", "last_age": "Age of latest reading", "last3_mean": "Mean of last 3 readings",
    "basin_last": "Water-body average", "tmean_3": "Temperature, 3 days", "tmean_7": "Temperature, 7 days",
    "tmean_14": "Temperature, 14 days", "tmax_7": "Hottest day, 7 days", "rain_3": "Rain, 3 days", "rain_7": "Rain, 7 days",
    "rain_14": "Rain, 14 days", "dry_days": "Days since real rain", "rad_7": "Sunshine, 7 days", "wind_7": "Wind, 7 days",
    "calm_7": "Calm days, 7 days", "trend": "Warming trend", "doy_sin": "Time of year (a)", "doy_cos": "Time of year (b)",
    "zone_idx": "Which zone", "nbr_last": "Neighbouring zones", "up_last": "Upstream zone", "nbr_diff": "Greener than neighbours",
    "mean30": "30-day mean", "last_change": "Last change", "prev_season_max": "Last season's peak",
}


# Inputs that tell the same kind of story, for group importance (scrambled together).
FEATURE_GROUPS = {
    "Recent satellite readings": ["last_ndci", "last_age", "last3_mean", "basin_last", "nbr_last", "up_last", "nbr_diff",
                                  "mean30", "last_change", "prev_season_max"],
    "Weather": list(features.WEATHER_COLS),
    "Time of year": ["doy_sin", "doy_cos"],
    "Which zone": ["zone_idx"],
}


def _label(train_y_threshold: float, y: np.ndarray) -> np.ndarray:
    return (y >= train_y_threshold).astype(int)


# Model settings the 1-7 day training search tries (spec "params"); "a" is the original recipe.
SETTINGS = {
    "a": {"max_depth": 3, "learning_rate": 0.05, "min_samples_leaf": 20},
    "b": {"max_depth": 2, "learning_rate": 0.05, "min_samples_leaf": 40},
    "c": {"max_depth": 4, "learning_rate": 0.03, "min_samples_leaf": 20},
}


def _hgb(params: Optional[Dict] = None):
    kw = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 20, "l2_regularization": 1.0,
          "random_state": 0}
    kw.update(params or {})
    return HistGradientBoostingRegressor(**kw)


class Scorer:
    """A fitted idea: call it on a feature table to get scores. A plain class (not a lambda) so it can be saved."""

    def __init__(self, method: str, model=None, cols=None, const: Optional[float] = None, train=None,
                 thr: float = 0.0, params: Optional[Dict] = None):
        self.method, self.model, self.cols, self.const = method, model, cols, const
        self.train, self.thr, self.params = train, thr, params

    def __call__(self, df: pd.DataFrame) -> np.ndarray:
        m = self.method
        if m == "persistence":
            return df["last_ndci"].to_numpy(dtype=float)
        if self.const is not None:
            return np.full(len(df), self.const)
        if m in ("climatology", "logit"):
            return _scores(m, self.train, df, self.thr, self.params)
        if m == "delta":
            return df["last_ndci"].to_numpy(dtype=float) + np.asarray(self.model.predict(df[self.cols]), dtype=float)
        if m == "classifier":
            return self.model.predict_proba(df[self.cols])[:, 1]
        return np.asarray(self.model.predict(df[self.cols]), dtype=float)


def fit_scorer(method: str, train: pd.DataFrame, thr_high: float, params: Optional[Dict] = None) -> Scorer:
    """Fit `method` on `train`; the result maps a feature table to scores (a predicted reading for regression ideas)."""
    if method == "persistence":
        return Scorer(method)
    if method in ("climatology", "logit"):
        return Scorer(method, train=train, thr=thr_high, params=params)
    cols = FEATURE_SETS.get(method)
    if cols is None:
        raise ValueError(f"unknown method {method!r}")
    if method == "delta":
        return Scorer(method, _hgb(params).fit(train[cols], train["y"] - train["last_ndci"]), cols)
    if method == "classifier":
        lab = _label(thr_high, train["y"].to_numpy())
        if lab.sum() == 0 or lab.sum() == len(lab):
            return Scorer(method, const=float(lab.mean()))
        kw = {"max_depth": 3, "learning_rate": 0.05, "max_iter": 300, "min_samples_leaf": 20, "l2_regularization": 1.0,
              "class_weight": "balanced", "random_state": 0, **(params or {})}
        return Scorer(method, HistGradientBoostingClassifier(**kw).fit(train[cols], lab), cols)
    return Scorer(method, _hgb(params).fit(train[cols], train["y"]), cols)


def _scores(method: str, train: pd.DataFrame, test: pd.DataFrame, thr_high: float, params: Optional[Dict] = None) -> np.ndarray:
    """Higher score = more likely to be a flagged reading `horizon` days later."""
    if method == "persistence":
        return test["last_ndci"].to_numpy(dtype=float)
    if method == "climatology":
        month = train["target_date"].dt.month
        zone_month = train.groupby(["zone_id", month])["y"].mean().to_dict()
        month_mean = train.groupby(month)["y"].mean().to_dict()
        overall = float(train["y"].mean())
        return np.array([zone_month.get((z, m), month_mean.get(m, overall)) for z, m in zip(test["zone_id"], test["target_date"].dt.month)])
    if method == "logit":
        m = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=500)).fit(
            train[LOGIT_FEATURES], _label(thr_high, train["y"].to_numpy()))
        return m.predict_proba(test[LOGIT_FEATURES])[:, 1]
    return fit_scorer(method, train, thr_high, params)(test)


def _best_cut(scores: np.ndarray, truth: np.ndarray) -> Optional[float]:
    cands = np.unique(np.quantile(scores, np.linspace(0.5, 0.99, 50)))
    best, best_cut = -1.0, None
    for c in cands:
        p = scores >= c
        tp = int((p & (truth == 1)).sum())
        fp = int((p & (truth == 0)).sum())
        fn = int((~p & (truth == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best:
            best, best_cut = f1, float(c)
    return best_cut


def _tuned_cutoff(method: str, train: pd.DataFrame, high_q: float, default: float, params: Optional[Dict] = None) -> float:
    """Pick the alert cutoff that maximises F1, using ONLY years inside the training window."""
    sc, tr = [], []
    for yr in sorted(train["year"].unique()):
        a, b = train[train["year"] < yr], train[train["year"] == yr]
        if len(a) < 120 or len(b) < 15:
            continue
        thr = float(np.quantile(a["y"], high_q))
        sc.append(_scores(method, a, b, thr, params))
        tr.append(_label(thr, b["y"].to_numpy()))
    if not sc:
        return default
    sc, tr = np.concatenate(sc), np.concatenate(tr)
    if tr.sum() < 8:
        return default
    cut = _best_cut(sc, tr)
    return default if cut is None else cut


def _f1(truth: np.ndarray, alert: np.ndarray) -> float:
    tp = int((alert & (truth == 1)).sum())
    fp = int((alert & (truth == 0)).sum())
    fn = int((~alert & (truth == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def _bootstrap_f1(dates: np.ndarray, truth: np.ndarray, alert: np.ndarray, n: int = 300, seed: int = 0) -> Optional[List[float]]:
    """95% range for F1, resampling whole dates (a river lights up all at once, so rows are not independent)."""
    uniq = np.unique(dates)
    if len(uniq) < 8 or truth.sum() < 5:
        return None
    idx_by_date = {d: np.nonzero(dates == d)[0] for d in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        rows = np.concatenate([idx_by_date[d] for d in pick])
        vals.append(_f1(truth[rows], alert[rows]))
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def load_table(ndci_csv: str, weather_csv: str, horizon: int, extra: bool = False) -> pd.DataFrame:
    obs = features.prepare_obs(pd.read_csv(ndci_csv))
    wfeat = features.weather_features(pd.read_csv(weather_csv, parse_dates=["date"]))
    table = features.build_training_table(obs, wfeat, horizon=horizon)
    return features.add_extra_features(table, obs) if extra else table


def _test_years(table: pd.DataFrame, spec: Dict) -> List[int]:
    years = sorted(int(y) for y in table["year"].unique())
    if spec.get("only_year") is not None:
        return [y for y in years if y == int(spec["only_year"])]
    if spec.get("holdout") is not None:
        return [y for y in years if y < int(spec["holdout"])]
    return years


def run_experiment(spec: Dict, ndci_csv: str, weather_csv: str) -> Dict:
    """Run one experiment and return plain-number results (JSON-safe)."""
    method, horizon, high_q = spec["method"], int(spec["horizon"]), float(spec.get("high_q", 0.9))
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    if method == "importance":
        return run_importance(spec, ndci_csv, weather_csv)
    table = load_table(ndci_csv, weather_csv, horizon, extra=method in NEEDS_EXTRA)
    params = SETTINGS.get(spec.get("setting") or "a")

    parts = []
    for yr in _test_years(table, spec):
        train, test = table[table["year"] < yr], table[table["year"] == yr]
        if train["year"].nunique() < 2 or len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        thr_high = float(np.quantile(train["y"], high_q))
        scores = _scores(method, train, test, thr_high, params)
        if method in TUNED:
            default = 0.5 if method in ("logit", "classifier") else thr_high
            cut = _tuned_cutoff(method, train, high_q, default=default, params=params)
        else:
            cut = thr_high   # persistence / climatology / hgb: flag when the predicted reading is itself 'high'
        regress = method not in ("logit", "classifier")
        parts.append(pd.DataFrame({
            "date": test["target_date"].dt.strftime("%Y-%m-%d").to_numpy(),
            "site": test["site"].to_numpy() if "site" in test.columns else "charles",
            "truth": (test["y"].to_numpy() >= thr_high).astype(int),
            "score": scores, "alert": scores >= cut,
            "err": np.abs(test["y"].to_numpy() - scores) if regress else np.nan,
        }))
    base = {"id": spec.get("id"), "label": spec.get("label"), "method": method, "horizon": horizon, "high_q": high_q,
            "stage": spec.get("stage", "screen"), "holdout": spec.get("holdout"), "only_year": spec.get("only_year"),
            "setting": spec.get("setting")}
    if not parts:
        return {**base, "ok": False, "reason": "not enough history to test this setup"}

    d = pd.concat(parts, ignore_index=True)
    truth, alert = d["truth"].to_numpy(), d["alert"].to_numpy().astype(bool)
    tp = int((alert & (truth == 1)).sum())
    fp = int((alert & (truth == 0)).sum())
    fn = int((~alert & (truth == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = _f1(truth, alert)
    two_classes = 0 < truth.sum() < len(truth)
    by_site = {}
    for site, g in d.groupby("site"):
        t, a = g["truth"].to_numpy(), g["alert"].to_numpy().astype(bool)
        by_site[str(site)] = {"n": int(len(g)), "n_flagged": int(t.sum()), "n_alerts": int(a.sum()),
                              "f1": _f1(t, a) if (t.sum() or a.sum()) else None}
    return {
        **base, "ok": True,
        "test_years": sorted({int(x[:4]) for x in d["date"]}),
        "n_test": int(len(d)), "n_flagged_readings": int(truth.sum()),
        "n_flagged_dates": int(d.loc[d["truth"] == 1, "date"].nunique()), "n_alerts": int(alert.sum()),
        "precision": precision, "recall": recall, "f1": f1,
        "auc": float(roc_auc_score(truth, d["score"])) if two_classes else None,
        "ap": float(average_precision_score(truth, d["score"])) if two_classes else None,
        "mae": float(d["err"].mean()) if d["err"].notna().any() else None,
        "f1_range": _bootstrap_f1(d["date"].to_numpy(), truth, alert),
        "by_site": by_site,
    }


def run_importance(spec: Dict, ndci_csv: str, weather_csv: str, repeats: int = 5) -> Dict:
    """Permutation importance of the trained model's recipe, measured where it matters: out of sample.

    For each test year, train on earlier years, score how well the model RANKS high readings (average precision),
    then scramble one input at a time and see how much that score drops. Bigger drop = the model leans on it more.
    Ranking (not F1) is used so the answer does not depend on where an alert cutoff happens to sit.
    """
    horizon, high_q = int(spec.get("horizon", 7)), float(spec.get("high_q", 0.9))
    cols = features.FEATURES + features.NEIGHBOR_COLS + features.MEMORY_COLS if spec.get("extra") else features.FEATURES
    table = load_table(ndci_csv, weather_csv, horizon, extra=bool(spec.get("extra")))
    rng = np.random.default_rng(0)
    drops: Dict[str, List[float]] = {c: [] for c in cols}
    group_drops: Dict[str, List[float]] = {}
    base_scores, n_rows = [], 0
    for yr in _test_years(table, spec):
        train, test = table[table["year"] < yr], table[table["year"] == yr]
        if train["year"].nunique() < 2 or len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        thr = float(np.quantile(train["y"], high_q))
        truth = _label(thr, test["y"].to_numpy())
        if not 0 < truth.sum() < len(truth):
            continue
        m = _hgb().fit(train[cols], train["y"])
        base = float(average_precision_score(truth, m.predict(test[cols])))
        base_scores.append(base)
        n_rows += len(test)
        for c in cols:
            vals = []
            for _ in range(repeats):
                shuffled = test[cols].copy()
                shuffled[c] = rng.permutation(shuffled[c].to_numpy())
                vals.append(base - float(average_precision_score(truth, m.predict(shuffled))))
            drops[c].append(float(np.mean(vals)))
        for g, gcols in FEATURE_GROUPS.items():             # scramble a whole group at once: overlapping inputs
            gcols = [c for c in gcols if c in cols]          # (latest reading vs mean of 3) cannot cover for each other
            if not gcols:
                continue
            vals = []
            for _ in range(repeats):
                shuffled = test[cols].copy()
                perm = rng.permutation(len(shuffled))        # the same shuffle for every column keeps each row's group intact
                shuffled[gcols] = shuffled[gcols].to_numpy()[perm]
                vals.append(base - float(average_precision_score(truth, m.predict(shuffled))))
            group_drops.setdefault(g, []).append(float(np.mean(vals)))
    out = {"id": spec.get("id"), "label": spec.get("label"), "method": "importance", "horizon": horizon, "high_q": high_q,
           "stage": spec.get("stage", "screen"), "holdout": spec.get("holdout"), "only_year": spec.get("only_year")}
    if not base_scores:
        return {**out, "ok": False, "reason": "not enough history to measure importance"}
    rows = [{"feature": c, "label": FEATURE_LABELS.get(c, c), "drop": float(np.mean(v)), "spread": float(np.std(v))}
            for c, v in drops.items() if v]
    rows.sort(key=lambda r: -r["drop"])
    groups = sorted(({"group": g, "drop": float(np.mean(v)), "spread": float(np.std(v)),
                      "inputs": [FEATURE_LABELS.get(c, c) for c in FEATURE_GROUPS[g] if c in cols]}
                     for g, v in group_drops.items()), key=lambda r: -r["drop"])
    return {**out, "ok": True, "base_ap": float(np.mean(base_scores)), "n_test": n_rows, "importance": rows, "groups": groups,
            "note": "Drop in how well the model ranks high readings (average precision) when one input is scrambled. "
                    "Inputs that carry the same information share the credit, so a low bar does not prove an input is useless."}


def main(argv: Optional[List[str]] = None) -> None:
    """python -m algaebro.labcore spec.json ndci.csv weather.csv  -> prints RESULT_JSON:{...}"""
    argv = sys.argv[1:] if argv is None else argv
    spec = json.load(open(argv[0]))
    res = run_experiment(spec, argv[1], argv[2])
    print("RESULT_JSON:" + json.dumps(res, default=lambda o: None if (isinstance(o, float) and math.isnan(o)) else float(o)))


if __name__ == "__main__":
    main()
