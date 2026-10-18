"""Train, honestly evaluate, and use the forecast model.

Evaluation is an EXPANDING WINDOW over years: to test year Y we train only on years before Y.
That mimics real use (never peeking at the future). We compare the model with two baselines
that any forecaster should beat:
  * persistence : "next reading = the last reading"
  * climatology : "the usual value for this zone in this month"

If the model does not beat persistence out of sample, the system says so and falls back to
persistence. Shipping a model that is worse than the trivial baseline would be dishonest.
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score

from . import config
from .features import FEATURES

MIN_TRAIN_ROWS = 150
MIN_TEST_ROWS = 20
BEAT_MARGIN = 0.98  # model MAE must be at least 2% lower than persistence MAE


def make_model(name: str = "hgb"):
    if name == "xgb":
        from xgboost import XGBRegressor

        return XGBRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
            min_child_weight=5, reg_lambda=2.0, random_state=0, n_jobs=1,
        )
    return HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=300, min_samples_leaf=20, l2_regularization=1.0, random_state=0
    )


def climatology_predictor(train: pd.DataFrame):
    month = train["target_date"].dt.month
    zone_month = train.groupby(["zone_id", month])["y"].mean().to_dict()
    month_mean = train.groupby(month)["y"].mean().to_dict()
    overall = float(train["y"].mean())

    def predict(df: pd.DataFrame) -> np.ndarray:
        months = df["target_date"].dt.month
        return np.array([zone_month.get((z, m), month_mean.get(m, overall)) for z, m in zip(df["zone_id"], months)])

    return predict


def _binary_scores(truth: np.ndarray, pred: np.ndarray, score: np.ndarray) -> dict:
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    prec = tp / (tp + fp) if (tp + fp) else None
    rec = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * prec * rec / (prec + rec)) if (prec and rec) else (0.0 if prec is not None and rec is not None else None)
    auc = float(roc_auc_score(truth, score)) if 0 < truth.sum() < len(truth) else None
    return {"precision": prec, "recall": rec, "f1": f1, "auc": auc, "tp": tp, "fp": fp, "fn": fn}


def evaluate(table: pd.DataFrame, model_name: str = "hgb") -> Tuple[Optional[pd.DataFrame], Optional[dict]]:
    """Expanding-window evaluation. Returns (out-of-sample predictions, report) or (None, None)."""
    parts = []
    for yr in sorted(table["year"].unique()):
        train = table[table["year"] < yr]
        test = table[table["year"] == yr]
        if train["year"].nunique() < 2 or len(train) < MIN_TRAIN_ROWS or len(test) < MIN_TEST_ROWS:
            continue
        model = make_model(model_name).fit(train[FEATURES], train["y"])
        clim = climatology_predictor(train)
        thr_el, thr_hi = np.quantile(train["y"], [config.TIER_ELEVATED_Q, config.TIER_HIGH_Q])
        t = test[["zone_id", "target_date", "year", "y", "last_ndci"]].copy()
        t["pred_model"] = model.predict(test[FEATURES])
        t["pred_persist"] = test["last_ndci"].to_numpy()
        t["pred_clim"] = clim(test)
        t["thr_high"] = thr_hi
        t["thr_elev"] = thr_el
        parts.append(t)
    if not parts:
        return None, None
    oos = pd.concat(parts, ignore_index=True)
    return oos, summarize(oos)


def summarize(oos: pd.DataFrame) -> dict:
    out: Dict = {
        "test_years": sorted(int(y) for y in oos["year"].unique()),
        "n_test_rows": int(len(oos)),
        "high_threshold_note": "'high' = at or above the 90th percentile of training-year readings (recomputed per fold)",
        "by_method": {},
        "by_year": {},
    }
    truth = (oos["y"] >= oos["thr_high"]).to_numpy()
    for name, col in (("model", "pred_model"), ("persistence", "pred_persist"), ("climatology", "pred_clim")):
        pred_hi = (oos[col] >= oos["thr_high"]).to_numpy()
        entry = {
            "mae": float(mean_absolute_error(oos["y"], oos[col])),
            "rmse": float(math.sqrt(mean_squared_error(oos["y"], oos[col]))),
        }
        entry.update(_binary_scores(truth, pred_hi, oos[col].to_numpy()))
        out["by_method"][name] = entry
    out["n_actual_high"] = int(truth.sum())
    # HIGH readings cluster (a whole basin lights up at once), so count distinct dates too
    out["n_actual_high_dates"] = int(oos.loc[truth, "target_date"].nunique())
    for yr, g in oos.groupby("year"):
        out["by_year"][str(int(yr))] = {
            "n": int(len(g)),
            "mae_model": float(mean_absolute_error(g["y"], g["pred_model"])),
            "mae_persistence": float(mean_absolute_error(g["y"], g["pred_persist"])),
            "mae_climatology": float(mean_absolute_error(g["y"], g["pred_clim"])),
        }
    m, p = out["by_method"]["model"]["mae"], out["by_method"]["persistence"]["mae"]
    out["skill_vs_persistence"] = float(1 - m / p) if p > 0 else None
    return out


def choose_method(report: Optional[dict]) -> Tuple[str, str]:
    """Pick what drives the alerts, judged on how well it CATCHES 'high' readings, not on average error.

    Average error rewards predictions that hug the middle, which rarely reach 'high' and so would
    almost never raise an alert. The model must beat persistence on F1 (precision and recall
    together) by a clear margin, rank risk at least as well (AUC), and not be worse on error.
    """
    if report is None:
        return "persistence", "Not enough history to validate a model, so alerts use the latest satellite reading (persistence)."
    m, p = report["by_method"]["model"], report["by_method"]["persistence"]
    m_f1, p_f1 = m.get("f1") or 0.0, p.get("f1") or 0.0
    m_auc, p_auc = m.get("auc") or 0.0, p.get("auc") or 0.0
    wins = m_f1 > p_f1 * 1.10 and m_auc >= p_auc - 0.01 and m["mae"] <= p["mae"] * 1.02
    if wins:
        return "model", f"Model beat persistence out of sample at catching HIGH readings (F1 {m_f1:.2f} vs {p_f1:.2f}, AUC {m_auc:.2f} vs {p_auc:.2f})."
    return "persistence", (
        f"The model did not beat simply carrying the latest satellite reading forward at catching HIGH readings "
        f"(F1 {m_f1:.2f} vs {p_f1:.2f}, AUC {m_auc:.2f} vs {p_auc:.2f}), so alerts use persistence."
    )


def tier_record(oos: pd.DataFrame, col: str) -> Dict[str, dict]:
    """What actually happened a week after each predicted tier, in years the method never trained on."""
    pred = oos[col].to_numpy()
    thr_hi, thr_el = oos["thr_high"].to_numpy(), oos["thr_elev"].to_numpy()
    tiers = np.where(pred >= thr_hi, "high", np.where(pred >= thr_el, "elevated", "low"))
    actual_high = (oos["y"].to_numpy() >= thr_hi)
    actual_elev = (oos["y"].to_numpy() >= thr_el)
    out = {}
    for t in ("high", "elevated", "low"):
        m = tiers == t
        n = int(m.sum())
        out[t] = {
            "n": n,
            "next_high": float(actual_high[m].mean()) if n else None,
            "next_elevated_or_high": float(actual_elev[m].mean()) if n else None,
        }
    return out


def train_and_save(table: pd.DataFrame, model_name: str = "hgb") -> dict:
    oos, report = evaluate(table, model_name)
    method, why = choose_method(report)
    model = make_model(model_name).fit(table[FEATURES], table["y"])
    thr_el, thr_hi = (float(v) for v in np.quantile(table["y"], [config.TIER_ELEVATED_Q, config.TIER_HIGH_Q]))

    record = None
    if oos is not None:
        col = "pred_model" if method == "model" else "pred_persist"
        resid = (oos["y"] - oos[col]).to_numpy()
        record = tier_record(oos, col)
    else:
        resid = (table["y"] - table["last_ndci"]).to_numpy()

    in_season = table[table["target_date"].dt.month.isin(config.SEASON_MONTHS)]
    ref = {
        "tmean_7_q80": float(in_season["tmean_7"].quantile(0.80)),
        "wind_7_q25": float(in_season["wind_7"].quantile(0.25)),
        "rad_7_q75": float(in_season["rad_7"].quantile(0.75)),
        "dry_days_q75": float(in_season["dry_days"].quantile(0.75)),
    }
    bundle = {
        "model": model,
        "model_name": model_name,
        "method": method,
        "method_reason": why,
        "features": FEATURES,
        "thr_elev": thr_el,
        "thr_high": thr_hi,
        "resid": resid,
        "weather_ref": ref,
        "tier_record": record,
        "horizon": config.HORIZON_DAYS,
        "trained_rows": int(len(table)),
        "trained_through": str(table["target_date"].max().date()),
        "data_source": config.data_source(),
        "report": report,
    }
    joblib.dump(bundle, config.MODEL_PATH)
    meta = {k: v for k, v in bundle.items() if k not in ("model", "resid")}
    meta["residual_quantiles"] = {str(q): float(np.quantile(resid, q)) for q in (0.1, 0.5, 0.9)}
    config.METRICS_JSON.write_text(json.dumps(meta, indent=1, default=str))
    if oos is not None:
        oos.to_csv(config.DATA_DIR / "oos_predictions.csv", index=False)
    return bundle


def load_bundle() -> dict:
    return joblib.load(config.MODEL_PATH)


def predict_rows(bundle: dict, rows: List[dict]) -> np.ndarray:
    if not rows:
        return np.array([])
    if bundle["method"] == "persistence":
        return np.array([r["last_ndci"] for r in rows], dtype=float)
    X = pd.DataFrame(rows)[bundle["features"]]
    return np.asarray(bundle["model"].predict(X), dtype=float)


def tier_for(bundle: dict, value: float) -> str:
    if value >= bundle["thr_high"]:
        return "high"
    if value >= bundle["thr_elev"]:
        return "elevated"
    return "low"


# NOTE: there is deliberately no "probability of high" helper. Checked against what actually happened,
# a probability built from past errors was badly overconfident at the top end (it said 89% when the
# real rate was 46%). Users get the measured track record (tier_record) in plain words instead.
