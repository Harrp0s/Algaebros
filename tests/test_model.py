import numpy as np
import pandas as pd

from algaebro import config, features, model, weather


def _table():
    obs = features.prepare_obs(pd.read_csv(config.NDCI_CSV))
    return features.build_training_table(obs, features.weather_features(weather.load()))


def test_expanding_window_never_trains_on_the_test_year_or_later(synth):
    table = _table()
    oos, report = model.evaluate(table)
    assert oos is not None and report is not None
    first_test_year = min(oos["year"])
    assert first_test_year >= table["year"].min() + 2, "needs at least two training years before the first test year"
    # the 'high' threshold for each fold must come from EARLIER years only
    yr = int(sorted(oos["year"].unique())[1])
    expected = np.quantile(table[table["year"] < yr]["y"], config.TIER_HIGH_Q)
    got = oos[oos["year"] == yr]["thr_high"].iloc[0]
    assert abs(expected - got) < 1e-9


def test_report_has_all_baselines_and_honest_fields(synth):
    _, report = model.evaluate(_table())
    assert set(report["by_method"]) == {"model", "persistence", "climatology"}
    for m in report["by_method"].values():
        assert m["mae"] > 0 and m["rmse"] >= m["mae"]
    assert report["skill_vs_persistence"] is not None


def _rep(m_f1, p_f1, m_auc=0.8, p_auc=0.8, m_mae=0.04, p_mae=0.045):
    return {"by_method": {"model": {"f1": m_f1, "auc": m_auc, "mae": m_mae}, "persistence": {"f1": p_f1, "auc": p_auc, "mae": p_mae}}}


def test_choose_method_judges_alert_quality_not_average_error():
    # the real-data situation: the model has lower average error but almost never reaches HIGH
    assert model.choose_method(_rep(m_f1=0.13, p_f1=0.44, m_mae=0.0423, p_mae=0.0446))[0] == "persistence"
    # clearly better at catching HIGH, ranks at least as well, error not worse: switch
    assert model.choose_method(_rep(m_f1=0.60, p_f1=0.44))[0] == "model"
    # a small F1 edge is within noise: stay with the simple method
    assert model.choose_method(_rep(m_f1=0.46, p_f1=0.44))[0] == "persistence"
    # better F1 but worse at ranking risk: stay
    assert model.choose_method(_rep(m_f1=0.60, p_f1=0.44, m_auc=0.70, p_auc=0.80))[0] == "persistence"
    # better F1 but clearly worse error: stay
    assert model.choose_method(_rep(m_f1=0.60, p_f1=0.44, m_mae=0.06, p_mae=0.045))[0] == "persistence"
    assert model.choose_method(None)[0] == "persistence"
    assert model.choose_method(_rep(m_f1=None, p_f1=None))[0] == "persistence"


def test_tiers_are_consistent(synth):
    b = model.load_bundle()
    assert b["thr_elev"] < b["thr_high"]
    assert model.tier_for(b, b["thr_high"] + 0.01) == "high"
    assert model.tier_for(b, (b["thr_elev"] + b["thr_high"]) / 2) == "elevated"
    assert model.tier_for(b, b["thr_elev"] - 0.01) == "low"


def test_no_probability_helper_exists():
    """A residual-based 'chance of high' was badly overconfident when checked; it must not come back unnoticed."""
    assert not hasattr(model, "p_high") and not hasattr(model, "interval80")


def test_tier_record_reports_what_actually_happened_a_week_later():
    oos = pd.DataFrame({
        "y":          [0.20, 0.05, 0.20, 0.01, 0.15, 0.00],
        "pred":       [0.20, 0.20, 0.09, 0.09, 0.01, 0.02],
        "thr_high":   [0.13] * 6, "thr_elev": [0.07] * 6,
    })
    rec = model.tier_record(oos, "pred")
    assert rec["high"]["n"] == 2 and rec["high"]["next_high"] == 0.5 and rec["high"]["next_elevated_or_high"] == 0.5
    assert rec["elevated"]["n"] == 2 and rec["elevated"]["next_high"] == 0.5
    assert rec["low"]["n"] == 2 and rec["low"]["next_high"] == 0.5


def test_training_records_how_clustered_the_high_readings_are(synth):
    _, report = model.evaluate(_table())
    assert 0 < report["n_actual_high_dates"] <= report["n_actual_high"]


def test_bundle_remembers_it_was_trained_on_synthetic_data(synth):
    assert model.load_bundle()["data_source"] == "synthetic"
