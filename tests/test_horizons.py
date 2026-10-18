"""1-7 day training: the search, the honest choice of what drives alerts, the saved models, and the ML view."""
import time

import pytest

from algaebro import config, forecast, horizons, labcore, server
from algaebro.messenger import MemoryMessenger


def _r(method, h, f1, rng=None, stage="screen", setting=None, ap=0.3):
    return {"ok": True, "method": method, "horizon": h, "f1": f1, "f1_range": rng, "stage": stage, "setting": setting,
            "ap": ap, "recall": 0.5, "precision": 0.5, "mae": 0.04}


def test_the_search_covers_every_day_idea_and_setting_plus_the_rule():
    specs = horizons.search_specs(2026)
    assert len(specs) == 7 * (1 + len(horizons.ML_IDEAS) * len(horizons.SETTINGS))
    assert {s["horizon"] for s in specs} == set(range(1, 8))
    assert len({s["id"] for s in specs}) == len(specs), "every sandbox has its own id"
    assert all(s["holdout"] == 2026 for s in specs), "the latest season is never screened on"


def test_ml_drives_alerts_only_when_clearly_better_on_both_screen_and_holdout():
    res = [_r("persistence", 1, 0.40), _r("delta", 1, 0.60, [0.50, 0.70], setting="b"), _r("memory", 1, 0.45, [0.3, 0.6]),
           _r("delta", 1, 0.55, [0.45, 0.65], stage="confirm", setting="b"), _r("persistence", 1, 0.35, stage="confirm"),
           _r("persistence", 2, 0.40), _r("delta", 2, 0.60, [0.50, 0.70]),
           _r("delta", 2, 0.45, [0.30, 0.60], stage="confirm"), _r("persistence", 2, 0.40, stage="confirm")]
    s = horizons.summarise(res, 2026)
    d1, d2 = s["days"]
    assert d1["ml_method"] == "delta" and d1["ml_setting"] == "b" and d1["use_ml"], "won both: alerts use it"
    assert not d2["use_ml"], "won the screen but not clearly on the unseen season: alerts keep the rule"
    assert "1" in s["headline"] and "2" not in s["headline"].split("ahead")[0]
    specs = horizons.confirm_specs(res, 2026)
    assert {(c["method"], c["horizon"], c["only_year"]) for c in specs} >= {("delta", 1, 2026), ("persistence", 1, 2026)}


def test_scorers_can_be_saved_and_used_again(synth, tmp_path):
    import joblib

    table = labcore.load_table(str(config.NDCI_CSV), str(config.WEATHER_CSV), 3)
    m = horizons.fit_final("delta", "b", table)
    joblib.dump(m, tmp_path / "m.joblib")
    back = joblib.load(tmp_path / "m.joblib")
    assert len(back["scorer"](table.head(5))) == 5 and back["thr_elev"] <= back["thr_high"]


@pytest.fixture()
def trained(synth, monkeypatch):
    """Train only days 2 and 7 with two quick ideas, locally."""
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    job = {"log": [], "done": 0, "total": 0}
    p = horizons.run_training(job, prefer_daytona=True, horizons=[2, 7], ideas=["hgb_tuned", "memory"], settings=["a"])
    return p, job


def test_training_runs_confirms_refits_and_says_where_it_ran(trained):
    p, job = trained
    assert "Daytona was not used" in p["runner"] and p["notes"], "never claims Daytona when it ran locally"
    assert job["done"] == job["total"] == 2 * 3 + 2 * 2, "3 per day, then 2 per day to confirm"
    assert {d["days"] for d in p["summary"]["days"]} == {2, 7}
    models = horizons.load_models()
    assert set(models) == {2, 7} and all("scorer" in m for m in models.values())


def test_the_forecast_carries_both_views_and_alerts_follow_the_rule_unless_ml_won(trained):
    inp = forecast.Inputs()
    fc = forecast.make_forecast("2026-07-20", inputs=inp, horizon=2)
    assert fc["horizon_days"] == 2 and fc["ml"]["available"]
    with_ml = [z for z in fc["zones"] if "ml" in z]
    assert with_ml and all(set(z["ml"]) == {"value", "tier", "display"} for z in with_ml)
    for z in with_ml:
        expect = z["ml"]["tier"] if inp.horizon_models[2]["use_ml"] else z["rule_tier"]
        assert z["tier"] == expect
    assert forecast.make_forecast("2026-07-20", inputs=inp, horizon=4)["ml"] == {"available": False}, "no model for day 4"


def test_the_server_serves_days_ahead_and_the_ml_heat_view(trained, tmp_path):
    app = server.App(messenger=MemoryMessenger(), as_of="2026-07-20", db_path=tmp_path / "s.db", refresh_weather=False)
    st = app.state("2")
    assert st["horizon"] == 2 and st["ml"]["available"] and st["ml_days"] == [2, 7]
    assert st["target_date"] == "2026-07-22" and {s["days"] for s in st["skill"]} == {2, 7}
    assert app.state("99")["horizon"] == config.HORIZON_DAYS, "nonsense falls back to the standard horizon"
    assert app.forecast_now()["horizon_days"] == config.HORIZON_DAYS, "alerts and the bot stay on the standard horizon"
    rule, ml = app.heat_overlay("auto", "2", "rule"), app.heat_overlay("auto", "2", "ml")
    assert rule["url"] != ml["url"] and ml["source"] == "interpolated" and "ML forecast" in ml["note"]
    assert app.science()["horizons"]["summary"]["days"]


def test_training_from_the_dashboard_reloads_the_models(synth, tmp_path, monkeypatch):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    real = horizons.run_training
    monkeypatch.setattr(horizons, "run_training", lambda job, **kw: real(job, horizons=[3], ideas=["hgb_tuned"], settings=["a"], **kw))
    app = server.App(messenger=MemoryMessenger(), as_of="2026-07-20", db_path=tmp_path / "s.db", refresh_weather=False)
    jid = app.train_start()
    assert app.train_start() == jid
    for _ in range(300):
        if app.jobs.get(jid)["status"] != "running":
            break
        time.sleep(0.5)
    assert app.jobs.get(jid)["status"] == "done", app.jobs.get(jid)["error"]
    assert 3 in app.inputs.horizon_models and app.state("3")["ml"]["available"]


def test_low_confidence_water_is_shown_but_never_alerted(replay_fc, synth):
    from algaebro import proximity, zones as zones_mod

    assert config.SITE_BY_KEY["upper_charles"]["low_confidence"]
    recs = zones_mod.load_zone_records()
    z0 = recs[0]
    fc = {**replay_fc, "zones": [{**z, "tier": "high", "low_confidence": z["id"] == z0["id"]} for z in replay_fc["zones"]]}
    near = proximity.flagged_near(fc, recs, z0["lat"], z0["lon"], radius_m=1)
    assert all(fz["id"] != z0["id"] for _, fz in near), "a low-confidence area never counts as flagged for alerts"
