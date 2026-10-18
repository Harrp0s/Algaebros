"""The most important tests: the forecast must never see anything after its issue date."""
import copy

import numpy as np
import pandas as pd

from algaebro import config, features, weather


def _load(synth):
    obs = features.prepare_obs(pd.read_csv(config.NDCI_CSV))
    wdf = weather.load()
    wfeat = features.weather_features(wdf)
    return obs, wdf, wfeat


def test_training_table_matches_live_forecast_row(synth):
    """Two independent code paths (batch merge vs one-row lookup) must produce identical features."""
    obs, _, wfeat = _load(synth)
    basin = features.basin_series(obs)
    table = features.build_training_table(obs, wfeat)
    assert len(table) > 500
    sample = table.sample(60, random_state=0)
    for _, r in sample.iterrows():
        as_of = r["target_date"] - pd.Timedelta(days=config.HORIZON_DAYS)
        row, why = features.feature_row(obs, basin, wfeat, r["zone_id"], as_of, r["target_date"])
        assert row is not None, why
        for f in features.FEATURES:
            assert np.isclose(row[f], r[f], atol=1e-9), f"{f}: live {row[f]} vs table {r[f]} ({r['zone_id']} {r['target_date'].date()})"


def test_future_satellite_readings_cannot_change_a_forecast(synth):
    obs, _, wfeat = _load(synth)
    as_of = pd.Timestamp("2026-07-20")
    target = as_of + pd.Timedelta(days=7)
    before = {}
    for zid in obs["zone_id"].unique():
        row, _ = features.feature_row(obs, features.basin_series(obs), wfeat, zid, as_of, target)
        before[zid] = row
    poisoned = obs.copy()
    poisoned.loc[poisoned["date"] > as_of, "ndci"] = 99.0
    poisoned["roll3"] = poisoned.groupby("zone_id")["ndci"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    pb = features.basin_series(poisoned)
    for zid, row in before.items():
        row2, _ = features.feature_row(poisoned, pb, wfeat, zid, as_of, target)
        assert row == row2, f"future satellite data leaked into {zid}"


def test_future_weather_cannot_change_a_forecast(synth):
    obs, wdf, wfeat = _load(synth)
    as_of = pd.Timestamp("2026-07-20")
    target = as_of + pd.Timedelta(days=7)
    basin = features.basin_series(obs)
    zid = obs["zone_id"].iloc[0]
    base, _ = features.feature_row(obs, basin, wfeat, zid, as_of, target)
    poisoned = wdf.copy()
    mask = poisoned["date"] > as_of
    poisoned.loc[mask, ["tmean", "tmax", "tmin", "precip", "rad", "wind"]] = 99.0
    wfeat2 = features.weather_features(poisoned)
    after, _ = features.feature_row(obs, basin, wfeat2, zid, as_of, target)
    assert base == after, "weather after the issue date leaked into the features"


def test_scene_taken_inside_the_processing_delay_is_not_used(synth):
    obs, _, wfeat = _load(synth)
    zid = "Z01"
    as_of = pd.Timestamp("2026-07-20")
    extra = pd.DataFrame([{"date": as_of - pd.Timedelta(days=1), "zone_id": zid, "item_id": "x", "ndci": 5.0, "p25": 5, "p75": 5, "n_valid": 100, "frac_valid": 1.0}])
    obs2 = features.prepare_obs(pd.concat([pd.read_csv(config.NDCI_CSV), extra.assign(date=extra["date"].dt.strftime("%Y-%m-%d"))]))
    row, _ = features.feature_row(obs2, features.basin_series(obs2), wfeat, zid, as_of, as_of + pd.Timedelta(days=7))
    assert row is not None and row["last_ndci"] != 5.0
    assert row["last_age"] >= config.OBS_LATENCY_DAYS


def test_stale_observation_gives_no_forecast_with_a_reason(synth):
    obs, _, wfeat = _load(synth)
    row, why = features.feature_row(obs, features.basin_series(obs), wfeat, "Z01", pd.Timestamp("2026-01-15"), pd.Timestamp("2026-01-22"))
    assert row is None and "days old" in why or "no satellite" in why
