import pandas as pd

from algaebro import features, forecast


def test_live_forecast_shape(synth):
    fc = forecast.make_forecast(None)
    assert fc["live"] and not fc["replay"]
    assert fc["data_source"] == "synthetic"
    assert len(fc["zones"]) == 10
    assert fc["weather"]["heat_wave_ahead"] in (True, False, None)
    for z in fc["zones"]:
        assert z["tier"] in {"low", "elevated", "high", "unknown"}
        if z["tier"] != "unknown":
            assert "p_high" not in z, "uncalibrated probabilities must not be shown"
            assert z["track"] is None or z["track"]["n"] > 0
            assert z["last_obs"] is not None and z["obs_age_days"] >= 2


def test_replay_is_labelled_and_does_not_look_at_future_weather(replay_fc):
    assert replay_fc["replay"] and not replay_fc["live"]
    assert replay_fc["weather"]["heat_wave_ahead"] is None


def test_replay_attaches_what_the_satellite_actually_saw(replay_fc):
    seen = [z for z in replay_fc["zones"] if "verification" in z]
    assert seen, "expected at least one zone with a later satellite reading"
    for z in seen:
        assert z["verification"]["observed_tier"] in {"low", "elevated", "high"}


def test_replay_forecast_is_unchanged_by_poisoning_everything_after_the_issue_date(synth):
    inp = forecast.Inputs()
    base = forecast.make_forecast("2026-07-20", inputs=inp)
    as_of = pd.Timestamp("2026-07-20")
    inp.obs.loc[inp.obs["date"] > as_of, "ndci"] = 50.0
    inp.obs["roll3"] = inp.obs.groupby("zone_id")["ndci"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    inp.basin = features.basin_series(inp.obs)
    inp.weather.loc[inp.weather["date"] > as_of, ["tmean", "tmax", "tmin", "precip", "rad", "wind"]] = 99.0
    inp.wfeat = features.weather_features(inp.weather)
    after = forecast.make_forecast("2026-07-20", inputs=inp)
    for a, b in zip(base["zones"], after["zones"]):
        assert a["tier"] == b["tier"] and a.get("ndci_pred") == b.get("ndci_pred"), f"{a['id']} changed"


def test_replay_season_frames(synth):
    rp = forecast.make_replay(2025)
    assert len(rp["frames"]) >= 15
    assert rp["data_source"] == "synthetic"
    assert all("points" in z for z in rp["zones"])


def test_map_page_needs_no_api_key_and_falls_back(synth):
    from algaebro import mapview
    html = mapview.build_map_html({"year": 2025, "data_source": "real", "method": "persistence", "zones": [], "frames": []}, None)
    assert "basemaps.cartocdn.com" not in html, "that provider now asks for an API key"
    assert "tile.openstreetmap.org" in html and "needsHttp" in html, "OSM only when served over http"
    assert "arcgisonline.com" in html and "tileerror" in html and "--serve" in html
    assert "bestN" in html


def test_map_can_be_served_over_http(synth, tmp_path):
    import importlib.util, threading, urllib.request
    from pathlib import Path
    from algaebro import config, forecast

    forecast.make_replay(2025)
    spec = importlib.util.spec_from_file_location("make_map", Path(__file__).resolve().parent.parent / "scripts" / "06_make_map.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    mod.build()
    httpd = mod.make_server(0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        body = urllib.request.urlopen(f"http://127.0.0.1:{httpd.server_address[1]}/map.html", timeout=5).read().decode()
        assert "Charles River bloom-risk forecast" in body
    finally:
        httpd.shutdown(); httpd.server_close()
