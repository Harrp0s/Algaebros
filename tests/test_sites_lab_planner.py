"""Several water bodies, the idea lab (holdout + confirmation), Science mode, and the cleanup planner."""
import io

import numpy as np
import pandas as pd
import pytest

from algaebro import config, features, heat, lab, labcore, nessie, planner, population, weather, zones


# ---------------------------------------------------------------------- several water bodies
def _grid(x0, y0, w, h, label):
    x = x0 + 20.0 * np.arange(w)
    y = y0 - 20.0 * np.arange(h)
    lab_ = np.zeros((h, w), "int16")
    lab_[1:-1, 1:-1] = label
    return lab_, x, y, np.full((h, w), 0.9, "float32")


def test_site_grids_paste_into_one_grid_at_their_own_place():
    a = _grid(323710.0, 4694190.0, 5, 4, 1)
    b = _grid(323810.0, 4694390.0, 3, 3, 11)          # 100 m east, 200 m north, same 20 m grid
    labels, x, y, _ = zones.mosaic([a, b])
    assert x[0] == 323710.0 and y[0] == 4694390.0 and labels.shape == (len(y), len(x))
    assert (labels == 1).sum() == (a[0] == 1).sum() and (labels == 11).sum() == (b[0] == 11).sum()
    r, c = np.argwhere(labels == 11)[0]
    assert x[c] == 323830.0 and y[r] == 4694370.0, "the second site lands where its own coordinates say"


def test_grids_that_are_not_on_the_same_20m_grid_are_refused():
    a = _grid(323710.0, 4694190.0, 5, 4, 1)
    b = _grid(323717.0, 4694190.0, 3, 3, 11)
    with pytest.raises(ValueError, match="same 20 m grid"):
        zones.mosaic([a, b])


def _two_sites(synth):
    """The synthetic river plus a made-up lake: same dates, other zone numbers, higher readings."""
    base = pd.read_csv(config.NDCI_CSV)
    lake = base[base["zone_id"].isin(["Z01", "Z02"])].copy()
    lake["zone_id"] = lake["zone_id"].map({"Z01": "Z21", "Z02": "Z22"})
    lake["ndci"] = lake["ndci"] + 0.08
    lake["site"] = "mystic_lakes"
    return features.prepare_obs(pd.concat([base.assign(site="charles"), lake], ignore_index=True))


def test_the_water_body_average_is_per_site_and_matches_the_live_row(synth):
    obs = _two_sites(synth)
    wfeat = features.weather_features(weather.load())
    table = features.build_training_table(obs, wfeat)
    basin = features.basin_series(obs)
    lake = table[table["site"] == "mystic_lakes"]
    river = table[table["site"] == "charles"]
    assert len(lake) > 50 and lake["basin_last"].mean() > river["basin_last"].mean() + 0.05, "the lake is not averaged with the river"
    for _, r in lake.sample(25, random_state=0).iterrows():
        row, why = features.feature_row(obs, basin, wfeat, r["zone_id"], r["target_date"] - pd.Timedelta(days=config.HORIZON_DAYS), r["target_date"])
        assert row is not None, why
        for f in features.FEATURES:
            assert np.isclose(row[f], r[f]), f


def test_extra_inputs_never_see_after_the_issue_date(synth):
    obs = _two_sites(synth)
    wfeat = features.weather_features(weather.load())
    table = features.build_training_table(obs, wfeat)
    extra = features.add_extra_features(table, obs)
    cols = features.NEIGHBOR_COLS + features.MEMORY_COLS
    assert extra[cols].notna().all().all()
    pick = extra.iloc[len(extra) // 2]
    poisoned = obs.copy()
    later = poisoned["date"] > pick["T_obs"]
    poisoned.loc[later, "ndci"] = 9.9                       # rewrite everything after the issue date
    again = features.add_extra_features(table, poisoned)
    row = again[(again["zone_id"] == pick["zone_id"]) & (again["target_date"] == pick["target_date"])].iloc[0]
    for c in ["nbr_last", "up_last", "nbr_diff", "mean30", "last_change"]:
        assert np.isclose(row[c], pick[c]), c


def test_neighbours_stay_on_their_own_water_body(synth):
    obs = _two_sites(synth)
    table = features.build_training_table(obs, features.weather_features(weather.load()))
    extra = features.add_extra_features(table, obs)
    z21 = extra[extra["zone_id"] == "Z21"]
    # Z21's only same-site neighbour is Z22 (Z20 does not exist, and Z10 is the river): no river reading leaks in
    assert (z21["up_last"] == z21["last_ndci"]).all(), "no upstream zone on the lake, so its own reading stands in"


def test_heat_blends_only_within_a_water_body():
    fc = {"zones": [{"number": 1, "last_obs": 0.0}, {"number": 2, "last_obs": 0.3}]}
    centers = [{"number": 1, "lat": 42.36, "lon": -71.10, "site": "charles"}, {"number": 2, "lat": 42.361, "lon": -71.10, "site": "mystic_lakes"}]
    v = heat.interpolate_values(fc, np.array([-71.10, -71.10]), np.array([42.3605, 42.3605]), np.array([1, 2]), centers)
    assert v[0] == pytest.approx(0.0) and v[1] == pytest.approx(0.3), "a pixel between two sites keeps its own site's reading"


# ---------------------------------------------------------------------- the idea lab
def test_holdout_screens_on_earlier_years_and_confirm_tests_only_the_held_out_year(synth):
    spec = {"id": "x", "method": "persistence", "horizon": 7, "high_q": 0.9}
    screened = labcore.run_experiment({**spec, "holdout": 2026}, str(config.NDCI_CSV), str(config.WEATHER_CSV))
    confirmed = labcore.run_experiment({**spec, "only_year": 2026, "stage": "confirm"}, str(config.NDCI_CSV), str(config.WEATHER_CSV))
    assert screened["ok"] and 2026 not in screened["test_years"] and max(screened["test_years"]) == 2025
    assert confirmed["ok"] and confirmed["test_years"] == [2026] and confirmed["stage"] == "confirm"
    assert set(screened["by_site"]) == {"charles"}


@pytest.mark.parametrize("method", ["delta", "classifier", "no_weather", "neighbors", "memory"])
def test_every_new_idea_runs_and_reports_a_score(synth, method):
    r = labcore.run_experiment({"id": method, "method": method, "horizon": 7, "high_q": 0.9, "holdout": 2026},
                               str(config.NDCI_CSV), str(config.WEATHER_CSV))
    assert r["ok"] and 0 <= r["f1"] <= 1 and r["auc"] is not None


def test_importance_ranks_inputs_with_plain_names(synth):
    r = labcore.run_experiment({"id": "imp", "method": "importance", "horizon": 7, "holdout": 2026}, str(config.NDCI_CSV), str(config.WEATHER_CSV))
    assert r["ok"] and len(r["importance"]) == len(features.FEATURES)
    drops = [x["drop"] for x in r["importance"]]
    assert drops == sorted(drops, reverse=True) and all(x["label"] != x["feature"] for x in r["importance"])
    assert {g["group"] for g in r["groups"]} == {"Recent satellite readings", "Weather", "Time of year", "Which zone"}


def _res(method, h, f1, lo, hi, stage="screen", year=None):
    return {"id": f"h{h}-{method}-{stage}", "ok": True, "method": method, "horizon": h, "high_q": 0.9, "f1": f1, "f1_range": [lo, hi],
            "recall": f1, "precision": f1, "stage": stage, "only_year": year}


def test_confirmation_picks_the_best_idea_per_window_against_the_simple_rule():
    screen = [_res("persistence", 7, 0.5, 0.4, 0.6), _res("delta", 7, 0.7, 0.6, 0.8), _res("memory", 7, 0.6, 0.5, 0.7),
              {"id": "imp", "ok": True, "method": "importance", "horizon": 7, "high_q": 0.9}]
    specs = lab.confirm_specs(screen, 2026)
    assert [(s["method"], s["only_year"], s["stage"]) for s in specs] == [("delta", 2026, "confirm"), ("persistence", 2026, "confirm")]
    assert lab.confirm_specs(screen, None) == []


def test_a_screened_winner_that_fails_on_the_held_out_year_is_not_called_a_winner():
    rs = [_res("persistence", 7, 0.5, 0.4, 0.6), _res("delta", 7, 0.8, 0.65, 0.9),
          _res("delta", 7, 0.45, 0.2, 0.7, "confirm", 2026), _res("persistence", 7, 0.5, 0.3, 0.7, "confirm", 2026)]
    s = lab.summarise(rs)
    assert "did not clearly hold up on the unseen 2026" in s["headline"] and s["confirmations"][0]["verdict"] == "did not hold up on 2026"
    rs[2] = _res("delta", 7, 0.8, 0.6, 0.9, "confirm", 2026)
    s = lab.summarise(rs)
    assert "also held up on the unseen 2026" in s["headline"] and s["confirmations"][0]["clear"]


def test_the_lab_specs_cover_every_idea_and_the_holdout():
    q, f = lab.quick_specs(2026), lab.full_specs(2026)
    assert {s["method"] for s in q} == set(lab.IDEAS) | {"importance"} and all(s["holdout"] == 2026 for s in q)
    assert len(f) == 3 * len(lab.IDEAS) + 1 and len({s["id"] for s in f}) == len(f)


def test_daytona_waits_and_retries_when_the_account_is_at_its_sandbox_limit(monkeypatch):
    monkeypatch.setattr(lab.time, "sleep", lambda s: None)

    class Client:
        n = 0

        def create(self, params, timeout=None):
            Client.n += 1
            if Client.n < 3:
                raise RuntimeError("Total CPU limit exceeded")
            return "sandbox"

    r = lab.DaytonaRunner.__new__(lab.DaytonaRunner)
    r._client = Client()
    r.params = lambda block_network=False: {}
    assert r._create(False) == "sandbox" and Client.n == 3


# ---------------------------------------------------------------------- the cleanup planner
def test_costs_follow_the_published_ranges():
    alum = planner.OPTION_BY_ID["alum"]
    assert planner.cost_10y(alum, 100) == [100 * 280 * 1, 100 * 2970 * 3], "lasts 4-20 years: once at best, three times at worst in 10 years"
    assert planner.cost_10y(planner.OPTION_BY_ID["ultrasonic"], 120) == [3 * 47000, 3 * 54000]
    assert planner.cost_10y(planner.OPTION_BY_ID["watershed"], 999, lbs=10) == [317000, 1380000]
    assert planner.cost_10y(planner.OPTION_BY_ID["peroxide"], 100) is None
    assert all(o["sources"] and all(u.startswith("https://") for _, u in o["sources"]) for o in planner.OPTIONS)


def _facts(kind="lake", people=20000, drinking=False):
    return {"site": "x", "name": "Test Pond", "kind": kind, "drinking_water": drinking, "zones": [], "area_km2": 0.5,
            "acres": 123.6, "people": people, "density_per_km2": 3000, "flagged_share": 0.1, "n_readings": 100}


def test_a_river_is_steered_to_the_root_cause_and_a_lake_can_use_alum():
    assert planner.plan(_facts("river"))["recommended"] == "watershed"
    assert planner.plan(_facts("lake"))["recommended"] in ("alum", "watershed")
    rows = {r["id"]: r for r in planner.plan(_facts("lake"))["options"]}
    assert rows["aeration"]["fit"] == "risky" and "Carmi" in rows["aeration"]["fit_why"]


def test_the_budget_decides_what_is_feasible():
    tiny = planner.plan(_facts("lake"), budget=1000)
    assert tiny["recommended"] is None and "Nothing that fits" in tiny["recommendation"]
    rows = {r["id"]: r for r in planner.plan(_facts("lake"), budget=200000)["options"]}
    assert rows["alum"]["within_budget"] == "maybe" and rows["ultrasonic"]["within_budget"] == "yes" and rows["dredging"]["within_budget"] == "no"


def test_drinking_water_gets_the_copper_warning():
    rows = {r["id"]: r for r in planner.plan(_facts(drinking=True))["options"]}
    assert "drinking-water" in rows["copper"]["warning"] and planner.plan(_facts())["options"][0].get("warning") is None


def test_without_grok_a_cost_question_is_answered_from_the_table():
    p = planner.plan(_facts("lake"))
    out = planner.answer("Would aeration cost more than alum here?", p)
    assert out["source"] == "fixed" and "Alum" in out["reply"] and "cheaper" in out["reply"]


def test_grok_is_given_only_the_computed_numbers():
    p = planner.plan(_facts("lake"), budget=300000)
    seen = {}

    def chat(msgs, temperature=0.2):
        seen["msgs"] = msgs
        return "ok"

    assert planner.answer("which is cheapest?", p, chat=chat) == {"reply": "ok", "source": "grok"}
    text = " ".join(m["content"] for m in seen["msgs"])
    assert "never invent a cost" in text and "300000" in text and "+1" not in text


def test_population_counts_tracts_near_the_water():
    tracts = [{"geoid": "25017000100", "lat": 42.3600, "lon": -71.1000, "land_km2": 1.0},
              {"geoid": "25017000200", "lat": 42.3650, "lon": -71.1000, "land_km2": 2.0},
              {"geoid": "25017000300", "lat": 42.4500, "lon": -71.1000, "land_km2": 1.0}]
    acs = population.parse_acs(io.StringIO("GEO_ID|B01003_E001|B01003_M001\n1400000US25017000100|4000|10\n1400000US25017000200|3000|9\n"
                                           "1400000US25017000300|9999|9\n1400000US36061000100|7|1\n"))
    assert acs == {"25017000100": 4000, "25017000200": 3000, "25017000300": 9999}
    zs = [{"id": "Z01", "site": "charles", "lat": 42.3605, "lon": -71.1, "points": [[-71.1, 42.3605]]}]
    out = population.exposure(zs, tracts, acs, radius_m=1000)
    assert out["zones"]["Z01"]["people"] == 7000 and out["sites"]["charles"]["density_per_km2"] == round(7000 / 3)


# ---------------------------------------------------------------------- the cleanup fund (Nessie or local)
def test_local_ledger_funds_spends_and_refuses_to_overspend(monkeypatch, tmp_path):
    monkeypatch.delenv("NESSIE_API_KEY", raising=False)
    monkeypatch.setattr(nessie, "LOCAL_JSON", tmp_path / "ledger.json")
    led = nessie.Ledger()
    assert led.backend == "local" and "not connected" in led.label
    with pytest.raises(ValueError, match="budget for this water body first"):
        led.spend("charles", 10, "x")
    assert led.fund("charles", 50000)["remaining"] == 50000
    assert led.spend("charles", 20000, "Alum")["remaining"] == 30000
    with pytest.raises(ValueError, match="only \\$30,000 is left"):
        led.spend("charles", 40000, "Dredging")


def test_with_a_key_the_fund_is_a_nessie_account_and_plans_are_withdrawals(monkeypatch, tmp_path):
    monkeypatch.setenv("NESSIE_API_KEY", "nk-test")
    monkeypatch.setattr(nessie, "IDS_JSON", tmp_path / "ids.json")
    calls, wd = [], []

    def fake(method, path, body=None, params=None, timeout=20):
        calls.append((method, path, body))
        if path == "/customers":
            return {"code": 201, "objectCreated": {"_id": "cust1"}}
        if path == "/customers/cust1/accounts":
            return {"code": 201, "objectCreated": {"_id": "acc1", "balance": body["balance"]}}
        if path == "/accounts/acc1/withdrawals" and method == "POST":
            wd.append({"amount": body["amount"], "description": body["description"], "status": "pending", "transaction_date": body["transaction_date"]})
            return {"code": 201, "objectCreated": {"_id": "w1"}}
        if path == "/accounts/acc1/withdrawals":
            return wd
        if path == "/accounts/acc1":
            return {"_id": "acc1", "balance": 100000}
        raise AssertionError(path)

    monkeypatch.setattr(nessie, "_call", fake)
    led = nessie.Ledger()
    st = led.fund("mystic_lakes", 100000)
    assert led.backend == "nessie" and st["account_id"] == "acc1" and st["funded"] == 100000
    st = led.spend("mystic_lakes", 25000, "Alum treatment (Mystic Lakes)")
    assert st["spent"] == 25000 and st["remaining"] == 75000 and st["transactions"][0]["memo"].startswith("Alum")
    sent = repr([c[2] for c in calls])
    assert "Savings" in sent and "+1" not in sent and "phone" not in sent.lower(), "only budgets and plan names go to Nessie"


def test_businesses_need_a_key_and_failures_are_explained(monkeypatch):
    monkeypatch.delenv("NESSIE_API_KEY", raising=False)
    assert nessie.merchants_near(42.36, -71.1)["ok"] is False
    monkeypatch.setenv("NESSIE_API_KEY", "nk-test")
    monkeypatch.setattr(nessie, "_call", lambda *a, **k: [{"name": "Boathouse Cafe", "category": ["food"], "geocode": {"lat": 42.36, "lng": -71.1}}])
    out = nessie.merchants_near(42.36, -71.1)
    assert out["ok"] and out["merchants"][0] == {"name": "Boathouse Cafe", "category": "food", "lat": 42.36, "lon": -71.1}
    assert "sandbox" in out["note"]


# ---------------------------------------------------------------------- the server
def test_science_and_planner_endpoints(synth, tmp_path, monkeypatch):
    import json
    import threading
    import urllib.request

    from algaebro import server

    monkeypatch.delenv("NESSIE_API_KEY", raising=False)
    monkeypatch.setattr(nessie, "LOCAL_JSON", tmp_path / "ledger.json")
    app = server.App(as_of="2026-07-20", db_path=tmp_path / "s.db", refresh_weather=False)
    httpd = server.build_server(app, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    try:
        code, sci = call("/api/science")
        assert code == 200 and sci["report"]["by_method"] and sci["scatter"] and sci["sites"][0]["key"] == "charles"
        assert {f["id"] for f in sci["features"]} == set(features.FEATURES)
        assert call("/api/plan/sites")[1] == [{"key": "charles", "name": "Lower Charles River", "kind": "river"}]
        code, p = call("/api/plan?site=charles&budget=5000000")
        assert code == 200 and p["facts"]["acres"] > 0 and p["ledger"]["backend"] == "local" and p["recommended"]
        assert call("/api/plan?site=nowhere")[0] == 400
        assert call("/api/plan/fund", {"site": "charles", "amount": 100000})[1]["remaining"] == 100000
        code, st = call("/api/plan/spend", {"site": "charles", "option": "watershed", "amount": 40000})
        assert code == 200 and st["remaining"] == 60000
        assert call("/api/plan/spend", {"site": "charles", "option": "watershed", "amount": 90000})[0] == 409
        assert call("/api/plan/spend", {"site": "charles", "option": "magic", "amount": 1})[0] == 400
        code, a = call("/api/plan/ask", {"site": "charles", "question": "Is alum cheaper than dredging?"})
        assert code == 200 and a["reply"]
        state = call("/api/state")[1]
        z = next(z for z in state["zones"] if z["tier"] != "unknown")
        assert z["site"] == "charles" and set(z["inputs"]) == set(features.FEATURES)
    finally:
        httpd.shutdown()
        httpd.server_close()
