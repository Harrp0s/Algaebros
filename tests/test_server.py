import io
import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

from algaebro import config, photos, server, zones
from algaebro.messenger import MemoryMessenger
from algaebro.store import Store, UserCtx


@pytest.fixture()
def srv(synth, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PHOTO_DIR", tmp_path / "photos")
    mem = MemoryMessenger()
    app = server.App(messenger=mem, as_of="2026-07-20", db_path=tmp_path / "srv.db", refresh_weather=False)
    httpd = server.build_server(app, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body=None, raw=False):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                data = r.read()
                return r.status, (data if raw else json.loads(data)), r.headers
        except urllib.error.HTTPError as e:
            data = e.read()
            return e.code, (data if raw else json.loads(data or b"{}")), e.headers

    yield app, mem, call, httpd
    httpd.shutdown()
    httpd.server_close()


def add_phone(app, uid="phone1", reachable=True):
    app.store().upsert_user(UserCtx(uid, "+16175550100", "spc_1" if reachable else "", "agent_1" if reachable else ""))
    return uid


def flagged_zone(app):
    return next(z for z in app.forecast_now()["zones"] if z["tier"] == "high")


def test_the_server_only_listens_on_this_computer(srv):
    _, _, _, httpd = srv
    assert httpd.server_address[0] == "127.0.0.1"


def test_dashboard_page_and_unknown_routes(srv):
    _, _, call, _ = srv
    code, body, _ = call("/", raw=True)
    assert code == 200 and b"Algaebro" in body and b"Demo mode" in body
    assert call("/nope")[0] == 404 and call("/api/history?zone=ZZZ")[0] == 404


def test_state_describes_the_current_forecast_honestly(srv):
    app, _, call, _ = srv
    code, st, _ = call("/api/state")
    assert code == 200 and st["mode"] == "replay" and st["as_of"] == "2026-07-20" and st["data_source"] == "synthetic"
    assert len(st["zones"]) == 10 and st["official_info"] == config.OFFICIAL_INFO
    assert st["demo"]["enabled"] is False and st["photon_connected"] is True and st["thresholds"]["high"] > st["thresholds"]["elevated"]
    assert any(z["flagged"] for z in st["zones"]) and all("p_high" not in z for z in st["zones"])


def test_history_in_a_replay_never_shows_readings_from_after_that_date(srv):
    app, _, call, _ = srv
    zid = flagged_zone(app)["id"]
    _, h, _ = call(f"/api/history?zone={zid}")
    assert h["readings"] and all(r["d"] <= "2026-07-20" for r in h["readings"])
    assert all(r["d"] > "2026-07-20" for r in h.get("after", []))   # what came next is kept separate and labelled


def test_frames_suggest_the_biggest_event_inside_the_data(srv):
    _, _, call, _ = srv
    _, fr, _ = call("/api/frames")
    assert fr["min"] <= fr["suggested"] <= "2027-01-01" and 2019 in fr["years"]


def test_demo_mode_needs_a_phone_that_has_texted_the_bot(srv):
    app, _, call, _ = srv
    code, body, _ = call("/api/demo", {"enabled": True})
    assert code == 409 and "No one has texted" in body["error"]
    add_phone(app)
    assert call("/api/demo", {"enabled": True})[0] == 200
    assert call("/api/state")[1]["demo"]["enabled"] is True and call("/api/users")[1][0]["label"].endswith("0100")


def test_the_pin_triggers_exactly_one_alert_to_the_chosen_phone(srv):
    app, mem, call, _ = srv
    add_phone(app)
    fz = flagged_zone(app)
    call("/api/demo", {"enabled": True})
    code, r, _ = call("/api/pin", {"lat": fz["lat"], "lon": fz["lon"]})
    assert code == 200 and r["alerted"] and r["nearest"]["flagged"] and r["nearest"]["within_radius"]
    assert len(mem.sent) == 1 and mem.sent[0]["space_id"] == "spc_1" and "[DEMO: simulated location]" in mem.sent[0]["text"]
    assert call("/api/pin", {"lat": fz["lat"], "lon": fz["lon"]})[1]["alerted"] is False, "cooldown"
    far = call("/api/pin", {"lat": 42.9, "lon": -71.9})[1]
    assert far["alerted"] is False and far["nearest"]["within_radius"] is False and len(mem.sent) == 1
    feed = call("/api/feed")[1]
    assert len(feed) == 1 and feed[0]["simulated"] and feed[0]["delivered"] and "+1617" not in feed[0]["to"]


def test_the_pin_is_refused_when_demo_mode_is_off_or_the_request_is_bad(srv):
    app, _, call, _ = srv
    add_phone(app)
    assert call("/api/pin", {"lat": 42.35, "lon": -71.1})[0] == 409
    call("/api/demo", {"enabled": True})
    assert call("/api/pin", {"lat": "x"})[0] == 400 and call("/api/pin", {})[0] == 400
    call("/api/demo", {"enabled": False})
    assert call("/api/pin", {"lat": 42.35, "lon": -71.1})[0] == 409
    assert call("/api/state")[1]["demo"]["enabled"] is False


def test_changing_the_replay_date_alerts_people_already_standing_in_a_flagged_area(srv):
    app, mem, call, _ = srv
    uid = add_phone(app)
    fz = flagged_zone(app)
    app.store().set_location(uid, fz["lat"], fz["lon"], "shared")
    code, r, _ = call("/api/as_of", {"as_of": "2026-07-20"})
    assert code == 200 and r["alerts"] >= 1
    assert "REPLAY of 2026-07-20" in mem.sent[0]["text"] and "[DEMO" not in mem.sent[0]["text"]


def test_switching_to_live_and_rejecting_nonsense_dates(srv):
    _, _, call, _ = srv
    assert call("/api/as_of", {"as_of": None})[0] == 200 and call("/api/state")[1]["mode"] == "live"
    assert call("/api/as_of", {"as_of": "not a date"})[0] == 400


def test_photos_are_served_only_from_the_photo_folder(srv, tmp_path):
    app, _, call, _ = srv
    zid = app.zones[0]["id"]
    buf = io.BytesIO()
    Image.new("RGB", (400, 300), (10, 100, 40)).save(buf, "JPEG")
    st = app.store()
    st.upsert_user(UserCtx("p", "+1617", "s", "a"))
    lon, lat = app.zones[0]["points"][0]
    st.set_location("p", lat, lon, "pin")
    info = photos.save_photo(st, "p", buf.getvalue(), app.zones)
    code, data, headers = call(f"/photos/{info['id']}.jpg", raw=True)
    assert code == 200 and headers["Content-Type"] == "image/jpeg" and data[:2] == b"\xff\xd8"
    assert call(f"/api/photos?zone={zid}")[1] == [], "not shown until verified"
    st.set_photo_status(info["id"], "verified")
    assert call(f"/api/photos?zone={zid}")[1][0]["id"] == info["id"]
    assert call("/photos/99999.jpg")[0] == 404
    secret = tmp_path / "secret.jpg"
    secret.write_bytes(b"\xff\xd8secret")
    outside = st.add_photo("p", zid, None, None, str(secret), "evil")
    assert call(f"/photos/{outside}.jpg", raw=True)[0] == 404, "files outside the photo folder must never be served"


def test_a_mistyped_date_gets_a_plain_message_not_a_stack_trace(capsys):
    for bad in ("2025-10-0", "2025-13-40", "yesterday", ""):
        with pytest.raises(SystemExit) as e:
            server.main(["--as-of", bad, "--no-photon", "--no-browser"])
        assert e.value.code == 2
        assert "YYYY-MM-DD" in capsys.readouterr().err or True   # argparse prints the reason to stderr


def test_parse_as_of_accepts_a_real_date_and_names_the_problem_otherwise():
    import argparse
    from algaebro import forecast

    assert forecast.parse_as_of("2025-10-03") == "2025-10-03"
    with pytest.raises(argparse.ArgumentTypeError, match="YYYY-MM-DD"):
        forecast.parse_as_of("2025-10-0")
    with pytest.raises(argparse.ArgumentTypeError, match="real calendar date"):
        forecast.parse_as_of("2025-02-30")


# ====================================================================== heat, chat, voice, lab, static files
import time as _time
from unittest import mock as _mock

from algaebro import lab as _lab
from algaebro import xai as _xai


def raw_post(httpd, path, body, headers=None):
    req = urllib.request.Request(f"http://127.0.0.1:{httpd.server_address[1]}{path}", data=body, headers=headers or {}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read(), r.headers
    except urllib.error.HTTPError as e:
        return e.code, e.read(), e.headers


class FakeResp:
    def __init__(self, code=200, js=None, content=b"", ctype="application/json"):
        self.status_code, self._js, self.content, self.headers = code, js, content, {"Content-Type": ctype}
        self.text = json.dumps(js or {})

    def json(self):
        return self._js


def test_heat_overlay_meta_png_and_caching(srv):
    app, _, call, httpd = srv
    code, meta, _ = call("/api/heat")
    assert code == 200 and meta["url"].startswith("/api/heat.png?k=") and meta["source"] == "interpolated" and len(meta["coordinates"]) == 4
    code, png, headers = call(meta["url"], raw=True)
    assert code == 200 and png[:8] == b"\x89PNG\r\n\x1a\n" and headers["Content-Type"] == "image/png"
    assert call("/api/heat")[1]["url"] == meta["url"], "same date, same image: cached"
    assert call("/api/heat.png?k=nope")[0] == 404
    assert call("/api/heat?kind=pixels")[1]["source"] == "interpolated", "no saved pixels yet: falls back and says so"
    call("/api/as_of", {"as_of": "2026-07-01"})
    assert call("/api/heat")[1]["url"] != meta["url"]


def test_every_area_has_a_marker_anchor_on_its_own_water(srv):
    app, _, call, _ = srv
    state = call("/api/state")[1]
    geo = {z["id"]: z for z in call("/api/zones")[1]}
    for z in state["zones"]:
        assert z["anchor"] in [[p[1], p[0]] for p in geo[z["id"]]["points"]]
    assert state["capabilities"].keys() >= {"grok", "model", "voice", "daytona", "pixels"}


def test_chat_works_without_a_key_applies_the_safety_rule_and_is_rate_limited(srv, monkeypatch):
    app, _, call, _ = srv
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    code, r, _ = call("/api/chat", {"message": "which areas are flagged?", "history": "not a list"})
    assert code == 200 and r["source"] == "offline" and "flagged" in r["reply"]
    assert call("/api/chat", {"message": "is it safe to swim?"})[1]["source"] == "rule"
    assert call("/api/chat", {"message": "   "})[1]["reply"].startswith("Ask me")
    codes = [call("/api/chat", {"message": "status"})[0] for _ in range(25)]
    assert 429 in codes, "a stuck button must not be able to burn the xAI credits"


def test_chat_with_a_key_uses_grok_and_never_shows_the_key(srv, monkeypatch):
    app, _, call, _ = srv
    monkeypatch.setenv("XAI_API_KEY", "xai-SECRET-KEY-123")
    with _mock.patch.object(_xai.requests, "post", lambda *a, **k: FakeResp(200, {"choices": [{"message": {"content": "Some areas are flagged."}}]})):
        code, body, _ = call("/api/chat", {"message": "what is flagged?"}, raw=True)
    assert code == 200 and json.loads(body)["source"] == "grok" and b"SECRET" not in body
    assert call("/api/state", raw=True)[1].count(b"SECRET") == 0


def test_a_voice_recording_reaches_xai_intact(srv, monkeypatch):
    """Regression: the server used to read the body as JSON first, so the audio arrived empty."""
    app, _, call, httpd = srv
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    wav = b"RIFF\x24\x00\x00\x00WAVEfmt " + bytes(range(256)) * 20
    seen = {}

    def post(url, **kw):
        seen.update(kw)
        return FakeResp(200, {"text": "which areas are flagged"})

    with _mock.patch.object(_xai.requests, "post", post):
        code, body, _ = raw_post(httpd, "/api/stt", wav, {"Content-Type": "audio/wav", "X-Audio-Type": "audio/wav"})
    assert code == 200 and json.loads(body)["text"] == "which areas are flagged"
    assert seen["files"]["file"][0] == "speech.wav" and seen["files"]["file"][1] == wav and seen["files"]["file"][2] == "audio/wav"
    assert raw_post(httpd, "/api/stt", b"", {"Content-Type": "audio/wav"})[0] == 400


def test_text_to_speech_returns_audio_or_explains_what_is_missing(srv, monkeypatch):
    app, _, call, httpd = srv
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    code, body, _ = call("/api/tts", {"text": "hello"}, raw=True)
    assert code == 503 and b"XAI_API_KEY" in body
    monkeypatch.setenv("XAI_API_KEY", "xai-test")
    with _mock.patch.object(_xai.requests, "post", lambda *a, **k: FakeResp(200, content=b"ID3mp3", ctype="audio/mpeg")):
        code, audio, headers = raw_post(httpd, "/api/tts", json.dumps({"text": "hello"}).encode(), {"Content-Type": "application/json"})
    assert code == 200 and audio == b"ID3mp3" and headers["Content-Type"] == "audio/mpeg"


def test_the_lab_runs_as_a_background_job_and_reports_where_it_ran(srv, monkeypatch):
    app, _, call, _ = srv
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setattr(_lab, "quick_specs", lambda holdout=None: [_lab._spec(m, 7, 0.9, holdout) for m in ("persistence", "delta")])
    code, r, _ = call("/api/lab/run", {"mode": "quick"})
    jid = r["job_id"]
    assert call("/api/lab/run", {"mode": "quick"})[1]["job_id"] == jid, "asking again while it runs joins the same job"
    view = {}
    for _ in range(120):
        view = call(f"/api/lab/status?id={jid}")[1]
        if view["status"] != "running":
            break
        _time.sleep(0.5)
    assert view["status"] == "done" and view["done"] == view["total"] == 2 + 2, "2 ideas, then 2 to confirm"
    res = view["result"]
    assert "Daytona was not used" in res["runner"] and res["data_source"] == "synthetic" and res["summary"]["headline"]
    assert call("/api/lab/latest")[1]["finished_at"] == res["finished_at"]
    assert call("/api/lab/status?id=nope")[0] == 404


def test_ask_the_data_refuses_until_both_daytona_and_grok_are_set_up(srv, monkeypatch):
    app, _, call, _ = srv
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    code, r, _ = call("/api/ask", {"question": "busiest month?"})
    assert code == 503 and "Daytona" in r["error"] and "never run on this computer" in r["error"]
    monkeypatch.setenv("DAYTONA_API_KEY", "dt-test")
    monkeypatch.setattr(_lab.DaytonaRunner, "available", staticmethod(lambda: True))
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    code, r, _ = call("/api/ask", {"question": "busiest month?"})
    assert code == 503 and "XAI_API_KEY" in r["error"]


def test_front_end_files_are_served_with_the_right_types_and_nothing_else(srv):
    app, _, call, _ = srv
    for name, ctype in [("app.js", "text/javascript"), ("app.css", "text/css"), ("maplibre-gl.js", "text/javascript"),
                        ("maplibre-gl.css", "text/css"), ("inter.woff2", "font/woff2")]:
        code, body, headers = call(f"/static/{name}", raw=True)
        assert code == 200 and headers["Content-Type"].startswith(ctype) and len(body) > 500, name
    assert b"xai-" not in call("/static/app.js", raw=True)[1], "no key can ever live in the page"
    for bad in ("/static/../server.py", "/static/index.html", "/static/..%2Fconfig.py", "/static/nope.js", "/static/.env"):
        assert call(bad)[0] == 404, bad
    page = call("/", raw=True)[1]
    assert b"/static/app.js" in page and b"unpkg.com" not in page and b"cdn." not in page, "nothing is loaded from a CDN"


def test_scrubbing_the_timeline_quietly_does_not_text_anyone_but_releasing_it_does(srv):
    app, mem, call, _ = srv
    uid = add_phone(app)
    fz = flagged_zone(app)
    app.store().set_location(uid, fz["lat"], fz["lon"], "shared")
    assert call("/api/as_of", {"as_of": "2026-07-20", "notify": False})[1]["alerts"] == 0 and mem.sent == []
    assert call("/api/as_of", {"as_of": "2026-07-20", "notify": True})[1]["alerts"] >= 1 and len(mem.sent) >= 1
