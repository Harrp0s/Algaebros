"""Grok's photo check, the moderation panel's actions, and the story bubbles."""
import io
import json
import threading
import urllib.error
import urllib.request

import pytest
from PIL import Image

from algaebro import config, photos, screening, server, xai
from algaebro.messenger import Incoming
from algaebro.store import Store, UserCtx

ANS = {"is_water_photo": True, "kind": "water", "offensive": False, "confidence": 0.9, "bloom": "bloom", "description": "Green scum near the shore."}


def jpeg(color=(30, 120, 60), size=(400, 300)):
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, "JPEG")
    return buf.getvalue()


def grok(answer):
    """A fake chat() that records what it was sent and replies with `answer` (dict -> JSON, str -> as is)."""
    seen = []

    def chat(msgs, temperature=0.0):
        seen.append(msgs)
        return answer if isinstance(answer, str) else "Sure:\n```json\n" + json.dumps(answer) + "\n```"
    chat.seen = seen
    return chat


@pytest.fixture(autouse=True)
def photo_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PHOTO_DIR", tmp_path / "photos")


# ---------------------------------------------------------------------- the verdict
@pytest.mark.parametrize("answer, status", [
    (ANS, "verified"),
    ({**ANS, "is_water_photo": False, "kind": "meme", "confidence": 0.95}, "rejected"),
    ({**ANS, "offensive": True}, "rejected"),
    ({**ANS, "kind": "satellite", "confidence": 0.9}, "rejected"),
    ({**ANS, "confidence": 0.4}, "review"),
    ({**ANS, "is_water_photo": False, "kind": "screenshot", "confidence": 0.5}, "review"),
    ("I think it is a lake", "review"),
])
def test_grok_answers_become_a_status(answer, status):
    d = screening.screen(jpeg(), "Spy Pond", chat=grok(answer))
    assert d["status"] == status
    if status == "rejected" and isinstance(answer, dict) and not answer.get("offensive"):
        assert answer["kind"] in d["reason"]


def test_the_photo_goes_to_grok_as_an_image_with_the_place_and_no_safety_question():
    chat = grok(ANS)
    screening.screen(jpeg(), "Mystic Lakes", chat=chat)
    content = chat.seen[0][0]["content"]
    assert content[0]["type"] == "image_url" and content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert "Mystic Lakes" in content[1]["text"] and "Never say whether the water is safe" in content[1]["text"]


def test_a_bad_bloom_label_or_confidence_is_made_safe():
    v = screening.parse(json.dumps({**ANS, "bloom": "definitely toxic", "confidence": "very"}))
    assert v["bloom"] == "unclear" and v["confidence"] == 0.0


def test_without_grok_or_when_it_fails_the_photo_waits_for_a_person(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    assert screening.screen(jpeg())["status"] == "review"

    def down(msgs, temperature=0.0):
        raise xai.XAIError("xAI says too many requests")
    d = screening.screen(jpeg(), chat=down)
    assert d["status"] == "review" and "too many requests" in d["reason"]


def test_the_vision_model_can_be_set_separately(monkeypatch):
    monkeypatch.setenv("XAI_MODEL", "grok-text")
    monkeypatch.delenv("XAI_VISION_MODEL", raising=False)
    assert xai.vision_model() == "grok-text"
    monkeypatch.setenv("XAI_VISION_MODEL", "grok-eyes")
    assert xai.vision_model() == "grok-eyes"


# ---------------------------------------------------------------------- through the bot
class Msgr:
    def __init__(self, files):
        self.files = files

    def download_attachment(self, att):
        return self.files[att]


def _send(store, recs, att, data, chat, user="u1"):
    return photos.handle_photo_message(store, Msgr({att: data}), recs, Incoming("m" + att, user, "+16175550100", None, "s", "a", attachment_id=att),
                                       screener=lambda j, p: screening.screen(j, p, chat=chat))


def test_a_troll_is_rejected_hidden_and_told_politely(store, synth):
    from algaebro import zones
    recs = zones.load_zone_records()
    reply = _send(store, recs, "t1", jpeg((200, 0, 200)), grok({**ANS, "is_water_photo": False, "kind": "meme", "confidence": 0.97}))
    row = store.db.execute("SELECT * FROM photos").fetchone()
    assert "doesn't look like a photo of the water" in reply and row["status"] == "rejected" and row["decided_by"] == "grok"
    assert store.photos_for_zone(row["zone_id"] or "Z01") == [], "rejected photos never reach the public gallery"


def test_a_real_photo_is_verified_placed_and_labelled(store, synth):
    from algaebro import zones
    recs = zones.load_zone_records()
    lon, lat = recs[3]["points"][0]
    store.upsert_user(UserCtx("u1", "+16175550100", "s", "a"))
    store.set_location("u1", lat, lon, "pin")
    reply = _send(store, recs, "g1", jpeg(), grok(ANS))
    assert "added to the map near" in reply and "not a safety test" in reply
    shown = store.photos_for_zone(recs[3]["id"])
    assert len(shown) == 1 and shown[0]["bloom"] == "bloom" and shown[0]["description"].startswith("Green")


def test_duplicates_tiny_images_and_floods_are_refused_before_grok_is_asked(store, synth, monkeypatch):
    from algaebro import zones
    recs = zones.load_zone_records()
    chat = grok(ANS)
    _send(store, recs, "d1", jpeg(), chat)
    assert "already have that photo" in _send(store, recs, "d2", jpeg(), chat)
    assert "too small" in _send(store, recs, "d3", jpeg(size=(120, 90)), chat)
    monkeypatch.setattr(photos, "MAX_PER_DAY", 2)
    _send(store, recs, "d4", jpeg((10, 90, 140)), chat)
    assert "a lot of photos today" in _send(store, recs, "d5", jpeg((90, 10, 40)), chat)
    assert len(chat.seen) == 2, "Grok was only asked about the two photos that were actually stored"


def test_screening_old_pending_photos(store, synth):
    from algaebro import zones
    recs = zones.load_zone_records()
    for i, c in enumerate([(30, 120, 60), (31, 121, 61)]):
        info = photos.save_photo(store, "u1", jpeg(c), recs, f"old{i}")
        store.db.execute("UPDATE photos SET status='unverified' WHERE id=?", (info["id"],))   # as stored before screening existed
    store.db.commit()
    counts = photos.screen_pending(store, recs, screener=lambda j, p: screening.screen(j, p, chat=grok(ANS)))
    assert counts["verified"] == 2 and store.photo_counts()["pending"] == 0


def test_an_old_database_gets_the_new_photo_columns(tmp_path):
    import sqlite3
    db = sqlite3.connect(tmp_path / "old.db")
    db.executescript("CREATE TABLE photos(id INTEGER PRIMARY KEY AUTOINCREMENT, user_id TEXT, zone_id TEXT, lat REAL, lon REAL, "
                     "received_at TEXT, path TEXT, attachment_id TEXT, status TEXT DEFAULT 'unverified');"
                     "INSERT INTO photos(user_id, path, status) VALUES('u', 'x.jpg', 'unverified');")
    db.commit()
    db.close()
    st = Store(tmp_path / "old.db")
    assert st.photo_counts()["pending"] == 1 and "bloom" in st.get_photo(1).keys()


# ---------------------------------------------------------------------- moderation panel and stories, over HTTP
@pytest.fixture()
def http(synth, tmp_path):
    app = server.App(as_of="2026-07-20", db_path=tmp_path / "srv.db", refresh_weather=False)
    httpd = server.build_server(app, 0)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"

    def call(path, body=None):
        req = urllib.request.Request(base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="GET" if body is None else "POST")
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, json.loads(r.read())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read() or b"{}")

    yield app, call
    httpd.shutdown()
    httpd.server_close()


def test_moderation_moves_photos_between_lists_and_delete_removes_the_file(http):
    app, call = http
    st = app.store()
    z = app.zones[2]
    st.upsert_user(UserCtx("u1", "+16175550100", "s", "a"))
    st.set_location("u1", z["points"][0][1], z["points"][0][0], "pin")
    info = photos.save_photo(st, "u1", jpeg(), app.zones, "m1")
    st.set_photo_review(info["id"], "review", "Grok was unsure", "unclear", "", "grok")
    code, q = call("/api/photos/queue?status=review")
    assert code == 200 and q["photos"][0]["id"] == info["id"] and q["photos"][0]["from"].endswith("0100") and "+1617" not in json.dumps(q)
    assert call("/api/photos/moderate", {"id": info["id"], "action": "verify", "bloom": "bloom"})[1]["status"] == "verified"
    stories = call("/api/stories")[1]
    assert stories[0]["zone"] == z["id"] and stories[0]["bloom"] is True and stories[0]["photos"][0]["label"] == "Looks like a bloom"
    assert call("/api/photos/moderate", {"id": info["id"], "action": "reject"})[1]["status"] == "rejected"
    assert call("/api/stories")[1] == [], "rejected photos leave the map"
    path = st.get_photo(info["id"])["path"]
    assert call("/api/photos/moderate", {"id": info["id"], "action": "delete"})[1]["deleted"] == info["id"]
    assert st.get_photo(info["id"]) is None and not __import__("os").path.exists(path)
    assert call("/api/photos/moderate", {"id": info["id"], "action": "verify"})[0] == 404
    assert call("/api/photos/moderate", {"id": 1, "action": "explode"})[0] in (400, 404)
    assert call("/api/photos/queue?status=nonsense")[0] == 400


def test_screening_button_needs_grok(http, monkeypatch):
    _, call = http
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    code, r = call("/api/photos/screen", {})
    assert code == 503 and "XAI_API_KEY" in r["error"]
