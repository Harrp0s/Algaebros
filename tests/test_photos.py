import io

import pytest
from PIL import Image

from algaebro import config, photos, zones
from algaebro.messenger import Incoming
from algaebro.run_bot import handle_incoming
from algaebro.store import UserCtx
from test_photon_fake import AGENT, HUMAN, FakePhoton, _received_photo  # noqa: F401
from test_photon_fake import fake, messenger  # noqa: F401  (fixtures)
from algaebro.bot import Bot


def make_jpeg(size=(300, 200), orientation=None, gps=False):
    img = Image.new("RGB", size, (20, 120, 50))
    ex = Image.Exif()
    if orientation:
        ex[0x0112] = orientation
    if gps:
        ex[0x8825] = {1: "N", 2: (42.0, 21.0, 0.0)}
    buf = io.BytesIO()
    img.save(buf, "JPEG", exif=ex)
    return buf.getvalue()


@pytest.fixture()
def recs():
    return zones.load_zone_records()


@pytest.fixture(autouse=True)
def photo_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PHOTO_DIR", tmp_path / "photos")


def test_location_metadata_is_removed_and_rotation_is_applied():
    out = Image.open(io.BytesIO(photos.normalise_image(make_jpeg(orientation=6, gps=True))))
    assert out.size == (200, 300), "the phone's rotation must be honoured"
    assert len(out.getexif()) == 0, "no EXIF (including GPS) may survive"


def test_iphone_heic_becomes_a_normal_jpeg():
    if not photos.HEIC_OK:
        pytest.skip("pillow-heif not installed")
    buf = io.BytesIO()
    Image.new("RGB", (64, 48), (30, 140, 60)).save(buf, format="HEIF")
    out = Image.open(io.BytesIO(photos.normalise_image(buf.getvalue())))
    assert out.format == "JPEG" and out.size == (64, 48)


def test_big_photos_are_shrunk_and_non_images_are_rejected(store):
    big = Image.open(io.BytesIO(photos.normalise_image(make_jpeg(size=(4000, 3000)))))
    assert max(big.size) <= photos.MAX_SIDE
    with pytest.raises(ValueError):
        photos.save_photo(store, "u", b"definitely not an image", [])


def test_photo_is_filed_under_the_nearest_area_to_the_person(store, recs):
    lon, lat = recs[4]["points"][0]
    store.upsert_user(UserCtx("u", "+1617", "s", "a"))
    store.set_location("u", lat, lon, "pin")
    info = photos.save_photo(store, "u", make_jpeg(), recs, "att1")
    assert info["placed"] and info["zone_id"] == recs[4]["id"]
    row = store.get_photo(info["id"])
    assert row["path"].startswith(str(config.PHOTO_DIR)) and open(row["path"], "rb").read()[:2] == b"\xff\xd8"


def test_a_photo_from_someone_with_no_location_waits_to_be_placed(store, recs):
    store.upsert_user(UserCtx("u", "+1617", "s", "a"))
    info = photos.save_photo(store, "u", make_jpeg(), recs)
    assert not info["placed"] and store.unplaced_photos("u")


def test_end_to_end_through_photon_a_photo_is_downloaded_stored_and_acknowledged(fake, messenger, store, recs, replay_fc):
    lon, lat = recs[2]["points"][0]
    store.upsert_user(UserCtx(HUMAN, "+16175550100", "spc_1", AGENT))
    store.set_location(HUMAN, lat, lon, "shared")
    fake.attachments["att_7"] = make_jpeg(gps=True)
    fake.not_ready_left = 2               # Photon says 'not ready' twice first
    messenger.attachment_wait_s = 0
    fake.inbox = [_received_photo("msg_ph", "att_7", "2026-07-20T15:00:00.000Z")]
    inc = messenger.poll()[0]
    assert inc.is_attachment and inc.attachment_id == "att_7" and not inc.is_text
    bot = Bot(store, lambda: replay_fc, recs)
    ok = lambda jpeg, place: {"status": "verified", "reason": "photo of the water", "bloom": "bloom", "description": "green scum"}  # noqa: E731
    n = handle_incoming(store, bot, messenger, inc, photo_handler=lambda i: photos.handle_photo_message(store, messenger, recs, i, screener=ok))
    assert n == 1
    text = fake.posts[0]["body"]["content"]["text"]
    assert "added to the map near" in text and "may show algae" in text and "DELETE PHOTOS" in text
    assert len(store.photos_for_zone(recs[2]["id"])) == 1 and fake.not_ready_left == 0


def test_a_photo_that_cannot_be_opened_gets_a_polite_failure_not_a_crash(fake, messenger, store, recs, replay_fc):
    store.upsert_user(UserCtx(HUMAN, "+1617", "spc_1", AGENT))
    fake.attachments["bad"] = b"%PDF-1.4 not a photo"
    messenger.attachment_wait_s = 0
    fake.inbox = [_received_photo("msg_bad", "bad", "2026-07-20T15:00:00.000Z")]
    inc = messenger.poll()[0]
    handle_incoming(store, Bot(store, lambda: replay_fc, recs), messenger, inc,
                    photo_handler=lambda i: photos.handle_photo_message(store, messenger, recs, i))
    assert "couldn't open that photo" in fake.posts[0]["body"]["content"]["text"] and store.photos_for_zone("Z01") == []


def test_the_same_photo_is_not_stored_twice(fake, messenger, store, recs):
    store.upsert_user(UserCtx(HUMAN, "+1617", "spc_1", AGENT))
    fake.attachments["a1"] = make_jpeg()
    inc = Incoming("m", HUMAN, "+1617", None, "spc_1", AGENT, attachment_id="a1")
    photos.handle_photo_message(store, messenger, recs, inc)
    photos.handle_photo_message(store, messenger, recs, inc)
    assert len(store.db.execute("SELECT * FROM photos").fetchall()) == 1
