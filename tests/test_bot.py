import re
from pathlib import Path

import pytest

from algaebro import config, proximity, zones
from algaebro.bot import Bot
from algaebro.store import UserCtx

CTX = UserCtx(user_id="u1", handle="+16175550100", space_id="spc_1", agent_id="usr_a")


@pytest.fixture()
def recs():
    return zones.load_zone_records()


@pytest.fixture()
def bot(store, replay_fc, recs):
    store.upsert_user(CTX)  # a returning person, so replies are not prefixed by the first-contact welcome
    return Bot(store, lambda: replay_fc, recs)


def say(bot, text, ctx=CTX):
    return "\n".join(bot.handle(ctx, text))


def flagged_point(fc, recs):
    fz = next(z for z in fc["zones"] if z["tier"] == "high")
    lon, lat = next(z for z in recs if z["id"] == fz["id"])["points"][0]
    return fz, lat, lon


def low_fc(fc):
    return dict(fc, zones=[dict(z, tier="low") for z in fc["zones"]])


def test_first_contact_gets_a_short_welcome_then_only_what_they_asked(store, replay_fc, recs):
    fresh = Bot(store, lambda: replay_fc, recs)
    new = UserCtx("new1", "+16175550199", "s", "a")
    first = fresh.handle(new, "status")
    assert len(first) == 2 and first[0].startswith("Hi, I'm Algaebro") and "Send your location" in first[0]
    assert fresh.handle(UserCtx("new2", "+16175550198", "s", "a"), "hi") == [first[0]]
    assert fresh.handle(UserCtx("new3", "+16175550197", "s", "a"), "gibberish") == [first[0]]
    assert "Algaebro: alerts" in "\n".join(fresh.handle(new, "hi")), "a returning person gets help, not the welcome again"


def test_help_is_minimal_and_honest(bot):
    text = say(bot, "help")
    assert len(text.splitlines()) <= 6
    assert all(w in text for w in ("STATUS", "WHY", "STOP", "START", "DELETE PHOTOS"))
    assert "not a safety test" in text and config.OFFICIAL_INFO in text


def test_a_shared_location_is_saved_and_acknowledged(bot, store, replay_fc, recs):
    _, lat, lon = flagged_point(replay_fc, recs)
    out = say(bot, f"{lat}, {lon}")
    assert out.startswith("Got your location") and "stretch I watch" not in out
    u = store.get_user("u1")
    assert u["lat"] == pytest.approx(lat) and u["loc_source"] == "shared"


def test_a_location_far_from_the_river_says_so(bot):
    out = say(bot, "40.7000, -74.0000")
    assert "km from the stretch I watch" in out


def test_status_without_a_location_lists_what_is_flagged(bot, replay_fc):
    out = say(bot, "status")
    names = [z["name"] for z in replay_fc["zones"] if z["tier"] == "high"]
    assert "Flagged now:" in out and names[0] in out and "Send your location" in out and "not a safety check" in out


def test_status_next_to_a_flagged_area_says_so(bot, replay_fc, recs):
    _, lat, lon = flagged_point(replay_fc, recs)
    bot.handle(CTX, f"{lat}, {lon}")
    out = say(bot, "status")
    assert "m from" in out and "high chlorophyll reading (possible algae)" in out


def test_status_when_nothing_is_flagged_never_sounds_like_an_all_clear(store, replay_fc, recs):
    store.upsert_user(CTX)
    quiet = Bot(store, lambda: low_fc(replay_fc), recs)
    out = say(quiet, "status")
    assert "Nothing flagged" in out and "not a safety check" in out


def test_why_explains_the_flag_and_its_track_record(bot, replay_fc, recs):
    _, lat, lon = flagged_point(replay_fc, recs)
    bot.handle(CTX, f"{lat}, {lon}")
    out = say(bot, "why")
    assert "Why:" in out and "chlorophyll index" in out and "In past years" in out
    assert "not a toxin test or a safety check" in out and config.OFFICIAL_INFO in out


def test_why_when_nothing_is_flagged(store, replay_fc, recs):
    store.upsert_user(CTX)
    quiet = Bot(store, lambda: low_fc(replay_fc), recs)
    assert "nothing to explain" in say(quiet, "why")


def test_stop_and_start_toggle_alerts_and_are_safe_to_repeat(bot, store):
    assert "Alerts off" in say(bot, "STOP") and store.get_user("u1")["alerts_off"] == 1
    assert "Alerts off" in say(bot, "stop")
    assert "Alerts on" in say(bot, "start") and store.get_user("u1")["alerts_off"] == 0


def test_delete_photos_removes_rows_and_files(bot, store, tmp_path):
    files = []
    for i in range(2):
        f = tmp_path / f"p{i}.jpg"
        f.write_bytes(b"x")
        files.append(f)
        store.add_photo("u1", "Z01", None, None, str(f), f"att{i}")
    other = tmp_path / "other.jpg"
    other.write_bytes(b"x")
    store.add_photo("someone-else", "Z01", None, None, str(other), "att9")
    assert "Deleted 2 photos" in say(bot, "delete photos")
    assert not any(f.exists() for f in files) and other.exists(), "only this person's photos go"
    assert "no saved photos" in say(bot, "delete photos")


def test_a_location_places_photos_sent_before_it(bot, store, replay_fc, recs, tmp_path):
    f = tmp_path / "early.jpg"
    f.write_bytes(b"x")
    pid = store.add_photo("u1", None, None, None, str(f), "att1")
    fz, lat, lon = flagged_point(replay_fc, recs)
    bot.handle(CTX, f"{lat}, {lon}")
    assert store.get_photo(pid)["zone_id"] == fz["id"]


def test_older_commands_still_work(bot, store):
    assert "Following Zone 3" in say(bot, "subscribe 3") and store.get_subs("u1") == ["Z03"]
    assert len(re.findall(r"^\d+\. ", say(bot, "zones"), re.M)) == 10
    assert "unverified report" in say(bot, "report green scum by the dock")
    assert "couldn't find" in say(bot, "subscribe 99")


def test_unknown_text_points_to_help_and_never_guesses(bot):
    assert "Text HELP" in say(bot, "is the water ok to swim in?")


def test_no_reply_ever_declares_water_safe(bot, replay_fc, recs):
    _, lat, lon = flagged_point(replay_fc, recs)
    bot.handle(CTX, f"{lat}, {lon}")
    risky = re.compile(r"\b(is|are|it's|its) safe\b|safe to (swim|drink|row|go)|all[- ]clear\b", re.I)
    for msg in ["help", "hi", "status", "why", "stop", "start", "zones", "mine", "is it safe?", "report clear water", "delete photos"]:
        for reply in bot.handle(CTX, msg):
            assert not risky.search(reply), f"{msg!r} -> {reply[:90]}"


def test_replies_carry_the_synthetic_and_replay_tags(bot):
    out = say(bot, "status")
    assert "SYNTHETIC DEMO DATA" in out and "REPLAY of 2026-07-20" in out
