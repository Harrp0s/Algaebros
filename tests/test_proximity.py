from datetime import datetime, timedelta, timezone

import pytest

from algaebro import config, proximity, zones
from algaebro.messenger import MemoryMessenger
from algaebro.store import UserCtx

NOON = datetime(2026, 7, 20, 16, 0, tzinfo=timezone.utc)     # 12:00 in Boston
NIGHT = datetime(2026, 7, 21, 3, 0, tzinfo=timezone.utc)     # 23:00 in Boston


@pytest.fixture()
def recs():
    return zones.load_zone_records()


def live(fc):
    return dict(fc, replay=False, live=True)


def flagged(fc):
    return next(z for z in fc["zones"] if z["tier"] == "high")


def place(store, recs, fc, uid="a", offset_deg=0.0, reachable=True, off=False):
    fz = flagged(fc)
    lon, lat = next(z for z in recs if z["id"] == fz["id"])["points"][0]
    store.upsert_user(UserCtx(uid, "+1617555" + uid.zfill(4)[-4:], "spc_" + uid if reachable else "", "agent" if reachable else ""))
    store.set_location(uid, lat + offset_deg, lon, "pin")
    if off:
        store.set_alerts_off(uid, True)
    return fz, store.get_user(uid)


def test_flagged_near_respects_the_radius(replay_fc, recs):
    fz = flagged(replay_fc)
    lon, lat = next(z for z in recs if z["id"] == fz["id"])["points"][0]
    assert proximity.flagged_near(replay_fc, recs, lat, lon)[0][1]["id"] == fz["id"]
    assert proximity.flagged_near(replay_fc, recs, lat + 0.5, lon) == []
    assert proximity.flagged_near(replay_fc, recs, lat + 0.5, lon, radius_m=1e9)


def test_alert_names_the_place_and_how_far(store, replay_fc, recs):
    fz, u = place(store, recs, replay_fc, offset_deg=0.0016)
    mem = MemoryMessenger()
    res = proximity.check_user(store, mem, replay_fc, recs, u, now=NOON, memo={})
    assert len(res["sent"]) == 1 and mem.sent[0]["parent_id"] is None
    text = mem.sent[0]["text"]
    assert text.count("\n") >= 3 and f"near {fz['name']}" in text and "m from you" in text
    assert "ALERT: High chlorophyll reading (possible algae)" in text and "not a safety test" in text
    assert "no flash, sun behind you" in text and "WHY for details, STOP to quit" in text
    feed = store.recent_feed(1)[0]
    from algaebro.store import mask_handle
    assert feed["delivered"] == 1 and feed["user_label"] == mask_handle(u["handle"]) and "+1617" not in feed["user_label"]


def test_standing_right_on_it_says_right_here(store, replay_fc, recs):
    _, u = place(store, recs, replay_fc)
    mem = MemoryMessenger()
    proximity.check_user(store, mem, replay_fc, recs, u, now=NOON, memo={})
    assert "(right here)" in mem.sent[0]["text"]


def test_simulated_alerts_repeat_only_after_the_short_cooldown(store, replay_fc, recs):
    _, u = place(store, recs, replay_fc)
    mem, memo = MemoryMessenger(), {}
    proximity.check_user(store, mem, replay_fc, recs, u, now=NOON, memo=memo)
    r2 = proximity.check_user(store, mem, replay_fc, recs, u, now=NOON + timedelta(seconds=10), memo=memo)
    assert len(mem.sent) == 1 and "a moment ago" in r2["skipped"][0]["reason"]
    proximity.check_user(store, mem, replay_fc, recs, u, now=NOON + timedelta(seconds=config.DEMO_COOLDOWN_S + 5), memo=memo)
    assert len(mem.sent) == 2


def test_live_alerts_use_the_real_48_hour_cooldown(store, replay_fc, recs):
    fc = live(replay_fc)
    fz, u = place(store, recs, fc)
    mem = MemoryMessenger()
    proximity.check_user(store, mem, fc, recs, u, now=NOON)
    proximity.check_user(store, mem, fc, recs, u, now=NOON + timedelta(hours=5))
    assert len(mem.sent) == 1 and store.last_alert("a", fz["id"]) is not None
    proximity.check_user(store, mem, fc, recs, u, now=NOON + timedelta(hours=49))
    assert len(mem.sent) == 2


def test_a_demo_pin_never_uses_up_the_real_cooldown_and_is_tagged(store, replay_fc, recs):
    fc = live(replay_fc)
    fz, u = place(store, recs, fc)
    mem = MemoryMessenger()
    proximity.check_user(store, mem, fc, recs, u, now=NOON, memo={}, demo=True)
    assert "[DEMO: simulated location]" in mem.sent[0]["text"]
    assert store.last_alert("a", fz["id"]) is None


def test_nobody_is_alerted_when_they_should_not_be(store, replay_fc, recs):
    mem = MemoryMessenger()
    _, off = place(store, recs, replay_fc, uid="1", off=True)
    assert "off" in proximity.check_user(store, mem, replay_fc, recs, off, now=NOON, memo={})["skipped"][0]["reason"]
    store.upsert_user(UserCtx("2", "+16175550002", "s", "a"))
    assert "no location" in proximity.check_user(store, mem, replay_fc, recs, store.get_user("2"), now=NOON, memo={})["skipped"][0]["reason"]
    _, far = place(store, recs, replay_fc, uid="3", offset_deg=0.5)
    assert "no flagged area" in proximity.check_user(store, mem, replay_fc, recs, far, now=NOON, memo={})["skipped"][0]["reason"]
    _, ok = place(store, recs, replay_fc, uid="4")
    assert "outside bloom season" in proximity.check_user(store, mem, dict(replay_fc, in_season=False), recs, ok, now=NOON, memo={})["skipped"][0]["reason"]
    assert mem.sent == []


def test_live_quiet_hours_hold_alerts_back_but_replay_does_not(store, replay_fc, recs):
    mem = MemoryMessenger()
    _, u = place(store, recs, live(replay_fc))
    assert "quiet" in proximity.check_user(store, mem, live(replay_fc), recs, u, now=NIGHT)["skipped"][0]["reason"]
    assert proximity.check_user(store, mem, replay_fc, recs, u, now=NIGHT, memo={})["sent"]


def test_someone_we_cannot_reach_is_logged_not_crashed(store, replay_fc, recs):
    _, u = place(store, recs, replay_fc, reachable=False)
    res = proximity.check_user(store, MemoryMessenger(), replay_fc, recs, u, now=NOON, memo={})
    assert res["sent"] == [] and "no way to reach" in res["skipped"][0]["reason"]
    assert store.recent_feed(1)[0]["delivered"] == 0


def test_a_failed_send_is_reported_and_can_be_retried(store, replay_fc, recs):
    class Boom:
        def send(self, *a, **k):
            raise RuntimeError("network down")

    _, u = place(store, recs, replay_fc)
    memo = {}
    res = proximity.check_user(store, Boom(), replay_fc, recs, u, now=NOON, memo=memo)
    assert res["sent"] == [] and "send failed" in res["skipped"][0]["reason"] and memo == {}


def test_with_no_photon_connected_the_demo_still_shows_the_alert_in_the_feed(store, replay_fc, recs):
    _, u = place(store, recs, replay_fc)
    res = proximity.check_user(store, None, replay_fc, recs, u, now=NOON, memo={}, demo=True)
    assert len(res["sent"]) == 1 and res["sent"][0]["delivered"] is False
    f = store.recent_feed(1)[0]
    assert f["delivered"] == 0 and "Photon not connected" in f["note"]


def test_dry_run_sends_and_records_nothing(store, replay_fc, recs):
    _, u = place(store, recs, replay_fc)
    mem = MemoryMessenger()
    res = proximity.check_user(store, mem, replay_fc, recs, u, now=NOON, memo={}, dry_run=True)
    assert len(res["sent"]) == 1 and mem.sent == [] and store.recent_feed(1) == []
