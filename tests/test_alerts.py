from datetime import datetime, timedelta, timezone

import pytest

from algaebro import alerts
from algaebro.messenger import MemoryMessenger
from algaebro.store import UserCtx

NOON = datetime(2026, 7, 20, 16, 0, tzinfo=timezone.utc)      # 12:00 in Boston
NIGHT = datetime(2026, 7, 21, 3, 0, tzinfo=timezone.utc)      # 23:00 in Boston


def _live(fc):
    f = dict(fc)
    f["replay"], f["live"] = False, True
    return f


def _high_zone(fc):
    z = next(z for z in fc["zones"] if z["tier"] == "high")
    return z


@pytest.fixture()
def setup(store, replay_fc):
    hz = _high_zone(replay_fc)
    store.upsert_user(UserCtx("a", "+1617", "spc_a", "agent_1"))
    store.upsert_user(UserCtx("b", "+1618", "spc_b", "agent_1"))
    store.add_sub("a", hz["id"])
    store.add_sub("b", "Z99")  # follows a zone that is not high
    return store, hz


def test_only_followers_of_a_high_zone_are_alerted(setup, replay_fc):
    store, hz = setup
    mem = MemoryMessenger()
    res = alerts.run_alerts(store, mem, replay_fc, now=NOON)
    assert [m["space_id"] for m in mem.sent] == ["spc_a"]
    assert len(res["sent"]) == 1
    text = mem.sent[0]["text"]
    assert hz["name"] in text and "not a safety test" in text and "STOP" in text and "ALERT: High chlorophyll reading (possible algae)" in text
    assert "SYNTHETIC DEMO DATA" in text and "REPLAY" in text
    assert mem.sent[0]["parent_id"] is None, "a proactive alert is not a reply"


def test_quiet_hours_hold_alerts_back(setup, replay_fc):
    store, _ = setup
    mem = MemoryMessenger()
    res = alerts.run_alerts(store, mem, replay_fc, now=NIGHT)
    assert mem.sent == [] and "quiet hours" in res["skipped"][0]["reason"]
    res = alerts.run_alerts(store, mem, replay_fc, now=NIGHT, ignore_quiet=True)
    assert len(mem.sent) == 1


def test_off_season_sends_nothing(setup, replay_fc):
    store, _ = setup
    mem = MemoryMessenger()
    fc = dict(replay_fc, in_season=False)
    alerts.run_alerts(store, mem, fc, now=NOON)
    assert mem.sent == []


def test_cooldown_prevents_repeat_alerts_for_live_forecasts(setup, replay_fc):
    store, _ = setup
    fc = _live(replay_fc)
    mem = MemoryMessenger()
    alerts.run_alerts(store, mem, fc, now=NOON)
    alerts.run_alerts(store, mem, fc, now=NOON + timedelta(hours=3))
    assert len(mem.sent) == 1, "second check inside the cooldown must not re-send"
    alerts.run_alerts(store, mem, fc, now=NOON + timedelta(hours=49))
    assert len(mem.sent) == 2, "after the cooldown it may alert again"


def test_replay_alerts_do_not_poison_the_real_cooldown(setup, replay_fc):
    store, hz = setup
    alerts.run_alerts(store, MemoryMessenger(), replay_fc, now=NOON)
    assert store.last_alert("a", hz["id"]) is None


def test_dry_run_sends_and_logs_nothing(setup, replay_fc):
    store, hz = setup
    mem = MemoryMessenger()
    res = alerts.run_alerts(store, mem, _live(replay_fc), now=NOON, dry_run=True)
    assert mem.sent == [] and len(res["sent"]) == 1 and store.last_alert("a", hz["id"]) is None


def test_user_we_cannot_reach_is_skipped_not_crashed(store, replay_fc):
    hz = _high_zone(replay_fc)
    store.upsert_user(UserCtx("c", "+1619", "", ""))
    store.add_sub("c", hz["id"])
    mem = MemoryMessenger()
    res = alerts.run_alerts(store, mem, replay_fc, now=NOON)
    assert mem.sent == [] and "no way to reach" in res["skipped"][0]["reason"]


def test_failed_send_is_not_logged_so_it_retries_next_time(setup, replay_fc):
    store, hz = setup

    class Boom:
        def send(self, *a, **k):
            raise RuntimeError("network down")

    res = alerts.run_alerts(store, Boom(), _live(replay_fc), now=NOON)
    assert res["sent"] == [] and "send failed" in res["skipped"][0]["reason"]
    assert store.last_alert("a", hz["id"]) is None


def test_only_user_limits_an_alert_to_that_person(store, replay_fc):
    hz = _high_zone(replay_fc)
    for uid, space in (("a", "spc_a"), ("b", "spc_b")):
        store.upsert_user(UserCtx(uid, "+1", space, "agent_1"))
        store.add_sub(uid, hz["id"])
    mem = MemoryMessenger()
    alerts.run_alerts(store, mem, replay_fc, now=NOON, only_user="b")
    assert [m["space_id"] for m in mem.sent] == ["spc_b"]


def test_replay_memo_stops_the_same_demo_alert_repeating(setup, replay_fc):
    store, _ = setup
    mem, memo = MemoryMessenger(), set()
    alerts.run_alerts(store, mem, replay_fc, now=NOON, sent_memo=memo)
    alerts.run_alerts(store, mem, replay_fc, now=NOON + timedelta(minutes=30), sent_memo=memo)
    assert len(mem.sent) == 1, "a replay alert must not repeat every check"
    # without a memo (old behaviour) it would repeat; and a dry run must not fill the memo
    memo2 = set()
    alerts.run_alerts(store, MemoryMessenger(), replay_fc, now=NOON, dry_run=True, sent_memo=memo2)
    assert memo2 == set()
