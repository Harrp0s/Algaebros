import importlib.util
from pathlib import Path

import httpx

from algaebro.photon_messenger import PhotonMessenger
from test_photon_fake import AGENT, HUMAN, FakePhoton, _received

spec = importlib.util.spec_from_file_location("photon_ping", Path(__file__).resolve().parent.parent / "scripts" / "photon_ping.py")
photon_ping = importlib.util.module_from_spec(spec)
spec.loader.exec_module(photon_ping)


def _messenger(fake, store):
    return PhotonMessenger("testkey", "proj", store, base_url="https://photon.test",
                           client=httpx.Client(transport=httpx.MockTransport(fake.handler)))


def test_ping_replies_pong_and_remembers_how_to_reach_the_sender(store):
    fake = FakePhoton()
    fake.inbox = [_received("ping_1", "ping", "2026-07-20T15:00:00.000Z")]
    rc = photon_ping.run_ping(_messenger(fake, store), store, window_s=5, sleep=lambda s: None)
    assert rc == 0
    assert len(fake.posts) == 1 and "pong" in fake.posts[0]["body"]["content"]["text"]
    assert fake.posts[0]["body"]["parentMessageId"] == "ping_1"
    u = store.get_user(HUMAN)
    assert u["space_id"] == "spc_1" and u["agent_id"] == AGENT
    assert store.already_seen("ping_1"), "the real bot must not answer this message a second time"


def test_ping_times_out_with_help_when_nothing_arrives(store, capsys):
    fake = FakePhoton()
    rc = photon_ping.run_ping(_messenger(fake, store), store, window_s=0.2, sleep=lambda s: None)
    assert rc == 1 and fake.posts == []
    assert "Send & Receive" in capsys.readouterr().out


def test_failed_ping_reply_is_not_marked_seen_so_it_can_be_retried(store):
    import pytest
    from photon_api import ApiError, Photon

    fake = FakePhoton()
    fake.fail_posts_with = 409
    fake.inbox = [_received("ping_2", "ping", "2026-07-20T15:00:00.000Z")]
    m = _messenger(fake, store)
    m.photon = Photon(headers={"Authorization": "Bearer testkey"}, base_url="https://photon.test", max_attempts=1,
                      client=httpx.Client(transport=httpx.MockTransport(fake.handler)))
    with pytest.raises(ApiError):
        photon_ping.run_ping(m, store, window_s=5, sleep=lambda s: None)
    assert not store.already_seen("ping_2")
    assert store.get_user(HUMAN) is not None, "still remember how to reach them"
    fake.fail_posts_with = None
    assert photon_ping.run_ping(m, store, window_s=5, sleep=lambda s: None) == 0
    assert store.already_seen("ping_2") and fake.posts[0]["idem"].startswith("ping:ping_2:")
