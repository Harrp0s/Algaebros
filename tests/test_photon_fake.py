"""Drive the REAL photon_api client against an in-process fake of Photon's HTTP API.

This proves our request building and response parsing match Photon's published models. It does
NOT prove the live service accepts our key, line, or plan: run scripts/00_check_apis.py --photon.
"""
import json
from datetime import datetime, timezone

import httpx
import pytest

from algaebro import alerts, zones
from algaebro.bot import Bot
from algaebro.photon_messenger import PhotonMessenger
from algaebro.run_bot import handle_incoming

HUMAN = "usr_human"
AGENT = "usr_agent"


def _user(uid, handle, kind):
    return {"id": uid, "handle": handle, "kind": kind, "platform": "imessage", "projectId": "proj", "metadata": {}, "details": {}}


def _space(kind="direct"):
    return {"id": "spc_1", "kind": kind, "platform": "imessage", "projectId": "proj", "metadata": {}, "details": {},
            "participantIds": [HUMAN, AGENT]}


def _received(mid, text, created, kind="direct"):
    return {
        "id": mid, "platform": "imessage", "status": "received", "projectId": "proj", "createdAt": created, "updatedAt": created,
        "metadata": {}, "timestamps": {"receivedAt": created},
        "sentBy": _user(HUMAN, "+16175550100", "human"), "receivedBy": _user(AGENT, "+16282679185", "agent"),
        "space": _space(kind), "content": {"type": "text", "format": "plain", "text": text},
    }


def _received_photo(mid, attachment_id, created):
    m = _received(mid, "x", created)
    m["content"] = {"type": "attachment", "attachmentId": attachment_id}
    return m


def _accepted(mid, text, created):
    return {
        "id": mid, "platform": "imessage", "status": "accepted", "projectId": "proj", "createdAt": created, "updatedAt": created,
        "metadata": {}, "timestamps": {"acceptedAt": created, "dispatchedAt": created},
        "acceptance": {"code": "delivery_confirmation_unavailable", "message": "n/a"},
        "sentBy": _user(AGENT, "+16282679185", "agent"), "space": _space(),
        "content": {"type": "text", "format": "plain", "text": text},
    }


class FakePhoton:
    def __init__(self):
        self.inbox = []          # messages the 'server' will return on every list call
        self.posts = []          # what we sent
        self.gets = []
        self.fail_lists_with = None   # keep failing every list call until cleared
        self.fail_posts_with = None   # keep failing every send until cleared
        self.space_unsendable = None  # "with_parent": refuse threaded replies; "always": refuse everything (SPACE_NOT_SENDABLE)
        self.attempts = 0             # every POST received, accepted or not
        self.attachments = {}         # attachment id -> bytes
        self.not_ready_left = 0       # answer ATTACHMENT_NOT_READY this many times first

    def handler(self, request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer testkey"
        if request.url.path.startswith("/v1/projects/proj/attachments/"):
            aid = request.url.path.split("/")[-2]
            if self.not_ready_left > 0:
                self.not_ready_left -= 1
                return httpx.Response(409, json={"code": "ATTACHMENT_NOT_READY", "title": "Attachment Not Ready", "status": 409})
            if aid not in self.attachments:
                return httpx.Response(404, json={"code": "ATTACHMENT_NOT_FOUND", "title": "Not Found", "status": 404})
            return httpx.Response(200, content=self.attachments[aid], headers={"content-type": "image/jpeg"})
        assert request.url.path == "/v1/projects/proj/messages", request.url.path
        now = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        if request.method == "GET":
            self.gets.append(dict(request.url.params))
            if self.fail_lists_with:
                code = self.fail_lists_with
                return httpx.Response(code, json={"type": "about:blank", "title": "slow down", "status": code})
            return httpx.Response(200, json={"items": list(self.inbox)})
        body = json.loads(request.content)
        self.attempts += 1
        if self.space_unsendable == "always" or (self.space_unsendable == "with_parent" and "parentMessageId" in body):
            return httpx.Response(409, json={"code": "SPACE_NOT_SENDABLE", "detail": "Replies are not supported for this Space",
                                             "status": 409, "title": "Space Not Sendable",
                                             "type": "https://photon.codes/docs/problems/space-not-sendable"})
        if self.fail_posts_with:
            code = self.fail_posts_with
            return httpx.Response(code, json={"type": "https://example.test/problems/conflict", "title": "Conflict", "status": code,
                                              "detail": "fake reason: request already in progress"})
        self.posts.append({"body": body, "idem": request.headers.get("idempotency-key")})
        return httpx.Response(202, json=_accepted(f"out_{len(self.posts)}", body["content"]["text"], now))


@pytest.fixture()
def fake():
    return FakePhoton()


@pytest.fixture()
def messenger(fake, store):
    client = httpx.Client(transport=httpx.MockTransport(fake.handler))
    return PhotonMessenger("testkey", "proj", store, base_url="https://photon.test", client=client)


@pytest.fixture()
def bot(store, replay_fc):
    from algaebro.store import UserCtx

    store.upsert_user(UserCtx(HUMAN, "+16175550100", "spc_1", AGENT))   # a returning person: no welcome message
    return Bot(store, lambda: replay_fc, zones.load_zone_records())


def test_poll_parses_a_real_photon_message(fake, messenger):
    fake.inbox = [_received("msg_1", "subscribe 3", "2026-07-20T15:00:00.000Z")]
    got = messenger.poll()
    assert len(got) == 1
    m = got[0]
    assert (m.message_id, m.user_id, m.handle, m.text, m.space_id, m.agent_id) == ("msg_1", HUMAN, "+16175550100", "subscribe 3", "spc_1", AGENT)
    q = fake.gets[0]
    assert q["direction"] == "received" and q["platform"] == "imessage" and "createdSince" in q


def test_group_chats_are_ignored(fake, messenger):
    fake.inbox = [_received("msg_g", "hi", "2026-07-20T15:00:00.000Z", kind="group")]
    assert messenger.poll() == []


def test_reply_uses_the_incoming_space_agent_and_parent(fake, messenger, store, bot):
    fake.inbox = [_received("msg_1", "subscribe 3", "2026-07-20T15:00:00.000Z")]
    for inc in messenger.poll():
        handle_incoming(store, bot, messenger, inc)
    assert len(fake.posts) == 1
    sent = fake.posts[0]
    assert sent["body"]["spaceId"] == "spc_1" and sent["body"]["senderId"] == AGENT
    assert sent["body"]["parentMessageId"] == "msg_1"
    assert sent["body"]["content"] == {"type": "text", "format": "plain", "text": sent["body"]["content"]["text"]}
    assert "Following Zone 3" in sent["body"]["content"]["text"]
    assert sent["idem"] == "reply:msg_1:0"
    assert store.get_subs(HUMAN) == ["Z03"]
    u = store.get_user(HUMAN)
    assert u["space_id"] == "spc_1" and u["agent_id"] == AGENT, "needed later to send proactive alerts"


def test_overlapping_polls_do_not_answer_twice(fake, messenger, store, bot):
    fake.inbox = [_received("msg_1", "help", "2026-07-20T15:00:00.000Z")]
    for _ in range(3):  # the fake keeps returning the same message, like the 5-second overlap window does
        for inc in messenger.poll():
            handle_incoming(store, bot, messenger, inc)
    assert len(fake.posts) == 1


def test_rate_limit_backs_off_instead_of_crashing(fake, messenger):
    from photon_api import Photon

    # no client-side retries, so the 429 reaches our own back-off code straight away
    messenger.photon = Photon(headers={"Authorization": "Bearer testkey"}, base_url="https://photon.test", max_attempts=1,
                              client=httpx.Client(transport=httpx.MockTransport(fake.handler)))
    fake.fail_lists_with = 429
    assert messenger.poll() == []
    assert messenger.backoff_until > 0, "a 429 must start a back-off"
    fake.fail_lists_with = None
    fake.inbox = [_received("msg_1", "help", "2026-07-20T15:00:00.000Z")]
    assert messenger.poll() == [], "still backing off, so no request is made yet"


def test_other_api_errors_propagate_so_the_runner_can_log_them(fake, messenger):
    from photon_api import ApiError, Photon

    messenger.photon = Photon(headers={"Authorization": "Bearer testkey"}, base_url="https://photon.test", max_attempts=1,
                              client=httpx.Client(transport=httpx.MockTransport(fake.handler)))
    fake.fail_lists_with = 403
    with pytest.raises(ApiError):
        messenger.poll()


def test_proactive_alert_has_no_parent_and_reuses_saved_ids(fake, messenger, store, bot, replay_fc):
    fake.inbox = [_received("msg_1", "subscribe all", "2026-07-20T15:00:00.000Z")]
    for inc in messenger.poll():
        handle_incoming(store, bot, messenger, inc)
    fake.posts.clear()
    res = alerts.run_alerts(store, messenger, replay_fc, now=datetime(2026, 7, 20, 16, 0, tzinfo=timezone.utc))
    assert res["sent"], "subscribe-all must produce at least one alert for a HIGH zone"
    for p in fake.posts:
        b = p["body"]
        assert "parentMessageId" not in b and b["spaceId"] == "spc_1" and b["senderId"] == AGENT
        assert p["idem"].startswith("alert:") and p["idem"].endswith(":replay")
        assert "ALERT: High chlorophyll reading (possible algae) near" in b["content"]["text"]


def test_first_contact_gets_a_welcome_before_the_answer(fake, messenger, store, replay_fc):
    from algaebro.store import UserCtx  # noqa: F401  (a brand-new person: nothing registered)

    fresh = Bot(store, lambda: replay_fc, zones.load_zone_records())
    fake.inbox = [_received("msg_new", "status", "2026-07-20T15:00:00.000Z")]
    inc = messenger.poll()[0]
    assert handle_incoming(store, fresh, messenger, inc) == 2
    texts = [p["body"]["content"]["text"] for p in fake.posts]
    assert texts[0].startswith("Hi, I'm Algaebro") and "Satellite-based" in texts[1]


def test_a_reaction_or_unknown_content_gets_no_reply(fake, messenger, store, bot):
    from algaebro.messenger import Incoming

    inc = Incoming("msg_x", HUMAN, "+1617", None, "spc_1", AGENT)   # no text, no attachment
    assert handle_incoming(store, bot, messenger, inc) == 0 and fake.posts == [] and store.already_seen("msg_x")


@pytest.mark.parametrize("raw", [
    "2026-07-20T15:00:00Z", "2026-07-20T15:00:00.5Z", "2026-07-20T15:00:00.123Z",
    "2026-07-20T15:00:00.123456789Z", "2026-07-20T11:00:00-04:00", "2026-07-20T15:00:00",
])
def test_timestamp_parser_accepts_any_iso_format(raw):
    from algaebro.photon_messenger import parse_ts

    got = parse_ts(raw)
    assert got.tzinfo is not None and got.replace(microsecond=0) == datetime(2026, 7, 20, 15, 0, tzinfo=timezone.utc)


def test_subscribing_triggers_an_immediate_alert_for_that_person_only(fake, messenger, store, bot, replay_fc):
    from algaebro.run_bot import handle_incoming as hi

    fired = []
    fake.inbox = [_received("msg_s", "subscribe all", "2026-07-20T15:00:00.000Z")]
    for inc in messenger.poll():
        hi(store, bot, messenger, inc, on_new_subscription=fired.append)
    assert fired == [HUMAN]
    fired.clear()
    for inc in messenger.poll():          # the same message again: nothing new, so no new trigger
        hi(store, bot, messenger, inc, on_new_subscription=fired.append)
    assert fired == []


def test_failed_reply_is_retried_then_abandoned_and_shows_photons_reason(fake, messenger, store, bot, capsys):
    fake.fail_posts_with = 409
    fake.inbox = [_received("msg_f", "report 4 green scum by the dock", "2026-07-20T15:00:00.000Z")]
    inc = messenger.poll()[0]
    assert handle_incoming(store, bot, messenger, inc) == 0 and not store.already_seen("msg_f")
    out = capsys.readouterr().out
    assert "HTTP 409" in out and "fake reason: request already in progress" in out and "testkey" not in out
    assert handle_incoming(store, bot, messenger, inc) == 0 and not store.already_seen("msg_f")
    assert handle_incoming(store, bot, messenger, inc) == 0 and store.already_seen("msg_f"), "gives up after 3 tries"
    assert "giving up" in capsys.readouterr().out
    assert store.db.execute("SELECT COUNT(*) AS n FROM reports").fetchone()["n"] == 1, "a retry must not apply 'report' again"


def test_a_reply_that_fails_once_then_works_is_sent_exactly_once(fake, messenger, store, bot):
    fake.inbox = [_received("msg_r", "subscribe 3", "2026-07-20T15:00:00.000Z")]
    inc = messenger.poll()[0]
    fake.fail_posts_with = 409
    assert handle_incoming(store, bot, messenger, inc) == 0 and fake.posts == []
    fake.fail_posts_with = None
    assert handle_incoming(store, bot, messenger, inc) == 1
    assert len(fake.posts) == 1 and store.already_seen("msg_r")
    assert handle_incoming(store, bot, messenger, inc) == 0 and len(fake.posts) == 1


def test_describe_error_includes_photons_text_and_never_the_key(fake, messenger):
    from photon_api import ApiError, Photon
    from algaebro.messenger import describe_error

    messenger.photon = Photon(headers={"Authorization": "Bearer testkey"}, base_url="https://photon.test", max_attempts=1,
                              client=httpx.Client(transport=httpx.MockTransport(fake.handler)))
    fake.fail_posts_with = 409
    with pytest.raises(ApiError) as caught:
        messenger.send("spc_1", AGENT, "hi", key="k1")
    text = describe_error(caught.value)
    assert "HTTP 409" in text and "request already in progress" in text and "testkey" not in text
    assert "ValueError: boom" in describe_error(ValueError("boom"))


def test_poll_window_overlaps_two_minutes_so_failed_messages_can_be_retried(fake, messenger, store):
    fake.inbox = [_received("msg_w", "help", "2026-07-20T15:00:00.000Z")]
    messenger.poll()
    since = store.kv_get("photon_since")
    assert since.startswith("2026-07-20T14:58:00"), since


def test_immediate_alert_waits_for_the_reply_to_actually_go_out(fake, messenger, store, bot):
    """The 'alert me now' hook must survive the retry logic, and must not fire if the reply failed."""
    fired = []
    fake.inbox = [_received("msg_n", "subscribe all", "2026-07-20T15:00:00.000Z")]
    inc = messenger.poll()[0]
    fake.fail_posts_with = 409
    handle_incoming(store, bot, messenger, inc, on_new_subscription=fired.append)
    assert fired == [], "no alert if the confirmation reply failed"
    fake.fail_posts_with = None
    handle_incoming(store, bot, messenger, inc, on_new_subscription=fired.append)
    assert fired == [HUMAN]
    handle_incoming(store, bot, messenger, inc, on_new_subscription=fired.append)
    assert fired == [HUMAN], "and only once"


def test_threaded_reply_refused_by_photon_falls_back_to_a_normal_message_and_remembers(fake, messenger):
    fake.space_unsendable = "with_parent"
    messenger.send("spc_1", AGENT, "first", parent_id="m1", key="k1")
    assert len(fake.posts) == 1 and "parentMessageId" not in fake.posts[0]["body"] and fake.posts[0]["idem"] == "k1:plain"
    assert messenger.threaded_replies is False
    before = fake.attempts
    messenger.send("spc_1", AGENT, "second", parent_id="m2", key="k2")
    assert fake.attempts == before + 1, "after learning, it must not try a threaded reply again"
    assert "parentMessageId" not in fake.posts[1]["body"]


def test_a_space_that_refuses_everything_still_raises_after_one_fallback(fake, messenger):
    from photon_api import ApiError

    fake.space_unsendable = "always"
    with pytest.raises(ApiError) as caught:
        messenger.send("spc_1", AGENT, "hi", parent_id="m1", key="k1")
    assert fake.attempts == 2, "one threaded try and one plain try, no loop"
    assert "SPACE_NOT_SENDABLE" in caught.value.raw_body


def test_unrelated_409_is_not_mistaken_for_the_threading_problem(fake, messenger):
    from photon_api import ApiError

    fake.fail_posts_with = 409
    with pytest.raises(ApiError):
        messenger.send("spc_1", AGENT, "hi", parent_id="m1", key="k1")
    assert fake.attempts == 1 and messenger.threaded_replies is True
