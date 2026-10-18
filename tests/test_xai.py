import json
from unittest import mock

import pytest
import requests

from algaebro import config, xai


class Resp:
    def __init__(self, code=200, js=None, text=None, content=b"", ctype="application/json"):
        self.status_code, self._js, self.content, self.headers = code, js, content, {"Content-Type": ctype}
        self.text = text if text is not None else json.dumps(js if js is not None else {})

    def json(self):
        return self._js


def chat_ok(text):
    return Resp(200, {"choices": [{"message": {"content": text}}]})


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "xai-SECRET-KEY-123")
    monkeypatch.delenv("XAI_MODEL", raising=False)
    monkeypatch.setattr(xai, "_resolved", None)


@pytest.fixture()
def state(replay_fc):
    return {"mode": "replay", "as_of": "2026-07-20", "data_source": "synthetic", "method": "persistence", "in_season": True,
            "thresholds": replay_fc["thresholds"], "official_info": config.OFFICIAL_INFO,
            "reliability": {"test_years": [2024, 2025], "precision": 0.42, "recall": 0.46, "n_high": 38, "n_high_dates": 9},
            "zones": [{"id": z["id"], "name": z["name"], "tier": z["tier"], "flagged": z["tier"] == "high", "last_obs": z.get("last_obs"),
                       "obs_age_days": z.get("obs_age_days"), "track": z.get("track")} for z in replay_fc["zones"]]}


def test_grounded_answer_comes_from_grok_with_the_dashboard_data_and_nothing_personal(state):
    sent = {}

    def post(url, **kw):
        sent.update(kw)
        return chat_ok("Two areas are flagged.")

    with mock.patch.object(xai.requests, "post", post):
        r = xai.answer("how many areas are flagged?", state, [{"role": "user", "content": "hi"}, {"role": "evil", "content": "x"}])
    assert r["source"] == "grok" and r["reply"] == "Two areas are flagged."
    prompt = json.dumps(sent["json"]["messages"])
    assert "DASHBOARD DATA" in prompt and "SYNTHETIC" in prompt and "xai-SECRET" not in prompt and "+1" not in prompt
    assert [m["role"] for m in sent["json"]["messages"]] == ["system", "system", "user", "user"], "unknown history roles are dropped"
    assert sent["headers"]["Authorization"] == "Bearer xai-SECRET-KEY-123", "the key goes only in the header"


@pytest.mark.parametrize("q", ["is it safe to swim?", "can my dog go in the water", "Is the river clean?", "is it ok to kayak today",
                               "will my kid get sick", "is it toxic", "safe for rowing?"])
def test_safety_questions_get_the_fixed_careful_answer_without_calling_grok(state, q):
    with mock.patch.object(xai.requests, "post", side_effect=AssertionError("Grok must not be called")):
        r = xai.answer(q, state)
    assert r["source"] == "rule" and "can't tell you whether the water is safe" in r["reply"] and config.OFFICIAL_INFO in r["reply"]


@pytest.mark.parametrize("bad", ["The river looks fine and is safe to row today.", "It's clean, no risk.", "All clear for swimming.",
                                 "You can swim there.", "Nothing to worry about.", "Feel free to swim.", "It is okay to row today.",
                                 "No need to worry.", "Go ahead and kayak.", "You could safely wade in."])
def test_a_model_answer_that_sounds_like_a_safety_verdict_is_replaced(state, bad):
    with mock.patch.object(xai.requests, "post", lambda *a, **k: chat_ok(bad)):
        r = xai.answer("what is the outlook?", state)
    assert r["replaced"] is True and "can't tell you whether the water is safe" in r["reply"]


def test_a_sensible_answer_is_not_replaced(state):
    ok = "The outlook is the latest satellite reading carried forward, and about 4 in 10 flagged readings stay flagged a week later."
    with mock.patch.object(xai.requests, "post", lambda *a, **k: chat_ok(ok)):
        assert xai.answer("what is the outlook?", state)["replaced"] is False


def test_without_a_key_it_answers_from_the_data_with_fixed_wording(state, monkeypatch):
    monkeypatch.delenv("XAI_API_KEY")
    r = xai.answer("which areas are flagged?", state)
    assert r["source"] == "offline" and "flagged for possible algal bloom activity" in r["reply"] and "not connected" in r["note"]
    assert "outlook" in xai.answer("can you predict next week?", state)["reply"]
    assert "top 10%" in xai.answer("why is it flagged?", state)["reply"]


def test_when_xai_is_unreachable_the_chat_keeps_answering(state):
    with mock.patch.object(xai.requests, "post", side_effect=requests.ConnectionError("boom")):
        r = xai.answer("which areas are flagged?", state)
    assert r["source"] == "offline" and "Grok did not answer" in r["note"]


def test_a_rejected_model_name_is_replaced_with_the_newest_plain_text_model(state):
    tried = []

    def post(url, **kw):
        tried.append(kw["json"]["model"])
        return Resp(404, text='{"error": "model grok-4.7 not found"}') if kw["json"]["model"] == "grok-4.7" else chat_ok("ok then")

    models = Resp(200, {"data": [{"id": "grok-4.3"}, {"id": "grok-4.9"}, {"id": "grok-4.9-fast"}, {"id": "grok-imagine-1"}, {"id": "grok-voice-latest"}]})
    with mock.patch.object(xai.requests, "post", post), mock.patch.object(xai.requests, "get", lambda *a, **k: models):
        assert xai.answer("which areas are flagged?", state)["model"] == "grok-4.9"
    assert tried == ["grok-4.7", "grok-4.9"] and xai.model() == "grok-4.9"


def test_pick_text_model_ignores_variants():
    assert xai.pick_text_model(["grok-4", "grok-4.10", "grok-4.9", "grok-build-0.1", "grok-4.9-fast"]) == "grok-4.10"
    assert xai.pick_text_model(["grok-voice-latest"]) is None


def test_errors_never_contain_the_key_and_say_what_to_do(state):
    for code, words in [(401, "XAI_API_KEY"), (404, "XAI_MODEL"), (429, "credits")]:
        with mock.patch.object(xai.requests, "post", lambda *a, c=code, **k: Resp(c, text="Authorization: Bearer xai-SECRET-KEY-123")):
            with pytest.raises(xai.XAIError) as e:
                xai.chat([{"role": "user", "content": "hi"}])
        assert words in str(e.value) and "SECRET" not in str(e.value) or code == 404 and "SECRET" not in str(e.value)


def test_text_to_speech_and_speech_to_text_use_the_documented_requests(monkeypatch):
    monkeypatch.setenv("XAI_VOICE", "sal")
    seen = {}

    def post(url, **kw):
        seen[url.rsplit("/", 1)[1]] = kw
        return Resp(200, content=b"ID3mp3", ctype="audio/mpeg") if url.endswith("/tts") else Resp(200, {"text": " which areas are flagged "})

    with mock.patch.object(xai.requests, "post", post):
        audio, ctype = xai.speak("x" * 2000)
        text = xai.transcribe(b"RIFFdata", "speech.wav", "audio/wav")
    assert audio == b"ID3mp3" and ctype == "audio/mpeg" and text == "which areas are flagged"
    assert seen["tts"]["json"]["voice_id"] == "sal" and seen["tts"]["json"]["language"] == "en" and len(seen["tts"]["json"]["text"]) <= 700
    assert seen["stt"]["data"]["model"] == "grok-voice-transcribe-2.0" and seen["stt"]["files"]["file"][0] == "speech.wav"


def test_no_key_means_voice_endpoints_explain_instead_of_crashing(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY")
    with pytest.raises(xai.XAIError, match="XAI_API_KEY"):
        xai.speak("hello")
    with pytest.raises(xai.XAIError, match="XAI_API_KEY"):
        xai.transcribe(b"x")


def test_the_safety_answer_does_not_read_out_a_long_list(state):
    for z in state["zones"]:
        z["flagged"] = True
    text = xai.safety_answer(state)
    assert "flags 10 areas" in text and "and 6 more" in text and text.count("Zone") <= 4
