import importlib.util
import sys
from pathlib import Path
from unittest import mock

import pytest

from algaebro import xai

SPEC = importlib.util.spec_from_file_location("check_ai", Path(__file__).resolve().parent.parent / "scripts" / "check_ai.py")
check_ai = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(check_ai)


def test_without_keys_everything_is_skipped_not_failed(monkeypatch, capsys):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    assert check_ai.check_grok() == 0 and check_ai.check_daytona() == 0
    out = capsys.readouterr().out
    assert out.count("SKIP") == 2 and "XAI_API_KEY" in out and "DAYTONA_API_KEY" in out


def test_a_working_setup_passes_and_the_key_is_never_printed(monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "xai-SECRET-KEY-123")
    with mock.patch.object(xai, "chat", return_value="ready"), mock.patch.object(xai, "speak", return_value=(b"ID3", "audio/mpeg")), \
            mock.patch.object(xai, "transcribe", return_value="Algaebro voice check"):
        assert check_ai.check_grok() == 0
    out = capsys.readouterr().out
    assert out.count("PASS") == 3 and "SECRET" not in out


def test_a_failure_says_what_to_fix_and_counts_as_a_problem(monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "xai-SECRET-KEY-123")
    with mock.patch.object(xai, "chat", side_effect=xai.XAIError("xAI rejected the API key for chat (HTTP 401). Check XAI_API_KEY in .env.")), \
            mock.patch.object(xai, "speak", side_effect=xai.XAIError("no credits")):
        assert check_ai.check_grok() == 2
    out = capsys.readouterr().out
    assert out.count("FAIL") == 2 and "Check XAI_API_KEY" in out and "SECRET" not in out


def test_speech_that_comes_back_wrong_is_reported(monkeypatch, capsys):
    monkeypatch.setenv("XAI_API_KEY", "xai-x")
    with mock.patch.object(xai, "chat", return_value="ready"), mock.patch.object(xai, "speak", return_value=(b"ID3", "audio/mpeg")), \
            mock.patch.object(xai, "transcribe", return_value="banana"):
        assert check_ai.check_grok() == 1
    assert "not what was said" in capsys.readouterr().out
