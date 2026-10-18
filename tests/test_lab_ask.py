import os
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

from algaebro import ask, config, lab


# ---------------------------------------------------------------------- a fake Daytona that behaves like the real one
class FakeFS:
    def __init__(self, calls):
        self.calls = calls

    def create_folder(self, path, mode):
        os.makedirs(path, exist_ok=True)
        self.calls.append(("mkdir", path))

    def upload_file(self, data, path):
        assert os.path.isdir(os.path.dirname(path)), f"the real service would refuse: parent folder missing for {path}"
        with open(path, "wb") as f:
            f.write(data)
        self.calls.append(("upload", path))


class FakeProcess:
    def __init__(self, calls, fail):
        self.calls, self.fail = calls, fail

    def exec(self, command, timeout=None):
        self.calls.append(("exec", command))
        if self.fail:
            raise RuntimeError("sandbox crashed")
        cmd = command.replace(" python ", f" {sys.executable} ", 1)
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}      # only the uploaded files may be importable
        p = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout or 120, env=env)
        return SimpleNamespace(exit_code=p.returncode, result=p.stdout + p.stderr)


class FakeSandbox:
    def __init__(self, home, calls, fail=False):
        self.home, self.deleted, self.id = home, False, "sbx-test"
        self.fs, self.process = FakeFS(calls), FakeProcess(calls, fail)

    def get_user_home_dir(self):
        return str(self.home)

    def delete(self):
        self.deleted = True


class FakeClient:
    def __init__(self, tmp_path, fail=False):
        self.tmp, self.fail, self.made, self.calls = tmp_path, fail, [], []

    def create(self, params, timeout=None):
        home = self.tmp / f"home{len(self.made)}"
        home.mkdir()
        sb = FakeSandbox(home, self.calls, self.fail)
        self.made.append((params, sb))
        return sb


def runner_with(client):
    r = lab.DaytonaRunner(client=client)
    r.params = lambda block_network=False: {"block_network": block_network}      # the real SDK object is checked in its own test
    return r


# ---------------------------------------------------------------------- the lab
def test_each_experiment_gets_its_own_sandbox_with_only_the_files_it_needs(synth, tmp_path):
    client = FakeClient(tmp_path)
    results = runner_with(client).run(lab.quick_specs()[:2], lambda *a: None)
    assert [r["ok"] for r in results] == [True, True], [r.get("reason") for r in results]
    assert len(client.made) == 2 and all(sb.deleted for _, sb in client.made)
    uploaded = {os.path.relpath(p, sb.home).replace(os.sep, "/") for (kind, p) in client.calls if kind == "upload" for _, sb in client.made if p.startswith(str(sb.home))}
    assert uploaded == {"work/algaebro/__init__.py", "work/algaebro/config.py", "work/algaebro/features.py", "work/algaebro/labcore.py",
                        "work/ndci.csv", "work/weather.csv", "work/spec.json"}, "no database, photos, phone numbers or keys"
    assert all(p["block_network"] is False for p, _ in client.made), "the lab's own trusted code may use the network (pip installs)"


def test_the_sandbox_is_deleted_even_when_the_run_crashes(synth, tmp_path):
    client = FakeClient(tmp_path, fail=True)
    results = runner_with(client).run(lab.quick_specs()[:1], lambda *a: None)
    assert results[0]["ok"] is False and "sandbox crashed" in results[0]["reason"]
    assert all(sb.deleted for _, sb in client.made)


def test_the_real_sdk_accepts_our_sandbox_parameters():
    pytest.importorskip("daytona")
    p = lab.DaytonaRunner.params(SimpleNamespace(), block_network=True)
    assert p.network_block_all is True and p.ephemeral is True
    assert lab.DaytonaRunner.params(SimpleNamespace(), block_network=False).network_block_all in (None, False)


def test_when_daytona_gives_nothing_the_lab_says_so_and_runs_locally(synth, monkeypatch):
    monkeypatch.setattr(lab, "IDEAS", ["persistence", "delta"])
    class Broken:
        label = "Daytona sandboxes"

        def run(self, specs, progress):
            return [{"id": s["id"], "ok": False, "method": s["method"], "horizon": s["horizon"], "high_q": 0.9, "reason": "no credits"} for s in specs]

    job = {"id": "t", "log": [], "done": 0, "total": 0, "runner": None}
    out = lab.run_lab(job, "quick", daytona_runner=Broken(), data_source="synthetic")
    assert "no credits" in out["notes"][0] and "this computer" in out["runner"] and out["data_source"] == "synthetic"
    assert any(r.get("ok") for r in out["results"])


def test_without_daytona_configured_the_lab_runs_locally_and_labels_it(synth, monkeypatch):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    monkeypatch.setattr(lab, "IDEAS", ["persistence", "delta"])
    job = {"id": "t", "log": [], "done": 0, "total": 0, "runner": None}
    out = lab.run_lab(job, "quick", data_source="synthetic")
    assert "Daytona was not used" in out["runner"] and "not set up" in out["notes"][0]
    assert job["done"] == job["total"] == 3 + 2 and lab.latest()["finished_at"] == out["finished_at"]
    assert out["holdout"] == 2026 and out["summary"]["confirmations"][0]["year"] == 2026
    assert out["summary"]["importance"]["importance"], "the importance sandbox ranks the inputs"


def test_available_needs_both_the_key_and_the_package(monkeypatch):
    monkeypatch.delenv("DAYTONA_API_KEY", raising=False)
    assert lab.DaytonaRunner.available() is False
    monkeypatch.setenv("DAYTONA_API_KEY", "dt-secret")
    assert lab.DaytonaRunner.available() is (lab.importlib.util.find_spec("daytona") is not None)


def result(method, h, f1, lo, hi, q=0.9):
    return {"id": f"h{h}-{method}", "ok": True, "method": method, "horizon": h, "high_q": q, "f1": f1, "f1_range": [lo, hi], "recall": f1, "precision": f1}


def test_a_method_is_called_clearly_better_only_when_its_whole_range_beats_the_baseline():
    s = lab.summarise([result("persistence", 7, 0.50, 0.40, 0.60), result("logit", 7, 0.80, 0.65, 0.90),
                       result("persistence", 3, 0.60, 0.50, 0.70), result("logit", 3, 0.66, 0.52, 0.78),
                       result("persistence", 14, 0.40, 0.30, 0.50), result("logit", 14, 0.30, 0.20, 0.40)])
    verdicts = {g["horizon"]: g["verdict"] for g in s["groups"]}
    assert verdicts[7].startswith("clearly better") and "within noise" in verdicts[3] and verdicts[14] == "not better than the simple rule"
    assert "did clearly better" in s["headline"] and "7 days" in s["headline"] and "falls from" in s["trend"]


def test_no_winner_gives_the_honest_headline_and_no_baseline_is_handled():
    s = lab.summarise([result("persistence", 7, 0.5, 0.4, 0.6), result("logit", 7, 0.45, 0.3, 0.6)])
    assert s["headline"].startswith("No method clearly beat")
    assert lab.summarise([result("logit", 7, 0.5, 0.4, 0.6)])["groups"][0]["verdict"] == "no baseline to compare with"
    assert lab.summarise([])["headline"] == "No experiment produced a result."


def test_jobs_report_progress_success_and_failure():
    jobs = lab.Jobs()
    gate = threading.Event()
    jid = jobs.start("lab", lambda job: (gate.wait(5), {"ok": 1})[1])
    assert jobs.running("lab") == jid and jobs.get(jid)["status"] == "running"
    gate.set()
    for _ in range(100):
        if jobs.get(jid)["status"] != "running":
            break
        threading.Event().wait(0.05)
    assert jobs.get(jid)["status"] == "done" and jobs.get(jid)["result"] == {"ok": 1} and jobs.running("lab") is None
    bad = jobs.start("ask", lambda job: 1 / 0)
    for _ in range(100):
        if jobs.get(bad)["status"] != "running":
            break
        threading.Event().wait(0.05)
    assert jobs.get(bad)["status"] == "error" and "ZeroDivisionError" in jobs.get(bad)["error"]


# ---------------------------------------------------------------------- ask the data
class FakeAskRunner:
    label = "Daytona sandboxes"
    ndci, weather = b"date,zone_id,ndci\n", b"date,tmean_c\n"

    def __init__(self, outputs=None, raise_first=None):
        self.calls, self.outputs, self.raise_first = [], outputs or [(0, "median flagged month: 7")], raise_first

    def exec_in_sandbox(self, files, command, timeout=300, block_network=False):
        self.calls.append({"files": sorted(files), "command": command, "block_network": block_network})
        if self.raise_first and len(self.calls) == 1:
            raise RuntimeError(self.raise_first)
        return self.outputs[min(len(self.calls) - 1, len(self.outputs) - 1)]


GOOD = "```python\nimport pandas as pd\ndf = pd.read_csv('ndci.csv')\nprint('median flagged month: 7')\n```"
TH = {"high": 0.128, "elevated": 0.065}


def run_ask(reply, runner, monkeypatch, available=True):
    monkeypatch.setattr(lab.DaytonaRunner, "available", staticmethod(lambda: available))
    job = {"log": [], "runner": None}
    return ask.run_ask(job, "Which month has the most flagged readings?", TH, runner=runner, chat=lambda msgs, temperature=0.1: reply), job


def test_ai_written_code_runs_only_in_the_sandbox_with_the_network_blocked(monkeypatch):
    runner = FakeAskRunner()
    out, _ = run_ask(GOOD, runner, monkeypatch)
    assert out["ran"] and out["runner"] == "Daytona sandboxes" and "median flagged month" in out["output"] and out["refused"] is None
    assert runner.calls == [{"files": ["analysis.py", "ndci.csv", "weather.csv"], "command": "python analysis.py", "block_network": True}]


@pytest.mark.parametrize("code", ["import os\nprint(os.listdir('/'))", "import requests", "from subprocess import run", "open('x','w').write('1')",
                                  "eval('1+1')", "exec('x=1')", "df.to_csv('out.csv')", "__import__('os')", "x = input()"])
def test_forbidden_code_is_never_run(code, monkeypatch):
    runner = FakeAskRunner()
    out, _ = run_ask(f"```python\n{code}\n```", runner, monkeypatch)
    assert out["ran"] is False and out["refused"].startswith("Not run") and out["code"].strip() == code and runner.calls == []


def test_without_daytona_nothing_runs_and_nothing_is_asked(monkeypatch):
    def chat(*a, **k):
        raise AssertionError("Grok must not even be asked")

    monkeypatch.setattr(lab.DaytonaRunner, "available", staticmethod(lambda: False))
    with pytest.raises(RuntimeError, match="will not run AI-written code on this computer"):
        ask.run_ask({"log": []}, "anything", TH, chat=chat)


def test_if_the_sdk_cannot_block_the_network_it_retries_once_and_says_so(monkeypatch):
    runner = FakeAskRunner(raise_first="network_block_all is not supported on this plan")
    out, job = run_ask(GOOD, runner, monkeypatch)
    assert out["ran"] and [c["block_network"] for c in runner.calls] == [True, False]
    assert any("network blocking was not accepted" in line for line in job["log"])


def test_other_sandbox_errors_are_not_swallowed(monkeypatch):
    with pytest.raises(RuntimeError, match="out of credits"):
        run_ask(GOOD, FakeAskRunner(raise_first="out of credits"), monkeypatch)


def test_a_reply_without_a_code_block_runs_nothing(monkeypatch):
    runner = FakeAskRunner()
    with pytest.raises(RuntimeError, match="did not return a code block"):
        run_ask("Sure! The answer is July.", runner, monkeypatch)
    assert runner.calls == []


def test_the_prompt_carries_the_thresholds_but_no_personal_data():
    msgs = ask.build_messages("what is the busiest month?", TH)
    text = " ".join(m["content"] for m in msgs)
    assert "+0.128" in text and "ndci.csv" in text and "+1" not in text and "phone" not in text.lower()


def test_long_output_is_cut(monkeypatch):
    out, _ = run_ask(GOOD, FakeAskRunner(outputs=[(0, "x" * 20000)]), monkeypatch)
    assert len(out["output"]) == ask.MAX_OUTPUT_CHARS
