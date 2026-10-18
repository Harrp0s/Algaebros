"""Forecast Lab: test many forecasting ideas in parallel and say honestly which (if any) beats the simple rule.

Two stages, because testing many ideas makes it likely that one wins by luck:
  1. SCREEN  every idea on every season except the latest one (which is held out);
  2. CONFIRM the best idea for each time window against the simple rule on the held-out season, which no idea saw.
An idea only counts as a winner if it wins both. Alongside, one sandbox measures which inputs the model relies on.

Runners
  * DaytonaRunner : each experiment runs in its own Daytona sandbox (isolated, reproducible environment, in parallel).
                    Only the satellite readings and weather files are uploaded: no phone numbers, photos or keys.
  * LocalRunner   : the same experiment code on this computer. Used only as a fallback, and the result says so.

Every result carries the name of the runner that actually produced it, so the dashboard never claims Daytona
when the work ran locally.
"""
from __future__ import annotations

import concurrent.futures as cf
import importlib.util
import random
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import pandas as pd

from . import config, labcore

PKG_DIR = Path(__file__).resolve().parent
LAB_FILES = ["__init__.py", "config.py", "features.py", "labcore.py"]   # everything an experiment imports
SANDBOX_PACKAGES = ["pandas", "numpy", "scikit-learn"]
METHOD_NAMES = {
    "persistence": "Latest reading carried forward",
    "climatology": "Seasonal average",
    "hgb": "Weather model (plain)",
    "hgb_tuned": "Weather model (tuned alerts)",
    "logit": "Logistic model (tuned alerts)",
    "delta": "Predict the change, not the level",
    "classifier": "Predict 'high' directly",
    "no_weather": "Model without weather",
    "neighbors": "Model + neighbouring zones",
    "memory": "Model + longer memory",
    "importance": "Which inputs matter",
}
IDEAS = ["persistence", "hgb_tuned", "delta", "classifier", "no_weather", "neighbors", "memory"]
BASELINES = ("persistence", "climatology")
IDEA_WHY = {
    "hgb_tuned": "The trained model's recipe, with an alert cutoff tuned on earlier years.",
    "delta": "The plain model hugs the average and rarely says 'high'. Predicting the change from the latest reading avoids that.",
    "classifier": "Instead of predicting a number and hoping it crosses the line, predict the yes/no question, weighting the rare high cases more.",
    "no_weather": "The same model with every weather input removed. If it does as well, weather is not helping.",
    "neighbors": "Blooms spread along the water, so add the zones either side and the zone upstream.",
    "memory": "Add a longer look back: the 30-day mean, whether the last reading rose or fell, and last season's peak.",
}


# ---------------------------------------------------------------------- experiment grids
def _spec(method: str, horizon: int, q: float, holdout: Optional[int] = None, only_year: Optional[int] = None,
          stage: str = "screen", **extra) -> Dict:
    sid = f"h{horizon}-{method}-q{int(round(q * 100))}" + (f"-y{only_year}" if only_year else "")
    label = (f"{METHOD_NAMES[method]}, {horizon} days ahead" if method == "importance"
             else f"{METHOD_NAMES[method]}, {horizon} days ahead, flag top {int(round((1 - q) * 100))}%")
    if only_year:
        label += f" (confirm on {only_year})"
    return {"id": sid, "method": method, "horizon": horizon, "high_q": q, "label": label, "stage": stage,
            "holdout": holdout, "only_year": only_year, **extra}


def holdout_year(ndci_csv: Path = None, min_earlier_years: int = 4) -> Optional[int]:
    """The latest season, kept aside for confirmation, if enough earlier seasons remain to screen on."""
    try:
        years = sorted({int(str(d)[:4]) for d in pd.read_csv(ndci_csv or config.NDCI_CSV, usecols=["date"])["date"]})
    except (OSError, ValueError, KeyError):
        return None
    return years[-1] if len(years) > min_earlier_years else None


def quick_specs(holdout: Optional[int] = None) -> List[Dict]:
    """Every idea at 7 days ahead, plus the input-importance run: 8 sandboxes."""
    return [_spec(m, 7, 0.9, holdout) for m in IDEAS] + [_spec("importance", 7, 0.9, holdout)]


def full_specs(holdout: Optional[int] = None) -> List[Dict]:
    """Every idea at 3, 7 and 14 days ahead, plus importance with the extra inputs: 22 sandboxes."""
    return ([_spec(m, h, 0.9, holdout) for h in (3, 7, 14) for m in IDEAS]
            + [_spec("importance", 7, 0.9, holdout, extra=True)])


def confirm_specs(screen_results: List[Dict], holdout: Optional[int]) -> List[Dict]:
    """For each time window: the best screened idea and the simple rule, tested on the held-out season only."""
    if holdout is None:
        return []
    out = []
    for h in sorted({r["horizon"] for r in screen_results if r.get("ok") and r["method"] != "importance"}):
        rs = [r for r in screen_results if r.get("ok") and r["horizon"] == h and r["method"] not in BASELINES + ("importance",)]
        if not rs:
            continue
        best = max(rs, key=lambda r: r["f1"] or 0)
        out += [_spec(best["method"], h, best["high_q"], only_year=holdout, stage="confirm"),
                _spec("persistence", h, best["high_q"], only_year=holdout, stage="confirm")]
    return out


# ---------------------------------------------------------------------- honest summary
def summarise(results: List[Dict]) -> Dict:
    """Group by (horizon, threshold); compare every method with 'latest reading carried forward'.
    Then check any screened winner on the held-out season (stage 'confirm')."""
    ok = [r for r in results if r.get("ok") and r.get("stage", "screen") == "screen" and r["method"] != "importance"]
    confirm = [r for r in results if r.get("ok") and r.get("stage") == "confirm"]
    importance = next((r for r in results if r.get("ok") and r["method"] == "importance"), None)
    groups: Dict[Tuple[int, float], List[Dict]] = {}
    for r in ok:
        groups.setdefault((r["horizon"], r["high_q"]), []).append(r)
    out_groups, winners = [], []
    for (h, q), rs in sorted(groups.items()):
        base = next((r for r in rs if r["method"] == "persistence"), None)
        rows = sorted(rs, key=lambda r: -(r["f1"] or 0))
        verdict, best = "no baseline to compare with", None
        if base is not None:
            challengers = [r for r in rs if r["method"] not in ("persistence", "climatology")]
            best = max(challengers, key=lambda r: r["f1"] or 0, default=None)
            if best is None:
                verdict = "only the baseline was run"
            elif best["f1_range"] and best["f1_range"][0] > base["f1"]:
                verdict = "clearly better than the simple rule"
                winners.append((h, q, best, base))
            elif (best["f1"] or 0) > (base["f1"] or 0):
                verdict = "slightly higher, but within noise: not clearly better"
            else:
                verdict = "not better than the simple rule"
        out_groups.append({"horizon": h, "high_q": q, "rows": rows, "verdict": verdict,
                           "best_challenger": best["id"] if best else None})
    persist_by_h = {g["horizon"]: next((r["f1"] for r in g["rows"] if r["method"] == "persistence"), None)
                    for g in out_groups if abs(g["high_q"] - 0.9) < 1e-9}
    trend = ""
    hs = sorted(h for h, v in persist_by_h.items() if v is not None)
    if len(hs) >= 2:
        trend = (f"The simple rule's score falls from {persist_by_h[hs[0]]:.2f} at {hs[0]} days to "
                 f"{persist_by_h[hs[-1]]:.2f} at {hs[-1]} days: the further ahead, the less any method can promise.")
    confirmations = []
    for h in sorted({r["horizon"] for r in confirm}):
        rs = [r for r in confirm if r["horizon"] == h]
        base = next((r for r in rs if r["method"] == "persistence"), None)
        chal = next((r for r in rs if r["method"] != "persistence"), None)
        if base is None or chal is None:
            continue
        year = chal.get("only_year")
        held = (chal["f1"] or 0) > (base["f1"] or 0)
        clear = bool(chal.get("f1_range")) and chal["f1_range"][0] > (base["f1"] or 0)
        verdict = (f"held up on {year}, clearly" if clear else f"held up on {year}, but within noise" if held
                   else f"did not hold up on {year}")
        confirmations.append({"horizon": h, "year": year, "method": chal["method"], "name": METHOD_NAMES[chal["method"]],
                              "f1": chal["f1"], "f1_range": chal.get("f1_range"), "base_f1": base["f1"],
                              "verdict": verdict, "clear": clear})
    confirmed = {(c["horizon"], c["method"]) for c in confirmations if c["clear"]}
    if winners:
        h, q, best, base = max(winners, key=lambda w: w[2]["f1"])
        headline = (f"{METHOD_NAMES[best['method']]} did clearly better than the simple rule at {h} days "
                    f"(score {best['f1']:.2f} vs {base['f1']:.2f}, range {best['f1_range'][0]:.2f} to {best['f1_range'][1]:.2f}).")
        if confirmations and (h, best["method"]) not in confirmed:
            year = confirmations[0]["year"]
            headline = (f"{METHOD_NAMES[best['method']]} looked clearly better at {h} days on earlier seasons, "
                        f"but did not clearly hold up on the unseen {year} season, so alerts keep the simple rule.")
        elif confirmations:
            headline += f" It also held up on the unseen {confirmations[0]['year']} season."
    elif ok:
        headline = "No method clearly beat simply using the latest satellite reading. That is why alerts use it."
    else:
        headline = "No experiment produced a result."
    return {"headline": headline, "trend": trend, "groups": out_groups, "confirmations": confirmations,
            "importance": importance, "ideas": {m: IDEA_WHY[m] for m in IDEA_WHY},
            "caution": "Flagged readings cluster on a few dates, so differences smaller than the shown range are noise."}


# ---------------------------------------------------------------------- runners
class LocalRunner:
    label = "this computer (Daytona was not used)"

    def __init__(self, ndci_csv: Path = None, weather_csv: Path = None, workers: int = 4):
        self.ndci_csv = str(ndci_csv or config.NDCI_CSV)
        self.weather_csv = str(weather_csv or config.WEATHER_CSV)
        self.workers = workers

    def run(self, specs: List[Dict], progress: Callable[[int, int, str], None]) -> List[Dict]:
        results, done = [], 0
        with cf.ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self._one, s): s for s in specs}
            for fut in cf.as_completed(futs):
                results.append(fut.result())
                done += 1
                progress(done, len(specs), futs[fut]["id"])
        return results

    def _one(self, spec: Dict) -> Dict:
        try:
            return labcore.run_experiment(spec, self.ndci_csv, self.weather_csv)
        except Exception as e:
            return {"id": spec["id"], "ok": False, "method": spec["method"], "horizon": spec["horizon"],
                    "high_q": spec.get("high_q", 0.9), "reason": f"{type(e).__name__}: {str(e)[:200]}"}


class DaytonaRunner:
    label = "Daytona sandboxes"

    def __init__(self, ndci_csv: Path = None, weather_csv: Path = None, max_parallel: Optional[int] = None, client=None):
        self.ndci = Path(ndci_csv or config.NDCI_CSV).read_bytes()
        self.weather = Path(weather_csv or config.WEATHER_CSV).read_bytes()
        self.max_parallel = max_parallel or int(os.environ.get("DAYTONA_MAX_PARALLEL", "8"))
        self._client = client

    @staticmethod
    def available() -> bool:
        return bool(os.environ.get("DAYTONA_API_KEY")) and importlib.util.find_spec("daytona") is not None

    # -- SDK access, separated so tests can supply a fake
    def client(self):
        if self._client is None:
            from daytona import Daytona, DaytonaConfig

            self._client = Daytona(DaytonaConfig(api_key=os.environ["DAYTONA_API_KEY"]))
        return self._client

    def params(self, block_network: bool = False):
        from daytona import CreateSandboxFromImageParams, Image

        kwargs = dict(image=Image.debian_slim("3.12").pip_install(SANDBOX_PACKAGES), ephemeral=True,
                      auto_stop_interval=10, labels={"app": "algaebro"})
        if block_network:
            kwargs["network_block_all"] = True     # code written by an AI gets no internet access
        return CreateSandboxFromImageParams(**kwargs)

    def exec_in_sandbox(self, files: Dict[str, bytes], command: str, timeout: int = 300,
                        block_network: bool = False) -> Tuple[int, str]:
        """Create a fresh sandbox, upload `files` (relative path -> bytes), run `command` from their folder,
        return (exit_code, output), and ALWAYS delete the sandbox afterwards."""
        sandbox = None
        try:
            sandbox = self._create(block_network)
            home = sandbox.get_user_home_dir() if hasattr(sandbox, "get_user_home_dir") else "/home/daytona"
            base = f"{home.rstrip('/')}/work"
            folders = {base} | {f"{base}/{rel.rsplit('/', 1)[0]}" for rel in files if "/" in rel}
            for folder in sorted(folders, key=len):          # parents first; "already exists" is fine
                try:
                    sandbox.fs.create_folder(folder, "755")
                except Exception:
                    pass
            for rel, data in files.items():
                sandbox.fs.upload_file(data, f"{base}/{rel}")
            resp = sandbox.process.exec(f"cd {base} && {command}", timeout=timeout)
            return int(resp.exit_code), resp.result or ""
        finally:
            if sandbox is not None:
                try:
                    sandbox.delete()
                except Exception:   # an ephemeral sandbox stops itself; never fail the run over cleanup
                    pass

    def _create(self, block_network: bool, tries: int = 4):
        """Create a sandbox; if the account is at its limit of running sandboxes, wait for one to finish and retry."""
        for attempt in range(tries):
            try:
                return self.client().create(self.params(block_network=block_network), timeout=300)
            except Exception as e:
                busy = any(w in str(e).lower() for w in ("limit", "quota", "exceed", "too many", "429", "insufficient"))
                if not busy or attempt == tries - 1:
                    raise
                time.sleep(8 * (attempt + 1) + random.random() * 4)

    # -- one experiment in one sandbox
    def _one(self, spec: Dict) -> Dict:
        fail = lambda why: {"id": spec["id"], "ok": False, "method": spec["method"], "horizon": spec["horizon"],  # noqa: E731
                            "high_q": spec.get("high_q", 0.9), "reason": why[:300]}
        files = {f"algaebro/{name}": (PKG_DIR / name).read_bytes() for name in LAB_FILES}
        files.update({"ndci.csv": self.ndci, "weather.csv": self.weather, "spec.json": json.dumps(spec).encode()})
        try:
            code, out = self.exec_in_sandbox(
                files, "python -m algaebro.labcore spec.json ndci.csv weather.csv", timeout=300)
        except Exception as e:
            return fail(f"{type(e).__name__}: {e}")
        for line in out.splitlines():
            if line.startswith("RESULT_JSON:"):
                return json.loads(line[len("RESULT_JSON:"):])
        return fail(f"sandbox exited {code} without a result: {out[-200:]}")

    def run(self, specs: List[Dict], progress: Callable[[int, int, str], None]) -> List[Dict]:
        results, done = [], 0
        if not specs:
            return results
        # The first sandbox may need to build the environment; do it alone so the others reuse it.
        first = self._one(specs[0])
        results.append(first)
        done += 1
        progress(done, len(specs), specs[0]["id"])
        with cf.ThreadPoolExecutor(max_workers=self.max_parallel) as ex:
            futs = {ex.submit(self._one, s): s for s in specs[1:]}
            for fut in cf.as_completed(futs):
                results.append(fut.result())
                done += 1
                progress(done, len(specs), futs[fut]["id"])
        return results


# ---------------------------------------------------------------------- jobs (background work the dashboard polls)
class Jobs:
    def __init__(self):
        self._jobs: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def start(self, kind: str, target: Callable[[Dict], Dict]) -> str:
        jid = uuid.uuid4().hex[:10]
        job = {"id": jid, "kind": kind, "status": "running", "done": 0, "total": 0, "runner": None, "log": [],
               "result": None, "error": None, "started": time.time(), "finished": None}
        with self._lock:
            self._jobs[jid] = job

        def work():
            try:
                job["result"] = target(job)
                job["status"] = "done"
            except Exception as e:
                job["status"] = "error"
                job["error"] = f"{type(e).__name__}: {str(e)[:300]}"
            finally:
                job["finished"] = time.time()

        threading.Thread(target=work, daemon=True).start()
        return jid

    def get(self, jid: str) -> Optional[Dict]:
        return self._jobs.get(jid)

    def running(self, kind: str) -> Optional[str]:
        with self._lock:
            return next((j["id"] for j in self._jobs.values() if j["kind"] == kind and j["status"] == "running"), None)


def run_lab(job: Dict, mode: str = "quick", prefer_daytona: bool = True, local_runner: LocalRunner = None,
            daytona_runner: Optional[DaytonaRunner] = None, data_source: Optional[str] = None) -> Dict:
    """The job body: pick the runner, run, summarise, remember. Falls back to local with a clear note."""
    holdout = holdout_year()
    specs = full_specs(holdout) if mode == "full" else quick_specs(holdout)
    job["total"] = len(specs)
    job["stage"] = "screen"
    offset = {"n": 0}

    def progress(done: int, total: int, label: str) -> None:
        job["done"] = offset["n"] + done
        job["log"].append(f"finished {label} ({offset['n'] + done}/{job['total']})")

    notes, results, runner_label = [], None, None
    if prefer_daytona and (daytona_runner is not None or DaytonaRunner.available()):
        runner = daytona_runner or DaytonaRunner()
        job["runner"] = runner.label
        job["log"].append(f"starting {len(specs)} experiments in Daytona sandboxes (the first run may build the environment)")
        results = runner.run(specs, progress)
        runner_label = runner.label
        n_ok = sum(1 for r in results if r.get("ok"))
        if n_ok == 0:
            notes.append("Daytona produced no results (" + (next((r.get("reason") for r in results if r.get("reason")), "unknown") or "unknown")
                         + "), so the experiments were run on this computer instead.")
            results = None
    elif prefer_daytona:
        notes.append("Daytona is not set up (no DAYTONA_API_KEY or the 'daytona' package is missing), so this ran on this computer.")
    if results is None:
        runner = local_runner or LocalRunner()
        job["runner"] = runner.label
        job["done"] = 0
        job["log"].append(f"running {len(specs)} experiments on this computer")
        results = runner.run(specs, progress)
        runner_label = runner.label

    more = confirm_specs(results, holdout)
    if more:
        job["stage"] = "confirm"
        offset["n"] = len(specs)
        job["total"] = len(specs) + len(more)
        job["log"].append(f"confirming the best idea per time window on the unseen {holdout} season ({len(more)} more sandboxes)")
        results = results + runner.run(more, progress)
        specs = specs + more

    payload = {"mode": mode, "runner": runner_label, "notes": notes, "n": len(specs), "holdout": holdout,
               "data_source": data_source or config.data_source(),
               "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"),
               "summary": summarise(results), "results": sorted(results, key=lambda r: (r.get("horizon", 0), r.get("id", "")))}
    try:
        (config.DATA_DIR / "lab_latest.json").write_text(json.dumps(payload, default=str))
    except OSError:
        pass
    return payload


def latest() -> Dict:
    p = config.DATA_DIR / "lab_latest.json"
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}
