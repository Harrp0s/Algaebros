"""'Ask the data': Grok writes a small analysis, Daytona runs it in a sandbox, you see the code AND the result.

The generated code is untrusted, so it is NEVER run on this computer. If Daytona is not available the
question is refused. The sandbox gets only the satellite readings and weather files (no phone numbers or
photos) and, when the SDK supports it, no internet access.
"""
from __future__ import annotations

import re
import time
from typing import Dict, List, Optional

from . import config, lab, xai

MAX_OUTPUT_CHARS = 4000
FORBIDDEN = [
    (r"\bimport\s+(os|sys|subprocess|socket|shutil|requests|urllib|http|ftplib|smtplib|ctypes|multiprocessing|pickle)\b", "system or network modules"),
    (r"\bfrom\s+(os|sys|subprocess|socket|shutil|requests|urllib|http|ctypes)\b", "system or network modules"),
    (r"__import__|\beval\s*\(|\bexec\s*\(|\bcompile\s*\(|\bopen\s*\(", "dynamic code or file access"),
    (r"\.to_(csv|excel|pickle|parquet|json|sql)\s*\(", "writing files"),
    (r"\binput\s*\(", "interactive input"),
]

SCHEMA = """Files in the current folder (read them with pandas; do not write files):
  ndci.csv    satellite chlorophyll index readings of the Charles River, one row per area per clear-sky date
              columns: date (YYYY-MM-DD), zone_id (like Z01..Z10, west to east), ndci (the index), plus a few quality columns
  weather.csv daily weather: date, tmean_c (mean temp C), tmax_c, precip_mm, wind_ms (mean wind m/s), plus a few others
'Flagged' ('high') means ndci >= {high:+.3f}; 'watch' means ndci >= {elev:+.3f}. Print column names first if unsure."""

INSTRUCTIONS = (
    "Write ONE short Python 3 script that answers the question using only pandas and numpy. "
    "Read the CSV files named above from the current folder. Print a SHORT plain-English result (at most 25 lines) with the key "
    "numbers. No plots, no file writing, no network, no input(). Reply with exactly one ```python code block and nothing else."
)


def extract_code(text: str) -> Optional[str]:
    m = re.search(r"```(?:python|py)?\s*\n(.*?)```", text, re.S | re.I)
    return m.group(1).strip() if m else None


def scan(code: str) -> Optional[str]:
    """Return a reason to refuse, or None. This is a second fence: the sandbox is the real boundary."""
    for pat, why in FORBIDDEN:
        if re.search(pat, code):
            return why
    return None


def build_messages(question: str, thresholds: Dict[str, float]) -> List[Dict[str, str]]:
    return [
        {"role": "system", "content": INSTRUCTIONS + "\n\n" + SCHEMA.format(high=thresholds["high"], elev=thresholds["elevated"])},
        {"role": "user", "content": question.strip()[:500]},
    ]


def run_ask(job: Dict, question: str, thresholds: Dict[str, float], runner: Optional["lab.DaytonaRunner"] = None,
            chat=xai.chat) -> Dict:
    question = (question or "").strip()
    if not question:
        raise ValueError("Type a question first.")
    if runner is None and not lab.DaytonaRunner.available():
        raise RuntimeError("Ask-the-data needs Daytona (set DAYTONA_API_KEY and install the 'daytona' package). "
                           "I will not run AI-written code on this computer.")
    job["log"].append("asking Grok to write the analysis")
    t0 = time.time()
    reply = chat(build_messages(question, thresholds), temperature=0.1)
    code = extract_code(reply)
    if not code:
        raise RuntimeError("Grok did not return a code block, so nothing was run. Try rephrasing the question.")
    why = scan(code)
    if why:
        return {"question": question, "code": code, "output": None, "ran": False, "runner": None,
                "refused": f"Not run: the generated code used {why}.", "seconds": round(time.time() - t0, 1)}
    runner = runner or lab.DaytonaRunner()
    job["runner"] = runner.label
    job["log"].append("running it in a Daytona sandbox")
    files = {"analysis.py": code.encode(), "ndci.csv": runner.ndci, "weather.csv": runner.weather}
    try:
        exit_code, out = runner.exec_in_sandbox(files, "python analysis.py", timeout=90, block_network=True)
    except Exception as e:   # some SDK versions may not support blocking the network; say so and retry once
        if "network" not in str(e).lower():
            raise
        job["log"].append("network blocking was not accepted; running without it (the code was already screened)")
        exit_code, out = runner.exec_in_sandbox(files, "python analysis.py", timeout=90, block_network=False)
    return {"question": question, "code": code, "output": out[:MAX_OUTPUT_CHARS], "exit_code": exit_code, "ran": True,
            "runner": runner.label, "refused": None, "seconds": round(time.time() - t0, 1)}
