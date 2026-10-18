"""Check the AI pieces without opening the dashboard. Never prints a key.

    python scripts/check_ai.py              # Grok chat, voice (speech out and back in), Nessie, Daytona
    python scripts/check_ai.py --no-daytona # skip the Daytona test (the first one builds a sandbox image and can take minutes)

What each line means:
  PASS  it worked.
  FAIL  it did not; the line after it says what to fix.
  SKIP  the key for it is not in .env yet.
"""
import argparse
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algaebro import config, lab, nessie, screening, xai
from algaebro.run_bot import load_dotenv


def line(status: str, what: str, detail: str = "") -> None:
    print(f"{status:<5} {what}" + (f"\n        {detail}" if detail else ""))


def check_grok() -> int:
    if not xai.configured():
        line("SKIP", "Grok chat, text-to-speech, speech-to-text", "Add XAI_API_KEY=... to .env (get one at console.x.ai).")
        return 0
    fails = 0
    try:
        t0 = time.time()
        reply = xai.chat([{"role": "user", "content": "Reply with the single word: ready"}])
        line("PASS", f"Grok chat with model '{xai.model()}'", f"it said: {reply[:60]!r} ({time.time() - t0:.1f}s)")
    except xai.XAIError as e:
        line("FAIL", "Grok chat", str(e))
        fails += 1
    try:
        audio, ctype = xai.speak("Algaebro voice check. The river is being watched.")
        line("PASS", f"text-to-speech with voice '{xai.voice()}'", f"{len(audio) // 1024} KB of {ctype}")
    except xai.XAIError as e:
        line("FAIL", "text-to-speech", str(e))
        fails += 1
        return fails
    try:
        ext = "mp3" if "mpeg" in ctype or "mp3" in ctype else "wav"
        text = xai.transcribe(audio, f"check.{ext}", ctype)
        ok = re.search(r"algae|bro|voice|river|check", text, re.I) is not None
        line("PASS" if ok else "FAIL", "speech-to-text (listened to its own voice)", f"heard: {text[:80]!r}" + ("" if ok else "  <- not what was said"))
        fails += 0 if ok else 1
    except xai.XAIError as e:
        line("FAIL", "speech-to-text", str(e))
        fails += 1
    return fails


def check_vision() -> int:
    """Can the configured model look at a photo? (The photo check needs it.)"""
    if not xai.configured():
        line("SKIP", "Grok photo check", "Add XAI_API_KEY=... to .env.")
        return 0
    import io

    from PIL import Image, ImageDraw

    img = Image.new("RGB", (480, 320), (135, 185, 230))                       # sky
    d = ImageDraw.Draw(img)
    d.rectangle([0, 150, 480, 320], fill=(40, 110, 70))                      # green water
    d.polygon([(0, 150), (120, 120), (260, 140), (480, 115), (480, 150)], fill=(60, 90, 50))   # far shore
    buf = io.BytesIO()
    img.save(buf, "JPEG")
    try:
        t0 = time.time()
        v = screening.parse(screening.ask_grok(buf.getvalue(), "a test pond"))
    except xai.XAIError as e:
        line("FAIL", "Grok photo check", f"{e} If the model cannot take images, set XAI_VISION_MODEL in .env to one that can.")
        return 1
    if v is None:
        line("FAIL", "Grok photo check", "the model answered, but not with the JSON the photo check needs")
        return 1
    line("PASS", f"Grok photo check with model '{xai.vision_model()}'",
         f"drawing of a pond -> water photo={v['is_water_photo']}, bloom={v['bloom']} ({time.time() - t0:.1f}s)")
    return 0


def check_nessie() -> int:
    if not nessie.configured():
        line("SKIP", "Nessie (cleanup fund, nearby businesses)", "Add NESSIE_API_KEY=... to .env (get one at api.nessieisreal.com).")
        return 0
    try:
        t0 = time.time()
        nessie._call("GET", "/customers")
        got = nessie.merchants_near(42.357, -71.10, miles=1.0)
        line("PASS", "Nessie key accepted", f"{got.get('count', 0)} sandbox merchants within 1 mile of the Charles ({time.time() - t0:.1f}s)")
        return 0
    except nessie.NessieError as e:
        line("FAIL", "Nessie", str(e))
        return 1


def check_daytona() -> int:
    import importlib.util
    import os

    if not os.environ.get("DAYTONA_API_KEY"):
        line("SKIP", "Daytona sandbox", "Add DAYTONA_API_KEY=... to .env (get one at app.daytona.io).")
        return 0
    if importlib.util.find_spec("daytona") is None:
        line("FAIL", "Daytona sandbox", "The 'daytona' package is missing. Run: pip install daytona")
        return 1
    print("      starting a sandbox (the first time it builds the environment, which can take a few minutes)...")
    t0 = time.time()
    try:
        runner = lab.DaytonaRunner()
        code, out = runner.exec_in_sandbox({"hello.py": b"import pandas, sklearn; print('daytona ok', pandas.__version__)"}, "python hello.py", timeout=600)
    except Exception as e:
        line("FAIL", "Daytona sandbox", f"{type(e).__name__}: {str(e)[:220]}")
        return 1
    ok = code == 0 and "daytona ok" in out
    line("PASS" if ok else "FAIL", "Daytona sandbox (created, ran code, deleted)", f"{out.strip()[:80]!r} in {time.time() - t0:.0f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-daytona", action="store_true")
    args = ap.parse_args()
    load_dotenv(config.ROOT / ".env")
    failures = check_grok() + check_vision() + check_nessie() + (0 if args.no_daytona else check_daytona())
    print("\nAll good." if not failures else f"\n{failures} problem(s). Fix the FAIL lines above and run this again.")
    sys.exit(1 if failures else 0)
