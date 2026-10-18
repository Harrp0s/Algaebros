"""Talk to the bot in the terminal, no phone needed. Great for trying commands and wording.

    python scripts/chat_console.py                      # live forecast
    python scripts/chat_console.py --as-of 2026-07-20   # replay a past date

Type messages like: help, zones, status 4, subscribe 4, 42.355, -71.110, report 4 green scum, stop.
Type /alerts to run the alert engine and see what would be texted to you, /quit to leave.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algaebro import alerts, forecast, zones
from algaebro.bot import Bot
from algaebro.messenger import MemoryMessenger
from algaebro.store import Store, UserCtx

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--as-of")
    args = ap.parse_args()
    store = Store()
    bot = Bot(store, lambda: forecast.get_forecast(args.as_of), zones.load_zone_records())
    ctx = UserCtx(user_id="console", handle="console", space_id="console-space", agent_id="console-agent")
    mem = MemoryMessenger()
    fc = forecast.get_forecast(args.as_of)
    print(f"Console chat. data={fc['data_source']} method={fc['method']} as_of={fc['as_of']}{' REPLAY' if fc['replay'] else ''}. Try: help")
    while True:
        try:
            line = input("\nyou> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if line == "/quit":
            break
        if line == "/alerts":
            res = alerts.run_alerts(store, mem, forecast.get_forecast(args.as_of), ignore_quiet=True)
            for m in mem.sent:
                print("\nbot (alert)>", m["text"])
            if not res["sent"]:
                print("(no alert would be sent:", [s.get("reason") for s in res["skipped"]] or "no followed zone is forecast HIGH", ")")
            mem.sent.clear()
            continue
        for reply in bot.handle(ctx, line):
            print("\nbot>", reply)
