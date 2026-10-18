"""Run the iMessage bot: answer texts, and send alerts when a followed zone is forecast HIGH.

    python -m algaebro.run_bot                      # live
    python -m algaebro.run_bot --as-of 2026-07-20   # REPLAY: pretend it is that date (messages say so)
    python -m algaebro.run_bot --once --dry-run     # one pass, send nothing (good first test)
"""
from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from . import alerts, config, forecast, messages, zones as zones_mod
from .bot import Bot
from .messenger import describe_error
from .store import Store, UserCtx


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _loc_state(store: Store, user_id: str):
    u = store.get_user(user_id)
    if u is None:
        return None, 0
    return ((u["lat"], u["lon"]) if u["lat"] is not None else None), int(u["alerts_off"] or 0)


def handle_incoming(store: Store, bot: Bot, messenger, inc, dry_run: bool = False, on_new_subscription=None,
                    max_tries: int = 3, photo_handler=None) -> int:
    """Process one inbound message. Returns how many replies were sent.

    * A message is marked 'seen' only after its replies were sent (or after max_tries failures), so a
      temporary Photon error does not silently lose the message.
    * on_new_subscription(user_id) is called once the reply went out, if this message changed what that person
      should be alerted about (followed a new area, shared a new location, turned alerts back on), so an alert
      for an already-flagged area reaches them now, not at the next periodic check.
    * photo_handler(inc) -> reply text handles photos; it runs once even if the reply has to be retried.
    """
    if store.already_seen(inc.message_id):
        return 0
    tries = int(store.kv_get(f"tries:{inc.message_id}", "0") or 0) + 1
    store.kv_set(f"tries:{inc.message_id}", str(tries))
    ctx = UserCtx(user_id=inc.user_id, handle=inc.handle, space_id=inc.space_id, agent_id=inc.agent_id)

    cached = store.kv_get(f"replies:{inc.message_id}")
    if cached is not None:  # a retry: reuse the first answer so commands like 'delete' or a photo are not applied twice
        replies = json.loads(cached)
        state_before = tuple(json.loads(store.kv_get(f"state_before:{inc.message_id}", "[null, 0, []]")))
    else:
        loc_before, off_before = _loc_state(store, inc.user_id)
        subs_before = sorted(store.get_subs(inc.user_id))
        state_before = (loc_before, off_before, subs_before)
        store.kv_set(f"state_before:{inc.message_id}", json.dumps(state_before))
        if inc.is_text:
            replies = bot.handle(ctx, inc.text)
        elif inc.is_attachment and photo_handler is not None:
            store.upsert_user(ctx)
            replies = [photo_handler(inc)]
        else:  # a reaction or some content we do not use: remember how to reach them, say nothing
            store.upsert_user(ctx)
            replies = []
        store.kv_set(f"replies:{inc.message_id}", json.dumps(replies))

    try:
        for i, text in enumerate(replies):
            print(f"[bot] -> {inc.handle or inc.user_id}: {text[:70].replace(chr(10), ' ')}...")
            if not dry_run:
                messenger.send(inc.space_id, inc.agent_id, text, parent_id=inc.message_id, key=f"reply:{inc.message_id}:{i}")
    except Exception as e:
        print(f"[bot] reply failed (try {tries} of {max_tries}): {describe_error(e)}")
        if tries >= max_tries:
            store.mark_seen(inc.message_id)
            print("[bot] giving up on that message.")
        return 0
    store.mark_seen(inc.message_id)
    if on_new_subscription is not None:
        loc_after, off_after = _loc_state(store, inc.user_id)
        loc_before, off_before, subs_before = state_before
        loc_before = tuple(loc_before) if loc_before else None
        changed = (loc_after != loc_before) or (off_before and not off_after) or (set(store.get_subs(inc.user_id)) - set(subs_before))
        if changed:
            on_new_subscription(inc.user_id)
    return len(replies)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--as-of", type=forecast.parse_as_of, help="REPLAY a past date (YYYY-MM-DD) instead of using today")
    ap.add_argument("--once", action="store_true", help="poll and check alerts once, then exit")
    ap.add_argument("--dry-run", action="store_true", help="do not send anything")
    ap.add_argument("--base-url", help="override the Photon API address (for testing against a fake)")
    ap.add_argument("--ignore-quiet-hours", action="store_true")
    args = ap.parse_args(argv)

    load_dotenv(config.ROOT / ".env")
    store = Store()
    recs = zones_mod.load_zone_records()
    bot = Bot(store, lambda: forecast.get_forecast(args.as_of), recs)

    from .photon_messenger import PhotonMessenger

    messenger = PhotonMessenger.from_env(store, base_url=args.base_url)
    fc = forecast.get_forecast(args.as_of)
    print(f"Algaebro bot running. data={fc['data_source']} method={fc['method']} as_of={fc['as_of']}"
          f"{' (REPLAY)' if fc['replay'] else ''} dry_run={args.dry_run}")
    if fc["data_source"] != "real":
        print("WARNING: this model was not trained on real data. Messages will be tagged accordingly.")

    replay_sent = set() if args.as_of else None   # a replay demo should not repeat the same alert every check
    quiet_off = args.ignore_quiet_hours or bool(args.as_of)

    def alert_user(user_id: str) -> None:
        try:
            res = alerts.run_alerts(store, messenger, forecast.get_forecast(args.as_of), dry_run=args.dry_run,
                                    ignore_quiet=quiet_off, only_user=user_id, sent_memo=replay_sent)
            print(f"[alerts] new subscriber: sent={len(res['sent'])} skipped={[s.get('reason') for s in res['skipped']][:3]}")
        except Exception as e:
            print(f"[alerts] error: {describe_error(e)}")

    last_alert_check = 0.0
    while True:
        try:
            for inc in messenger.poll():
                handle_incoming(store, bot, messenger, inc, dry_run=args.dry_run, on_new_subscription=alert_user)
        except Exception as e:
            print(f"[bot] poll error: {describe_error(e)}")
        if args.once or time.time() - last_alert_check > config.ALERT_CHECK_MINUTES * 60:
            try:
                fc = forecast.get_forecast(args.as_of)
                res = alerts.run_alerts(store, messenger, fc, dry_run=args.dry_run, ignore_quiet=quiet_off, sent_memo=replay_sent)
                print(f"[alerts] sent={len(res['sent'])} skipped={len(res['skipped'])} {[s.get('reason') for s in res['skipped']][:3]}")
            except Exception as e:
                print(f"[alerts] error: {describe_error(e)}")
            last_alert_check = time.time()
        if args.once:
            break
        time.sleep(config.POLL_SECONDS)


if __name__ == "__main__":
    main()
