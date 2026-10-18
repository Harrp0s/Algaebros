"""Photon round-trip test. Run this, then text your Photon number from your phone.

    python scripts/photon_ping.py

It waits up to 3 minutes for a text, replies 'pong', and saves how to reach you (so the bot can
send you alerts later). If the reply lands on your phone, Photon works end to end.
It needs only your .env (PHOTON_API_KEY, PHOTON_PROJECT_ID); no satellite data required.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from algaebro import config
from algaebro.messenger import Incoming, describe_error
from algaebro.run_bot import load_dotenv
from algaebro.store import Store, UserCtx

TIMEOUT_HELP = """
No message arrived. Check, in this order:
  1. You texted the number Photon ASSIGNED to you (not your own number), as an iMessage (blue bubble, not SMS).
  2. On your iPhone: Settings > Apps > Messages > Send & Receive. The number you added in the Photon dashboard
     must be listed there, and you must choose it as the 'start new conversations from' identity.
  3. You started a NEW conversation after assigning the line (an older thread does not count).
  4. The sender you added is typed with +1 and the area code, for example +16175550123.
  5. The dashboard shows a shared iMessage seat on your plan. If not, ask a Photon person at the venue.
  6. In the Photon dashboard, open the iMessage page / API activity and look for the message or an error.
"""


def run_ping(messenger, store, window_s: float = 180.0, sleep=time.sleep, poll_s: float = 3.0) -> int:
    """Wait for one text, answer it, remember the sender. Returns 0 on success, 1 on timeout."""
    deadline = time.time() + window_s
    while time.time() < deadline:
        for inc in messenger.poll():
            if store.already_seen(inc.message_id):
                continue
            store.upsert_user(UserCtx(inc.user_id, inc.handle, inc.space_id, inc.agent_id))
            print(f"Got {inc.text!r} from {inc.handle}.")
            # a fresh idempotency key each run, so a retry after a failed attempt is never refused as a duplicate
            messenger.send(inc.space_id, inc.agent_id, "pong: Algaebro can hear you.",
                           parent_id=inc.message_id, key=f"ping:{inc.message_id}:{int(time.time())}")
            store.mark_seen(inc.message_id)
            print("Reply accepted by Photon. If 'pong' shows up on your phone, Photon works end to end.")
            print("(Accepted means queued: delivery can take a few seconds.)")
            return 0
        sleep(poll_s)
    print(TIMEOUT_HELP)
    return 1


def main() -> int:
    load_dotenv(config.ROOT / ".env")
    from photon_api import ApiError
    from algaebro.photon_messenger import PhotonMessenger

    store = Store()
    try:
        messenger = PhotonMessenger.from_env(store)
        print("Ready. From your phone, text anything (for example: ping) to your Photon number. Waiting up to 3 minutes...")
        return run_ping(messenger, store)
    except ApiError as e:
        print(f"Photon rejected the request: {describe_error(e)}")
        if getattr(e, "status", None) == 409:
            print("409 means 'conflict'. Check your phone first: the reply may already have been delivered (a slow send "
                  "that was retried can be refused as a duplicate). If no pong arrived, send a NEW text and run this again, "
                  "and send me the 'Photon said:' text above.")
        else:
            print("401/403 usually means the API key or its permissions (project:read, project:write) are wrong; "
                  "404 usually means the project ID is wrong.")
        return 1
    except RuntimeError as e:
        print(e)
        return 1


if __name__ == "__main__":
    sys.exit(main())
