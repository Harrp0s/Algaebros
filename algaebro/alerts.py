"""Decide who gets a 'heat wave style' alert, and send it.

An alert goes to a subscriber when a zone they follow is forecast HIGH, the target date is in
bloom season, it is not quiet hours, and they have not been alerted about that zone recently.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Optional, Set, Tuple
from zoneinfo import ZoneInfo

from . import config, messages
from .messenger import describe_error
from .store import Store


def in_quiet_hours(now_utc: datetime) -> bool:
    hour = now_utc.astimezone(ZoneInfo(config.TIMEZONE)).hour
    return hour >= config.QUIET_START_H or hour < config.QUIET_END_H


def run_alerts(store: Store, messenger, fc: dict, now: Optional[datetime] = None,
               dry_run: bool = False, ignore_quiet: bool = False,
               only_user: Optional[str] = None, sent_memo: Optional[Set[Tuple[str, str, str]]] = None) -> dict:
    """Returns {'sent': [...], 'skipped': [...]}. With dry_run nothing is sent or logged.

    only_user : alert just this person (used right after they subscribe).
    sent_memo : for REPLAY forecasts only, an in-memory set remembering (user, zone, target date) already
                alerted, so a demo does not repeat the same alert every check. Live alerts use the database cooldown.
    """
    now = now or datetime.now(timezone.utc)
    result = {"sent": [], "skipped": []}

    if not fc.get("in_season"):
        result["skipped"].append({"reason": "target date is outside bloom season"})
        return result
    if not ignore_quiet and in_quiet_hours(now):
        result["skipped"].append({"reason": f"quiet hours ({config.QUIET_START_H}:00 to {config.QUIET_END_H}:00 local)"})
        return result

    replay = bool(fc.get("replay"))
    for z in fc["zones"]:
        if z["tier"] != "high" or z.get("low_confidence"):   # noisy narrow water: shown on the map, never alerted
            continue
        for u in store.subscribers_for_zone(z["id"]):
            if only_user is not None and u["user_id"] != only_user:
                continue
            who = {"user": u["user_id"], "zone": z["id"]}
            if not u["space_id"] or not u["agent_id"]:
                result["skipped"].append({**who, "reason": "no way to reach this user yet"})
                continue
            last = store.last_alert(u["user_id"], z["id"])
            if not replay and last and (now - last) < timedelta(hours=config.ALERT_COOLDOWN_H):
                result["skipped"].append({**who, "reason": "alerted recently"})
                continue
            memo_key = (u["user_id"], z["id"], fc["target_date"])
            if replay and sent_memo is not None and memo_key in sent_memo:
                result["skipped"].append({**who, "reason": "already sent in this replay"})
                continue
            text = messages.alert_text(z, fc)
            key = f"alert:{z['id']}:{fc['target_date']}:{u['user_id']}" + (":replay" if replay else "")
            if not dry_run:
                try:
                    messenger.send(u["space_id"], u["agent_id"], text, key=key)
                except Exception as e:
                    result["skipped"].append({**who, "reason": f"send failed: {describe_error(e)}"})
                    store.log_feed("follow", u["handle"], z["id"], z["name"], None, text, False, f"send failed: {describe_error(e)}", replay, when=now)
                    continue
                store.log_feed("follow", u["handle"], z["id"], z["name"], None, text, True, "", replay, when=now)
                if not replay:
                    store.log_alert(u["user_id"], z["id"], z["tier"], fc["target_date"], when=now)
                elif sent_memo is not None:
                    sent_memo.add(memo_key)
            result["sent"].append({**who, "text": text})
    return result
