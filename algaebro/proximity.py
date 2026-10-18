"""Who is near a flagged area, and should they be told?

A 'flagged area' is a zone whose tier is HIGH in the current forecast. A person is 'near' it when their
location (a pin dragged on the dashboard during a demo, or a location they texted) is within
PROXIMITY_RADIUS_M of that zone's water.

Cooldowns: live alerts use the database (48 h per person and area). Replay/demo alerts use an in-memory
dict with a short cooldown, so the same alert is not repeated while a pin sits inside a hotspot, but
dragging out and back in after a short while alerts again.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from . import config, geo, messages
from .alerts import in_quiet_hours
from .messenger import describe_error
from .store import Store


def zone_distance_m(lat: float, lon: float, zone: dict) -> float:
    return min(geo.haversine_m(lat, lon, p[1], p[0]) for p in zone["points"])


def nearest_zones(zones: List[dict], lat: float, lon: float) -> List[Tuple[float, dict]]:
    return sorted(((zone_distance_m(lat, lon, z), z) for z in zones), key=lambda t: t[0])


def flagged_near(fc: dict, zones: List[dict], lat: float, lon: float, radius_m: float = None) -> List[Tuple[float, dict]]:
    """Flagged zones within radius of the point, nearest first. Entries are (distance_m, forecast zone entry)."""
    radius = config.PROXIMITY_RADIUS_M if radius_m is None else radius_m
    by_id = {z["id"]: z for z in zones}
    out = []
    for fz in fc["zones"]:
        if fz["tier"] != "high" or fz.get("low_confidence") or fz["id"] not in by_id:
            continue
        d = zone_distance_m(lat, lon, by_id[fz["id"]])
        if d <= radius:
            out.append((d, fz))
    return sorted(out, key=lambda t: t[0])


def check_user(store: Store, messenger, fc: dict, zones: List[dict], user, now: Optional[datetime] = None,
               memo: Optional[Dict[Tuple[str, str], datetime]] = None, ignore_quiet: bool = False,
               dry_run: bool = False, demo: bool = False) -> dict:
    """Alert one person if they are near a flagged area. Returns {'sent': [...], 'skipped': [...]}.

    demo=True means the location is a simulated pin: short cooldown, a visible [DEMO] tag on the message,
    quiet hours ignored, and nothing is written to the real 48-hour alert history.
    """
    now = now or datetime.now(timezone.utc)
    result = {"sent": [], "skipped": []}
    uid = user["user_id"]
    who = {"user": uid}

    if user["alerts_off"]:
        result["skipped"].append({**who, "reason": "alerts are off for this person"})
        return result
    if user["lat"] is None or user["lon"] is None:
        result["skipped"].append({**who, "reason": "no location yet"})
        return result
    if not fc.get("in_season"):
        result["skipped"].append({**who, "reason": "outside bloom season"})
        return result
    replay = bool(fc.get("replay"))
    simulated = replay or demo          # short cooldown, never touches the real alert history
    if not simulated and not ignore_quiet and in_quiet_hours(now):
        result["skipped"].append({**who, "reason": "quiet hours"})
        return result

    near = flagged_near(fc, zones, user["lat"], user["lon"])
    if not near:
        result["skipped"].append({**who, "reason": f"no flagged area within {config.PROXIMITY_RADIUS_M} m"})
        return result

    dist, z = near[0]
    who["zone"] = z["id"]
    if simulated:
        last = (memo or {}).get((uid, z["id"]))
        if last and (now - last).total_seconds() < config.DEMO_COOLDOWN_S:
            result["skipped"].append({**who, "reason": "alerted a moment ago"})
            return result
    else:
        last = store.last_alert(uid, z["id"])
        if last and (now - last) < timedelta(hours=config.ALERT_COOLDOWN_H):
            result["skipped"].append({**who, "reason": "alerted recently"})
            return result

    text = messages.alert_text(z, fc, distance_m=dist, demo=demo)
    reachable = bool(user["space_id"] and user["agent_id"]) and messenger is not None
    note = ""
    delivered = False
    if dry_run:
        note = "dry run"
    elif not reachable:
        note = "not sent: Photon not connected" if messenger is None else "not sent: no way to reach this person yet"
    else:
        try:
            messenger.send(user["space_id"], user["agent_id"], text,
                           key=f"prox:{uid}:{z['id']}:{fc['as_of']}:{int(now.timestamp())}")
            delivered = True
        except Exception as e:
            note = f"send failed: {describe_error(e)}"
    if not dry_run:
        store.log_feed("proximity", user["handle"], z["id"], z["name"], round(dist), text, delivered, note, simulated, when=now)
        if delivered or messenger is None:  # offline demo still counts as "alerted" so the feed does not repeat
            if simulated:
                if memo is not None:
                    memo[(uid, z["id"])] = now
            else:
                store.log_alert(uid, z["id"], z["tier"], fc["target_date"], when=now)
    entry = {**who, "text": text, "distance_m": round(dist), "delivered": delivered, "note": note}
    counts_as_sent = delivered or dry_run or messenger is None   # with no Photon at all, the demo still shows the alert in the feed
    if counts_as_sent:
        result["sent"].append(entry)
    else:
        result["skipped"].append({**who, "reason": note or "not delivered"})
    return result


def check_everyone(store: Store, messenger, fc: dict, zones: List[dict], now=None, memo=None, ignore_quiet=False,
                   dry_run=False, demo_user: Optional[str] = None) -> dict:
    total = {"sent": [], "skipped": []}
    for u in store.list_users():
        r = check_user(store, messenger, fc, zones, u, now=now, memo=memo, ignore_quiet=ignore_quiet, dry_run=dry_run,
                       demo=(demo_user is not None and u["user_id"] == demo_user))
        total["sent"] += r["sent"]
        total["skipped"] += r["skipped"]
    return total
