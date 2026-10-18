"""All user-facing wording lives here so it can be reviewed in one place.

Style: short and plain. Wording rules that never change:
  * Say what it is: a satellite reading, not a toxin test or a safety check. Never say water is safe.
  * Point to official sources.
  * If the data is synthetic or a replay, the message itself says so.
"""
from __future__ import annotations

from typing import List, Optional

from . import config

TIER_WORD = {"high": "high chlorophyll reading (possible algae)", "elevated": "watch", "low": "nothing unusual", "unknown": "unknown"}


def _tag(fc: dict) -> str:
    tags = []
    if fc.get("data_source") == "synthetic":
        tags.append("SYNTHETIC DEMO DATA, NOT REAL")
    if fc.get("replay"):
        tags.append(f"REPLAY of {fc['as_of']}")
    return f"[{' | '.join(tags)}]\n" if tags else ""


def _age_phrase(z: dict) -> str:
    age = z.get("obs_age_days")
    if age is None:
        return "a recent satellite reading"
    return "a satellite reading from today" if age <= 0 else f"a satellite reading {age} day{'s' if age != 1 else ''} old"


def _distance_phrase(distance_m: Optional[float]) -> str:
    if distance_m is None:
        return ""
    if distance_m < 50:
        return " (right here)"
    return f" (~{int(round(distance_m / 10.0) * 10)} m from you)"


def place(z: dict) -> str:
    return z["name"]


# ---------------------------------------------------------------- conversation
def welcome_text() -> str:
    return (
        "Hi, I'm Algaebro. I watch the Charles River for possible algal bloom activity "
        "and text you when you're near some.\n"
        "Send your location to get alerts near you, or a photo of the river. Text HELP for commands."
    )


def help_text() -> str:
    return (
        "Algaebro: alerts for possible algal bloom activity on the Charles.\n"
        "Send your location: get alerts near you\n"
        "Send a photo of the water: checked by an AI (xAI), then shown on the map\n"
        "STATUS, WHY, STOP, START, DELETE PHOTOS\n"
        f"Satellite-based, not a safety test. Official info: {config.OFFICIAL_INFO}"
    )


def location_saved(far_from: Optional[tuple] = None) -> str:
    if far_from:
        name, km = far_from
        return (f"Got your location. You're about {km:.1f} km from the stretch I watch (nearest: {name}). "
                "I'll message you if you get close to possible algal bloom activity.")
    return "Got your location. I'll message you if you're near possible algal bloom activity."


def alert_text(z: dict, fc: dict, distance_m: Optional[float] = None, demo: bool = False) -> str:
    return (
        _tag(fc)
        + ("[DEMO: simulated location]\n" if demo else "")
        + f"ALERT: High chlorophyll reading (possible algae) near {place(z)}{_distance_phrase(distance_m)}.\n"
        + f"Based on {_age_phrase(z)}. Avoid green or scummy water and keep pets out. This is not a safety test.\n"
        + "Send a photo to help (no flash, sun behind you). WHY for details, STOP to quit."
    )


def _track_sentence(z: dict) -> str:
    """What happened in past years after a zone looked like this (measured, not modelled)."""
    tr = z.get("track")
    if not tr or not tr.get("n") or tr.get("next_high") is None:
        return ""
    hi, el = tr["next_high"], tr["next_elevated_or_high"]
    if z["tier"] == "high":
        return (f"In past years, about {round(hi * 10)} in 10 readings this high were still flagged a week later, "
                f"and about {round(el * 10)} in 10 were at least elevated (from {tr['n']} past readings).")
    if z["tier"] == "elevated":
        return f"In past years, about {round(hi * 100)}% of readings at this level reached the flag a week later (from {tr['n']} past readings)."
    return f"In past years, about {round(hi * 100)}% of readings this low reached the flag a week later (from {tr['n']} past readings)."


def why_text(z: dict, fc: dict) -> str:
    high = fc["thresholds"]["high"]
    how = ("Weather did not improve forecasts in our tests, so this uses the latest satellite reading only."
           if fc.get("method") == "persistence" else "This blends the latest reading with recent weather.")
    return (
        _tag(fc)
        + f"Why: {place(z)} was flagged from {_age_phrase(z)}. Its chlorophyll index is {z['last_obs']:+.2f}; "
        + f"the flag starts at {high:+.2f} (the top 10% of this river's past readings).\n"
        + _track_sentence(z) + "\n"
        + how + f" It is not a toxin test or a safety check. Official info: {config.OFFICIAL_INFO}"
    ).replace("\n\n", "\n")


def status_text(fc: dict, near: Optional[tuple] = None, flagged_names: Optional[List[str]] = None) -> str:
    """near = (zone_name, distance_m, is_flagged) for the closest watched area to the person, if their location is known."""
    lines = []
    if not fc.get("in_season"):
        lines.append("Outside the usual bloom season, so alerts are paused.")
    if near:
        name, dist, flagged = near
        if flagged:
            lines.append(f"You're ~{int(round(dist / 10.0) * 10)} m from {name}: high chlorophyll reading (possible algae).")
        else:
            lines.append(f"Nothing flagged within {config.PROXIMITY_RADIUS_M} m of you.")
    if flagged_names:
        lines.append("Flagged now: " + ", ".join(flagged_names) + ".")
    elif not near or not near[2]:
        lines.append("Nothing flagged in the latest satellite reading.")
    if not near:
        lines.append("Send your location to get alerts near you.")
    lines.append("Satellite-based, not a safety check.")
    return _tag(fc) + "\n".join(lines)


def stopped() -> str:
    return "Alerts off. Text START to turn them back on."


def started() -> str:
    return "Alerts on. Send your location to get alerts near you."


def didnt_get_that() -> str:
    return "Sorry, I didn't get that. Text HELP for commands."


# ---------------------------------------------------------------- photos
def photo_saved(place_name: Optional[str]) -> str:
    if place_name:
        return f"Thanks, saved as an unverified report near {place_name}. Text DELETE PHOTOS to remove yours."
    return "Thanks, saved. Send your location so I can place it on the map."


def photo_verified(place_name: Optional[str], bloom: str) -> str:
    where = f" near {place_name}" if place_name else ""
    seen = " It looks like it may show algae, so thanks for flagging it." if bloom == "bloom" else ""
    tail = "" if place_name else " Send your location so I can place it on the map."
    return (f"Thanks, your photo was added to the map{where}.{seen} It's a community report, not a safety test."
            f"{tail} Text DELETE PHOTOS to remove yours.")


def photo_in_review(place_name: Optional[str]) -> str:
    tail = "" if place_name else " Send your location so I can place it on the map."
    return f"Thanks, got your photo. It will be checked before it appears on the map.{tail}"


def photo_rejected() -> str:
    return "Thanks, but this doesn't look like a photo of the water, so it wasn't added. Send a photo of the lake or river itself."


def photo_refused(why: str) -> str:
    return {"duplicate": "I already have that photo, thanks.",
            "too small": "That image is too small to be useful. Please send the full photo.",
            "limit": "Thanks! You've sent a lot of photos today, so I'll take more tomorrow."}.get(why, photo_failed())


def photo_failed() -> str:
    return "Sorry, I couldn't open that photo. Please try sending it again."


def photos_deleted(n: int) -> str:
    return f"Deleted {n} photo{'s' if n != 1 else ''}." if n else "You have no saved photos."


# ---------------------------------------------------------------- legacy commands (still work, not advertised)
def zones_list(fc: dict) -> str:
    lines = ["Watched areas, west to east:"]
    for z in fc["zones"]:
        lines.append(f"{z['number']}. {z['name']}: {TIER_WORD.get(z['tier'], z['tier'])}")
    return _tag(fc) + "\n".join(lines)


def zone_status(z: dict, fc: dict) -> str:
    if z["tier"] == "unknown":
        return f"{place(z)}: no reading right now ({z.get('reason', 'not enough data')})."
    return _tag(fc) + f"{place(z)}: {TIER_WORD.get(z['tier'], z['tier'])}. Based on {_age_phrase(z)}. WHY for details."


def subscribed(zone_names: List[str]) -> str:
    return f"Following {', '.join(zone_names)}. I'll text you if it's flagged. STOP to quit."


def unsubscribed(n: int) -> str:
    return stopped()


def not_a_zone(text: str) -> str:
    return f"I couldn't find '{text}'. Text STATUS to see what's flagged."


def report_logged(zone_name: Optional[str]) -> str:
    where = f" near {zone_name}" if zone_name else ""
    return f"Logged as an unverified report{where}. If it looks like a bloom, also tell {config.OFFICIAL_INFO}."
