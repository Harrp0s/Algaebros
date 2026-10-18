"""The brain of the iMessage bot. Pure Python: text in, list of reply texts out.

It knows nothing about Photon, so it can be tested and demoed without any phone.
Rule-based on purpose: a safety-adjacent bot should be predictable and never improvise.

Everyday use is deliberately tiny: send a location, send a photo, STATUS, WHY, STOP, START, DELETE PHOTOS.
Older zone commands (subscribe, zones, mine, report) still work but are not advertised.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Callable, List, Optional, Tuple

from . import config, geo, messages, proximity
from .store import Store, UserCtx

STOP_WORDS = {"stop", "unsubscribe", "cancel", "quit", "end", "unfollow"}
START_WORDS = {"start", "resume", "on"}
GREETINGS = {"hi", "hello", "hey", "yo"}


class Bot:
    def __init__(self, store: Store, forecast_fn: Callable[[], dict], zones: List[dict]):
        self.store = store
        self.forecast_fn = forecast_fn
        self.zones = zones

    # ---------------------------------------------------------------- helpers
    def resolve_zone(self, token: str) -> Optional[dict]:
        t = token.strip().lower()
        if not t:
            return None
        m = re.fullmatch(r"(?:zone\s*|z)?0*(\d{1,2})", t)
        if m:
            n = int(m.group(1))
            return next((z for z in self.zones if z["number"] == n), None)
        for z in self.zones:
            if t in z["name"].lower():
                return z
        return None

    def _zone_numbers_in(self, text: str) -> List[dict]:
        found = []
        for n in re.findall(r"\d{1,2}", text):
            z = self.resolve_zone(n)
            if z and z not in found:
                found.append(z)
        return found

    @staticmethod
    def _fc_zone(fc: dict, zone_id: str) -> dict:
        return next(z for z in fc["zones"] if z["id"] == zone_id)

    def _user_point(self, user_id: str) -> Optional[Tuple[float, float]]:
        u = self.store.get_user(user_id)
        if u is not None and u["lat"] is not None and u["lon"] is not None:
            return u["lat"], u["lon"]
        return None

    def _focus_zone(self, user_id: str, fc: dict) -> Optional[Tuple[Optional[float], dict]]:
        """The flagged area that matters to this person: the nearest one within reach, else the nearest flagged one."""
        flagged = [z for z in fc["zones"] if z["tier"] == "high" and not z.get("low_confidence")]
        if not flagged:
            return None
        pt = self._user_point(user_id)
        if pt is None:
            return None, max(flagged, key=lambda z: z.get("last_obs", 0))
        near = proximity.flagged_near(fc, self.zones, pt[0], pt[1], radius_m=1e9)
        d, z = near[0]
        return d, z

    # ---------------------------------------------------------------- main entry
    def handle(self, ctx: UserCtx, text: Optional[str]) -> List[str]:
        is_new = self.store.get_user(ctx.user_id) is None
        self.store.upsert_user(ctx)
        replies, recognized = self._route(ctx, text)
        if is_new:
            # first contact: a short welcome, then the answer only if they actually asked for something
            return [messages.welcome_text()] + (replies if recognized and not self._is_help(text) else [])
        return replies

    @staticmethod
    def _is_help(text: Optional[str]) -> bool:
        t = (text or "").strip().lower()
        return t in ("", "help", "menu", "?", "about") or t in GREETINGS

    def _route(self, ctx: UserCtx, text: Optional[str]) -> Tuple[List[str], bool]:
        raw = (text or "").strip()
        if not raw:
            return [messages.help_text()], True
        low = raw.lower()
        cmd, _, rest = low.partition(" ")
        rest = rest.strip()

        if low in ("stop alerts", "turn off") or cmd in STOP_WORDS:
            self.store.clear_subs(ctx.user_id)
            self.store.set_alerts_off(ctx.user_id, True)
            return [messages.stopped()], True
        if cmd in START_WORDS and not rest:
            self.store.set_alerts_off(ctx.user_id, False)
            return [messages.started()], True
        if cmd in ("help", "menu", "?", "about") or cmd in GREETINGS:
            return [messages.help_text()], True
        if cmd == "why":
            return self._why(ctx), True
        if cmd in ("status", "check", "forecast"):
            return self._status(ctx, rest), True
        if cmd == "delete":
            return self._delete_photos(ctx), True
        if cmd in ("zones", "list"):
            return [messages.zones_list(self.forecast_fn())], True
        if cmd in ("subscribe", "sub", "follow", "alert", "alerts"):
            return self._subscribe(ctx, rest), True
        if cmd in ("mine", "my", "following"):
            return self._mine(ctx), True
        if cmd == "report":
            return self._report(ctx, rest), True

        loc = geo.parse_location(raw)
        if loc:
            return self._near(ctx, loc[0], loc[1]), True
        return [messages.didnt_get_that()], False

    # ---------------------------------------------------------------- commands
    def _status(self, ctx: UserCtx, arg: str) -> List[str]:
        fc = self.forecast_fn()
        if arg:  # legacy: status 4
            z = self.resolve_zone(arg)
            if not z:
                return [messages.not_a_zone(arg[:30])]
            return [messages.zone_status(self._fc_zone(fc, z["id"]), fc)]
        flagged_names = [z["name"] for z in fc["zones"] if z["tier"] == "high" and not z.get("low_confidence")]
        near = None
        pt = self._user_point(ctx.user_id)
        if pt is not None:
            d, zr = proximity.nearest_zones(self.zones, pt[0], pt[1])[0]
            fz = self._fc_zone(fc, zr["id"])
            near = (fz["name"], d, fz["tier"] == "high" and not fz.get("low_confidence") and d <= config.PROXIMITY_RADIUS_M)
        return [messages.status_text(fc, near, flagged_names)]

    def _why(self, ctx: UserCtx) -> List[str]:
        fc = self.forecast_fn()
        focus = self._focus_zone(ctx.user_id, fc)
        if focus is None:
            return [messages._tag(fc) + "Nothing is flagged in the latest satellite reading, so there's nothing to explain. "
                    "That is not a safety check."]
        return [messages.why_text(focus[1], fc)]

    def _delete_photos(self, ctx: UserCtx) -> List[str]:
        paths = self.store.delete_user_photos(ctx.user_id)
        for p in paths:
            try:
                Path(p).unlink(missing_ok=True)
            except OSError:
                pass
        return [messages.photos_deleted(len(paths))]

    def _near(self, ctx: UserCtx, lat: float, lon: float) -> List[str]:
        ranked = proximity.nearest_zones(self.zones, lat, lon)
        self.store.set_location(ctx.user_id, lat, lon, "shared")
        d, z = ranked[0]
        self.place_pending_photos(ctx.user_id, lat, lon)
        far = (z["name"], d / 1000.0) if d > 2000 else None
        return [messages.location_saved(far)]

    def place_pending_photos(self, user_id: str, lat: float, lon: float) -> int:
        """Photos sent before the bot knew where the person was get placed once it does."""
        ranked = proximity.nearest_zones(self.zones, lat, lon)
        d, z = ranked[0]
        if d > config.ALERT_RADIUS_M:
            return 0
        n = 0
        for row in self.store.unplaced_photos(user_id):
            self.store.place_photo(row["id"], z["id"], lat, lon)
            n += 1
        return n

    # ---------------------------------------------------------------- older commands (not advertised)
    def _subscribe(self, ctx: UserCtx, arg: str) -> List[str]:
        if arg.strip() in ("all", "everything"):
            targets = list(self.zones)
        else:
            targets = self._zone_numbers_in(arg)
            if not targets:
                z = self.resolve_zone(arg)
                targets = [z] if z else []
        if not targets:
            return [messages.not_a_zone(arg[:30]) if arg else "Which area? Text STATUS to see what's flagged, or send your location."]
        self.store.set_alerts_off(ctx.user_id, False)
        for z in targets:
            self.store.add_sub(ctx.user_id, z["id"], "manual")
        return [messages.subscribed([z["name"] for z in targets])]

    def _mine(self, ctx: UserCtx) -> List[str]:
        subs = self.store.get_subs(ctx.user_id)
        if not subs:
            return ["You're not following any areas. Send your location to get alerts near you."]
        fc = self.forecast_fn()
        lines = ["You're following:"]
        for zid in subs:
            z = self._fc_zone(fc, zid)
            lines.append(f"{z['name']}: {messages.TIER_WORD.get(z['tier'], z['tier'])}")
        return ["\n".join(lines)]

    def _report(self, ctx: UserCtx, arg: str) -> List[str]:
        first, _, note = arg.partition(" ")
        zone = self.resolve_zone(first) if first else None
        if not zone:
            note = arg
        if not note.strip():
            return ["Tell me what you saw, for example: report green scum by the dock."]
        self.store.add_report(ctx.user_id, zone["id"] if zone else None, note.strip())
        return [messages.report_logged(zone["name"] if zone else None)]
