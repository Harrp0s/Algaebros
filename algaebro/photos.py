"""Citizen photos: take what a person sent through Photon, make it safe to store and show, file it by place.

* HEIC (iPhone) is converted to JPEG so any browser can show it.
* The image is re-encoded, which drops EXIF metadata (including GPS and camera details). Only the place the
  bot already knew (their pin or shared location) is recorded.
* Photos are stored on this computer (data/photos) with a row in the SQLite database.
* Each photo is sent to Grok (xAI) once, already stripped of metadata, to check it is a real photo of the water and
  to describe what the water looks like (screening.py). Confident trolls are rejected (kept, hidden, reversible);
  unsure ones wait for a person in the dashboard's Photos panel. Duplicates and more than MAX_PER_DAY a day are refused.
* Photos never change a forecast or trigger an alert; verified ones show on the map as community reports.
"""
from __future__ import annotations

import hashlib
import io
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

from . import config, messages, proximity, screening
from .store import Store

MAX_SIDE = 1600
MAX_PER_DAY = 10            # per person; more than this in 24 hours is refused before anything is stored or sent to Grok
MIN_SIDE = 200              # thumbnails, stickers and tiny images are not useful reports


class Refused(ValueError):
    """A photo we deliberately do not store (duplicate, daily limit, too small); the message says why."""
try:  # iPhone photos are usually HEIC; without this converter only JPEG/PNG work
    import pillow_heif

    pillow_heif.register_heif_opener()
    HEIC_OK = True
except Exception:  # pragma: no cover - depends on the installed packages
    HEIC_OK = False


def normalise_image(data: bytes) -> bytes:
    """Any common photo format -> upright JPEG, at most MAX_SIDE px, with metadata removed."""
    from PIL import Image, ImageOps

    img = Image.open(io.BytesIO(data))
    img = ImageOps.exif_transpose(img)           # honour the phone's rotation before the metadata is dropped
    img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    img.save(out, format="JPEG", quality=85)     # no exif= argument: metadata is not carried over
    return out.getvalue()


def save_photo(store: Store, user_id: str, data: bytes, zones: List[dict], attachment_id: str = "") -> dict:
    """Store one photo (status 'pending'). Returns {'id', 'zone_id', 'zone_name', 'placed', 'jpeg'}.

    Raises ValueError if it is not an image, and Refused for duplicates, tiny images and the daily limit."""
    try:
        jpeg = normalise_image(data)
    except Exception as e:
        raise ValueError(f"not a readable image: {type(e).__name__}") from e
    digest = hashlib.sha256(jpeg).hexdigest()
    if store.photo_hash_known(digest):
        raise Refused("duplicate")
    from PIL import Image

    if min(Image.open(io.BytesIO(jpeg)).size) < MIN_SIDE:
        raise Refused("too small")
    since = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(timespec="seconds")
    if store.photos_since(user_id, since) >= MAX_PER_DAY:
        raise Refused("limit")
    config.PHOTO_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PHOTO_DIR / f"{uuid.uuid4().hex}.jpg"
    path.write_bytes(jpeg)

    zone_id = zone_name = None
    lat = lon = None
    u = store.get_user(user_id)
    if u is not None and u["lat"] is not None and u["lon"] is not None:
        d, z = proximity.nearest_zones(zones, u["lat"], u["lon"])[0]
        if d <= config.ALERT_RADIUS_M:
            zone_id, zone_name, lat, lon = z["id"], z["name"], u["lat"], u["lon"]
    pid = store.add_photo(user_id, zone_id, lat, lon, str(path), attachment_id, sha256=digest)
    return {"id": pid, "zone_id": zone_id, "zone_name": zone_name, "placed": zone_id is not None, "jpeg": jpeg}


def screen_photo(store: Store, photo_id: int, jpeg: bytes, place: str = "", screener=None) -> dict:
    """Ask Grok about one stored photo and record the verdict. Returns the decision."""
    screener = screener or screening.screen
    d = screener(jpeg, place)
    store.set_photo_review(photo_id, d["status"], d["reason"], d["bloom"], d["description"],
                           "none" if d["reason"].startswith("not screened") else "grok")
    return d


def screen_pending(store: Store, zones: List[dict], screener=None, limit: int = 50) -> dict:
    """Run every pending (or pre-screening 'unverified') photo through Grok. For the Photos panel button."""
    names = {z["id"]: z.get("site_name") or z["name"] for z in zones}
    counts = {"verified": 0, "rejected": 0, "review": 0, "missing": 0}
    for row in store.photos_with_status(("pending", "unverified"), limit=limit):
        try:
            with open(row["path"], "rb") as fh:
                jpeg = fh.read()
        except OSError:
            counts["missing"] += 1
            continue
        d = screen_photo(store, row["id"], jpeg, names.get(row["zone_id"] or "", ""), screener)
        counts[d["status"]] = counts.get(d["status"], 0) + 1
    return counts


def handle_photo_message(store: Store, messenger, zones: List[dict], inc, screener=None) -> str:
    """Reply text for an incoming photo message: download, store, ask Grok, and say what happened."""
    if inc.attachment_id and store.photo_attachment_known(inc.attachment_id):
        return messages.photo_in_review(None)  # a retry of one we already stored
    try:
        data = messenger.download_attachment(inc.attachment_id)
        info = save_photo(store, inc.user_id, data, zones, inc.attachment_id)
    except Refused as e:
        return messages.photo_refused(str(e))
    except Exception as e:
        print(f"[photos] could not store photo: {type(e).__name__}: {str(e)[:160]}")
        return messages.photo_failed()
    site = next((z.get("site_name") or z["name"] for z in zones if z["id"] == info["zone_id"]), "")
    d = screen_photo(store, info["id"], info["jpeg"], site, screener)
    if d["status"] == "rejected":
        return messages.photo_rejected()
    if d["status"] == "verified":
        return messages.photo_verified(info["zone_name"], d["bloom"])
    return messages.photo_in_review(info["zone_name"])
