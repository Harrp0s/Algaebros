"""Grok looks at each texted photo: is it a real photo of the water, and does the water look like a bloom?

One vision call per photo returns a small JSON verdict. The policy that turns it into a status is plain code here:

  rejected   Grok is confident it is not a real outdoor photo of water (meme, screenshot, selfie, text, unrelated)
             or it is offensive. Kept on disk but hidden, so a person can overturn it in the Photos panel.
  verified   Grok is confident it is a real photo of a lake, pond or river. Shown on the map and in the details panel.
  review     anything in between, or Grok was unavailable: waits for a person.

"Verified" means a plausible photo of water, NOT proof it was taken at that place or today (metadata is stripped and
there is no reverse image search). The bloom label is what the picture LOOKS like: green water can be duckweed,
pollen, reflections or sediment, and toxins are invisible. Nothing here ever says water is safe, and no photo
changes a forecast or sends an alert.
"""
from __future__ import annotations

import base64
import json
import re
from typing import Callable, Dict, Optional

from . import xai

VERIFY_AT = 0.6          # confidence needed to verify without a person
REJECT_AT = 0.7          # confidence needed to reject without a person
BLOOM_LABELS = {"bloom": "Looks like a bloom", "no_bloom": "No bloom visible", "unclear": "Can't tell from this photo"}
REJECT_KINDS = ("satellite", "map", "meme", "screenshot", "selfie", "text", "unrelated", "indoor", "offensive", "ai_generated")

PROMPT = """You check photos that members of the public text to a community algal-bloom watch for {place}.
Answer with ONLY a JSON object, no other text:
{{"is_water_photo": true|false,      // a real photograph of a lake, pond or river, taken by a person AT the water
                                     // (from the shore, a bridge, a boat or a low drone). Satellite or aerial
                                     // imagery, maps, charts, diagrams and photos of screens are NOT.
  "kind": "water"|"satellite"|"map"|"meme"|"screenshot"|"selfie"|"text"|"unrelated"|"indoor"|"offensive"|"ai_generated",
  "offensive": true|false,           // nudity, gore, hate symbols or harassment
  "confidence": 0.0-1.0,             // how sure you are about is_water_photo
  "bloom": "bloom"|"no_bloom"|"unclear",
  "description": "one short neutral sentence about what the water looks like"}}
"bloom" means visible green or blue-green scum, paint-like streaks or pea-soup water. Use "unclear" when the water
is small in the frame, it is dark, glare hides it, or it could be duckweed, pollen or reflections.
Never say whether the water is safe. Do not describe people in the photo."""


def _data_url(jpeg: bytes) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(jpeg).decode("ascii")


def ask_grok(jpeg: bytes, place: str, chat: Callable = None) -> str:
    chat = chat or xai.chat
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": _data_url(jpeg), "detail": "high"}},
        {"type": "text", "text": PROMPT.format(place=place or "the Charles River and nearby lakes")},
    ]}]
    return chat(msgs, temperature=0.0)


def parse(reply: str) -> Optional[dict]:
    """Pull the JSON object out of Grok's reply (it sometimes wraps it in a code block or a sentence)."""
    m = re.search(r"\{.*\}", reply or "", re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except ValueError:
        return None
    try:
        conf = float(d.get("confidence", 0))
    except (TypeError, ValueError):
        conf = 0.0
    bloom = d.get("bloom") if d.get("bloom") in BLOOM_LABELS else "unclear"
    kind = str(d.get("kind") or ("water" if d.get("is_water_photo") else "unrelated")).lower()
    if kind in REJECT_KINDS:          # the kind wins if the two disagree (e.g. a satellite view 'of water')
        d["is_water_photo"] = False
    return {"is_water_photo": bool(d.get("is_water_photo")), "kind": kind, "offensive": bool(d.get("offensive")),
            "confidence": max(0.0, min(1.0, conf)), "bloom": bloom, "description": str(d.get("description") or "")[:200]}


def decide(v: Optional[dict]) -> Dict[str, str]:
    """Grok's answer -> {'status', 'reason', 'bloom', 'description'} using the thresholds above."""
    if v is None:
        return {"status": "review", "reason": "Grok's answer could not be read", "bloom": "unclear", "description": ""}
    if v["offensive"]:
        return {"status": "rejected", "reason": "offensive content", "bloom": "unclear", "description": ""}
    if not v["is_water_photo"] and v["confidence"] >= REJECT_AT:
        return {"status": "rejected", "reason": f"not a photo of the water ({v['kind']})", "bloom": "unclear", "description": v["description"]}
    if v["is_water_photo"] and v["confidence"] >= VERIFY_AT:
        return {"status": "verified", "reason": f"photo of the water (confidence {v['confidence']:.2f})", "bloom": v["bloom"],
                "description": v["description"]}
    return {"status": "review", "reason": f"Grok was unsure ({v['kind']}, confidence {v['confidence']:.2f})", "bloom": v["bloom"],
            "description": v["description"]}


def screen(jpeg: bytes, place: str = "", chat: Callable = None) -> Dict[str, str]:
    """Full check for one photo. Never raises: if Grok is unavailable the photo simply waits for a person."""
    if chat is None and not xai.configured():
        return {"status": "review", "reason": "not screened: Grok is not set up", "bloom": "unclear", "description": ""}
    try:
        return decide(parse(ask_grok(jpeg, place, chat)))
    except xai.XAIError as e:
        return {"status": "review", "reason": f"not screened: {str(e)[:120]}", "bloom": "unclear", "description": ""}
