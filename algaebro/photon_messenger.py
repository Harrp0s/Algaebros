"""Photon 2.0 (Beta platform) iMessage transport, using the official Python client `photonhq-api`.

How this follows Photon's docs (https://photon.codes/docs/beta/get-started/imessage):
  * Receive : poll list_messages(direction='received') every few seconds. Their docs call polling a
              connection test, not a production inbox, and recommend the Socket.IO live-events feed
              for long-running agents. For a hackathon, polling every ~3 s is fine; we back off on 429.
  * Send    : send_message(spaceId, senderId, content). A reply sets parentMessageId.
              A proactive alert omits parentMessageId and reuses the space and agent ids we saved
              when the person first texted us.
  * Rules   : a shared line CANNOT send the first message. People must text the agent number first
              (that is also our opt-in). Only handles you added to the project can reach a shared line.

UNTESTED against the live service: it was tested against a local fake that returns Photon's own
response models (tests/test_photon_fake.py). Run scripts/00_check_apis.py --photon with your keys first.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pandas as pd

from .messenger import Incoming
from .store import Store


def parse_ts(value: str) -> datetime:
    """Parse any ISO-8601 timestamp (any number of fractional digits, 'Z' or offset) into aware UTC.

    datetime.fromisoformat is picky on Python 3.10/3.11, so we let pandas do it.
    """
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.to_pydatetime()


class PhotonMessenger:
    def __init__(self, api_key: str, project_id: str, store: Store, base_url: Optional[str] = None, client=None):
        from photon_api import Photon  # imported here so the rest of the app works without the package

        kwargs = {"headers": {"Authorization": f"Bearer {api_key}"}}
        if base_url:
            kwargs["base_url"] = base_url
        if client is not None:
            kwargs["client"] = client
        self.photon = Photon(**kwargs)
        self.project_id = project_id
        self.store = store
        self.backoff_until = 0.0
        self.threaded_replies = True  # switched off automatically if Photon refuses threaded replies
        self.attachment_wait_s = 3.0  # pause between 'photo not ready yet' retries

    @classmethod
    def from_env(cls, store: Store, base_url: Optional[str] = None) -> "PhotonMessenger":
        key = os.environ.get("PHOTON_API_KEY")
        project = os.environ.get("PHOTON_PROJECT_ID")
        if not key or not project:
            raise RuntimeError("Set PHOTON_API_KEY and PHOTON_PROJECT_ID (see .env.example).")
        return cls(key, project, store, base_url=base_url)

    # ------------------------------------------------------------------ receive
    def poll(self) -> List[Incoming]:
        from photon_api import ApiError
        from photon_api.generated.models import AttachmentContent, TextContent, UserSnapshot
        from photon_api.rpc_generated import ListMessagesInput

        if time.monotonic() < self.backoff_until:
            return []
        since = self.store.kv_get("photon_since")
        if not since:
            since = (datetime.now(timezone.utc) - timedelta(minutes=10)).isoformat(timespec="milliseconds").replace("+00:00", "Z")

        out: List[Incoming] = []
        newest = None
        query = {"createdSince": since, "direction": "received", "order": "asc", "pageSize": 100, "platform": "imessage"}
        try:
            while True:
                page = self.photon.projects.list_messages(
                    ListMessagesInput.model_validate({"path": {"id": self.project_id}, "query": query})
                )
                for item in page.items:
                    m = item.root.root
                    if m.platform != "imessage" or m.status != "received":
                        continue
                    if not isinstance(m.receivedBy, UserSnapshot):
                        continue
                    sender, agent, space = m.sentBy.root, m.receivedBy.root, m.space.root
                    if sender.kind != "human" or agent.kind != "agent" or space.kind != "direct":
                        continue
                    content = m.content
                    text = content.text if isinstance(content, TextContent) else None
                    attachment_id = content.attachmentId if isinstance(content, AttachmentContent) else None
                    created = parse_ts(m.createdAt)
                    newest = created if newest is None or created > newest else newest
                    out.append(
                        Incoming(
                            message_id=m.id, user_id=sender.id, handle=sender.handle, text=text,
                            space_id=space.id, agent_id=agent.id, created_at=created,
                            attachment_id=attachment_id,
                        )
                    )
                token = page.nextPageToken
                if not isinstance(token, str):
                    break
                query["pageToken"] = token
        except ApiError as e:
            if getattr(e, "status", None) == 429:
                self.backoff_until = time.monotonic() + 30
                print("[photon] rate limited (429); backing off for 30 s")
                return []
            raise
        if newest is not None:
            # Re-read 2 minutes of overlap next time: a message whose reply failed gets retried, and the
            # 'seen' table (filled only after a reply succeeds or we give up) removes duplicates.
            self.store.kv_set("photon_since", (newest - timedelta(seconds=120)).isoformat(timespec="milliseconds").replace("+00:00", "Z"))
        return out

    # ------------------------------------------------------------------ send
    def _post(self, space_id: str, agent_id: str, text: str, parent_id: Optional[str], key: str) -> None:
        from photon_api.generated.models import SendMessageRequest, SendTextContent
        from photon_api.rpc_generated import SendMessageHeader, SendMessageInput, SendMessagePath

        body = dict(spaceId=space_id, senderId=agent_id, content=SendTextContent(type="text", format="plain", text=text))
        if parent_id:
            body["parentMessageId"] = parent_id
        self.photon.projects.send_message(
            SendMessageInput(
                path=SendMessagePath(id=self.project_id),
                headers=SendMessageHeader.model_validate({"idempotency-key": key}),
                body=SendMessageRequest(**body),
            )
        )

    def send(self, space_id: str, agent_id: str, text: str, *, parent_id: Optional[str] = None, key: Optional[str] = None) -> None:
        """Send a message. A threaded reply is tried first; if Photon says this Space cannot take replies
        (SPACE_NOT_SENDABLE), resend once as a normal message and remember that for next time."""
        from photon_api import ApiError

        idem = key or f"msg:{space_id}:{abs(hash(text))}"
        use_parent = parent_id if self.threaded_replies else None
        try:
            self._post(space_id, agent_id, text, use_parent, idem)
        except ApiError as e:
            refused = getattr(e, "status", None) == 409 and "SPACE_NOT_SENDABLE" in (getattr(e, "raw_body", "") or "")
            if not (use_parent and refused):
                raise
            print("[photon] Photon refused a threaded reply in this chat; sending as a normal message from now on.")
            self.threaded_replies = False
            self._post(space_id, agent_id, text, None, f"{idem}:plain")

    # ------------------------------------------------------------------ photos
    def download_attachment(self, attachment_id: str, tries: int = 4, wait_s: Optional[float] = None, sleep=time.sleep) -> bytes:
        """Fetch a photo's bytes. Photon answers ATTACHMENT_NOT_READY for a few seconds after a photo arrives, so retry."""
        from photon_api import ApiError
        from photon_api.rpc_generated import DownloadAttachmentInput

        last = None
        for attempt in range(tries):
            try:
                res = self.photon.projects.download_attachment(
                    DownloadAttachmentInput.model_validate({"path": {"id": self.project_id, "attachmentId": attachment_id}})
                )
                return bytes(getattr(res, "data", res))   # the client hands back the bytes themselves; accept a wrapper too
            except ApiError as e:
                last = e
                if "ATTACHMENT_NOT_READY" in (getattr(e, "raw_body", "") or "") and attempt < tries - 1:
                    sleep(self.attachment_wait_s if wait_s is None else wait_s)
                    continue
                raise
        raise last
