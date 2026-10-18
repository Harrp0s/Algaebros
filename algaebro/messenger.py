"""The messaging interface the rest of the app uses, plus an in-memory version for tests and demos."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Optional


@dataclass
class Incoming:
    message_id: str
    user_id: str
    handle: str
    text: Optional[str]          # None when the message is not plain text (a photo, etc.)
    space_id: str
    agent_id: str
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    attachment_id: Optional[str] = None   # set when the message is a photo or file

    @property
    def is_text(self) -> bool:
        return self.text is not None

    @property
    def is_attachment(self) -> bool:
        return self.attachment_id is not None


class MemoryMessenger:
    """No network. Queue messages with inject(); sent replies pile up in .sent."""

    def __init__(self):
        self.inbox: List[Incoming] = []
        self.sent: List[dict] = []

    def inject(self, incoming: Incoming) -> None:
        self.inbox.append(incoming)

    def poll(self) -> List[Incoming]:
        out, self.inbox = self.inbox, []
        return out

    def send(self, space_id: str, agent_id: str, text: str, *, parent_id: Optional[str] = None, key: Optional[str] = None) -> None:
        self.sent.append({"space_id": space_id, "agent_id": agent_id, "text": text, "parent_id": parent_id, "key": key})


def describe_error(e: Exception) -> str:
    """One readable line for any error, including Photon's own explanation when it sent one.

    Never includes request headers, so an API key cannot leak into logs or the terminal.
    """
    status = getattr(e, "status", None)
    if status is None:
        return f"{type(e).__name__}: {str(e)[:300]}"
    request_id = getattr(e, "request_id", None) or "unknown"
    body = (getattr(e, "raw_body", "") or "").strip().replace("\n", " ")[:800]
    return f"HTTP {status} (request id {request_id}). Photon said: {body or '(no details)'}"
