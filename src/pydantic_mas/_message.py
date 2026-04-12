import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class MessageType(StrEnum):
    REQUEST = "request"
    REPLY = "reply"
    NOTIFICATION = "notification"


class Message(BaseModel, frozen=True):
    """Envelope for all inter-agent communication."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    type: MessageType
    content: str
    depth: int = 0
    in_reply_to: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
