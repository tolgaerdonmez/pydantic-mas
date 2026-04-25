import uuid
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class MessageType(StrEnum):
    REQUEST = "request"
    REPLY = "reply"
    NOTIFICATION = "notification"


class Message(BaseModel, frozen=True):
    """Envelope for all inter-agent communication.

    `metadata` is an open-ended bag for runtime payloads — output schemas,
    live structured outputs, caller policy tags, anything callers and
    factories want to thread between hops. Values are *not* validated and
    may be arbitrary Python objects (classes, BaseModel instances, etc.).

    On serialization, metadata is dumped as a string (`repr`) — it is a
    runtime-only concern and is not expected to round-trip through
    history persistence or wire transport.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender: str
    receiver: str
    type: MessageType
    content: str
    depth: int = 0
    in_reply_to: str | None = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_serializer("metadata")
    def _serialize_metadata(self, v: dict[str, Any]) -> str:
        return repr(v)
