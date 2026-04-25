"""Tests for Message model."""

import uuid
from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from pydantic_mas._message import Message, MessageType


class TestMessageType:
    def test_request_value(self):
        assert MessageType.REQUEST == "request"

    def test_reply_value(self):
        assert MessageType.REPLY == "reply"

    def test_notification_value(self):
        assert MessageType.NOTIFICATION == "notification"


class TestMessage:
    def test_create_with_required_fields(self):
        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            type=MessageType.REQUEST,
            content="hello",
        )
        assert msg.sender == "agent_a"
        assert msg.receiver == "agent_b"
        assert msg.type == MessageType.REQUEST
        assert msg.content == "hello"

    def test_id_auto_generated(self):
        msg = Message(sender="a", receiver="b", type=MessageType.REQUEST, content="hi")
        # Should be a valid UUID string
        uuid.UUID(msg.id)

    def test_id_unique_per_message(self):
        msg1 = Message(sender="a", receiver="b", type=MessageType.REQUEST, content="hi")
        msg2 = Message(sender="a", receiver="b", type=MessageType.REQUEST, content="hi")
        assert msg1.id != msg2.id

    def test_timestamp_auto_generated(self):
        before = datetime.now(timezone.utc)
        msg = Message(sender="a", receiver="b", type=MessageType.REQUEST, content="hi")
        after = datetime.now(timezone.utc)
        assert before <= msg.timestamp <= after

    def test_depth_defaults_to_zero(self):
        msg = Message(sender="a", receiver="b", type=MessageType.REQUEST, content="hi")
        assert msg.depth == 0

    def test_depth_can_be_set(self):
        msg = Message(
            sender="a", receiver="b", type=MessageType.REQUEST, content="hi", depth=3
        )
        assert msg.depth == 3

    def test_in_reply_to_defaults_to_none(self):
        msg = Message(sender="a", receiver="b", type=MessageType.REQUEST, content="hi")
        assert msg.in_reply_to is None

    def test_in_reply_to_can_be_set(self):
        msg = Message(
            sender="a",
            receiver="b",
            type=MessageType.REPLY,
            content="response",
            in_reply_to="msg-123",
        )
        assert msg.in_reply_to == "msg-123"

    def test_frozen_immutable(self):
        msg = Message(sender="a", receiver="b", type=MessageType.REQUEST, content="hi")
        with pytest.raises(ValidationError):
            msg.content = "changed"

    def test_content_must_be_string(self):
        with pytest.raises(ValidationError):
            Message(sender="a", receiver="b", type=MessageType.REQUEST, content=123)

    def test_serialization_roundtrip(self):
        """Non-metadata fields round-trip cleanly. Metadata is one-way
        (dumped as a string for observability; not expected to validate back)."""
        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            type=MessageType.REQUEST,
            content="hello",
            depth=2,
            in_reply_to="prev-id",
        )
        data = msg.model_dump()
        data.pop("metadata", None)
        restored = Message.model_validate(data)
        assert restored.sender == msg.sender
        assert restored.receiver == msg.receiver
        assert restored.type == msg.type
        assert restored.content == msg.content
        assert restored.depth == msg.depth
        assert restored.in_reply_to == msg.in_reply_to
        assert restored.id == msg.id
