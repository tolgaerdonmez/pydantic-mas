"""Tests for default_message_formatter."""

from pydantic_mas._formatter import default_message_formatter
from pydantic_mas._message import Message, MessageType


class TestDefaultMessageFormatter:
    def test_system_message(self):
        msg = Message(
            sender="system",
            receiver="agent_a",
            type=MessageType.REQUEST,
            content="Analyze this data.",
        )
        result = default_message_formatter(msg)
        assert result.startswith("[System | New task]")
        assert "Analyze this data." in result

    def test_request_message(self):
        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            type=MessageType.REQUEST,
            content="Please help.",
        )
        result = default_message_formatter(msg)
        assert "agent_a" in result
        assert "request" in result
        assert msg.id in result
        assert "Please help." in result

    def test_reply_message(self):
        msg = Message(
            sender="agent_b",
            receiver="agent_a",
            type=MessageType.REPLY,
            content="Here is the answer.",
            in_reply_to="msg-123",
        )
        result = default_message_formatter(msg)
        assert "Reply from" in result
        assert "agent_b" in result
        assert "msg-123" in result
        assert msg.id in result
        assert "Here is the answer." in result

    def test_notification_message(self):
        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            type=MessageType.NOTIFICATION,
            content="FYI: task complete.",
        )
        result = default_message_formatter(msg)
        assert "agent_a" in result
        assert "notification" in result
        assert "FYI: task complete." in result
