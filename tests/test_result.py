"""Tests for MASResult and TerminationReason."""

from pydantic_mas._budget import BudgetSnapshot
from pydantic_mas._message import Message, MessageType
from pydantic_mas._result import MASResult, TerminationReason


class TestTerminationReason:
    def test_values(self):
        assert TerminationReason.COMPLETED == "completed"
        assert TerminationReason.BUDGET_EXCEEDED == "budget_exceeded"
        assert TerminationReason.TIMEOUT == "timeout"

    def test_is_str(self):
        assert isinstance(TerminationReason.COMPLETED, str)


class TestMASResult:
    def test_construction(self):
        msg = Message(
            sender="system",
            receiver="agent_a",
            type=MessageType.REQUEST,
            content="hello",
        )
        snapshot = BudgetSnapshot(
            total_messages=1,
            per_agent_messages={"system": 1},
            max_depth_seen=0,
        )
        result = MASResult(
            message_log=[msg],
            agent_histories={"agent_a": []},
            termination_reason=TerminationReason.COMPLETED,
            budget_usage=snapshot,
        )

        assert result.message_log == [msg]
        assert result.agent_histories == {"agent_a": []}
        assert result.termination_reason == TerminationReason.COMPLETED
        assert result.budget_usage == snapshot

    def test_frozen(self):
        snapshot = BudgetSnapshot(
            total_messages=0,
            per_agent_messages={},
            max_depth_seen=0,
        )
        result = MASResult(
            message_log=[],
            agent_histories={},
            termination_reason=TerminationReason.COMPLETED,
            budget_usage=snapshot,
        )
        import pytest

        with pytest.raises(Exception):
            result.termination_reason = TerminationReason.TIMEOUT  # type: ignore[misc]
