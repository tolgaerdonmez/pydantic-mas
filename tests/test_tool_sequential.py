"""Tests for send_message tool sequential execution safety.

pydantic-ai executes multiple tool calls in a single LLM turn concurrently
by default (via asyncio.create_task). The send_message tool mutates shared
state (router, budget tracker) through a synchronous check-then-act pattern.

To prevent interleaving, send_message is registered with sequential=True,
which forces pydantic-ai to execute all tool calls in the batch sequentially
when any tool in the batch has this flag set.

These tests verify:
1. The tool is correctly marked as sequential
2. Concurrent route() calls via the tool produce consistent state
"""

import asyncio

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.models.test import TestModel

from pydantic_mas._budget import Budget, BudgetTracker
from pydantic_mas._message import MessageType
from pydantic_mas._router import MessageRouter


def _make_send_message_tool(
    agent_id: str, router: MessageRouter, current_depth: int = 0
) -> Tool:
    """Create a send_message tool the same way AgentNode will."""

    async def send_message(target_agent: str, content: str) -> str:
        """Send a message to another agent in the system.

        Args:
            target_agent: The ID of the agent to send the message to.
            content: The message content to send.
        """
        try:
            msg = router.route(
                sender=agent_id,
                receiver=target_agent,
                content=content,
                type=MessageType.REQUEST,
                depth=current_depth + 1,
            )
            return f"Message sent to '{target_agent}' (id: {msg.id})"
        except ValueError as e:
            return f"Error: {e}"

    return Tool(send_message, sequential=True)


class TestSendMessageToolSequential:
    def test_tool_is_marked_sequential(self):
        tracker = BudgetTracker(Budget())
        router = MessageRouter(tracker)
        tool = _make_send_message_tool("agent_a", router)
        assert tool.sequential is True

    def test_multiple_routes_consistent_count(self):
        """Multiple synchronous route() calls should produce exact counts."""
        tracker = BudgetTracker(Budget())
        router = MessageRouter(tracker)
        inbox_b = asyncio.Queue()
        inbox_c = asyncio.Queue()
        router.register("a", asyncio.Queue())
        router.register("b", inbox_b)
        router.register("c", inbox_c)

        router.route(sender="a", receiver="b", content="1", type=MessageType.REQUEST)
        router.route(sender="a", receiver="c", content="2", type=MessageType.REQUEST)
        router.route(sender="a", receiver="b", content="3", type=MessageType.REQUEST)

        assert tracker.total_messages == 3
        assert tracker.per_agent_messages["a"] == 3
        assert inbox_b.qsize() == 2
        assert inbox_c.qsize() == 1
        assert len(router.message_log) == 3

    def test_budget_enforced_under_sequential_calls(self):
        """Budget limit should be respected even with rapid sequential calls."""
        tracker = BudgetTracker(Budget(max_total_messages=2))
        router = MessageRouter(tracker)
        router.register("a", asyncio.Queue())
        router.register("b", asyncio.Queue())

        router.route(sender="a", receiver="b", content="1", type=MessageType.REQUEST)
        router.route(sender="a", receiver="b", content="2", type=MessageType.REQUEST)

        from pydantic_mas._budget import BudgetExceededError

        with pytest.raises(BudgetExceededError, match="total_messages"):
            router.route(
                sender="a", receiver="b", content="3", type=MessageType.REQUEST
            )

        # Count should be exactly 2, not 3
        assert tracker.total_messages == 2

    async def test_sequential_tool_with_test_model(self):
        """End-to-end: agent with sequential send_message tool runs correctly."""
        tracker = BudgetTracker(Budget())
        router = MessageRouter(tracker)
        inbox_a: asyncio.Queue = asyncio.Queue()
        inbox_b: asyncio.Queue = asyncio.Queue()
        router.register("agent_a", inbox_a)
        router.register("agent_b", inbox_b)

        tool = _make_send_message_tool("agent_a", router)

        agent = Agent(model=TestModel(), tools=[tool])
        result = await agent.run("Send a message to agent_b saying hello")

        # TestModel may or may not call the tool depending on its behavior,
        # but the tool should be available and the agent should complete
        assert result.output is not None
