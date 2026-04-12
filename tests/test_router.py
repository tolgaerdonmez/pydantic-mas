"""Tests for MessageRouter."""

import asyncio

import pytest

from pydantic_mas._budget import Budget, BudgetExceededError, BudgetTracker
from pydantic_mas._message import Message, MessageType
from pydantic_mas._router import MessageRouter


@pytest.fixture
def budget_tracker() -> BudgetTracker:
    return BudgetTracker(Budget())


@pytest.fixture
def router(budget_tracker: BudgetTracker) -> MessageRouter:
    return MessageRouter(budget_tracker)


class TestRegister:
    def test_register_agent(self, router: MessageRouter):
        inbox: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_a", inbox)
        assert "agent_a" in router.agent_ids

    def test_register_multiple_agents(self, router: MessageRouter):
        router.register("a", asyncio.Queue())
        router.register("b", asyncio.Queue())
        router.register("c", asyncio.Queue())
        assert set(router.agent_ids) == {"a", "b", "c"}

    def test_register_duplicate_raises(self, router: MessageRouter):
        router.register("agent_a", asyncio.Queue())
        with pytest.raises(ValueError, match="already registered"):
            router.register("agent_a", asyncio.Queue())


class TestRoute:
    def test_route_creates_message(self, router: MessageRouter):
        router.register("sender", asyncio.Queue())
        router.register("receiver", asyncio.Queue())

        msg = router.route(
            sender="sender",
            receiver="receiver",
            content="hello",
            type=MessageType.REQUEST,
        )

        assert isinstance(msg, Message)
        assert msg.sender == "sender"
        assert msg.receiver == "receiver"
        assert msg.content == "hello"
        assert msg.type == MessageType.REQUEST

    def test_route_delivers_to_inbox(self, router: MessageRouter):
        inbox: asyncio.Queue[Message] = asyncio.Queue()
        router.register("sender", asyncio.Queue())
        router.register("receiver", inbox)

        msg = router.route(
            sender="sender",
            receiver="receiver",
            content="hello",
            type=MessageType.REQUEST,
        )

        assert not inbox.empty()
        delivered = inbox.get_nowait()
        assert delivered == msg

    def test_route_to_unknown_receiver_raises(self, router: MessageRouter):
        router.register("sender", asyncio.Queue())
        with pytest.raises(ValueError, match="Unknown receiver"):
            router.route(
                sender="sender",
                receiver="nonexistent",
                content="hello",
                type=MessageType.REQUEST,
            )

    def test_route_with_depth(self, router: MessageRouter):
        router.register("a", asyncio.Queue())
        router.register("b", asyncio.Queue())

        msg = router.route(
            sender="a", receiver="b", content="hi", type=MessageType.REQUEST, depth=3
        )
        assert msg.depth == 3

    def test_route_with_in_reply_to(self, router: MessageRouter):
        router.register("a", asyncio.Queue())
        router.register("b", asyncio.Queue())

        msg = router.route(
            sender="a",
            receiver="b",
            content="reply",
            type=MessageType.REPLY,
            in_reply_to="original-msg-id",
        )
        assert msg.in_reply_to == "original-msg-id"

    def test_route_respects_budget_limit(self):
        tracker = BudgetTracker(Budget(max_total_messages=1))
        router = MessageRouter(tracker)
        router.register("a", asyncio.Queue())
        router.register("b", asyncio.Queue())

        router.route(sender="a", receiver="b", content="1", type=MessageType.REQUEST)
        with pytest.raises(BudgetExceededError):
            router.route(
                sender="a", receiver="b", content="2", type=MessageType.REQUEST
            )


class TestMessageLog:
    def test_message_log_accumulates(self, router: MessageRouter):
        router.register("a", asyncio.Queue())
        router.register("b", asyncio.Queue())

        msg1 = router.route(
            sender="a", receiver="b", content="first", type=MessageType.REQUEST
        )
        msg2 = router.route(
            sender="b", receiver="a", content="second", type=MessageType.REPLY
        )

        assert router.message_log == [msg1, msg2]

    def test_message_log_empty_initially(self, router: MessageRouter):
        assert router.message_log == []

    def test_failed_route_does_not_log(self, router: MessageRouter):
        router.register("a", asyncio.Queue())
        with pytest.raises(ValueError):
            router.route(
                sender="a",
                receiver="nonexistent",
                content="hi",
                type=MessageType.REQUEST,
            )
        assert router.message_log == []
