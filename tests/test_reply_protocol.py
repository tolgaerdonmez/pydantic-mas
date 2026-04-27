"""Tests for reply-protocol enforcement.

When B is processing a REQUEST from A, and B's model calls
`send_message(A, ...)` instead of letting the auto-reply fire from
last_output, that send is reinterpreted: it is routed as a REPLY to A,
B's run terminates, and the normal last_output auto-reply is suppressed.

This prevents the common model bug where B issues both a tool-driven
REQUEST back to A *and* a separate auto-generated REPLY from its final
output, doubling the message graph.
"""

import asyncio
import json
from typing import Callable

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_mas._agent_node import AgentNode
from pydantic_mas._budget import Budget, BudgetTracker
from pydantic_mas._config import AgentConfig
from pydantic_mas._result import TerminationReason
from pydantic_mas._mas import MAS
from pydantic_mas._message import Message, MessageType
from pydantic_mas._router import MessageRouter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


ModelHandler = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


def _make_b_node(
    model_handler: ModelHandler,
    enforce_reply_protocol: bool = True,
    interrupt_on_send: bool = False,
) -> tuple[AgentNode, MessageRouter, asyncio.Queue[Message]]:
    """Build an isolated AgentNode for `agent_b`, with `agent_a` pre-registered.

    Returns the node, router, and agent_a's inbox queue.
    """
    tracker = BudgetTracker(Budget())
    router = MessageRouter(tracker)
    agent: Agent = Agent(model=FunctionModel(model_handler))
    node: AgentNode = AgentNode(
        agent_id="agent_b",
        agent=agent,
        router=router,
        enforce_reply_protocol=enforce_reply_protocol,
        interrupt_on_send=interrupt_on_send,
    )
    router.register("agent_b", node.inbox)
    inbox_a: asyncio.Queue[Message] = asyncio.Queue()
    router.register("agent_a", inbox_a)
    return node, router, inbox_a


def _b_replies_via_send(
    target: str = "agent_a", content: str = "here is my reply"
) -> ModelHandler:
    """B's model: calls send_message(target, content) once, then trailing text."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last = messages[-1].parts[-1]
        if isinstance(last, UserPromptPart):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps({"target_agent": target, "content": content}),
                        tool_call_id="b-send-1",
                    ),
                ]
            )
        return ModelResponse(parts=[TextPart(content="trailing text from B")])

    return handler


async def _run_until_idle(node: AgentNode, sleep_seconds: float = 0.4) -> None:
    task = asyncio.create_task(node.run_loop())
    await asyncio.sleep(sleep_seconds)
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


class TestDefaultsAreOn:
    def test_mas_default_is_on(self):
        mas = MAS(agents={"a": AgentConfig(agent=Agent(model=TestModel()))})
        assert mas.enforce_reply_protocol is True

    def test_agent_node_default_is_on(self):
        tracker = BudgetTracker(Budget())
        router = MessageRouter(tracker)
        node: AgentNode = AgentNode(
            agent_id="a",
            agent=Agent(model=TestModel()),
            router=router,
        )
        assert node.enforce_reply_protocol is True


# ---------------------------------------------------------------------------
# Core: target == sender of current REQUEST is reinterpreted as a REPLY
# ---------------------------------------------------------------------------


class TestInterceptRoutesReply:
    async def test_single_reply_message_with_correct_envelope(self):
        node, router, inbox_a = _make_b_node(_b_replies_via_send())

        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            content="please help",
            type=MessageType.REQUEST,
            depth=1,
        )
        node.inbox.put_nowait(msg)
        await _run_until_idle(node)

        b_to_a = [
            m
            for m in router.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]

        # Exactly one message — not REQUEST + auto-REPLY
        assert len(b_to_a) == 1, [m.type for m in b_to_a]
        only = b_to_a[0]
        assert only.type == MessageType.REPLY
        assert only.in_reply_to == msg.id
        assert only.depth == msg.depth
        assert only.content == "here is my reply"
        # And it actually landed in agent_a's inbox
        assert not inbox_a.empty()


class TestInterceptTerminatesRun:
    async def test_b_llm_called_only_once(self):
        llm_calls = 0
        base = _b_replies_via_send()

        def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal llm_calls
            llm_calls += 1
            return base(messages, info)

        node, _, _ = _make_b_node(handler)

        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            content="x",
            type=MessageType.REQUEST,
            depth=1,
        )
        node.inbox.put_nowait(msg)
        await _run_until_idle(node)

        # No second LLM turn — run terminated after the intercepted tool batch
        assert llm_calls == 1


# ---------------------------------------------------------------------------
# Disabled: bug behavior is preserved (REQUEST + auto-REPLY)
# ---------------------------------------------------------------------------


class TestDisabledKeepsLegacyBehavior:
    async def test_disabled_produces_request_plus_auto_reply(self):
        node, router, _ = _make_b_node(
            _b_replies_via_send(), enforce_reply_protocol=False
        )

        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            content="x",
            type=MessageType.REQUEST,
            depth=1,
        )
        node.inbox.put_nowait(msg)
        await _run_until_idle(node, sleep_seconds=0.5)

        b_to_a = [
            m
            for m in router.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        types = [m.type for m in b_to_a]
        # The buggy double-send is intact: a tool-driven REQUEST and an auto-REPLY
        assert MessageType.REQUEST in types
        assert MessageType.REPLY in types


# ---------------------------------------------------------------------------
# Negative cases: intercept must NOT fire
# ---------------------------------------------------------------------------


class TestNoInterceptForOtherTargets:
    async def test_send_to_third_party_is_normal_request(self):
        def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            last = messages[-1].parts[-1]
            if isinstance(last, UserPromptPart):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_c", "content": "hi C"}
                            ),
                            tool_call_id="b-send-c",
                        ),
                    ]
                )
            return ModelResponse(parts=[TextPart(content="ok done")])

        node, router, _ = _make_b_node(handler)
        inbox_c: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_c", inbox_c)

        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            content="please",
            type=MessageType.REQUEST,
            depth=1,
        )
        node.inbox.put_nowait(msg)
        await _run_until_idle(node)

        # C got a normal REQUEST
        b_to_c = [
            m
            for m in router.message_log
            if m.sender == "agent_b" and m.receiver == "agent_c"
        ]
        assert len(b_to_c) == 1
        assert b_to_c[0].type == MessageType.REQUEST

        # A got the normal auto-reply (no intercept since target != sender)
        b_to_a = [
            m
            for m in router.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        assert len(b_to_a) == 1
        assert b_to_a[0].type == MessageType.REPLY


class TestNoInterceptForReplyTypeMessages:
    async def test_b_processing_reply_sends_normal_request(self):
        # B is processing a REPLY from A. send_message(A) is a brand-new REQUEST,
        # not a reply-to-a-reply. (Replies don't auto-reply, so no double-send risk.)
        node, router, _ = _make_b_node(_b_replies_via_send())

        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            content="here's my answer",
            type=MessageType.REPLY,
            in_reply_to="prev-id",
            depth=2,
        )
        node.inbox.put_nowait(msg)
        await _run_until_idle(node)

        b_to_a = [
            m
            for m in router.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        types = [m.type for m in b_to_a]
        assert MessageType.REQUEST in types
        assert MessageType.REPLY not in types


# ---------------------------------------------------------------------------
# Multi-target turn: only the back-to-sender call becomes a REPLY
# ---------------------------------------------------------------------------


class TestMultiTargetSameTurn:
    async def test_third_party_request_plus_single_reply(self):
        def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            last = messages[-1].parts[-1]
            if isinstance(last, UserPromptPart):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_c", "content": "for C"}
                            ),
                            tool_call_id="b-c",
                        ),
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {
                                    "target_agent": "agent_a",
                                    "content": "for A as reply",
                                }
                            ),
                            tool_call_id="b-a",
                        ),
                    ]
                )
            return ModelResponse(parts=[TextPart(content="should not happen")])

        node, router, _ = _make_b_node(handler)
        inbox_c: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_c", inbox_c)

        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            content="please",
            type=MessageType.REQUEST,
            depth=1,
        )
        node.inbox.put_nowait(msg)
        await _run_until_idle(node)

        b_to_c = [
            m
            for m in router.message_log
            if m.sender == "agent_b" and m.receiver == "agent_c"
        ]
        assert len(b_to_c) == 1
        assert b_to_c[0].type == MessageType.REQUEST

        b_to_a = [
            m
            for m in router.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        assert len(b_to_a) == 1
        reply = b_to_a[0]
        assert reply.type == MessageType.REPLY
        assert reply.in_reply_to == msg.id
        assert reply.content == "for A as reply"


# ---------------------------------------------------------------------------
# Repeated intercept attempts in the same turn
# ---------------------------------------------------------------------------


class TestRepeatedInterceptInSameTurn:
    async def test_only_first_send_to_sender_routes_a_reply(self):
        def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            last = messages[-1].parts[-1]
            if isinstance(last, UserPromptPart):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_a", "content": "first"}
                            ),
                            tool_call_id="b-1",
                        ),
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_a", "content": "second"}
                            ),
                            tool_call_id="b-2",
                        ),
                    ]
                )
            return ModelResponse(parts=[TextPart(content="ok")])

        node, router, _ = _make_b_node(handler)

        msg = Message(
            sender="agent_a",
            receiver="agent_b",
            content="please",
            type=MessageType.REQUEST,
            depth=1,
        )
        node.inbox.put_nowait(msg)
        await _run_until_idle(node)

        b_to_a = [
            m
            for m in router.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        assert len(b_to_a) == 1
        assert b_to_a[0].type == MessageType.REPLY
        assert b_to_a[0].content == "first"


# ---------------------------------------------------------------------------
# End-to-end through MAS — both sides of the flag
# ---------------------------------------------------------------------------


def _a_delegates_then_acks() -> ModelHandler:
    """A: first turn calls send_message(B), later turns return text."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        has_prior_tools = any(
            hasattr(p, "tool_name") for m in messages for p in m.parts
        )
        last = messages[-1].parts[-1]
        if isinstance(last, UserPromptPart) and not has_prior_tools:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {"target_agent": "agent_b", "content": "task for B"}
                        ),
                        tool_call_id="a-1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="A acknowledges")])

    return handler


class TestMASEndToEnd:
    async def test_default_yields_single_reply(self):
        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(model=FunctionModel(_a_delegates_then_acks()))
                ),
                "agent_b": AgentConfig(
                    agent=Agent(model=FunctionModel(_b_replies_via_send()))
                ),
            },
        )

        result = await mas.run(entry_agent="agent_a", prompt="start")

        b_to_a = [
            m
            for m in result.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        assert len(b_to_a) == 1
        assert b_to_a[0].type == MessageType.REPLY

    async def test_disabled_yields_request_plus_auto_reply(self):
        # Without enforcement the buggy double-send is what prevents B from
        # producing a single clean answer — it can also kick off a ping-pong
        # between A and B. Cap with a low message budget so the run terminates.
        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(model=FunctionModel(_a_delegates_then_acks()))
                ),
                "agent_b": AgentConfig(
                    agent=Agent(model=FunctionModel(_b_replies_via_send()))
                ),
            },
            enforce_reply_protocol=False,
            budget=Budget(max_total_messages=8),
        )

        result = await mas.run(entry_agent="agent_a", prompt="start")

        # Either completed or hit the budget — both are fine for this test.
        # What matters is that within B's first response it produced both a
        # tool-driven REQUEST AND an auto-generated REPLY back to A.
        assert result.termination_reason in (
            TerminationReason.COMPLETED,
            TerminationReason.BUDGET_EXCEEDED,
        )
        b_to_a = [
            m
            for m in result.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        types = [m.type for m in b_to_a]
        assert MessageType.REQUEST in types
        assert MessageType.REPLY in types
