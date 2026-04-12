"""Tests for MASHooks — communication tool interception."""

import asyncio
import json

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_mas._agent_node import AgentNode
from pydantic_mas._budget import Budget, BudgetTracker
from pydantic_mas._config import AgentConfig
from pydantic_mas._hooks import MASHooks, SendMessageHookContext
from pydantic_mas._mas import MAS
from pydantic_mas._message import Message, MessageType
from pydantic_mas._result import TerminationReason
from pydantic_mas._router import MessageRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_message(
    sender: str = "system",
    receiver: str = "agent_a",
    content: str = "Hello",
    type: MessageType = MessageType.REQUEST,
    depth: int = 0,
) -> Message:
    return Message(
        sender=sender, receiver=receiver, type=type, content=content, depth=depth
    )


def _send_model(target: str = "agent_b", content: str = "hello") -> FunctionModel:
    """Model that calls send_message once, then returns text."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for msg in messages:
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    if hasattr(part, "tool_name"):
                        return ModelResponse(parts=[TextPart(content="Done")])

        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="send_message",
                    args=json.dumps(
                        {"target_agent": target, "content": content}
                    ),
                    tool_call_id="send-1",
                )
            ]
        )

    return FunctionModel(handler)


def _make_node(
    agent: Agent | None = None,
    hooks: MASHooks | None = None,
    budget: Budget | None = None,
    interrupt_on_send: bool = False,
) -> tuple[AgentNode, MessageRouter, BudgetTracker]:
    tracker = BudgetTracker(budget or Budget())
    router = MessageRouter(tracker)
    node = AgentNode(
        agent_id="agent_a",
        agent=agent or Agent(model=TestModel()),
        router=router,
        hooks=hooks,
        interrupt_on_send=interrupt_on_send,
    )
    router.register("agent_a", node.inbox)
    return node, router, tracker


# ---------------------------------------------------------------------------
# Tests: before_send_message
# ---------------------------------------------------------------------------


class TestBeforeSendMessage:
    async def test_hook_is_called(self):
        """before_send_message fires for every send_message invocation."""
        calls: list[SendMessageHookContext] = []

        async def record(ctx: SendMessageHookContext) -> SendMessageHookContext:
            calls.append(ctx)
            return ctx

        hooks = MASHooks(before_send_message=record)
        node, router, _ = _make_node(
            agent=Agent(model=_send_model()), hooks=hooks
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert len(calls) == 1
        assert calls[0].sender_id == "agent_a"
        assert calls[0].receiver_id == "agent_b"
        assert calls[0].content == "hello"

    async def test_hook_can_modify_content(self):
        """before hook changes content; routed message has modified content."""
        async def modify(ctx: SendMessageHookContext) -> SendMessageHookContext:
            ctx.content = "MODIFIED: " + ctx.content
            return ctx

        hooks = MASHooks(before_send_message=modify)
        node, router, _ = _make_node(
            agent=Agent(model=_send_model(content="original")), hooks=hooks
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        msg = inbox_b.get_nowait()
        assert msg.content == "MODIFIED: original"

    async def test_hook_can_change_receiver(self):
        """before hook redirects to a different agent."""
        async def redirect(ctx: SendMessageHookContext) -> SendMessageHookContext:
            ctx.receiver_id = "agent_c"
            return ctx

        hooks = MASHooks(before_send_message=redirect)
        node, router, _ = _make_node(
            agent=Agent(model=_send_model(target="agent_b")), hooks=hooks
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        inbox_c: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)
        router.register("agent_c", inbox_c)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert inbox_b.empty(), "agent_b should NOT receive the message"
        assert not inbox_c.empty(), "agent_c should receive the redirected message"

    async def test_hook_can_block(self):
        """before hook returns None; message is not routed."""
        async def block(ctx: SendMessageHookContext) -> None:
            return None

        hooks = MASHooks(before_send_message=block)
        node, router, _ = _make_node(
            agent=Agent(model=_send_model()), hooks=hooks
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert inbox_b.empty(), "message should be blocked"
        assert len(router.message_log) == 0

    async def test_context_has_correct_current_message(self):
        """context.current_message matches the message being processed."""
        captured: list[SendMessageHookContext] = []

        async def record(ctx: SendMessageHookContext) -> SendMessageHookContext:
            captured.append(ctx)
            return ctx

        hooks = MASHooks(before_send_message=record)
        node, router, _ = _make_node(
            agent=Agent(model=_send_model()), hooks=hooks
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        original = _make_message(content="process this")
        node.inbox.put_nowait(original)
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert captured[0].current_message.content == "process this"
        assert captured[0].current_message.id == original.id

    async def test_context_has_correct_depth(self):
        """context.depth is current_depth + 1."""
        captured: list[SendMessageHookContext] = []

        async def record(ctx: SendMessageHookContext) -> SendMessageHookContext:
            captured.append(ctx)
            return ctx

        hooks = MASHooks(before_send_message=record)
        node, router, _ = _make_node(
            agent=Agent(model=_send_model()), hooks=hooks
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message(depth=3))
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert captured[0].depth == 4


# ---------------------------------------------------------------------------
# Tests: after_send_message
# ---------------------------------------------------------------------------


class TestAfterSendMessage:
    async def test_hook_is_called_with_message(self):
        """after_send_message fires after routing, receives the Message."""
        delivered: list[tuple[SendMessageHookContext, Message]] = []

        async def record(ctx: SendMessageHookContext, msg: Message) -> None:
            delivered.append((ctx, msg))

        hooks = MASHooks(after_send_message=record)
        node, router, _ = _make_node(
            agent=Agent(model=_send_model()), hooks=hooks
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert len(delivered) == 1
        ctx, msg = delivered[0]
        assert ctx.sender_id == "agent_a"
        assert msg.sender == "agent_a"
        assert msg.receiver == "agent_b"

    async def test_not_called_when_blocked(self):
        """after hook should NOT fire if before hook blocked the message."""
        after_calls: list[Message] = []

        async def block(ctx: SendMessageHookContext) -> None:
            return None

        async def after(ctx: SendMessageHookContext, msg: Message) -> None:
            after_calls.append(msg)

        hooks = MASHooks(before_send_message=block, after_send_message=after)
        node, router, _ = _make_node(
            agent=Agent(model=_send_model()), hooks=hooks
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert len(after_calls) == 0


# ---------------------------------------------------------------------------
# Tests: hook ordering
# ---------------------------------------------------------------------------


class TestHookOrdering:
    async def test_before_fires_then_after(self):
        """before_send_message fires before after_send_message."""
        log: list[str] = []

        async def before(ctx: SendMessageHookContext) -> SendMessageHookContext:
            log.append("before")
            return ctx

        async def after(ctx: SendMessageHookContext, msg: Message) -> None:
            log.append("after")

        hooks = MASHooks(before_send_message=before, after_send_message=after)
        node, router, _ = _make_node(
            agent=Agent(model=_send_model()), hooks=hooks
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert log == ["before", "after"]


# ---------------------------------------------------------------------------
# Tests: no hooks (default behavior)
# ---------------------------------------------------------------------------


class TestNoHooks:
    async def test_none_hooks_works(self):
        """hooks=None preserves default send_message behavior."""
        node, router, _ = _make_node(
            agent=Agent(model=_send_model()), hooks=None
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert not inbox_b.empty()


# ---------------------------------------------------------------------------
# Tests: hooks + interrupt_on_send
# ---------------------------------------------------------------------------


class TestHooksWithInterrupt:
    async def test_hooks_fire_with_interrupt(self):
        """Hooks fire even when interrupt_on_send is True."""
        log: list[str] = []

        async def before(ctx: SendMessageHookContext) -> SendMessageHookContext:
            log.append("before")
            return ctx

        async def after(ctx: SendMessageHookContext, msg: Message) -> None:
            log.append("after")

        hooks = MASHooks(before_send_message=before, after_send_message=after)
        node, router, _ = _make_node(
            agent=Agent(model=_send_model()),
            hooks=hooks,
            interrupt_on_send=True,
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert log == ["before", "after"]
        assert not inbox_b.empty()


# ---------------------------------------------------------------------------
# Tests: access control pattern
# ---------------------------------------------------------------------------


class TestAccessControlPattern:
    async def test_block_unauthorized_targets(self):
        """before hook blocks messages to unauthorized agents."""

        async def access_control(
            ctx: SendMessageHookContext,
        ) -> SendMessageHookContext | None:
            allowed = {"agent_a": ["agent_c"]}
            if ctx.receiver_id not in allowed.get(ctx.sender_id, []):
                return None
            return ctx

        hooks = MASHooks(before_send_message=access_control)
        node, router, _ = _make_node(
            agent=Agent(model=_send_model(target="agent_b")), hooks=hooks
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert inbox_b.empty(), "agent_b is not in allowed list, should be blocked"
        assert len(node.history) > 0, "agent should still complete processing"


# ---------------------------------------------------------------------------
# Tests: e2e with MAS
# ---------------------------------------------------------------------------


class TestMASWithHooks:
    async def test_logging_hook_captures_all_communication(self):
        """after_send_message hook as a logger in a full MAS.run()."""
        logged: list[tuple[str, str, str]] = []

        async def log_hook(ctx: SendMessageHookContext, msg: Message) -> None:
            logged.append((ctx.sender_id, ctx.receiver_id, ctx.content))

        hooks = MASHooks(after_send_message=log_hook)

        def handler(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            for msg in messages:
                if hasattr(msg, "parts"):
                    for part in msg.parts:
                        if hasattr(part, "tool_name"):
                            return ModelResponse(
                                parts=[TextPart(content="Done")]
                            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {
                                "target_agent": "agent_b",
                                "content": "task for you",
                            }
                        ),
                        tool_call_id="send-1",
                    )
                ]
            )

        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(model=FunctionModel(handler))
                ),
                "agent_b": AgentConfig(agent=Agent(model=TestModel())),
            },
            hooks=hooks,
        )

        result = await mas.run(entry_agent="agent_a", prompt="start")

        assert result.termination_reason == TerminationReason.COMPLETED
        # The logger should have captured agent_a -> agent_b
        assert any(
            s == "agent_a" and r == "agent_b" for s, r, _ in logged
        )

    async def test_content_modification_e2e(self):
        """before hook modifies content; MASResult message_log reflects change."""

        async def tag(ctx: SendMessageHookContext) -> SendMessageHookContext:
            ctx.content = f"[tagged] {ctx.content}"
            return ctx

        hooks = MASHooks(before_send_message=tag)

        def handler(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            for msg in messages:
                if hasattr(msg, "parts"):
                    for part in msg.parts:
                        if hasattr(part, "tool_name"):
                            return ModelResponse(
                                parts=[TextPart(content="Done")]
                            )
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {
                                "target_agent": "agent_b",
                                "content": "original content",
                            }
                        ),
                        tool_call_id="send-1",
                    )
                ]
            )

        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(model=FunctionModel(handler))
                ),
                "agent_b": AgentConfig(agent=Agent(model=TestModel())),
            },
            hooks=hooks,
        )

        result = await mas.run(entry_agent="agent_a", prompt="go")

        a_to_b = [
            m
            for m in result.message_log
            if m.sender == "agent_a" and m.receiver == "agent_b"
        ]
        assert len(a_to_b) == 1
        assert a_to_b[0].content == "[tagged] original content"
