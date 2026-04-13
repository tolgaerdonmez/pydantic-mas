"""Tests for interrupt-on-send behavior.

When interrupt_on_send=True, an agent that calls send_message is stopped
after the tool-call turn completes (all tools in the turn execute), but
before the next LLM call. This prevents the LLM from taking additional
turns after delegating work.
"""

import json

from pydantic_ai import Agent, Tool
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_mas._agent_node import AgentNode
from pydantic_mas._budget import Budget, BudgetTracker
from pydantic_mas._config import AgentConfig
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


def _make_node_with_interrupt(
    agent_id: str = "agent_a",
    agent: Agent | None = None,
    budget: Budget | None = None,
    interrupt_on_send: bool = True,
) -> tuple[AgentNode, MessageRouter, BudgetTracker]:
    tracker = BudgetTracker(budget or Budget())
    router = MessageRouter(tracker)
    node = AgentNode(
        agent_id=agent_id,
        agent=agent or Agent(model=TestModel()),
        router=router,
        interrupt_on_send=interrupt_on_send,
    )
    router.register(agent_id, node.inbox)
    return node, router, tracker


def _two_tool_model() -> FunctionModel:
    """Model that calls send_message + lookup in one turn, then returns text.

    Turn 1: send_message("agent_b", "hello") + lookup("data")
    Turn 2: TextPart("All done")  — should NOT be reached when interrupted.
    """

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last_part = messages[-1].parts[-1]

        if isinstance(last_part, UserPromptPart):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {"target_agent": "agent_b", "content": "hello"}
                        ),
                        tool_call_id="send-1",
                    ),
                    ToolCallPart(
                        tool_name="lookup",
                        args=json.dumps({"key": "data"}),
                        tool_call_id="lookup-1",
                    ),
                ]
            )

        if isinstance(last_part, ToolReturnPart):
            return ModelResponse(parts=[TextPart(content="All done — second LLM turn")])

        return ModelResponse(parts=[TextPart(content="fallback")])

    return FunctionModel(handler)


def _send_only_model(target: str = "agent_b", content: str = "hello") -> FunctionModel:
    """Model that calls send_message once, then returns text."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last_part = messages[-1].parts[-1]

        if isinstance(last_part, UserPromptPart):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps({"target_agent": target, "content": content}),
                        tool_call_id="send-1",
                    )
                ]
            )

        return ModelResponse(parts=[TextPart(content="Done")])

    return FunctionModel(handler)


def _multi_send_model() -> FunctionModel:
    """Model that calls send_message to B and C in one turn."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        last_part = messages[-1].parts[-1]

        if isinstance(last_part, UserPromptPart):
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {"target_agent": "agent_b", "content": "for B"}
                        ),
                        tool_call_id="send-b",
                    ),
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {"target_agent": "agent_c", "content": "for C"}
                        ),
                        tool_call_id="send-c",
                    ),
                ]
            )

        return ModelResponse(parts=[TextPart(content="Done")])

    return FunctionModel(handler)


# ---------------------------------------------------------------------------
# Tests: interrupt flag behavior
# ---------------------------------------------------------------------------


class TestInterruptFlagSet:
    async def test_send_message_sets_interrupt_flag(self):
        """When interrupt_on_send=True, send_message sets the interrupt flag."""
        import asyncio

        agent = Agent(model=_send_only_model())
        node, router, _ = _make_node_with_interrupt(agent=agent)
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # The message should have been routed to agent_b
        assert not inbox_b.empty()

    async def test_no_flag_when_interrupt_disabled(self):
        """When interrupt_on_send=False, send_message does NOT set the flag."""
        import asyncio

        agent = Agent(model=_send_only_model())
        node, router, _ = _make_node_with_interrupt(
            agent=agent, interrupt_on_send=False
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert not inbox_b.empty()
        # Flag should not be set (interrupt is disabled)
        assert not node._interrupt_requested


# ---------------------------------------------------------------------------
# Tests: processing stops after tool turn
# ---------------------------------------------------------------------------


class TestInterruptStopsProcessing:
    async def test_llm_not_called_after_interrupt(self):
        """With interrupt, the LLM should NOT get a second turn after send_message."""
        import asyncio

        llm_call_count = 0
        original_handler = _two_tool_model()

        def counting_handler(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal llm_call_count
            llm_call_count += 1
            return original_handler.function(messages, info)

        lookup_called = False

        async def lookup(key: str) -> str:
            """Look up data.

            Args:
                key: The key to look up.
            """
            nonlocal lookup_called
            lookup_called = True
            return f"data:{key}=value"

        agent = Agent(model=FunctionModel(counting_handler), tools=[Tool(lookup)])
        node, router, _ = _make_node_with_interrupt(agent=agent)
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # LLM called once (initial turn), not twice
        assert llm_call_count == 1
        # Both tools executed (send_message + lookup)
        assert lookup_called
        assert not inbox_b.empty()


class TestAllToolsExecuteBeforeInterrupt:
    async def test_fan_out_both_messages_delivered(self):
        """All send_message calls in one turn execute before interrupt."""
        import asyncio

        agent = Agent(model=_multi_send_model())
        node, router, _ = _make_node_with_interrupt(agent=agent)
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        inbox_c: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)
        router.register("agent_c", inbox_c)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert not inbox_b.empty(), "agent_b should have received a message"
        assert not inbox_c.empty(), "agent_c should have received a message"


# ---------------------------------------------------------------------------
# Tests: no auto-reply on interrupt
# ---------------------------------------------------------------------------


class TestNoAutoReplyOnInterrupt:
    async def test_no_reply_when_interrupted(self):
        """When interrupted, no auto-reply should be sent back to sender."""
        import asyncio

        agent = Agent(model=_send_only_model())
        node, router, _ = _make_node_with_interrupt(agent=agent)
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        inbox_sender: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)
        router.register("other_agent", inbox_sender)

        # Message from another agent (not system) — would normally trigger auto-reply
        msg = _make_message(sender="other_agent", receiver="agent_a", content="help me")
        node.inbox.put_nowait(msg)

        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # send_message delivered to agent_b
        assert not inbox_b.empty()
        # No auto-reply back to sender (interrupted before natural completion)
        assert inbox_sender.empty()


# ---------------------------------------------------------------------------
# Tests: history preserved after interrupt
# ---------------------------------------------------------------------------


class TestHistoryPreservedAfterInterrupt:
    async def test_history_has_three_messages(self):
        """After interrupt, history should contain:
        1. ModelRequest with UserPromptPart
        2. ModelResponse with ToolCallPart(s)
        3. ModelRequest with ToolReturnPart(s)
        """
        import asyncio

        agent = Agent(model=_send_only_model())
        node, router, _ = _make_node_with_interrupt(agent=agent)
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert len(node.history) == 3

        # Message 0: user prompt
        assert node.history[0].kind == "request"
        user_parts = [p for p in node.history[0].parts if isinstance(p, UserPromptPart)]
        assert len(user_parts) >= 1

        # Message 1: model response with tool call
        assert node.history[1].kind == "response"
        tool_calls = [p for p in node.history[1].parts if isinstance(p, ToolCallPart)]
        assert len(tool_calls) >= 1
        assert any(tc.tool_name == "send_message" for tc in tool_calls)

        # Message 2: tool returns
        assert node.history[2].kind == "request"
        tool_returns = [
            p for p in node.history[2].parts if isinstance(p, ToolReturnPart)
        ]
        assert len(tool_returns) >= 1

    async def test_next_message_sees_interrupted_history(self):
        """After interrupt, the next message to this agent should see the full history."""
        import asyncio

        llm_calls: list[list[ModelMessage]] = []

        def tracking_handler(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            llm_calls.append(list(messages))
            last_part = messages[-1].parts[-1]

            if isinstance(last_part, UserPromptPart) and len(messages) <= 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_b", "content": "hi"}
                            ),
                            tool_call_id="send-1",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="final answer")])

        agent = Agent(model=FunctionModel(tracking_handler))
        node, router, _ = _make_node_with_interrupt(agent=agent)
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        # First message — will be interrupted
        node.inbox.put_nowait(_make_message(content="first"))
        # Second message — should see interrupted history
        node.inbox.put_nowait(_make_message(content="second"))

        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.5)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # The second LLM call should have seen the interrupted history
        assert len(llm_calls) >= 2
        # Second call's messages should include history from interrupted first run
        second_call_msgs = llm_calls[1]
        assert len(second_call_msgs) > 2  # more than just the new user prompt


# ---------------------------------------------------------------------------
# Tests: no interrupt when disabled
# ---------------------------------------------------------------------------


class TestNoInterruptWhenDisabled:
    async def test_agent_completes_normally(self):
        """With interrupt_on_send=False, the agent runs to completion."""
        import asyncio

        llm_call_count = 0

        def counting_handler(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal llm_call_count
            llm_call_count += 1
            last_part = messages[-1].parts[-1]

            if isinstance(last_part, UserPromptPart):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_b", "content": "hi"}
                            ),
                            tool_call_id="send-1",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="Done after tools")])

        agent = Agent(model=FunctionModel(counting_handler))
        node, router, _ = _make_node_with_interrupt(
            agent=agent, interrupt_on_send=False
        )
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node.inbox.put_nowait(_make_message())
        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # LLM called twice: first for tool call, second for final text
        assert llm_call_count == 2


# ---------------------------------------------------------------------------
# Tests: end-to-end with MAS
# ---------------------------------------------------------------------------


class TestMASInterruptOnSend:
    async def test_e2e_interrupt_completes(self):
        """MAS with interrupt_on_send=True runs to completion."""

        def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            last_part = messages[-1].parts[-1]
            if isinstance(last_part, UserPromptPart) and "start" in str(last_part):
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
            return ModelResponse(parts=[TextPart(content="ok")])

        mas = MAS(
            agents={
                "agent_a": AgentConfig(agent=Agent(model=FunctionModel(handler))),
                "agent_b": AgentConfig(agent=Agent(model=TestModel())),
            },
            interrupt_on_send=True,
        )

        result = await mas.run(entry_agent="agent_a", prompt="start")

        assert result.termination_reason == TerminationReason.COMPLETED
        pairs = [(m.sender, m.receiver) for m in result.message_log]
        assert ("system", "agent_a") in pairs
        assert ("agent_a", "agent_b") in pairs

    async def test_e2e_interrupt_prevents_extra_llm_turn(self):
        """With interrupt, agent_a's first processing is stopped after send_message.

        The LLM for agent_a is called:
          - Once for the initial message (interrupted after send_message)
          - Once for B's reply (no send_message, completes normally)
        Without interrupt, the first processing would have 2 LLM calls
        (tool call + final text). With interrupt it's only 1.
        """
        llm_a_calls = 0

        def handler_a(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal llm_a_calls
            llm_a_calls += 1

            # Only send on the very first LLM call (no tool interactions yet)
            has_prior_tools = any(
                hasattr(p, "tool_name") for msg in messages for p in msg.parts
            )
            last_part = messages[-1].parts[-1]

            if isinstance(last_part, UserPromptPart) and not has_prior_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {
                                    "target_agent": "agent_b",
                                    "content": "delegated",
                                }
                            ),
                            tool_call_id="send-1",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="Acknowledged")])

        mas = MAS(
            agents={
                "agent_a": AgentConfig(agent=Agent(model=FunctionModel(handler_a))),
                "agent_b": AgentConfig(agent=Agent(model=TestModel())),
            },
            interrupt_on_send=True,
        )

        result = await mas.run(entry_agent="agent_a", prompt="start")

        assert result.termination_reason == TerminationReason.COMPLETED
        # First run: 1 LLM call (interrupted after send_message, no second turn)
        # Second run (B's reply): 1 LLM call (sees prior tools, returns text)
        assert llm_a_calls == 2
