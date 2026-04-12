"""Tests for AgentNode."""

import asyncio
import json

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_mas._agent_node import AgentNode, AgentState
from pydantic_mas._budget import Budget, BudgetTracker
from pydantic_mas._formatter import default_message_formatter
from pydantic_mas._message import Message, MessageType
from pydantic_mas._router import MessageRouter


def _make_message(
    sender: str = "system",
    receiver: str = "agent_a",
    content: str = "Hello",
    type: MessageType = MessageType.REQUEST,
    depth: int = 0,
    in_reply_to: str | None = None,
) -> Message:
    return Message(
        sender=sender,
        receiver=receiver,
        type=type,
        content=content,
        depth=depth,
        in_reply_to=in_reply_to,
    )


def _make_node(
    agent_id: str = "agent_a",
    budget: Budget | None = None,
    call_tools: list[str] | str = [],
) -> tuple[AgentNode, MessageRouter, BudgetTracker]:
    """Create an AgentNode with a TestModel agent, router, and tracker."""
    tracker = BudgetTracker(budget or Budget())
    router = MessageRouter(tracker)

    agent = Agent(model=TestModel(call_tools=call_tools))
    node = AgentNode(
        agent_id=agent_id,
        agent=agent,
        router=router,
        message_formatter=default_message_formatter,
    )
    router.register(agent_id, node.inbox)
    return node, router, tracker


class TestAgentNodeState:
    async def test_initial_state_is_idle(self):
        node, _, _ = _make_node()
        assert node.state == AgentState.IDLE

    async def test_idle_event_set_initially(self):
        node, _, _ = _make_node()
        assert node.idle_event.is_set()

    async def test_state_transitions_to_processing_and_back(self):
        node, _, _ = _make_node()
        node.inbox.put_nowait(_make_message())

        task = asyncio.create_task(node.run_loop())
        # Let the loop process one message
        await asyncio.sleep(0.1)

        # After processing, should be back to IDLE waiting for next message
        assert node.state == AgentState.IDLE
        assert node.idle_event.is_set()

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_cancellation_preserves_history(self):
        node, _, _ = _make_node()
        node.inbox.put_nowait(_make_message(content="First message"))

        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.1)

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # History should have been updated from the processed message
        assert len(node.history) > 0


class TestAgentNodeHistory:
    async def test_history_grows_after_processing(self):
        node, _, _ = _make_node()
        node.inbox.put_nowait(_make_message(content="First"))

        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.1)

        history_after_first = len(node.history)
        assert history_after_first > 0

        node.inbox.put_nowait(_make_message(content="Second"))
        await asyncio.sleep(0.1)

        assert len(node.history) > history_after_first

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

    async def test_current_message_tracked(self):
        node, _, _ = _make_node()
        msg = _make_message(content="Track me", depth=3)
        node.inbox.put_nowait(msg)

        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.1)

        # After processing, current_message should be the last processed
        assert node.current_message is not None
        assert node.current_message.content == "Track me"
        assert node.current_depth == 3

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


def _model_that_calls_tool(tool_name: str, args: dict) -> FunctionModel:
    """Create a FunctionModel that calls a specific tool then returns text."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        for msg in messages:
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    if hasattr(part, "tool_name"):
                        return ModelResponse(parts=[TextPart(content="Done")])

        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=tool_name,
                    args=json.dumps(args),
                    tool_call_id="call-1",
                )
            ]
        )

    return FunctionModel(handler)


def _model_that_calls_send_message(
    target: str, content: str
) -> FunctionModel:
    """Create a FunctionModel that calls send_message then returns text."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # If we already got tool results back, return final text
        for msg in messages:
            if hasattr(msg, "parts"):
                for part in msg.parts:
                    if hasattr(part, "tool_name"):
                        return ModelResponse(parts=[TextPart(content="Done")])

        # First call: invoke send_message
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="send_message",
                    args=json.dumps(
                        {"target_agent": target, "content": content}
                    ),
                    tool_call_id="call-1",
                )
            ]
        )

    return FunctionModel(handler)


class TestAgentNodeSendMessage:
    async def test_send_message_routes_to_target(self):
        """When the agent calls send_message, the message appears in the target's inbox."""
        tracker = BudgetTracker(Budget())
        router = MessageRouter(tracker)

        model = _model_that_calls_send_message("agent_b", "hello from a")
        agent = Agent(model=model)
        node_a = AgentNode(
            agent_id="agent_a",
            agent=agent,
            router=router,
            message_formatter=default_message_formatter,
        )
        router.register("agent_a", node_a.inbox)

        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node_a.inbox.put_nowait(_make_message(sender="system", receiver="agent_a"))

        task = asyncio.create_task(node_a.run_loop())
        await asyncio.sleep(0.3)

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # send_message should have routed a message to agent_b
        assert not inbox_b.empty()
        routed_msg = inbox_b.get_nowait()
        assert routed_msg.sender == "agent_a"
        assert routed_msg.receiver == "agent_b"
        assert routed_msg.content == "hello from a"
        assert routed_msg.type == MessageType.REQUEST
        assert routed_msg.depth == 1  # initial depth 0 + 1

    async def test_send_message_to_unknown_agent_returns_error(self):
        """send_message to a non-existent agent returns an error string, not an exception."""
        tracker = BudgetTracker(Budget())
        router = MessageRouter(tracker)

        model = _model_that_calls_send_message("nonexistent", "hello")
        agent = Agent(model=model)
        node = AgentNode(
            agent_id="agent_a",
            agent=agent,
            router=router,
            message_formatter=default_message_formatter,
        )
        router.register("agent_a", node.inbox)

        node.inbox.put_nowait(_make_message(sender="system", receiver="agent_a"))

        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # Agent should have completed without crashing
        assert len(node.history) > 0
        # No message should be in the log (route failed gracefully)
        assert len(router.message_log) == 0


class TestAgentNodeLastOutputReply:
    async def test_auto_reply_to_request_from_agent(self):
        """When processing a request from another agent, auto-reply with output."""
        node_a, router, _ = _make_node(agent_id="agent_a")
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        # Message from agent_b (not system) — should trigger auto-reply
        msg = _make_message(
            sender="agent_b", receiver="agent_a", content="Help me"
        )
        node_a.inbox.put_nowait(msg)

        task = asyncio.create_task(node_a.run_loop())
        await asyncio.sleep(0.2)

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # A reply should have been routed back to agent_b
        assert not inbox_b.empty()
        reply = inbox_b.get_nowait()
        assert reply.type == MessageType.REPLY
        assert reply.sender == "agent_a"
        assert reply.receiver == "agent_b"
        assert reply.in_reply_to == msg.id

    async def test_no_reply_to_system_message(self):
        """System messages should not trigger auto-reply."""
        node, router, tracker = _make_node(agent_id="agent_a")

        msg = _make_message(sender="system", receiver="agent_a")
        node.inbox.put_nowait(msg)

        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.2)

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # Only the initial system message should be in the log (no reply routed)
        # The system message itself is NOT routed through the router in this test,
        # so the router log should be empty (no auto-reply generated)
        assert len(router.message_log) == 0

    async def test_no_reply_to_reply_message(self):
        """Reply messages should not trigger auto-reply (prevents ping-pong)."""
        node_a, router, _ = _make_node(agent_id="agent_a")
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        msg = _make_message(
            sender="agent_b",
            receiver="agent_a",
            type=MessageType.REPLY,
            content="Here is your answer",
            in_reply_to="some-msg-id",
        )
        node_a.inbox.put_nowait(msg)

        task = asyncio.create_task(node_a.run_loop())
        await asyncio.sleep(0.2)

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # No reply should be sent back to agent_b
        assert inbox_b.empty()

    async def test_no_reply_to_notification(self):
        """Notification messages should not trigger auto-reply."""
        node_a, router, _ = _make_node(agent_id="agent_a")
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        msg = _make_message(
            sender="agent_b",
            receiver="agent_a",
            type=MessageType.NOTIFICATION,
            content="FYI",
        )
        node_a.inbox.put_nowait(msg)

        task = asyncio.create_task(node_a.run_loop())
        await asyncio.sleep(0.2)

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert inbox_b.empty()


class TestAgentNodePreservesDevTools:
    async def test_developer_tools_not_replaced_by_framework(self):
        """Framework-injected tools (send_message) must not replace developer tools."""
        tracker = BudgetTracker(Budget())
        router = MessageRouter(tracker)

        dev_tool_called = False

        async def my_dev_tool(query: str) -> str:
            """A developer-defined tool.

            Args:
                query: The search query.
            """
            nonlocal dev_tool_called
            dev_tool_called = True
            return f"result for: {query}"

        # Model calls the developer tool, not send_message
        model = _model_that_calls_tool("my_dev_tool", {"query": "test"})
        agent = Agent(model=model, tools=[Tool(my_dev_tool)])

        node = AgentNode(
            agent_id="agent_a",
            agent=agent,
            router=router,
            message_formatter=default_message_formatter,
        )
        router.register("agent_a", node.inbox)

        node.inbox.put_nowait(_make_message(sender="system", receiver="agent_a"))

        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert dev_tool_called, "Developer tool should have been called"

    async def test_both_dev_and_framework_tools_available(self):
        """Both developer tools and framework tools should be callable."""
        tracker = BudgetTracker(Budget())
        router = MessageRouter(tracker)

        dev_tool_called = False

        async def my_dev_tool(data: str) -> str:
            """A developer-defined tool.

            Args:
                data: Input data.
            """
            nonlocal dev_tool_called
            dev_tool_called = True
            return f"processed: {data}"

        # Model calls both tools in sequence
        call_count = 0

        def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="my_dev_tool",
                            args=json.dumps({"data": "hello"}),
                            tool_call_id="call-1",
                        )
                    ]
                )
            elif call_count == 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_b", "content": "from dev"}
                            ),
                            tool_call_id="call-2",
                        )
                    ]
                )
            else:
                return ModelResponse(parts=[TextPart(content="All done")])

        model = FunctionModel(handler)
        agent = Agent(model=model, tools=[Tool(my_dev_tool)])

        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        router.register("agent_b", inbox_b)

        node = AgentNode(
            agent_id="agent_a",
            agent=agent,
            router=router,
            message_formatter=default_message_formatter,
        )
        router.register("agent_a", node.inbox)

        node.inbox.put_nowait(_make_message(sender="system", receiver="agent_a"))

        task = asyncio.create_task(node.run_loop())
        await asyncio.sleep(0.3)

        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        assert dev_tool_called, "Developer tool should have been called"
        assert not inbox_b.empty(), "send_message should have routed to agent_b"


class TestAgentNodeFanOut:
    """Tests for an agent sending messages to multiple agents in one turn."""

    async def test_agent_sends_to_two_agents_in_one_turn(self):
        """Agent A sends to B and C in a single LLM turn (parallel tool calls)."""
        tracker = BudgetTracker(Budget())
        router = MessageRouter(tracker)

        call_count = 0

        def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # First call: send to both B and C in one turn
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_b", "content": "hello B"}
                            ),
                            tool_call_id="call-b",
                        ),
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_c", "content": "hello C"}
                            ),
                            tool_call_id="call-c",
                        ),
                    ]
                )
            return ModelResponse(parts=[TextPart(content="Done")])

        agent_a = Agent(model=FunctionModel(handler))
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        inbox_c: asyncio.Queue[Message] = asyncio.Queue()

        node_a = AgentNode(
            agent_id="agent_a",
            agent=agent_a,
            router=router,
            message_formatter=default_message_formatter,
        )
        router.register("agent_a", node_a.inbox)
        router.register("agent_b", inbox_b)
        router.register("agent_c", inbox_c)

        node_a.inbox.put_nowait(_make_message(sender="system", receiver="agent_a"))

        task = asyncio.create_task(node_a.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # Both B and C should have received messages
        assert not inbox_b.empty(), "agent_b should have a message"
        assert not inbox_c.empty(), "agent_c should have a message"

        msg_b = inbox_b.get_nowait()
        msg_c = inbox_c.get_nowait()

        assert msg_b.sender == "agent_a"
        assert msg_b.content == "hello B"
        assert msg_b.depth == 1

        assert msg_c.sender == "agent_a"
        assert msg_c.content == "hello C"
        assert msg_c.depth == 1

        # Budget should reflect exactly 2 messages (both sends)
        assert tracker.total_messages == 2
        assert tracker.per_agent_messages["agent_a"] == 2

    async def test_agent_sends_to_two_and_budget_counts_correctly(self):
        """Budget limit of 1 should block the second send in a fan-out."""
        tracker = BudgetTracker(Budget(max_total_messages=1))
        router = MessageRouter(tracker)

        def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Try to send to both B and C
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
                            {"target_agent": "agent_b", "content": "first"}
                        ),
                        tool_call_id="call-1",
                    ),
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {"target_agent": "agent_c", "content": "second"}
                        ),
                        tool_call_id="call-2",
                    ),
                ]
            )

        agent_a = Agent(model=FunctionModel(handler))
        inbox_b: asyncio.Queue[Message] = asyncio.Queue()
        inbox_c: asyncio.Queue[Message] = asyncio.Queue()

        node_a = AgentNode(
            agent_id="agent_a",
            agent=agent_a,
            router=router,
            message_formatter=default_message_formatter,
        )
        router.register("agent_a", node_a.inbox)
        router.register("agent_b", inbox_b)
        router.register("agent_c", inbox_c)

        node_a.inbox.put_nowait(_make_message(sender="system", receiver="agent_a"))

        task = asyncio.create_task(node_a.run_loop())
        await asyncio.sleep(0.3)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)

        # First send should succeed, second should fail (budget exceeded)
        assert not inbox_b.empty(), "first send should succeed"
        # The second send got a BudgetExceededError but the tool returns
        # an error string, so the agent didn't crash
        assert tracker.total_messages == 1
        assert len(node_a.history) > 0

    async def test_chain_a_sends_to_b_and_c_then_b_sends_to_c(self):
        """A sends to B and C. B then sends to C. All messages arrive correctly."""
        tracker = BudgetTracker(Budget(max_total_messages=20))
        router = MessageRouter(tracker)

        # Agent A: sends to B and C on first call
        def handler_a(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                if hasattr(msg, "parts"):
                    for part in msg.parts:
                        if hasattr(part, "tool_name"):
                            return ModelResponse(parts=[TextPart(content="A done")])

            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {"target_agent": "agent_b", "content": "task for B"}
                        ),
                        tool_call_id="a-to-b",
                    ),
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {"target_agent": "agent_c", "content": "task for C"}
                        ),
                        tool_call_id="a-to-c",
                    ),
                ]
            )

        # Agent B: sends to C when it receives a message
        def handler_b(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            for msg in messages:
                if hasattr(msg, "parts"):
                    for part in msg.parts:
                        if hasattr(part, "tool_name"):
                            return ModelResponse(parts=[TextPart(content="B done")])

            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {"target_agent": "agent_c", "content": "B forwarding to C"}
                        ),
                        tool_call_id="b-to-c",
                    )
                ]
            )

        # Agent C: just responds with text
        agent_c_pydantic = Agent(model=TestModel())

        node_a = AgentNode(
            agent_id="agent_a",
            agent=Agent(model=FunctionModel(handler_a)),
            router=router,
            message_formatter=default_message_formatter,
        )
        node_b = AgentNode(
            agent_id="agent_b",
            agent=Agent(model=FunctionModel(handler_b)),
            router=router,
            message_formatter=default_message_formatter,
        )
        node_c = AgentNode(
            agent_id="agent_c",
            agent=agent_c_pydantic,
            router=router,
            message_formatter=default_message_formatter,
        )

        router.register("agent_a", node_a.inbox)
        router.register("agent_b", node_b.inbox)
        router.register("agent_c", node_c.inbox)

        # Kick off: system message to A
        node_a.inbox.put_nowait(_make_message(sender="system", receiver="agent_a"))

        # Run all three agent loops concurrently
        task_a = asyncio.create_task(node_a.run_loop())
        task_b = asyncio.create_task(node_b.run_loop())
        task_c = asyncio.create_task(node_c.run_loop())

        await asyncio.sleep(0.5)

        task_a.cancel()
        task_b.cancel()
        task_c.cancel()
        await asyncio.gather(task_a, task_b, task_c, return_exceptions=True)

        # Verify the message flow:
        # 1. A -> B ("task for B")
        # 2. A -> C ("task for C")
        # 3. B -> C ("B forwarding to C")
        # Plus auto-replies from last_output strategy (B replies to A, C replies to A and B)

        # At minimum, the 3 explicit sends should be in the log
        senders_receivers = [
            (m.sender, m.receiver) for m in router.message_log
        ]
        assert ("agent_a", "agent_b") in senders_receivers
        assert ("agent_a", "agent_c") in senders_receivers
        assert ("agent_b", "agent_c") in senders_receivers

        # All agents should have non-empty history (they all processed messages)
        assert len(node_a.history) > 0
        assert len(node_b.history) > 0
        assert len(node_c.history) > 0

        # Budget should reflect all messages
        assert tracker.total_messages >= 3


class TestAgentNodeIdleEvent:
    async def test_idle_event_cleared_during_processing(self):
        """idle_event should be cleared while processing and set when idle."""
        node, _, _ = _make_node()

        processing_seen = False

        async def observe():
            nonlocal processing_seen
            # Wait for idle event to be cleared (processing started)
            while node.idle_event.is_set():
                await asyncio.sleep(0.01)
            processing_seen = True

        node.inbox.put_nowait(_make_message())

        task = asyncio.create_task(node.run_loop())
        observer = asyncio.create_task(observe())

        await asyncio.sleep(0.2)

        task.cancel()
        observer.cancel()
        await asyncio.gather(task, observer, return_exceptions=True)

        assert processing_seen
