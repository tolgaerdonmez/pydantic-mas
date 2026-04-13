"""Tests for MASInstance."""

import asyncio
import json

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_mas._agent_node import AgentNode
from pydantic_mas._budget import Budget, BudgetTracker
from pydantic_mas._instance import MASInstance
from pydantic_mas._message import MessageType
from pydantic_mas._result import MASResult, TerminationReason
from pydantic_mas._router import MessageRouter


def _make_instance(
    agents: dict[str, Agent],
    budget: Budget | None = None,
) -> MASInstance:
    """Create a MASInstance with test agents, router, and tracker."""
    tracker = BudgetTracker(budget or Budget())
    router = MessageRouter(tracker)
    nodes: list[AgentNode] = []
    for agent_id, agent in agents.items():
        node = AgentNode(
            agent_id=agent_id,
            agent=agent,
            router=router,
        )
        router.register(agent_id, node.inbox)
        nodes.append(node)
    return MASInstance(
        agent_nodes=nodes,
        router=router,
        budget_tracker=tracker,
    )


def _model_that_sends(target: str, content: str) -> FunctionModel:
    """Create a FunctionModel that calls send_message once, then returns text."""
    call_count = 0

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps({"target_agent": target, "content": content}),
                        tool_call_id="call-1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="Done")])

    return FunctionModel(handler)


class TestMASInstanceValidation:
    async def test_entry_agent_not_found_raises_value_error(self):
        """run() should raise ValueError if entry_agent is not registered."""
        instance = _make_instance({"agent_a": Agent(model=TestModel())})

        with pytest.raises(ValueError, match="not_registered"):
            await instance.run(entry_agent="not_registered", prompt="hi")

    async def test_entry_agent_found_does_not_raise(self):
        """run() should succeed when entry_agent is valid."""
        instance = _make_instance({"agent_a": Agent(model=TestModel())})

        result = await instance.run(entry_agent="agent_a", prompt="hi")
        assert isinstance(result, MASResult)


class TestMASInstanceTermination:
    async def test_single_agent_completes(self):
        """Single agent processes initial message and terminates COMPLETED."""
        instance = _make_instance({"agent_a": Agent(model=TestModel())})

        result = await instance.run(entry_agent="agent_a", prompt="do something")

        assert result.termination_reason == TerminationReason.COMPLETED

    async def test_two_agents_request_reply(self):
        """Agent A sends to B, B processes and auto-replies, all terminate COMPLETED."""
        agents = {
            "agent_a": Agent(model=_model_that_sends("agent_b", "hello B")),
            "agent_b": Agent(model=TestModel()),
        }
        instance = _make_instance(agents)

        result = await instance.run(entry_agent="agent_a", prompt="start")

        assert result.termination_reason == TerminationReason.COMPLETED

        # Message flow: system->A, A->B (send_message), B->A (auto-reply)
        senders_receivers = [(m.sender, m.receiver) for m in result.message_log]
        assert ("system", "agent_a") in senders_receivers
        assert ("agent_a", "agent_b") in senders_receivers
        assert ("agent_b", "agent_a") in senders_receivers

    async def test_timeout_terminates(self):
        """Agent blocked by a long-running tool, timeout fires, terminates TIMEOUT."""
        never_resolve = asyncio.Event()

        async def blocking_tool(value: str) -> str:
            """A tool that blocks until cancelled.

            Args:
                value: Dummy value.
            """
            await never_resolve.wait()
            return "done"

        agent = Agent(
            model=TestModel(call_tools=["blocking_tool"]),
            tools=[Tool(blocking_tool)],
        )
        instance = _make_instance({"agent_a": agent})

        result = await instance.run(
            entry_agent="agent_a",
            prompt="go",
            timeout=0.2,
        )

        assert result.termination_reason == TerminationReason.TIMEOUT

    async def test_timeout_from_budget_config(self):
        """Timeout configured in Budget is enforced."""
        never_resolve = asyncio.Event()

        async def blocking_tool(value: str) -> str:
            """Block forever.

            Args:
                value: Dummy value.
            """
            await never_resolve.wait()
            return "done"

        agent = Agent(
            model=TestModel(call_tools=["blocking_tool"]),
            tools=[Tool(blocking_tool)],
        )
        instance = _make_instance(
            {"agent_a": agent},
            budget=Budget(timeout_seconds=0.2),
        )

        result = await instance.run(entry_agent="agent_a", prompt="go")

        assert result.termination_reason == TerminationReason.TIMEOUT

    async def test_budget_exceeded_on_initial_message(self):
        """Budget with max_total_messages=0 causes immediate BUDGET_EXCEEDED."""
        instance = _make_instance(
            {"agent_a": Agent(model=TestModel())},
            budget=Budget(max_total_messages=0),
        )

        result = await instance.run(entry_agent="agent_a", prompt="hi")

        assert result.termination_reason == TerminationReason.BUDGET_EXCEEDED

    async def test_agent_crash_terminates_without_hanging(self):
        """MAS terminates COMPLETED (not TIMEOUT) when an agent crashes mid-processing.

        Regression test: if run_loop doesn't re-set _idle_event on crash,
        _monitor_termination hangs forever on the dead agent's idle_event.
        """

        def crashing_handler(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            raise RuntimeError("Simulated LLM API failure")

        agents = {
            "agent_a": Agent(model=FunctionModel(crashing_handler)),
            "agent_b": Agent(model=TestModel()),
        }
        instance = _make_instance(agents)

        # timeout=2 is a safety net to prevent infinite hang. With the bug,
        # the monitor waits on the dead agent's idle_event until timeout fires.
        result = await instance.run(entry_agent="agent_a", prompt="go", timeout=2)

        # With the fix, MAS detects quiescence immediately → COMPLETED, not TIMEOUT.
        assert result.termination_reason == TerminationReason.COMPLETED


class TestMASInstanceResult:
    async def test_result_message_log(self):
        """MASResult.message_log contains all routed messages."""
        instance = _make_instance({"agent_a": Agent(model=TestModel())})

        result = await instance.run(entry_agent="agent_a", prompt="hello")

        # At minimum, the initial system->agent_a message
        assert len(result.message_log) >= 1
        assert result.message_log[0].sender == "system"
        assert result.message_log[0].receiver == "agent_a"
        assert result.message_log[0].content == "hello"
        assert result.message_log[0].type == MessageType.REQUEST

    async def test_result_agent_histories(self):
        """MASResult.agent_histories has per-agent conversation histories."""
        agents = {
            "agent_a": Agent(model=_model_that_sends("agent_b", "hi B")),
            "agent_b": Agent(model=TestModel()),
        }
        instance = _make_instance(agents)

        result = await instance.run(entry_agent="agent_a", prompt="start")

        # Both agents should have non-empty histories
        assert "agent_a" in result.agent_histories
        assert "agent_b" in result.agent_histories
        assert len(result.agent_histories["agent_a"]) > 0
        assert len(result.agent_histories["agent_b"]) > 0

    async def test_result_budget_usage(self):
        """MASResult.budget_usage reflects actual budget consumption."""
        agents = {
            "agent_a": Agent(model=_model_that_sends("agent_b", "msg")),
            "agent_b": Agent(model=TestModel()),
        }
        instance = _make_instance(agents)

        result = await instance.run(entry_agent="agent_a", prompt="go")

        assert result.budget_usage.total_messages >= 1
        assert isinstance(result.budget_usage.per_agent_messages, dict)

    async def test_histories_preserved_after_timeout(self):
        """Agent histories are not lost after timeout termination."""
        call_count = 0

        async def slow_tool(value: str) -> str:
            """Block on second call, respond on first.

            Args:
                value: Dummy value.
            """
            nonlocal call_count
            call_count += 1
            if call_count > 1:
                await asyncio.Event().wait()  # block forever
            return "first call done"

        agent = Agent(
            model=TestModel(call_tools=["slow_tool"]),
            tools=[Tool(slow_tool)],
        )
        instance = _make_instance({"agent_a": agent})

        result = await instance.run(
            entry_agent="agent_a",
            prompt="process this",
            timeout=0.5,
        )

        # Even after timeout, the agent_a key should exist
        assert "agent_a" in result.agent_histories

    async def test_three_agents_chain(self):
        """A -> B -> C chain: all messages arrive and result is COMPLETED."""
        agents = {
            "agent_a": Agent(model=_model_that_sends("agent_b", "for B")),
            "agent_b": Agent(model=_model_that_sends("agent_c", "for C")),
            "agent_c": Agent(model=TestModel()),
        }
        instance = _make_instance(agents)

        result = await instance.run(entry_agent="agent_a", prompt="start chain")

        assert result.termination_reason == TerminationReason.COMPLETED

        senders_receivers = [(m.sender, m.receiver) for m in result.message_log]
        assert ("system", "agent_a") in senders_receivers
        assert ("agent_a", "agent_b") in senders_receivers
        assert ("agent_b", "agent_c") in senders_receivers

        # All agents should have histories
        for agent_id in ["agent_a", "agent_b", "agent_c"]:
            assert len(result.agent_histories[agent_id]) > 0
