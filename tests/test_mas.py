"""Tests for MAS orchestrator and AgentConfig."""

import json
from typing import Any

import pytest
from pydantic_ai import Agent, Tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_mas._budget import Budget
from pydantic_mas._config import AgentConfig, ReplyStrategy
from pydantic_mas._mas import MAS
from pydantic_mas._message import Message, MessageType
from pydantic_mas._result import TerminationReason


# ---------------------------------------------------------------------------
# AgentConfig
# ---------------------------------------------------------------------------


class TestAgentConfig:
    def test_static_deps(self):
        config = AgentConfig(agent=Agent(model=TestModel()), deps="my-deps")
        assert config.resolve_deps() == "my-deps"

    def test_no_deps_returns_none(self):
        config = AgentConfig(agent=Agent(model=TestModel()))
        assert config.resolve_deps() is None

    def test_deps_factory_called(self):
        call_count = 0

        def factory() -> str:
            nonlocal call_count
            call_count += 1
            return f"deps-{call_count}"

        config = AgentConfig(
            agent=Agent(model=TestModel()), deps_factory=factory
        )

        assert config.resolve_deps() == "deps-1"
        assert config.resolve_deps() == "deps-2"

    def test_factory_takes_precedence_over_static(self):
        config = AgentConfig(
            agent=Agent(model=TestModel()),
            deps="static",
            deps_factory=lambda: "from-factory",
        )
        assert config.resolve_deps() == "from-factory"


class TestReplyStrategy:
    def test_values(self):
        assert ReplyStrategy.LAST_OUTPUT == "last_output"
        assert ReplyStrategy.ANSWER_TOOL == "answer_tool"
        assert ReplyStrategy.SEND_MESSAGE == "send_message"

    def test_is_str(self):
        assert isinstance(ReplyStrategy.LAST_OUTPUT, str)


# ---------------------------------------------------------------------------
# MAS helpers
# ---------------------------------------------------------------------------


def _model_that_sends(target: str, content: str) -> FunctionModel:
    """FunctionModel that calls send_message once, then returns text."""
    call_count = 0

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal call_count
        call_count += 1

        if call_count == 1:
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
        return ModelResponse(parts=[TextPart(content="Done")])

    return FunctionModel(handler)


# ---------------------------------------------------------------------------
# MAS validation
# ---------------------------------------------------------------------------


class TestMASValidation:
    async def test_unknown_entry_agent_raises(self):
        mas = MAS(agents={"agent_a": AgentConfig(agent=Agent(model=TestModel()))})

        with pytest.raises(ValueError, match="not_here"):
            await mas.run(entry_agent="not_here", prompt="hi")


# ---------------------------------------------------------------------------
# MAS.run()
# ---------------------------------------------------------------------------


class TestMASRun:
    async def test_single_agent_completes(self):
        mas = MAS(
            agents={"agent_a": AgentConfig(agent=Agent(model=TestModel()))}
        )

        result = await mas.run(entry_agent="agent_a", prompt="hello")

        assert result.termination_reason == TerminationReason.COMPLETED
        assert len(result.message_log) >= 1

    async def test_two_agents_communicate(self):
        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(model=_model_that_sends("agent_b", "hi B"))
                ),
                "agent_b": AgentConfig(agent=Agent(model=TestModel())),
            }
        )

        result = await mas.run(entry_agent="agent_a", prompt="start")

        assert result.termination_reason == TerminationReason.COMPLETED
        pairs = [(m.sender, m.receiver) for m in result.message_log]
        assert ("system", "agent_a") in pairs
        assert ("agent_a", "agent_b") in pairs
        assert ("agent_b", "agent_a") in pairs

    async def test_budget_enforcement(self):
        mas = MAS(
            agents={
                "agent_a": AgentConfig(agent=Agent(model=TestModel())),
            },
            budget=Budget(max_total_messages=0),
        )

        result = await mas.run(entry_agent="agent_a", prompt="go")

        assert result.termination_reason == TerminationReason.BUDGET_EXCEEDED

    async def test_timeout_enforcement(self):
        import asyncio

        never = asyncio.Event()

        async def blocking_tool(value: str) -> str:
            """Block forever.

            Args:
                value: Dummy.
            """
            await never.wait()
            return "done"

        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(
                        model=TestModel(call_tools=["blocking_tool"]),
                        tools=[Tool(blocking_tool)],
                    )
                ),
            },
            budget=Budget(timeout_seconds=0.2),
        )

        result = await mas.run(entry_agent="agent_a", prompt="go")

        assert result.termination_reason == TerminationReason.TIMEOUT


class TestMASCustomFormatter:
    async def test_custom_formatter_is_used(self):
        formatted: list[str] = []

        def custom_formatter(message: Message) -> str:
            text = f"CUSTOM: {message.content}"
            formatted.append(text)
            return text

        mas = MAS(
            agents={"agent_a": AgentConfig(agent=Agent(model=TestModel()))},
            message_formatter=custom_formatter,
        )

        await mas.run(entry_agent="agent_a", prompt="hello")

        assert any("CUSTOM: hello" in f for f in formatted)


class TestMASDeps:
    async def test_static_deps_passed_to_agent(self):
        from pydantic_ai import RunContext

        received: dict[str, Any] = {}

        async def capture_tool(ctx: RunContext[str], value: str) -> str:
            """Capture deps from RunContext.

            Args:
                value: Dummy.
            """
            received["deps"] = ctx.deps
            return "captured"

        agent: Agent[str] = Agent(model=TestModel(call_tools=["capture_tool"]))
        agent.tool(capture_tool)

        mas = MAS(
            agents={"agent_a": AgentConfig(agent=agent, deps="my-deps")}
        )

        await mas.run(entry_agent="agent_a", prompt="go")

        assert received["deps"] == "my-deps"

    async def test_deps_factory_called_per_run(self):
        call_count = 0

        def factory() -> str:
            nonlocal call_count
            call_count += 1
            return f"run-{call_count}"

        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(model=TestModel()), deps_factory=factory
                )
            }
        )

        await mas.run(entry_agent="agent_a", prompt="first")
        await mas.run(entry_agent="agent_a", prompt="second")

        assert call_count == 2


class TestMASIsolation:
    async def test_multiple_runs_have_independent_state(self):
        mas = MAS(
            agents={"agent_a": AgentConfig(agent=Agent(model=TestModel()))},
            budget=Budget(max_total_messages=10),
        )

        result1 = await mas.run(entry_agent="agent_a", prompt="first")
        result2 = await mas.run(entry_agent="agent_a", prompt="second")

        assert result1.message_log[0].content == "first"
        assert result2.message_log[0].content == "second"
        # Budget counters are independent, not cumulative
        assert result1.budget_usage.total_messages == result2.budget_usage.total_messages
