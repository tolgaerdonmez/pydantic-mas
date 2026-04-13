"""Tests for OTel tracing integration."""

import json

import logfire
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_evals.otel._context_subtree import context_subtree

from pydantic_mas import MAS, AgentConfig, Budget

# Configure logfire once for the test module (no network calls).
logfire.configure(send_to_logfire=False)
logfire.instrument_pydantic_ai()


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
                        args=json.dumps({"target_agent": target, "content": content}),
                        tool_call_id="call-1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="Done")])

    return FunctionModel(handler)


class TestMASParentSpan:
    """MAS.run() should create a root OTel span wrapping all activity."""

    async def test_mas_run_creates_parent_span(self):
        mas = MAS(
            agents={
                "alpha": AgentConfig(agent=Agent(TestModel(custom_output_text="ok")))
            },
            budget=Budget(max_total_messages=5, timeout_seconds=5),
        )
        with context_subtree() as tree:
            await mas.run("alpha", "hello")

        roots = [s for s in tree.roots if s.name == "MAS run"]
        assert len(roots) == 1

    async def test_parent_span_has_attributes(self):
        mas = MAS(
            agents={
                "alpha": AgentConfig(agent=Agent(TestModel(custom_output_text="ok"))),
                "beta": AgentConfig(agent=Agent(TestModel(custom_output_text="ok"))),
            },
            budget=Budget(max_total_messages=5, timeout_seconds=5),
        )
        with context_subtree() as tree:
            await mas.run("alpha", "hello")

        root = next(s for s in tree.roots if s.name == "MAS run")
        assert root.attributes["mas.entry_agent"] == "alpha"
        assert root.attributes["mas.agent_count"] == 2

    async def test_agent_runs_nested_under_parent(self):
        mas = MAS(
            agents={
                "alpha": AgentConfig(agent=Agent(TestModel(custom_output_text="ok")))
            },
            budget=Budget(max_total_messages=5, timeout_seconds=5),
        )
        with context_subtree() as tree:
            await mas.run("alpha", "hello")

        root = next(s for s in tree.roots if s.name == "MAS run")
        agent_children = [c for c in root.children if c.name == "agent run"]
        assert len(agent_children) >= 1

    async def test_agent_names_in_spans(self):
        mas = MAS(
            agents={
                "coordinator": AgentConfig(
                    agent=Agent(TestModel(custom_output_text="ok"))
                )
            },
            budget=Budget(max_total_messages=5, timeout_seconds=5),
        )
        with context_subtree() as tree:
            await mas.run("coordinator", "hello")

        root = next(s for s in tree.roots if s.name == "MAS run")
        agent_span = root.children[0]
        assert agent_span.attributes["agent_name"] == "coordinator"

    async def test_multi_agent_all_nested(self):
        """With 2 agents communicating, all agent run spans share the same parent."""
        mas = MAS(
            agents={
                "sender": AgentConfig(agent=Agent(_model_that_sends("receiver", "hi"))),
                "receiver": AgentConfig(
                    agent=Agent(TestModel(custom_output_text="got it"))
                ),
            },
            budget=Budget(max_total_messages=10, timeout_seconds=5),
        )
        with context_subtree() as tree:
            await mas.run("sender", "start")

        root = next(s for s in tree.roots if s.name == "MAS run")
        agent_names = {
            c.attributes["agent_name"] for c in root.children if c.name == "agent run"
        }
        assert "sender" in agent_names
        assert "receiver" in agent_names


class TestSpanAttributes:
    """MAS span should record termination metadata."""

    async def test_span_records_completed(self):
        mas = MAS(
            agents={
                "alpha": AgentConfig(agent=Agent(TestModel(custom_output_text="ok")))
            },
            budget=Budget(max_total_messages=5, timeout_seconds=5),
        )
        with context_subtree() as tree:
            await mas.run("alpha", "hello")

        root = next(s for s in tree.roots if s.name == "MAS run")
        assert root.attributes["mas.termination_reason"] == "completed"
        assert root.attributes["mas.message_count"] == 1  # system -> alpha

    async def test_span_records_budget_exceeded(self):
        """Budget exceeded on initial message sets termination_reason on span."""
        mas = MAS(
            agents={
                "alpha": AgentConfig(agent=Agent(TestModel(custom_output_text="ok")))
            },
            budget=Budget(max_total_messages=0),
        )
        with context_subtree() as tree:
            await mas.run("alpha", "hello")

        root = next(s for s in tree.roots if s.name == "MAS run")
        assert root.attributes["mas.termination_reason"] == "budget_exceeded"

    async def test_span_records_timeout(self):
        async def block_forever(ctx) -> str:
            """Tool that never returns."""
            import asyncio

            await asyncio.Event().wait()
            return "never"  # pragma: no cover

        agent = Agent(
            TestModel(call_tools=["block_forever"]),
            tools=[block_forever],
        )
        mas = MAS(
            agents={"stuck": AgentConfig(agent=agent)},
            budget=Budget(timeout_seconds=0.2),
        )
        with context_subtree() as tree:
            await mas.run("stuck", "go")

        root = next(s for s in tree.roots if s.name == "MAS run")
        assert root.attributes["mas.termination_reason"] == "timeout"
