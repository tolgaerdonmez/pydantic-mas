"""Tests for AgentConfig.agent_factory and per-message Agent resolution."""

import json
from typing import Any

import pytest
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_mas._config import AgentConfig, FactoryContext
from pydantic_mas._hooks import MASHooks, MASInsertContext
from pydantic_mas._mas import MAS
from pydantic_mas._message import Message, MessageType
from pydantic_mas._result import TerminationReason


# ---------------------------------------------------------------------------
# AgentConfig API: factory branch + XOR
# ---------------------------------------------------------------------------


class TestAgentConfigFactoryBranch:
    def test_accepts_agent_factory(self):
        def factory(ctx: FactoryContext[Any]) -> Agent:
            return Agent(model=TestModel())

        cfg = AgentConfig(agent_factory=factory)
        assert cfg.agent_factory is factory
        assert cfg.agent is None

    def test_xor_neither_raises(self):
        with pytest.raises(ValueError):
            AgentConfig()

    def test_xor_both_raises(self):
        with pytest.raises(ValueError):

            def factory(ctx: FactoryContext[Any]) -> Agent:
                return Agent(model=TestModel())

            AgentConfig(agent=Agent(model=TestModel()), agent_factory=factory)

    def test_static_agent_still_works(self):
        a = Agent(model=TestModel())
        cfg = AgentConfig(agent=a)
        assert cfg.agent is a
        assert cfg.agent_factory is None


class TestFactoryContextShape:
    def test_fields_present(self):
        ctx = FactoryContext[Any](
            agent_id="x",
            incoming_message=Message(
                sender="s", receiver="x", type=MessageType.REQUEST, content="c"
            ),
            history=[],
            deps=None,
        )
        assert ctx.agent_id == "x"
        assert ctx.incoming_message.content == "c"
        assert ctx.history == []
        assert ctx.deps is None
        # router/current_depth default to None/0 when constructed bare;
        # AgentNode populates them with live values per message.
        assert ctx.router is None
        assert ctx.current_depth == 0


# ---------------------------------------------------------------------------
# Factory invocation: called once per incoming message, sees the message
# ---------------------------------------------------------------------------


class TestFactoryInvokedPerMessage:
    async def test_factory_called_with_incoming_message(self):
        captured: list[FactoryContext[Any]] = []

        baseline = Agent(model=TestModel())

        def factory(ctx: FactoryContext[Any]) -> Agent:
            captured.append(ctx)
            return baseline

        mas = MAS(agents={"solo": AgentConfig(agent_factory=factory)})
        result = await mas.run(entry_agent="solo", prompt="hello")

        assert result.termination_reason == TerminationReason.COMPLETED
        assert len(captured) == 1
        assert captured[0].agent_id == "solo"
        assert captured[0].incoming_message.content == "hello"
        assert captured[0].incoming_message.sender == "system"

    async def test_factory_can_be_async(self):
        called = 0

        async def factory(ctx: FactoryContext[Any]) -> Agent:
            nonlocal called
            called += 1
            return Agent(model=TestModel())

        mas = MAS(agents={"solo": AgentConfig(agent_factory=factory)})
        result = await mas.run(entry_agent="solo", prompt="hi")

        assert result.termination_reason == TerminationReason.COMPLETED
        assert called == 1

    async def test_factory_history_reflects_prior_turns(self):
        """Two REQUESTs into the same agent — second call sees non-empty history."""
        histories_seen: list[int] = []

        baseline = Agent(model=TestModel())

        def factory(ctx: FactoryContext[Any]) -> Agent:
            histories_seen.append(len(ctx.history))
            return baseline

        # Use a coordinator that pings 'solo' twice.
        call_count = 0

        def coord_handler(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {
                                    "target_agent": "solo",
                                    "content": f"ping-{call_count}",
                                }
                            ),
                            tool_call_id=f"c-{call_count}",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="done")])

        mas = MAS(
            agents={
                "coord": AgentConfig(agent=Agent(model=FunctionModel(coord_handler))),
                "solo": AgentConfig(agent_factory=factory),
            }
        )
        result = await mas.run(entry_agent="coord", prompt="kickoff")
        assert result.termination_reason == TerminationReason.COMPLETED

        # Factory invoked twice; second invocation must see history grown by turn 1.
        assert len(histories_seen) == 2
        assert histories_seen[0] == 0
        assert histories_seen[1] > 0


# ---------------------------------------------------------------------------
# Kickoff metadata propagation: mas.run(metadata=...) lands on the entry msg
# ---------------------------------------------------------------------------


class TestKickoffMetadata:
    async def test_metadata_lands_on_entry_message(self):
        captured: list[FactoryContext[Any]] = []

        def factory(ctx: FactoryContext[Any]) -> Agent:
            captured.append(ctx)
            return Agent(model=TestModel())

        mas = MAS(agents={"solo": AgentConfig(agent_factory=factory)})
        await mas.run(
            entry_agent="solo",
            prompt="hi",
            metadata={"output_schema": "MySchema", "tag": 42},
        )
        assert len(captured) == 1
        assert captured[0].incoming_message.metadata == {
            "output_schema": "MySchema",
            "tag": 42,
        }


# ---------------------------------------------------------------------------
# End-to-end: structured output via factory + metadata on REPLY
# ---------------------------------------------------------------------------


class TripPlan(BaseModel):
    destination: str
    days: int


class TestStructuredOutputE2E:
    async def test_reply_metadata_carries_live_structured_output(self):
        """coord -> planner with output_schema; planner produces TripPlan;
        the reply back to coord carries the live TripPlan in metadata
        and a string in content."""

        def planner_factory(ctx: FactoryContext[Any]) -> Agent:
            schema = ctx.incoming_message.metadata.get("output_schema")
            if schema is None:
                return Agent(model=TestModel())
            return Agent(model=TestModel(), output_type=schema)

        # Coordinator sends one structured request via a custom tool that
        # routes a message with metadata. We attach a *developer* tool that
        # uses the framework router; expose it via deps.

        # Easier path: use a FunctionModel that emits send_message, then have
        # an on_request_insert hook stamp metadata when caller=='coord'.
        def coord_handler(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            if not any(
                isinstance(p, ToolCallPart)
                for m in messages
                for p in getattr(m, "parts", [])
            ):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "planner", "content": "plan tokyo"}
                            ),
                            tool_call_id="c-1",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="ok")])

        async def stamp_schema(ctx: MASInsertContext[Any, Any]) -> Message:
            if (
                ctx.message.type == MessageType.REQUEST
                and ctx.caller_id == "coord"
                and ctx.callee_id == "planner"
            ):
                return ctx.message.model_copy(
                    update={"metadata": {"output_schema": TripPlan}}
                )
            return ctx.message

        reply_ctxs: list[MASInsertContext[Any, Any]] = []

        async def capture_reply(ctx: MASInsertContext[Any, Any]) -> Message:
            reply_ctxs.append(ctx)
            return ctx.message

        mas = MAS(
            agents={
                "coord": AgentConfig(agent=Agent(model=FunctionModel(coord_handler))),
                "planner": AgentConfig(agent_factory=planner_factory),
            },
            hooks=MASHooks(
                on_request_insert=stamp_schema, on_reply_insert=capture_reply
            ),
        )
        result = await mas.run(entry_agent="coord", prompt="kickoff")
        assert result.termination_reason == TerminationReason.COMPLETED

        # There must be a reply from planner to coord.
        assert len(reply_ctxs) == 1
        reply = reply_ctxs[0].message
        assert reply.sender == "planner"
        assert reply.receiver == "coord"
        assert reply.type == MessageType.REPLY

        # Content is a string (so the LLM can read it).
        assert isinstance(reply.content, str)
        assert reply.content  # non-empty

        # Metadata carries the live structured output object.
        live = reply.metadata.get("structured_output")
        assert isinstance(live, TripPlan)
