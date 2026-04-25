"""Tests for the user-facing pattern: an externally-defined tool that
routes a message with metadata, leveraging FactoryContext.router."""

import json
from typing import Any

from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_mas import (
    MAS,
    AgentConfig,
    FactoryContext,
    MASHooks,
    MASInsertContext,
    Message,
    MessageType,
)
from pydantic_mas._result import TerminationReason


class TripPlan(BaseModel):
    destination: str
    days: int


SCHEMA_REGISTRY: dict[str, type[BaseModel]] = {"TripPlan": TripPlan}


def _coord_handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    """Stateless handler: emit the external tool call only when no prior tool
    call exists in history; otherwise emit text. Robust across re-runs of the
    same agent (factory rebuilds the model each invocation)."""
    has_prior_tool_call = any(
        isinstance(p, ToolCallPart) for m in messages for p in getattr(m, "parts", [])
    )
    if not has_prior_tool_call:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="send_message_external",
                    args=json.dumps(
                        {
                            "target_agent": "planner",
                            "content": "plan tokyo",
                            "output_schema": "TripPlan",
                        }
                    ),
                    tool_call_id="ext-1",
                )
            ]
        )
    return ModelResponse(parts=[TextPart(content="ack")])


def _coord_factory(ctx: FactoryContext[Any]) -> Agent:
    """Wires send_message_external using ctx.router and ctx.current_depth."""
    agent = Agent(model=FunctionModel(_coord_handler))
    router = ctx.router
    sender_id = ctx.agent_id
    next_depth = ctx.current_depth + 1

    @agent.tool_plain
    async def send_message_external(
        target_agent: str,
        content: str,
        output_schema: str,
    ) -> str:
        """Send a message to another agent, requesting a structured response."""
        schema_cls = SCHEMA_REGISTRY[output_schema]
        msg = router.route(
            sender=sender_id,
            receiver=target_agent,
            content=content,
            type=MessageType.REQUEST,
            depth=next_depth,
            metadata={"output_schema": schema_cls},
        )
        return f"sent to {target_agent} (id: {msg.id})"

    return agent


def _planner_factory(ctx: FactoryContext[Any]) -> Agent:
    schema = ctx.incoming_message.metadata.get("output_schema")
    if schema is None:
        return Agent(model=TestModel())
    return Agent(model=TestModel(), output_type=schema)


class TestExternalSendMessageTool:
    async def test_routes_with_schema_metadata_and_target_returns_structured(self):
        reply_ctxs: list[MASInsertContext[Any, Any]] = []

        async def capture_reply(ctx: MASInsertContext[Any, Any]) -> Message:
            reply_ctxs.append(ctx)
            return ctx.message

        mas = MAS(
            agents={
                "coord": AgentConfig(agent_factory=_coord_factory),
                "planner": AgentConfig(agent_factory=_planner_factory),
            },
            hooks=MASHooks(on_reply_insert=capture_reply),
        )
        result = await mas.run(entry_agent="coord", prompt="kickoff")
        assert result.termination_reason == TerminationReason.COMPLETED

        # The coord -> planner request carried output_schema=TripPlan in metadata.
        coord_to_planner = [
            m
            for m in result.message_log
            if m.sender == "coord" and m.receiver == "planner"
        ]
        assert len(coord_to_planner) == 1
        assert coord_to_planner[0].metadata.get("output_schema") is TripPlan

        # The reply back to coord carries a live TripPlan in metadata.
        assert len(reply_ctxs) == 1
        reply = reply_ctxs[0].message
        assert reply.sender == "planner"
        assert reply.receiver == "coord"
        assert isinstance(reply.metadata.get("structured_output"), TripPlan)


class TestFactoryContextExposesRouterAndDepth:
    async def test_router_and_current_depth_present(self):
        seen: list[FactoryContext[Any]] = []

        def factory(ctx: FactoryContext[Any]) -> Agent:
            seen.append(ctx)
            return Agent(model=TestModel())

        mas = MAS(agents={"solo": AgentConfig(agent_factory=factory)})
        await mas.run(entry_agent="solo", prompt="hi")

        assert len(seen) == 1
        # The router is the same one used to log the kickoff message.
        assert seen[0].router is not None
        assert seen[0].current_depth == 0
