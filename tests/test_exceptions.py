"""Fail-fast exception propagation: agent crashes surface out of mas.run().

Design intent (see _instance.py): an unhandled exception inside any agent
must propagate to the caller of mas.run(). The runtime does not contain or
swallow agent-internal failures. This file pins that contract.
"""

import asyncio

import pytest
from pydantic_ai import Agent, RunContext
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_mas._agent_node import AgentNode
from pydantic_mas._budget import Budget, BudgetTracker
from pydantic_mas._config import AgentConfig
from pydantic_mas._instance import MASInstance
from pydantic_mas._mas import MAS
from pydantic_mas._message import Message, MessageType
from pydantic_mas._router import MessageRouter


class Boom(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# 1. Tool raising an exception that pydantic-ai propagates (BaseException so
#    pydantic-ai's `except Exception` retry loop cannot swallow it).
# ---------------------------------------------------------------------------


class _ToolBoom(BaseException):
    """BaseException so pydantic-ai's tool-retry layer cannot catch it."""


async def _crash_tool(ctx: RunContext[None]) -> str:
    """Tool that raises an uncatchable exception."""
    raise _ToolBoom("tool exploded")


class TestToolExceptionPropagates:
    async def test_tool_raise_surfaces_from_run(self):
        agent: Agent[None] = Agent(
            model=FunctionModel(
                lambda m, i: ModelResponse(
                    parts=[ToolCallPart(tool_name="_crash_tool", args={})]
                )
            )
        )
        agent.tool(_crash_tool)

        mas = MAS(agents={"a": AgentConfig(agent=agent)})

        with pytest.raises(_ToolBoom, match="tool exploded"):
            await asyncio.wait_for(mas.run(entry_agent="a", prompt="go"), timeout=4.0)


# ---------------------------------------------------------------------------
# 2. Model handler raising — covers any failure inside agent.run() that
#    isn't a tool.
# ---------------------------------------------------------------------------


class TestModelExceptionPropagates:
    async def test_model_raise_surfaces_from_run(self):
        def crashing(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise Boom("model boom")

        agent: Agent[None] = Agent(model=FunctionModel(crashing))
        mas = MAS(agents={"a": AgentConfig(agent=agent)})

        with pytest.raises(Boom, match="model boom"):
            await asyncio.wait_for(mas.run(entry_agent="a", prompt="go"), timeout=4.0)


# ---------------------------------------------------------------------------
# 3. Crash in a multi-agent setup. A sends to B, B crashes — exception must
#    propagate and A must be cancelled cleanly (no stall).
# ---------------------------------------------------------------------------


def _a_sends_to_b() -> FunctionModel:
    """A's model: send_message to b, then finish."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        has_tool = any(
            isinstance(p, ToolReturnPart)
            for m in messages
            for p in getattr(m, "parts", [])
        )
        if has_tool:
            return ModelResponse(parts=[TextPart(content="A done")])
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name="send_message",
                    args={"target_agent": "b", "content": "hi"},
                )
            ]
        )

    return FunctionModel(handler)


class TestMultiAgentCrashPropagates:
    async def test_b_crash_surfaces_and_a_is_cancelled(self):
        def b_crashes(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise Boom("b died")

        a: Agent[None] = Agent(model=_a_sends_to_b())
        b: Agent[None] = Agent(model=FunctionModel(b_crashes))

        mas = MAS(
            agents={
                "a": AgentConfig(agent=a),
                "b": AgentConfig(agent=b),
            }
        )

        with pytest.raises(Boom, match="b died"):
            await asyncio.wait_for(mas.run(entry_agent="a", prompt="go"), timeout=4.0)


# ---------------------------------------------------------------------------
# 4. Direct stall test: agent task dies with a pending message in its inbox.
#    Under the OLD design this hung _monitor_termination forever. Under the
#    new design the exception must propagate immediately.
# ---------------------------------------------------------------------------


class TestCrashedAgentWithPendingInboxDoesNotStall:
    async def test_stall_scenario_now_raises(self):
        def crashing(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            raise Boom("instant crash")

        budget = BudgetTracker(Budget())
        router = MessageRouter(budget)

        agent: Agent[None] = Agent(model=FunctionModel(crashing))
        peers: dict = {}
        node = AgentNode(
            agent_id="a",
            agent=agent,
            router=router,
            peers=peers,
            enforce_reply_protocol=False,
        )
        router.register("a", node.inbox)
        peers["a"] = node

        instance = MASInstance(agent_nodes=[node], router=router, budget_tracker=budget)

        # Pre-load a second message into the inbox. The first triggers the
        # crash; the second sits there. Old monitor would loop forever.
        await node.inbox.put(
            Message(
                sender="system",
                receiver="a",
                content="dangling",
                type=MessageType.REQUEST,
                depth=0,
            )
        )

        with pytest.raises(Boom, match="instant crash"):
            await asyncio.wait_for(
                instance.run(entry_agent="a", prompt="trigger"), timeout=4.0
            )
