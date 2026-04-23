"""Tests for MAS insertion hooks — on_request_insert / on_reply_insert."""

import json
from dataclasses import dataclass
from typing import Any

from pydantic_ai import Agent
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
    TextPart,
    ToolCallPart,
)
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_mas._config import AgentConfig
from pydantic_mas._hooks import MASHooks, MASInsertContext
from pydantic_mas._mas import MAS
from pydantic_mas._message import Message, MessageType
from pydantic_mas._result import TerminationReason


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class PlannerDeps:
    label: str = "planner-deps"


@dataclass
class ResearcherDeps:
    label: str = "researcher-deps"


def _model_that_sends(target: str, content: str) -> FunctionModel:
    """FunctionModel that calls send_message once to `target`, then returns text."""
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


def _two_agent_mas(
    hooks: MASHooks | None = None,
    request_content: str = "task for you",
) -> MAS:
    """Planner sends a request to researcher; researcher replies via LAST_OUTPUT."""
    planner_agent = Agent(
        model=_model_that_sends("researcher", request_content),
    )
    researcher_agent = Agent(
        model=TestModel(custom_output_text="the answer is 42"),
    )

    return MAS(
        agents={
            "planner": AgentConfig(agent=planner_agent, deps=PlannerDeps()),
            "researcher": AgentConfig(agent=researcher_agent, deps=ResearcherDeps()),
        },
        hooks=hooks,
    )


# ---------------------------------------------------------------------------
# Test 1 — on_request_insert fires with both deps populated
# ---------------------------------------------------------------------------


class TestOnRequestInsert:
    async def test_fires_with_both_deps(self):
        captured: list[MASInsertContext] = []

        async def record(ctx: MASInsertContext[Any, Any]) -> Message:
            captured.append(ctx)
            return ctx.message

        mas = _two_agent_mas(hooks=MASHooks(on_request_insert=record))
        result = await mas.run(entry_agent="planner", prompt="start")

        assert result.termination_reason == TerminationReason.COMPLETED

        # Expect two request insertions: system→planner, planner→researcher
        request_ctxs = [c for c in captured if c.message.type == MessageType.REQUEST]
        assert len(request_ctxs) == 2

        # First: system → planner. Caller is "system", deps are None.
        sys_call = request_ctxs[0]
        assert sys_call.caller_id == "system"
        assert sys_call.caller_deps is None
        assert sys_call.callee_id == "planner"
        assert isinstance(sys_call.callee_deps, PlannerDeps)

        # Second: planner → researcher. Both deps populated.
        p2r = request_ctxs[1]
        assert p2r.caller_id == "planner"
        assert isinstance(p2r.caller_deps, PlannerDeps)
        assert p2r.callee_id == "researcher"
        assert isinstance(p2r.callee_deps, ResearcherDeps)


# ---------------------------------------------------------------------------
# Test 2 — on_reply_insert: caller=originator, callee=responder
# ---------------------------------------------------------------------------


class TestOnReplyInsert:
    async def test_roles_stable_across_direction(self):
        captured: list[MASInsertContext] = []

        async def record(ctx: MASInsertContext[Any, Any]) -> Message:
            captured.append(ctx)
            return ctx.message

        mas = _two_agent_mas(hooks=MASHooks(on_reply_insert=record))
        result = await mas.run(entry_agent="planner", prompt="start")

        assert result.termination_reason == TerminationReason.COMPLETED

        # The reply from researcher → planner is the one insertion.
        assert len(captured) == 1
        ctx = captured[0]
        assert ctx.message.type == MessageType.REPLY
        # Roles stay oriented to the original request: planner initiated.
        assert ctx.caller_id == "planner"
        assert isinstance(ctx.caller_deps, PlannerDeps)
        assert ctx.callee_id == "researcher"
        assert isinstance(ctx.callee_deps, ResearcherDeps)


# ---------------------------------------------------------------------------
# Test 3 — mutation passes through
# ---------------------------------------------------------------------------


class TestMutationPassesThrough:
    async def test_request_content_modified(self):
        async def prefix(ctx: MASInsertContext[Any, Any]) -> Message:
            return ctx.message.model_copy(
                update={"content": f"PREFIXED: {ctx.message.content}"}
            )

        mas = _two_agent_mas(
            hooks=MASHooks(on_request_insert=prefix),
            request_content="original",
        )
        result = await mas.run(entry_agent="planner", prompt="go")

        # The researcher's history should contain the prefixed content.
        researcher_history = result.agent_histories["researcher"]
        assert any(
            "PREFIXED: original" in _flatten_parts(msg) for msg in researcher_history
        )


# ---------------------------------------------------------------------------
# Test 4 — hook raise aborts the insertion
# ---------------------------------------------------------------------------


class TestHookRaisePropagates:
    async def test_exception_surfaces(self):
        class Boom(RuntimeError):
            pass

        async def explode(ctx: MASInsertContext[Any, Any]) -> Message:
            raise Boom("no go")

        mas = _two_agent_mas(hooks=MASHooks(on_request_insert=explode))

        import pytest

        with pytest.raises(Boom):
            await mas.run(entry_agent="planner", prompt="go")


# ---------------------------------------------------------------------------
# Test 5 — system-originated entry message
# ---------------------------------------------------------------------------


class TestSystemEntry:
    async def test_system_sender_yields_none_caller_deps(self):
        captured: list[MASInsertContext] = []

        async def record(ctx: MASInsertContext[Any, Any]) -> Message:
            captured.append(ctx)
            return ctx.message

        mas = MAS(
            agents={
                "planner": AgentConfig(
                    agent=Agent(model=TestModel()), deps=PlannerDeps()
                ),
            },
            hooks=MASHooks(on_request_insert=record),
        )
        await mas.run(entry_agent="planner", prompt="first")

        assert len(captured) == 1
        ctx = captured[0]
        assert ctx.caller_id == "system"
        assert ctx.caller_deps is None
        assert ctx.caller_history == []
        assert ctx.callee_id == "planner"
        assert isinstance(ctx.callee_deps, PlannerDeps)


# ---------------------------------------------------------------------------
# Test 6 — sync hook accepted
# ---------------------------------------------------------------------------


class TestSyncHookAccepted:
    async def test_non_async_function_works(self):
        called: list[bool] = []

        def sync_hook(ctx: MASInsertContext[Any, Any]) -> Message:
            called.append(True)
            return ctx.message

        mas = MAS(
            agents={
                "planner": AgentConfig(
                    agent=Agent(model=TestModel()), deps=PlannerDeps()
                ),
            },
            hooks=MASHooks(on_request_insert=sync_hook),
        )
        result = await mas.run(entry_agent="planner", prompt="hi")

        assert result.termination_reason == TerminationReason.COMPLETED
        assert called == [True]


# ---------------------------------------------------------------------------
# Test 7 — no hooks baseline (no regression)
# ---------------------------------------------------------------------------


class TestNoHooksBaseline:
    async def test_mas_runs_without_hooks(self):
        mas = _two_agent_mas(hooks=None)
        result = await mas.run(entry_agent="planner", prompt="go")

        assert result.termination_reason == TerminationReason.COMPLETED
        pairs = [(m.sender, m.receiver) for m in result.message_log]
        assert ("planner", "researcher") in pairs
        assert ("researcher", "planner") in pairs


# ---------------------------------------------------------------------------
# Test 8 — histories are snapshots, not live refs
# ---------------------------------------------------------------------------


class TestHistorySnapshots:
    async def test_callee_history_is_snapshot(self):
        """Hook stores ctx.callee_history; subsequent turns must not mutate it."""
        snapshots: list[list[ModelMessage]] = []

        async def record(ctx: MASInsertContext[Any, Any]) -> Message:
            snapshots.append(ctx.callee_history)
            return ctx.message

        mas = _two_agent_mas(hooks=MASHooks(on_request_insert=record))
        result = await mas.run(entry_agent="planner", prompt="go")
        assert result.termination_reason == TerminationReason.COMPLETED

        # The first snapshot was taken when researcher's history was empty.
        # After the run, researcher has non-empty history.
        researcher_final_history = result.agent_histories["researcher"]
        assert len(researcher_final_history) > 0

        # The planner→researcher request was captured while researcher.history was []
        p2r_snapshots = [s for s, _ in zip(snapshots, range(len(snapshots)))]
        # The second snapshot corresponds to planner→researcher (researcher's empty history at that moment)
        assert p2r_snapshots[1] == []
        # Mutating the stored snapshot should NOT change what's in researcher's live history
        # (i.e., they are separate list objects)
        p2r_snapshots[1].append("tampered")  # type: ignore[arg-type]
        assert "tampered" not in result.agent_histories["researcher"]


# ---------------------------------------------------------------------------
# Helper to flatten message text parts for substring assertions
# ---------------------------------------------------------------------------


def _flatten_parts(msg: ModelMessage) -> str:
    parts = getattr(msg, "parts", [])
    chunks: list[str] = []
    for p in parts:
        for attr in ("content", "user_prompt"):
            v = getattr(p, attr, None)
            if isinstance(v, str):
                chunks.append(v)
    return " | ".join(chunks)
