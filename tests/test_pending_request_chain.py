"""Demonstrates that when B is invoked by A and then consults C/D before
returning, B loses track of A as the original requester.

Scenario:
    1. A sends REQUEST to B ("do task X")
    2. B sends REQUEST to C ("subtask")        — interrupt_on_send fires, B parks
    3. C produces an auto-REPLY to B ("result of subtask")
    4. B resumes, this time processing C's REPLY, and produces final text

Expected: B should send a REPLY back to A (the original requester) with the
final answer.

Actual (current behavior, the bug): When B resumes to process C's REPLY,
`current_message` is overwritten to that REPLY. The auto-reply path in
`_handle_last_output_reply` short-circuits because `original_message.type
!= REQUEST`. The original REQUEST from A is therefore never answered, and
A never receives the final result.

These tests are written red-first: they SHOULD fail on the current
implementation and pass once the agent tracks a pending request across
the inbox cycle.
"""

import json
from typing import Callable

from pydantic_ai import Agent
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

from pydantic_mas import (
    MAS,
    AgentConfig,
    Budget,
    MessageType,
    TerminationReason,
)


ModelHandler = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


# ---------------------------------------------------------------------------
# Model handlers
# ---------------------------------------------------------------------------


def _a_delegates_then_acks() -> ModelHandler:
    """A: first turn asks B to do the task; later turns just acknowledge."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        has_prior_tools = any(
            hasattr(p, "tool_name") for m in messages for p in m.parts
        )
        last = messages[-1].parts[-1]
        if isinstance(last, UserPromptPart) and not has_prior_tools:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {"target_agent": "agent_b", "content": "do task X"}
                        ),
                        tool_call_id="a-1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="A acknowledges")])

    return handler


def _b_consults_c_then_finalizes() -> ModelHandler:
    """B's behaviour:

    - On the FIRST turn (it just received a REQUEST from A and has no
      prior tool history), call send_message(agent_c, "subtask").
    - On the SECOND invocation (it has been resumed because C's REPLY
      arrived in the inbox), produce final text "FINAL ANSWER FOR A".
      It deliberately does NOT call send_message(agent_a, ...) — the
      framework's auto-reply is supposed to deliver the result.
    """

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        # Have we already invoked send_message in a prior turn? Check for
        # any ToolReturnPart for send_message in the message history.
        consulted_c_already = any(
            isinstance(p, ToolReturnPart) and p.tool_name == "send_message"
            for m in messages
            for p in m.parts
        )

        if not consulted_c_already:
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {"target_agent": "agent_c", "content": "subtask for C"}
                        ),
                        tool_call_id="b-1",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content="FINAL ANSWER FOR A")])

    return handler


def _c_replies() -> ModelHandler:
    """C: any input → text "result from C", which auto-replies to B."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content="result from C")])

    return handler


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBRepliesToAAfterConsultingC:
    """The core regression: A must receive B's final answer."""

    async def test_a_receives_final_answer_from_b(self):
        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(model=FunctionModel(_a_delegates_then_acks()))
                ),
                "agent_b": AgentConfig(
                    agent=Agent(model=FunctionModel(_b_consults_c_then_finalizes()))
                ),
                "agent_c": AgentConfig(agent=Agent(model=FunctionModel(_c_replies()))),
            },
            interrupt_on_send=True,
            budget=Budget(max_total_messages=20, timeout_seconds=5),
        )

        result = await mas.run(entry_agent="agent_a", prompt="please solve X")

        # 1. Run terminated cleanly (not by timeout/budget).
        assert result.termination_reason == TerminationReason.COMPLETED, (
            f"Run did not complete cleanly: {result.termination_reason}\n"
            f"messages: {[(m.sender, m.receiver, m.type) for m in result.message_log]}"
        )

        # 2. C did receive B's subtask.
        b_to_c = [
            m
            for m in result.message_log
            if m.sender == "agent_b" and m.receiver == "agent_c"
        ]
        assert len(b_to_c) == 1, "B should have sent exactly one REQUEST to C"
        assert b_to_c[0].type == MessageType.REQUEST

        # 3. B did receive C's reply.
        c_to_b = [
            m
            for m in result.message_log
            if m.sender == "agent_c" and m.receiver == "agent_b"
        ]
        assert len(c_to_b) == 1
        assert c_to_b[0].type == MessageType.REPLY

        # 4. **The bug**: B must reply to A with the final answer.
        b_to_a = [
            m
            for m in result.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        assert len(b_to_a) >= 1, (
            "B never sent anything back to A. After consulting C, B's "
            "final output was lost because the agent forgot it was still "
            "fulfilling A's original REQUEST.\n"
            f"All messages: "
            f"{[(m.sender, m.receiver, m.type, m.content[:40]) for m in result.message_log]}"
        )

        # 5. That message back to A must be a REPLY (not a fresh REQUEST),
        #    in_reply_to the original A→B request, with the final content.
        a_request = next(
            m
            for m in result.message_log
            if m.sender == "agent_a" and m.receiver == "agent_b"
        )
        final = b_to_a[-1]
        assert final.type == MessageType.REPLY, (
            f"B's message back to A should be a REPLY, got {final.type}. "
            "A fresh REQUEST means the framework lost the original-requester "
            "context and started a new conversation instead of completing one."
        )
        assert final.in_reply_to == a_request.id, (
            "B's REPLY to A should reference A's original REQUEST id"
        )
        assert "FINAL ANSWER FOR A" in final.content, (
            f"Wrong reply content: {final.content!r}"
        )


class TestExplicitFinalSendIsRoutedAsReply:
    """Variant: B explicitly calls send_message(agent_a, ...) in its final
    turn (after C's REPLY came back). This too should be reinterpreted as
    a REPLY to the original requester, not a fresh REQUEST.
    """

    async def test_explicit_send_to_original_requester_becomes_reply(self):
        def b_handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            consulted_c_already = any(
                isinstance(p, ToolReturnPart) and p.tool_name == "send_message"
                for m in messages
                for p in m.parts
            )
            if not consulted_c_already:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_c", "content": "sub"}
                            ),
                            tool_call_id="b-1",
                        )
                    ]
                )
            # After C replies, B explicitly addresses A.
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name="send_message",
                        args=json.dumps(
                            {
                                "target_agent": "agent_a",
                                "content": "explicit answer for A",
                            }
                        ),
                        tool_call_id="b-2",
                    )
                ]
            )

        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(model=FunctionModel(_a_delegates_then_acks()))
                ),
                "agent_b": AgentConfig(agent=Agent(model=FunctionModel(b_handler))),
                "agent_c": AgentConfig(agent=Agent(model=FunctionModel(_c_replies()))),
            },
            interrupt_on_send=True,
            budget=Budget(max_total_messages=20, timeout_seconds=5),
        )

        result = await mas.run(entry_agent="agent_a", prompt="please solve X")

        b_to_a = [
            m
            for m in result.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        assert len(b_to_a) == 1, (
            f"Expected exactly one B→A message; got {len(b_to_a)}: "
            f"{[(m.type, m.content[:40]) for m in b_to_a]}"
        )
        # Should be a REPLY to A's original REQUEST, since B is still
        # fulfilling it — even though current_message at this moment is
        # C's REPLY.
        a_request = next(
            m
            for m in result.message_log
            if m.sender == "agent_a" and m.receiver == "agent_b"
        )
        assert b_to_a[0].type == MessageType.REPLY, (
            f"Expected REPLY, got {b_to_a[0].type}. The framework treated "
            "B's send_message(agent_a, ...) as a fresh REQUEST because B's "
            "current_message at that moment was C's REPLY, not A's REQUEST."
        )
        assert b_to_a[0].in_reply_to == a_request.id


class TestOutOfOrderRequestersBothGetReplies:
    """Two requesters interleave on the same agent.

    Flow:
        1. A → B   (r1)
        2. B → C   (out_c, scoped to r1)              [B parks]
        3. D → B   (r3) arrives while B is parked
        4. B picks up r3 next, replies to D directly
        5. C replies to B (in_reply_to=out_c) → B re-attributes to r1
        6. B replies to A

    Both A and D must receive a REPLY tied to their own original
    REQUEST, in completion order, regardless of inbox arrival order.
    """

    async def test_both_requesters_receive_replies_with_correct_in_reply_to(self):
        # A delegates once to B, then acks.
        # D delegates once to B, then acks.
        def a_handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            has_prior_tools = any(
                hasattr(p, "tool_name") for m in messages for p in m.parts
            )
            last = messages[-1].parts[-1]
            if isinstance(last, UserPromptPart) and not has_prior_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_b", "content": "task from A"}
                            ),
                            tool_call_id="a-1",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="A acknowledges")])

        def d_handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            has_prior_tools = any(
                hasattr(p, "tool_name") for m in messages for p in m.parts
            )
            last = messages[-1].parts[-1]
            if isinstance(last, UserPromptPart) and not has_prior_tools:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_b", "content": "task from D"}
                            ),
                            tool_call_id="d-1",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="D acknowledges")])

        # B's behaviour:
        #   - First time it sees a REQUEST whose content mentions A, fire send to C.
        #   - Any other turn (REPLY from C, or D's REQUEST), produce final text
        #     that names whichever requester is being served.
        def b_handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            # Find the most recent UserPromptPart (the framework-formatted msg).
            last_user_text = ""
            for m in reversed(messages):
                for p in m.parts:
                    if isinstance(p, UserPromptPart):
                        last_user_text = str(p.content)
                        break
                if last_user_text:
                    break

            if "task from A" in last_user_text:
                # Serving A's REQUEST → consult C.
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_c", "content": "subtask"}
                            ),
                            tool_call_id="b-c",
                        )
                    ]
                )
            if "task from D" in last_user_text:
                # Serving D's REQUEST → quick final answer (no sub-call).
                return ModelResponse(parts=[TextPart(content="answer for D")])
            # Otherwise we're seeing C's REPLY → finalize for A.
            return ModelResponse(parts=[TextPart(content="answer for A")])

        mas = MAS(
            agents={
                "agent_a": AgentConfig(agent=Agent(model=FunctionModel(a_handler))),
                "agent_b": AgentConfig(agent=Agent(model=FunctionModel(b_handler))),
                "agent_c": AgentConfig(agent=Agent(model=FunctionModel(_c_replies()))),
                "agent_d": AgentConfig(agent=Agent(model=FunctionModel(d_handler))),
            },
            interrupt_on_send=True,
            budget=Budget(max_total_messages=40, timeout_seconds=5),
        )

        # Both A and D start in parallel by sending REQUESTs to B.
        # We do that by giving each entry-agent a separate run? No — single
        # MAS.run takes one entry. Instead we kick off B with two REQUESTs
        # by having both A and D be "primary" agents that each receive a
        # system prompt. That isn't supported, so we simulate the race by
        # running the MAS with `agent_a` as entry and pre-seeding D's run
        # via a custom-driving agent. The simplest path: have agent_a's
        # first turn ALSO trigger D's flow by putting a REQUEST in D's
        # inbox via a dedicated "kicker" agent.

        # Actually, the cleanest e2e: entry is `kicker`, which sends in
        # parallel to A and D in the same tool batch. Both then each
        # delegate to B.
        def kicker_handler(
            messages: list[ModelMessage], info: AgentInfo
        ) -> ModelResponse:
            last = messages[-1].parts[-1]
            if isinstance(last, UserPromptPart):
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_a", "content": "go"}
                            ),
                            tool_call_id="k-a",
                        ),
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_d", "content": "go"}
                            ),
                            tool_call_id="k-d",
                        ),
                    ]
                )
            return ModelResponse(parts=[TextPart(content="kicker done")])

        mas = MAS(
            agents={
                "kicker": AgentConfig(agent=Agent(model=FunctionModel(kicker_handler))),
                "agent_a": AgentConfig(agent=Agent(model=FunctionModel(a_handler))),
                "agent_b": AgentConfig(agent=Agent(model=FunctionModel(b_handler))),
                "agent_c": AgentConfig(agent=Agent(model=FunctionModel(_c_replies()))),
                "agent_d": AgentConfig(agent=Agent(model=FunctionModel(d_handler))),
            },
            interrupt_on_send=True,
            budget=Budget(max_total_messages=60, timeout_seconds=5),
        )
        result = await mas.run(entry_agent="kicker", prompt="start")

        assert result.termination_reason == TerminationReason.COMPLETED, (
            f"Run did not complete: {result.termination_reason}\n"
            f"messages: "
            f"{[(m.sender, m.receiver, m.type, m.content[:30]) for m in result.message_log]}"
        )

        a_request = next(
            m
            for m in result.message_log
            if m.sender == "agent_a" and m.receiver == "agent_b"
        )
        d_request = next(
            m
            for m in result.message_log
            if m.sender == "agent_d" and m.receiver == "agent_b"
        )

        b_to_a = [
            m
            for m in result.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        b_to_d = [
            m
            for m in result.message_log
            if m.sender == "agent_b" and m.receiver == "agent_d"
        ]

        # Each requester gets exactly one REPLY, tied to its own REQUEST.
        assert len(b_to_a) == 1
        assert b_to_a[0].type == MessageType.REPLY
        assert b_to_a[0].in_reply_to == a_request.id
        assert "answer for A" in b_to_a[0].content

        assert len(b_to_d) == 1
        assert b_to_d[0].type == MessageType.REPLY
        assert b_to_d[0].in_reply_to == d_request.id
        assert "answer for D" in b_to_d[0].content


class TestNoSpuriousMidConversationReplyToA:
    """Negative check: B should NOT auto-reply to A immediately after sending
    to C — only after the full sub-conversation has produced a final answer.

    With `interrupt_on_send=True` the run is parked after the tool turn, so
    no auto-reply fires mid-flight. This test pins that behaviour so a fix
    for the main bug doesn't accidentally regress and produce TWO replies
    to A (one premature + one final).
    """

    async def test_only_one_reply_reaches_a(self):
        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(model=FunctionModel(_a_delegates_then_acks()))
                ),
                "agent_b": AgentConfig(
                    agent=Agent(model=FunctionModel(_b_consults_c_then_finalizes()))
                ),
                "agent_c": AgentConfig(agent=Agent(model=TestModel())),
            },
            interrupt_on_send=True,
            budget=Budget(max_total_messages=20, timeout_seconds=5),
        )

        result = await mas.run(entry_agent="agent_a", prompt="please solve X")

        b_to_a = [
            m
            for m in result.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        assert len(b_to_a) == 1, f"Expected exactly one B→A message; got {len(b_to_a)}"
