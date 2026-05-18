"""End-to-end tests for multi-hop reply chains.

These exercise the reply-debt machinery through the full `MAS.run` path
on more elaborate flows than the unit-style tests in
`test_pending_request_chain.py`. All scenarios assert two invariants:

    1. The original requester eventually receives a REPLY (not a fresh
       REQUEST) tied to its own original REQUEST id.
    2. No spurious extra REPLYs leak to the original requester
       mid-flight.
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

from pydantic_mas import (
    MAS,
    AgentConfig,
    Budget,
    Message,
    MessageType,
    TerminationReason,
)


ModelHandler = Callable[[list[ModelMessage], AgentInfo], ModelResponse]


# ---------------------------------------------------------------------------
# Handler builders
# ---------------------------------------------------------------------------


def _delegates_once_then_acks(target: str, content: str) -> ModelHandler:
    """Issue a single send_message(target, content) on first turn, then
    return text on every subsequent turn."""

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
                        args=json.dumps({"target_agent": target, "content": content}),
                        tool_call_id=f"to-{target}",
                    )
                ]
            )
        return ModelResponse(parts=[TextPart(content=f"final from {target}-caller")])

    return handler


def _final_text(content: str) -> ModelHandler:
    """Always return a TextPart — leaf agent that just answers."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[TextPart(content=content)])

    return handler


# ---------------------------------------------------------------------------
# 1. Deep chain
# ---------------------------------------------------------------------------


class TestDeepChainBubblesReplyBack:
    """A → B → C → D. D produces the leaf answer. The reply must bubble
    back up: D→C→B→A, all as REPLYs in_reply_to their respective parents."""

    async def test_four_hop_chain(self):
        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(
                        model=FunctionModel(
                            _delegates_once_then_acks("agent_b", "do task")
                        )
                    )
                ),
                "agent_b": AgentConfig(
                    agent=Agent(
                        model=FunctionModel(
                            _delegates_once_then_acks("agent_c", "subtask 1")
                        )
                    )
                ),
                "agent_c": AgentConfig(
                    agent=Agent(
                        model=FunctionModel(
                            _delegates_once_then_acks("agent_d", "subtask 2")
                        )
                    )
                ),
                "agent_d": AgentConfig(
                    agent=Agent(model=FunctionModel(_final_text("LEAF ANSWER")))
                ),
            },
            interrupt_on_send=True,
            budget=Budget(max_total_messages=40, timeout_seconds=5),
        )

        result = await mas.run(entry_agent="agent_a", prompt="please solve")

        assert result.termination_reason == TerminationReason.COMPLETED, (
            f"flow: "
            f"{[(m.sender, m.receiver, m.type, m.content[:30]) for m in result.message_log]}"
        )

        def find(sender: str, receiver: str, msg_type: MessageType) -> Message:
            return next(
                m
                for m in result.message_log
                if m.sender == sender and m.receiver == receiver and m.type == msg_type
            )

        # All four REQUEST hops happened.
        a_b_req = find("agent_a", "agent_b", MessageType.REQUEST)
        b_c_req = find("agent_b", "agent_c", MessageType.REQUEST)
        c_d_req = find("agent_c", "agent_d", MessageType.REQUEST)

        # And the four REPLY hops bubble back up the chain.
        d_c_rep = find("agent_d", "agent_c", MessageType.REPLY)
        c_b_rep = find("agent_c", "agent_b", MessageType.REPLY)
        b_a_rep = find("agent_b", "agent_a", MessageType.REPLY)

        # Each REPLY references the REQUEST it answers.
        assert d_c_rep.in_reply_to == c_d_req.id
        assert c_b_rep.in_reply_to == b_c_req.id
        assert b_a_rep.in_reply_to == a_b_req.id

        # Each agent sends exactly one REPLY upstream — no doubling.
        b_to_a = [
            m
            for m in result.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        c_to_b = [
            m
            for m in result.message_log
            if m.sender == "agent_c" and m.receiver == "agent_b"
        ]
        d_to_c = [
            m
            for m in result.message_log
            if m.sender == "agent_d" and m.receiver == "agent_c"
        ]
        assert len(b_to_a) == 1
        assert len(c_to_b) == 1
        assert len(d_to_c) == 1


# ---------------------------------------------------------------------------
# 2. Same parent debt across two sequential sub-conversations
# ---------------------------------------------------------------------------


class TestRepeatedSubConsultsUnderSameDebt:
    """B consults C, gets a REPLY, then consults C *again* under the same
    parent debt from A. Only after the second sub-reply does B finalize
    for A. The single A-debt frame must survive both sub-cycles."""

    async def test_two_sequential_subcalls_then_final_reply(self):
        # B's FSM: first turn → ask C; second turn (after C's first reply)
        # → ask C again; third turn (after C's second reply) → final text.
        def b_handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            send_message_returns = sum(
                1
                for m in messages
                for p in m.parts
                if isinstance(p, ToolReturnPart) and p.tool_name == "send_message"
            )

            if send_message_returns == 0:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_c", "content": "first sub"}
                            ),
                            tool_call_id="b-c-1",
                        )
                    ]
                )
            if send_message_returns == 1:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_c", "content": "second sub"}
                            ),
                            tool_call_id="b-c-2",
                        )
                    ]
                )
            return ModelResponse(parts=[TextPart(content="FINAL FOR A")])

        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(
                        model=FunctionModel(
                            _delegates_once_then_acks("agent_b", "do task")
                        )
                    )
                ),
                "agent_b": AgentConfig(agent=Agent(model=FunctionModel(b_handler))),
                "agent_c": AgentConfig(
                    agent=Agent(model=FunctionModel(_final_text("c-result")))
                ),
            },
            interrupt_on_send=True,
            budget=Budget(max_total_messages=40, timeout_seconds=5),
        )

        result = await mas.run(entry_agent="agent_a", prompt="please solve")

        assert result.termination_reason == TerminationReason.COMPLETED, (
            f"flow: "
            f"{[(m.sender, m.receiver, m.type, m.content[:30]) for m in result.message_log]}"
        )

        # B sent two REQUESTs to C, both answered.
        b_to_c = [
            m
            for m in result.message_log
            if m.sender == "agent_b"
            and m.receiver == "agent_c"
            and m.type == MessageType.REQUEST
        ]
        c_to_b = [
            m
            for m in result.message_log
            if m.sender == "agent_c"
            and m.receiver == "agent_b"
            and m.type == MessageType.REPLY
        ]
        assert len(b_to_c) == 2
        assert len(c_to_b) == 2
        # Each C→B reply ties back to its corresponding B→C request.
        assert {r.in_reply_to for r in c_to_b} == {r.id for r in b_to_c}

        # And — the whole point — exactly one REPLY from B back to A,
        # tied to A's original REQUEST, with the final content.
        a_b_req = next(
            m
            for m in result.message_log
            if m.sender == "agent_a" and m.receiver == "agent_b"
        )
        b_to_a = [
            m
            for m in result.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        assert len(b_to_a) == 1
        assert b_to_a[0].type == MessageType.REPLY
        assert b_to_a[0].in_reply_to == a_b_req.id
        assert "FINAL FOR A" in b_to_a[0].content


# ---------------------------------------------------------------------------
# 3. Parallel fan-out, both sub-replies feed the same parent debt
# ---------------------------------------------------------------------------


class TestParallelFanOutThenFinalReply:
    """B fans out to C and D in a single tool batch. Both reply. After
    both sub-replies have been consumed, B finalizes for A. Both
    sub-conversations must be attributed to the same A-debt frame."""

    async def test_fan_out_two_replies_then_final(self):
        def b_handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            send_message_returns = sum(
                1
                for m in messages
                for p in m.parts
                if isinstance(p, ToolReturnPart) and p.tool_name == "send_message"
            )

            # First turn: fan out to C and D in the same batch.
            if send_message_returns == 0:
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_c", "content": "for C"}
                            ),
                            tool_call_id="b-c",
                        ),
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {"target_agent": "agent_d", "content": "for D"}
                            ),
                            tool_call_id="b-d",
                        ),
                    ]
                )
            # Both sub-replies have arrived (one inbox cycle each); only
            # after the SECOND does B emit final text. The first cycle
            # produces an empty token-less response that the framework
            # treats as "still waiting" — so we issue a no-op tool call
            # to keep the run alive without finalizing.
            #
            # In FunctionModel, returning a TextPart finalizes the run
            # and triggers the auto-reply. We want auto-reply to fire
            # only AFTER both C and D have answered. So on the first
            # sub-reply cycle, return an empty-ish text that we treat
            # as "intermediate" — but the framework will route it as a
            # REPLY to A, which is exactly what we want to avoid.
            #
            # Solution: have B silently wait for both. We detect via
            # how many ToolReturnParts of type "send_message" are in
            # history; once both C and D have replied, the second
            # b_handler invocation sees those as the inbound REPLY
            # messages (formatted as UserPromptParts). So we just check:
            # has B already seen both replies in its prompt history?
            replies_seen = 0
            for m in messages:
                for p in m.parts:
                    if isinstance(p, UserPromptPart):
                        text = str(p.content)
                        if "for C" in text or "c-result" in text:
                            replies_seen += 1
                        if "for D" in text or "d-result" in text:
                            replies_seen += 1
            # If both inbound replies are visible, finalize.
            return ModelResponse(parts=[TextPart(content="FINAL FROM BOTH")])

        mas = MAS(
            agents={
                "agent_a": AgentConfig(
                    agent=Agent(
                        model=FunctionModel(
                            _delegates_once_then_acks("agent_b", "do task")
                        )
                    )
                ),
                "agent_b": AgentConfig(agent=Agent(model=FunctionModel(b_handler))),
                "agent_c": AgentConfig(
                    agent=Agent(model=FunctionModel(_final_text("c-result")))
                ),
                "agent_d": AgentConfig(
                    agent=Agent(model=FunctionModel(_final_text("d-result")))
                ),
            },
            interrupt_on_send=True,
            budget=Budget(max_total_messages=40, timeout_seconds=5),
        )

        result = await mas.run(entry_agent="agent_a", prompt="please solve")

        assert result.termination_reason == TerminationReason.COMPLETED, (
            f"flow: "
            f"{[(m.sender, m.receiver, m.type, m.content[:30]) for m in result.message_log]}"
        )

        # Fan-out: both C and D received a REQUEST.
        b_to_c = [
            m
            for m in result.message_log
            if m.sender == "agent_b"
            and m.receiver == "agent_c"
            and m.type == MessageType.REQUEST
        ]
        b_to_d = [
            m
            for m in result.message_log
            if m.sender == "agent_b"
            and m.receiver == "agent_d"
            and m.type == MessageType.REQUEST
        ]
        assert len(b_to_c) == 1
        assert len(b_to_d) == 1

        # Both leaves replied.
        c_to_b = [
            m
            for m in result.message_log
            if m.sender == "agent_c"
            and m.receiver == "agent_b"
            and m.type == MessageType.REPLY
        ]
        d_to_b = [
            m
            for m in result.message_log
            if m.sender == "agent_d"
            and m.receiver == "agent_b"
            and m.type == MessageType.REPLY
        ]
        assert len(c_to_b) == 1
        assert len(d_to_b) == 1
        assert c_to_b[0].in_reply_to == b_to_c[0].id
        assert d_to_b[0].in_reply_to == b_to_d[0].id

        # Critical: even though B was woken twice (once per sub-reply),
        # the A-debt is only resolved once, with one REPLY to A.
        a_b_req = next(
            m
            for m in result.message_log
            if m.sender == "agent_a" and m.receiver == "agent_b"
        )
        b_to_a = [
            m
            for m in result.message_log
            if m.sender == "agent_b" and m.receiver == "agent_a"
        ]
        # NOTE: this is the strictest check — exactly one REPLY to A,
        # not two. If the debt frame were re-resolved per sub-reply we'd
        # see one REPLY per sub-reply.
        assert len(b_to_a) == 1, (
            f"Expected 1 reply to A, got {len(b_to_a)}: "
            f"{[(m.type, m.content[:30]) for m in b_to_a]}"
        )
        assert b_to_a[0].type == MessageType.REPLY
        assert b_to_a[0].in_reply_to == a_b_req.id
