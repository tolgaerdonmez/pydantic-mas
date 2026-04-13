"""End-to-end tests for MASInstance with multi-agent interaction."""

import json

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel

from pydantic_mas._agent_node import AgentNode
from pydantic_mas._budget import Budget, BudgetTracker
from pydantic_mas._instance import MASInstance
from pydantic_mas._message import MessageType
from pydantic_mas._result import TerminationReason
from pydantic_mas._router import MessageRouter


def _make_instance(
    agents: dict[str, Agent],
    budget: Budget | None = None,
) -> MASInstance:
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


class TestCoordinatorDelegatesToWorkers:
    """Coordinator receives a task, fans out to researcher + writer,
    both workers auto-reply, coordinator receives both replies.

    Expected message flow:
        1. system -> coordinator  (REQUEST, depth 0)
        2. coordinator -> researcher  (REQUEST, depth 1)  via send_message
        3. coordinator -> writer  (REQUEST, depth 1)  via send_message
        4. researcher -> coordinator  (REPLY, depth 1)  auto-reply
        5. writer -> coordinator  (REPLY, depth 1)  auto-reply
    """

    @staticmethod
    def _coordinator_model() -> FunctionModel:
        call_count = 0

        def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            nonlocal call_count
            call_count += 1

            if call_count == 1:
                # Fan-out: send to both workers
                return ModelResponse(
                    parts=[
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {
                                    "target_agent": "researcher",
                                    "content": "Research topic X",
                                }
                            ),
                            tool_call_id="to-researcher",
                        ),
                        ToolCallPart(
                            tool_name="send_message",
                            args=json.dumps(
                                {
                                    "target_agent": "writer",
                                    "content": "Draft section Y",
                                }
                            ),
                            tool_call_id="to-writer",
                        ),
                    ]
                )
            # All subsequent calls: just return text
            return ModelResponse(parts=[TextPart(content="Acknowledged")])

        return FunctionModel(handler)

    async def test_completes_successfully(self):
        instance = _make_instance(
            {
                "coordinator": Agent(model=self._coordinator_model()),
                "researcher": Agent(model=TestModel()),
                "writer": Agent(model=TestModel()),
            }
        )

        result = await instance.run(
            entry_agent="coordinator", prompt="Build a report on X"
        )

        assert result.termination_reason == TerminationReason.COMPLETED

    async def test_message_flow(self):
        instance = _make_instance(
            {
                "coordinator": Agent(model=self._coordinator_model()),
                "researcher": Agent(model=TestModel()),
                "writer": Agent(model=TestModel()),
            }
        )

        result = await instance.run(
            entry_agent="coordinator", prompt="Build a report on X"
        )

        pairs = [(m.sender, m.receiver) for m in result.message_log]

        # Initial message
        assert ("system", "coordinator") in pairs
        # Coordinator delegated to both workers
        assert ("coordinator", "researcher") in pairs
        assert ("coordinator", "writer") in pairs
        # Workers auto-replied
        assert ("researcher", "coordinator") in pairs
        assert ("writer", "coordinator") in pairs

    async def test_message_types(self):
        instance = _make_instance(
            {
                "coordinator": Agent(model=self._coordinator_model()),
                "researcher": Agent(model=TestModel()),
                "writer": Agent(model=TestModel()),
            }
        )

        result = await instance.run(
            entry_agent="coordinator", prompt="Build a report on X"
        )

        types_by_pair = {(m.sender, m.receiver): m.type for m in result.message_log}
        assert types_by_pair[("system", "coordinator")] == MessageType.REQUEST
        assert types_by_pair[("coordinator", "researcher")] == MessageType.REQUEST
        assert types_by_pair[("coordinator", "writer")] == MessageType.REQUEST
        assert types_by_pair[("researcher", "coordinator")] == MessageType.REPLY
        assert types_by_pair[("writer", "coordinator")] == MessageType.REPLY

    async def test_all_agents_have_history(self):
        instance = _make_instance(
            {
                "coordinator": Agent(model=self._coordinator_model()),
                "researcher": Agent(model=TestModel()),
                "writer": Agent(model=TestModel()),
            }
        )

        result = await instance.run(
            entry_agent="coordinator", prompt="Build a report on X"
        )

        for agent_id in ["coordinator", "researcher", "writer"]:
            assert len(result.agent_histories[agent_id]) > 0

    async def test_budget_tracks_all_messages(self):
        instance = _make_instance(
            {
                "coordinator": Agent(model=self._coordinator_model()),
                "researcher": Agent(model=TestModel()),
                "writer": Agent(model=TestModel()),
            }
        )

        result = await instance.run(
            entry_agent="coordinator", prompt="Build a report on X"
        )

        assert result.budget_usage.total_messages == len(result.message_log)
        assert "system" in result.budget_usage.per_agent_messages
        assert "coordinator" in result.budget_usage.per_agent_messages
        assert "researcher" in result.budget_usage.per_agent_messages
        assert "writer" in result.budget_usage.per_agent_messages


class TestPeerToPeerInteraction:
    """Workers communicate directly with each other, not just through coordinator.

    Scenario: coordinator sends to analyst. Analyst sends to reviewer.
    Reviewer auto-replies to analyst. Analyst auto-replies to coordinator.

    Expected message flow:
        1. system -> coordinator  (REQUEST, depth 0)
        2. coordinator -> analyst  (REQUEST, depth 1)  via send_message
        3. analyst -> reviewer  (REQUEST, depth 2)  via send_message
        4. analyst -> coordinator  (REPLY, depth 1)  auto-reply
        5. reviewer -> analyst  (REPLY, depth 2)  auto-reply
    """

    @staticmethod
    def _model_that_sends(target: str, content: str) -> FunctionModel:
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

    async def test_chain_completes(self):
        instance = _make_instance(
            {
                "coordinator": Agent(
                    model=self._model_that_sends("analyst", "analyze data")
                ),
                "analyst": Agent(
                    model=self._model_that_sends("reviewer", "review my analysis")
                ),
                "reviewer": Agent(model=TestModel()),
            }
        )

        result = await instance.run(entry_agent="coordinator", prompt="Start analysis")

        assert result.termination_reason == TerminationReason.COMPLETED

    async def test_depth_increases_along_chain(self):
        instance = _make_instance(
            {
                "coordinator": Agent(
                    model=self._model_that_sends("analyst", "analyze")
                ),
                "analyst": Agent(model=self._model_that_sends("reviewer", "review")),
                "reviewer": Agent(model=TestModel()),
            }
        )

        result = await instance.run(entry_agent="coordinator", prompt="Start")

        depths = {(m.sender, m.receiver): m.depth for m in result.message_log}
        assert depths[("system", "coordinator")] == 0
        assert depths[("coordinator", "analyst")] == 1
        assert depths[("analyst", "reviewer")] == 2

    async def test_max_depth_tracked(self):
        instance = _make_instance(
            {
                "coordinator": Agent(
                    model=self._model_that_sends("analyst", "analyze")
                ),
                "analyst": Agent(model=self._model_that_sends("reviewer", "review")),
                "reviewer": Agent(model=TestModel()),
            }
        )

        result = await instance.run(entry_agent="coordinator", prompt="Start")

        assert result.budget_usage.max_depth_seen == 2

    async def test_peer_messages_in_log(self):
        instance = _make_instance(
            {
                "coordinator": Agent(
                    model=self._model_that_sends("analyst", "analyze")
                ),
                "analyst": Agent(
                    model=self._model_that_sends("reviewer", "review this")
                ),
                "reviewer": Agent(model=TestModel()),
            }
        )

        result = await instance.run(entry_agent="coordinator", prompt="Start")

        pairs = [(m.sender, m.receiver) for m in result.message_log]
        # Peer-to-peer: analyst -> reviewer (not via coordinator)
        assert ("analyst", "reviewer") in pairs
        # Reviewer auto-replies to analyst
        assert ("reviewer", "analyst") in pairs


class TestBudgetConstrainedInteraction:
    """Multiple agents interact under a tight budget.

    With max_total_messages=3, the system can route the initial message plus
    two more before hitting the limit. Further sends fail gracefully.
    """

    @staticmethod
    def _model_that_sends(target: str, content: str) -> FunctionModel:
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

    async def test_budget_limits_message_count(self):
        """Only max_total_messages messages get routed; excess sends fail gracefully."""
        instance = _make_instance(
            {
                "coordinator": Agent(
                    model=self._model_that_sends("analyst", "analyze")
                ),
                "analyst": Agent(model=self._model_that_sends("reviewer", "review")),
                "reviewer": Agent(model=TestModel()),
            },
            budget=Budget(max_total_messages=3),
        )

        result = await instance.run(entry_agent="coordinator", prompt="Start")

        assert result.termination_reason == TerminationReason.COMPLETED
        assert result.budget_usage.total_messages == 3
        assert len(result.message_log) == 3

    async def test_depth_limited_interaction(self):
        """max_depth=1 allows initial (depth 0) and one hop (depth 1) but blocks depth 2."""
        instance = _make_instance(
            {
                "coordinator": Agent(
                    model=self._model_that_sends("analyst", "analyze")
                ),
                "analyst": Agent(model=self._model_that_sends("reviewer", "review")),
                "reviewer": Agent(model=TestModel()),
            },
            budget=Budget(max_depth=1),
        )

        result = await instance.run(entry_agent="coordinator", prompt="Start")

        assert result.termination_reason == TerminationReason.COMPLETED
        # analyst -> reviewer at depth 2 should have been blocked
        pairs = [(m.sender, m.receiver) for m in result.message_log]
        assert ("coordinator", "analyst") in pairs
        assert ("analyst", "reviewer") not in pairs
        assert result.budget_usage.max_depth_seen <= 1
