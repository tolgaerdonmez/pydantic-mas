import asyncio

from opentelemetry.trace import get_tracer_provider, use_span

from pydantic_mas._agent_node import AgentNode, AgentState
from pydantic_mas._budget import BudgetExceededError, BudgetTracker
from pydantic_mas._message import MessageType
from pydantic_mas._result import MASResult, TerminationReason
from pydantic_mas._router import MessageRouter

_tracer = get_tracer_provider().get_tracer("pydantic-mas")


class MASInstance:
    """A single isolated MAS runtime.

    Owns all AgentNode instances, the MessageRouter, and the BudgetTracker.
    Manages agent task lifecycle and detects termination.
    """

    def __init__(
        self,
        agent_nodes: list[AgentNode],
        router: MessageRouter,
        budget_tracker: BudgetTracker,
    ):
        self._agent_nodes = {node.agent_id: node for node in agent_nodes}
        self._router = router
        self._budget_tracker = budget_tracker
        self._tasks: dict[str, asyncio.Task[None]] = {}

    async def run(
        self,
        entry_agent: str,
        prompt: str,
        timeout: float | None = None,
    ) -> MASResult:
        """Run the instance to completion.

        Args:
            entry_agent: ID of the agent that receives the initial prompt.
            prompt: The initial user prompt.
            timeout: Optional timeout in seconds. Overrides budget.timeout_seconds.

        Raises:
            ValueError: If entry_agent is not registered.
        """
        if entry_agent not in self._agent_nodes:
            raise ValueError(
                f"Entry agent '{entry_agent}' not found. "
                f"Available agents: {list(self._agent_nodes.keys())}"
            )

        span = _tracer.start_span(
            "MAS run",
            attributes={
                "mas.entry_agent": entry_agent,
                "mas.agent_count": len(self._agent_nodes),
            },
        )

        with use_span(span, end_on_exit=True, record_exception=True):
            termination_reason = TerminationReason.COMPLETED

            try:
                self._router.route(
                    sender="system",
                    receiver=entry_agent,
                    content=prompt,
                    type=MessageType.REQUEST,
                    depth=0,
                )

                effective_timeout = (
                    timeout or self._budget_tracker.budget.timeout_seconds
                )
                if effective_timeout:
                    async with asyncio.timeout(effective_timeout):
                        await self._run_agents()
                else:
                    await self._run_agents()

            except TimeoutError:
                termination_reason = TerminationReason.TIMEOUT
            except BudgetExceededError:
                termination_reason = TerminationReason.BUDGET_EXCEEDED
            finally:
                await self._cancel_all_agents()

            span.set_attribute("mas.termination_reason", str(termination_reason))
            span.set_attribute("mas.message_count", len(self._router.message_log))

            return MASResult(
                message_log=self._router.message_log,
                agent_histories={
                    agent_id: list(node.history)
                    for agent_id, node in self._agent_nodes.items()
                },
                termination_reason=termination_reason,
                budget_usage=self._budget_tracker.snapshot(),
            )

    async def _run_agents(self) -> None:
        """Start all agent loops and wait for termination."""
        for agent_id, node in self._agent_nodes.items():
            task = asyncio.create_task(node.run_loop(), name=f"agent-{agent_id}")
            self._tasks[agent_id] = task

        await self._monitor_termination()

    async def _monitor_termination(self) -> None:
        """Wait until all agents are idle and all queues are empty.

        Uses event-driven detection: waits for all agents to signal idle,
        then checks the global quiescence condition.
        """
        while True:
            # Wait for every agent to be in IDLE state.
            # idle_event.wait() returns immediately if already set.
            await asyncio.gather(
                *(node.idle_event.wait() for node in self._agent_nodes.values())
            )

            # All agents are idle — check if queues are also empty
            if self._all_quiesced():
                return

            # Some queue still has items. Yield so agents can pick them up.
            await asyncio.sleep(0)

    def _all_quiesced(self) -> bool:
        """Check if all agents are idle with empty queues."""
        return all(
            node.state == AgentState.IDLE and node.inbox.empty()
            for node in self._agent_nodes.values()
        )

    async def _cancel_all_agents(self) -> None:
        """Cancel all running agent tasks and wait for cleanup."""
        for task in self._tasks.values():
            if not task.done():
                task.cancel()

        if self._tasks:
            await asyncio.gather(*self._tasks.values(), return_exceptions=True)
