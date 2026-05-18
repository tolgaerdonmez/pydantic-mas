import asyncio

from opentelemetry.trace import get_tracer_provider, use_span

from pydantic_mas._agent_node import AgentNode
from pydantic_mas._budget import BudgetExceededError, BudgetTracker
from pydantic_mas._message import MessageType
from pydantic_mas._result import MASResult, TerminationReason
from pydantic_mas._router import MessageRouter

_tracer = get_tracer_provider().get_tracer("pydantic-mas")


class MASInstance:
    """A single isolated MAS runtime.

    Owns all AgentNode instances, the MessageRouter, and the BudgetTracker.
    Supervises agent tasks with a fail-fast policy: if any agent raises an
    unhandled exception, the run is torn down and the exception propagates
    out of `run()` to the library user.
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
            BaseException: Any exception raised inside an agent's run loop
                (tool, model handler, hook, output validator, etc.) propagates
                out unchanged.
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
                        await self._run_supervised()
                else:
                    await self._run_supervised()

            except TimeoutError:
                termination_reason = TerminationReason.TIMEOUT
            except BudgetExceededError:
                termination_reason = TerminationReason.BUDGET_EXCEEDED

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

    async def _run_supervised(self) -> None:
        """Run all agent loops; return when the run quiesces or any agent raises.

        Termination contract:
          - Quiescence (router.outstanding == 0) signals clean completion. The
            watcher task fires, all agent tasks are cancelled, and we return.
          - If any agent task raises a non-CancelledError exception, that
            exception is captured and re-raised after cancelling the rest.
            This is the fail-fast guarantee: agent failures reach the caller
            of `mas.run()` unchanged.
        """
        agent_tasks: list[asyncio.Task[None]] = [
            asyncio.create_task(node.run_loop(), name=f"agent-{node.agent_id}")
            for node in self._agent_nodes.values()
        ]
        watcher = asyncio.create_task(self._router.wait_quiet(), name="quiescence")
        all_tasks: list[asyncio.Task[None]] = [*agent_tasks, watcher]

        first_exc: BaseException | None = None
        try:
            await asyncio.wait(all_tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in all_tasks:
                if not t.done():
                    t.cancel()
            # Drain everything so we can inspect for unsurfaced exceptions and
            # so no task is left dangling. `return_exceptions=True` prevents
            # a sibling's exception from masking another.
            results = await asyncio.gather(*all_tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, BaseException) and not isinstance(
                    r, asyncio.CancelledError
                ):
                    first_exc = r
                    break

        if first_exc is not None:
            raise first_exc
