import asyncio

from pydantic_mas._budget import BudgetTracker
from pydantic_mas._message import Message, MessageType


class MessageRouter:
    """In-process message delivery layer. One per MASInstance.

    Owns an outstanding-message counter used for quiescence detection:
    `route()` increments it (and clears `quiet`); `mark_consumed()` is
    called by each AgentNode after a message has been fully processed
    (success or crash). When the counter returns to zero, `quiet` is set
    and the supervisor can terminate the run.
    """

    def __init__(self, budget_tracker: BudgetTracker):
        self._budget_tracker = budget_tracker
        self._inboxes: dict[str, asyncio.Queue[Message]] = {}
        self._message_log: list[Message] = []
        self._outstanding: int = 0
        self._quiet: asyncio.Event = asyncio.Event()
        self._quiet.set()

    def register(self, agent_id: str, inbox: asyncio.Queue[Message]) -> None:
        """Register an agent's inbox queue."""
        if agent_id in self._inboxes:
            raise ValueError(f"Agent '{agent_id}' is already registered")
        self._inboxes[agent_id] = inbox

    def route(
        self,
        sender: str,
        receiver: str,
        content: str,
        type: MessageType,
        in_reply_to: str | None = None,
        depth: int = 0,
    ) -> Message:
        """Create and deliver a message. Synchronous (non-blocking).

        WARNING: This method MUST remain fully synchronous (no `await`). It mutates
        shared state (budget counters, message log, inbox queues) using a
        check-then-act pattern that is only safe because no yield points exist
        between the check and the mutation. If this method or any code it calls
        becomes async, concurrent tool calls could interleave and cause race
        conditions (e.g. two messages passing the budget check before either
        increments the counter). As an additional safeguard, the send_message tool
        is registered with `sequential=True` in pydantic-ai, forcing sequential
        execution even when the LLM returns multiple tool calls in one turn.

        Raises:
            BudgetExceededError: if any budget limit is hit.
            ValueError: if receiver is not registered.
        """
        if receiver not in self._inboxes:
            raise ValueError(f"Unknown receiver agent: '{receiver}'")

        self._budget_tracker.check_and_record_message(sender, depth)

        message = Message(
            sender=sender,
            receiver=receiver,
            content=content,
            type=type,
            in_reply_to=in_reply_to,
            depth=depth,
        )

        self._message_log.append(message)
        self._inboxes[receiver].put_nowait(message)
        self._outstanding += 1
        self._quiet.clear()

        return message

    def mark_consumed(self) -> None:
        """Called by AgentNode after a message has been processed (or crashed).

        Drops the outstanding counter and, if it reaches zero, signals
        quiescence so the supervisor can shut down the run.
        """
        self._outstanding -= 1
        if self._outstanding <= 0:
            self._outstanding = 0
            self._quiet.set()

    async def wait_quiet(self) -> None:
        """Block until the outstanding-message counter reaches zero."""
        await self._quiet.wait()

    @property
    def outstanding(self) -> int:
        return self._outstanding

    @property
    def message_log(self) -> list[Message]:
        return self._message_log

    @property
    def agent_ids(self) -> list[str]:
        return list(self._inboxes.keys())
