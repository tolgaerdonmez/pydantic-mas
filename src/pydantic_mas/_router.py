import asyncio

from pydantic_mas._budget import BudgetTracker
from pydantic_mas._message import Message, MessageType


class MessageRouter:
    """In-process message delivery layer. One per MASInstance."""

    def __init__(self, budget_tracker: BudgetTracker):
        self._budget_tracker = budget_tracker
        self._inboxes: dict[str, asyncio.Queue[Message]] = {}
        self._message_log: list[Message] = []

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

        return message

    @property
    def message_log(self) -> list[Message]:
        return self._message_log

    @property
    def agent_ids(self) -> list[str]:
        return list(self._inboxes.keys())
