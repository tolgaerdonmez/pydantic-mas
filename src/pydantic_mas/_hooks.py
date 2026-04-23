from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

from pydantic_ai.messages import ModelMessage

from pydantic_mas._message import Message

CallerDepsT = TypeVar("CallerDepsT")
CalleeDepsT = TypeVar("CalleeDepsT")


@dataclass
class MASInsertContext(Generic[CallerDepsT, CalleeDepsT]):
    """Context passed to MAS insertion hooks.

    An insertion point is the moment MAS is about to hand a Message envelope
    to a callee's `agent.run()`. `caller` and `callee` are oriented to the
    original request direction — the caller is the agent that initiated the
    exchange, the callee is the agent that responds. Those roles stay fixed
    across request and reply hooks; only the message flow direction differs.

    Attributes:
        caller_id: Initiating agent id. "system" for the entry prompt.
        caller_deps: Caller's deps. None when the caller is "system".
        caller_history: Snapshot of caller's history at the fire point.
        callee_id: Responding agent id.
        callee_deps: Callee's deps.
        callee_history: Snapshot of callee's history at the fire point.
        message: The envelope about to be inserted.
        depth: Message depth (mirrors message.depth).
    """

    caller_id: str
    caller_deps: CallerDepsT | None
    caller_history: list[ModelMessage]
    callee_id: str
    callee_deps: CalleeDepsT
    callee_history: list[ModelMessage]
    message: Message
    depth: int


InsertHookResult = Message | Awaitable[Message]
OnRequestInsertHook = Callable[[MASInsertContext], InsertHookResult]
OnReplyInsertHook = Callable[[MASInsertContext], InsertHookResult]


@dataclass
class MASHooks:
    """Developer-provided callbacks for MAS communication insertion points.

    Two hooks, one per direction of flow through the inbox bottleneck:

    on_request_insert: Fires when a REQUEST envelope is about to be handed
        to the callee's agent.run(). Return the (possibly modified) Message
        to proceed. Raise to abort.

    on_reply_insert: Fires when a REPLY envelope is about to be handed to
        the caller's agent.run(). Return the (possibly modified) Message
        to proceed. Raise to abort.

    Both sync and async callables are accepted.
    """

    on_request_insert: OnRequestInsertHook | None = None
    on_reply_insert: OnReplyInsertHook | None = None
