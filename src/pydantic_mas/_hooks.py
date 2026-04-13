from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Callable

from pydantic_mas._message import Message


@dataclass
class SendMessageHookContext:
    """Context passed to send_message hooks.

    before_send_message hooks can mutate fields (content, receiver_id, etc.)
    to modify the outgoing message, or return None to block it entirely.
    """

    sender_id: str
    receiver_id: str
    content: str
    current_message: Message
    depth: int


@dataclass
class AnswerHookContext:
    """Context passed to answer hooks (used with answer_tool reply strategy)."""

    agent_id: str
    content: str
    original_request: Message


BeforeSendHook = Callable[
    [SendMessageHookContext], Awaitable[SendMessageHookContext | None]
]
AfterSendHook = Callable[[SendMessageHookContext, Message], Awaitable[None]]
BeforeAnswerHook = Callable[[AnswerHookContext], Awaitable[AnswerHookContext | None]]
AfterAnswerHook = Callable[[AnswerHookContext, Message], Awaitable[None]]


@dataclass
class MASHooks:
    """Developer-provided callbacks for intercepting communication tools.

    before_send_message: Called before router delivers a send_message.
        Return the context (possibly modified) to proceed, or None to block.
    after_send_message: Called after router delivers the message. Observe only.
    before_answer: Called before the answer tool routes a reply.
        Same modify-or-block semantics as before_send_message.
    after_answer: Called after the reply is routed. Observe only.
    """

    before_send_message: BeforeSendHook | None = None
    after_send_message: AfterSendHook | None = None
    before_answer: BeforeAnswerHook | None = None
    after_answer: AfterAnswerHook | None = None
