from collections.abc import Awaitable
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Generic, TypeVar

from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessage

from pydantic_mas._message import Message


class ReplyStrategy(StrEnum):
    LAST_OUTPUT = "last_output"
    ANSWER_TOOL = "answer_tool"
    SEND_MESSAGE = "send_message"


DepsT = TypeVar("DepsT")


@dataclass
class FactoryContext(Generic[DepsT]):
    """Context handed to an `agent_factory` when a message arrives.

    The factory's job is to *produce the Agent* that will run this message.
    History stays owned by the AgentNode and is passed to `agent.run()` as
    `message_history`, so swapping the Agent per message preserves continuity.

    Attributes:
        agent_id: The receiving agent's id.
        incoming_message: The envelope about to be processed (post hooks).
        history: Snapshot of the AgentNode's conversation history at this point.
        deps: Resolved deps for this AgentNode.
    """

    agent_id: str
    incoming_message: Message
    history: list[ModelMessage] = field(default_factory=list)
    deps: DepsT | None = None


AgentFactoryResult = Agent[Any] | Awaitable[Agent[Any]]
AgentFactory = Callable[[FactoryContext[Any]], AgentFactoryResult]


class AgentConfig:
    """Configuration for a single agent in the MAS.

    Provide exactly one of:
      * `agent` — a single Agent instance reused for every incoming message
        (the static path; matches the original behavior).
      * `agent_factory` — a callable invoked per incoming message that returns
        an Agent. The factory sees the message (including its `metadata`) and
        can build a differently-shaped Agent — different `output_type`, tools,
        instructions, model — for that one run. History continuity is
        preserved by the AgentNode regardless.

    Not a Pydantic model because pydantic-ai's Agent cannot be embedded
    as a Pydantic field (its internal generics break schema generation).
    """

    def __init__(
        self,
        agent: Agent | None = None,
        agent_factory: AgentFactory | None = None,
        deps: Any = None,
        deps_factory: Callable[[], Any] | None = None,
    ):
        if (agent is None) == (agent_factory is None):
            raise ValueError(
                "AgentConfig requires exactly one of `agent` or `agent_factory`."
            )
        self.agent = agent
        self.agent_factory = agent_factory
        self.deps = deps
        self.deps_factory = deps_factory

    def resolve_deps(self) -> Any:
        """Get deps: use factory if provided, else return static deps."""
        if self.deps_factory is not None:
            return self.deps_factory()
        return self.deps
