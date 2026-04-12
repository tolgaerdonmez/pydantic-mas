from enum import StrEnum
from typing import Any, Callable

from pydantic_ai import Agent


class ReplyStrategy(StrEnum):
    LAST_OUTPUT = "last_output"
    ANSWER_TOOL = "answer_tool"
    SEND_MESSAGE = "send_message"


class AgentConfig:
    """Configuration for a single agent in the MAS.

    Not a Pydantic model because pydantic-ai's Agent cannot be embedded
    as a Pydantic field (its internal generics break schema generation).
    """

    def __init__(
        self,
        agent: Agent,
        deps: Any = None,
        deps_factory: Callable[[], Any] | None = None,
    ):
        self.agent = agent
        self.deps = deps
        self.deps_factory = deps_factory

    def resolve_deps(self) -> Any:
        """Get deps: use factory if provided, else return static deps."""
        if self.deps_factory is not None:
            return self.deps_factory()
        return self.deps
