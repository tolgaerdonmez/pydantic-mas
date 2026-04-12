"""pydantic-mas: Multi-agent system framework built on pydantic-ai."""

from pydantic_mas._agent_node import AgentNode, AgentState
from pydantic_mas._budget import Budget, BudgetExceededError, BudgetSnapshot, BudgetTracker
from pydantic_mas._config import AgentConfig, ReplyStrategy
from pydantic_mas._formatter import default_message_formatter
from pydantic_mas._instance import MASInstance
from pydantic_mas._mas import MAS
from pydantic_mas._message import Message, MessageType
from pydantic_mas._result import MASResult, TerminationReason
from pydantic_mas._router import MessageRouter

__all__ = [
    "AgentConfig",
    "AgentNode",
    "AgentState",
    "Budget",
    "BudgetExceededError",
    "BudgetSnapshot",
    "BudgetTracker",
    "MAS",
    "MASInstance",
    "MASResult",
    "Message",
    "MessageRouter",
    "MessageType",
    "ReplyStrategy",
    "TerminationReason",
    "default_message_formatter",
]
