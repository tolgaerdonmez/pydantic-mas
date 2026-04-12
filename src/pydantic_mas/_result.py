from enum import StrEnum

from pydantic import BaseModel, ConfigDict
from pydantic_ai.messages import ModelMessage

from pydantic_mas._budget import BudgetSnapshot
from pydantic_mas._message import Message


class TerminationReason(StrEnum):
    COMPLETED = "completed"
    BUDGET_EXCEEDED = "budget_exceeded"
    TIMEOUT = "timeout"


class MASResult(BaseModel, frozen=True):
    """Result of a single MAS instance run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    message_log: list[Message]
    agent_histories: dict[str, list[ModelMessage]]
    termination_reason: TerminationReason
    budget_usage: BudgetSnapshot
