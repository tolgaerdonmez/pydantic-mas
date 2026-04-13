from pydantic import BaseModel, ConfigDict
from pydantic_ai.usage import UsageLimits


class Budget(BaseModel, frozen=True):
    """Immutable budget configuration (MAS-level limits)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    max_total_messages: int | None = None
    max_agent_messages: int | None = None
    max_depth: int | None = None
    timeout_seconds: float | None = None
    usage_limits: UsageLimits | None = None


class BudgetExceededError(Exception):
    """Raised when a MAS-level budget limit is exceeded."""

    def __init__(self, metric: str, current: int, limit: int):
        self.metric = metric
        self.current = current
        self.limit = limit
        super().__init__(f"Budget exceeded: {metric} = {current}, limit = {limit}")


class BudgetSnapshot(BaseModel, frozen=True):
    """Snapshot of MAS-level budget usage at a point in time."""

    total_messages: int
    per_agent_messages: dict[str, int]
    max_depth_seen: int


class BudgetTracker:
    """Mutable runtime tracker for MAS-level resource consumption.

    Tracks message count, per-agent counts, and depth.
    Token/request tracking is handled by pydantic-ai's RunUsage.
    """

    def __init__(self, budget: Budget):
        self.budget = budget
        self.total_messages: int = 0
        self.per_agent_messages: dict[str, int] = {}
        self.max_depth_seen: int = 0

    def check_and_record_message(self, sender: str, depth: int) -> None:
        """Check limits and record a message. Raises BudgetExceededError if any limit is hit."""
        if (
            self.budget.max_total_messages is not None
            and self.total_messages >= self.budget.max_total_messages
        ):
            raise BudgetExceededError(
                "total_messages", self.total_messages, self.budget.max_total_messages
            )

        agent_count = self.per_agent_messages.get(sender, 0)
        if (
            self.budget.max_agent_messages is not None
            and agent_count >= self.budget.max_agent_messages
        ):
            raise BudgetExceededError(
                f"agent_messages[{sender}]",
                agent_count,
                self.budget.max_agent_messages,
            )

        if self.budget.max_depth is not None and depth > self.budget.max_depth:
            raise BudgetExceededError("depth", depth, self.budget.max_depth)

        self.total_messages += 1
        self.per_agent_messages[sender] = agent_count + 1
        self.max_depth_seen = max(self.max_depth_seen, depth)

    def snapshot(self) -> BudgetSnapshot:
        """Return a typed snapshot of current usage."""
        return BudgetSnapshot(
            total_messages=self.total_messages,
            per_agent_messages=dict(self.per_agent_messages),
            max_depth_seen=self.max_depth_seen,
        )
