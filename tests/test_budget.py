"""Tests for Budget, BudgetSnapshot, BudgetTracker, and BudgetExceededError."""

import pytest

from pydantic_mas._budget import (
    Budget,
    BudgetExceededError,
    BudgetSnapshot,
    BudgetTracker,
)


class TestBudget:
    def test_defaults_are_none(self):
        budget = Budget()
        assert budget.max_total_messages is None
        assert budget.max_agent_messages is None
        assert budget.max_depth is None
        assert budget.timeout_seconds is None
        assert budget.usage_limits is None

    def test_frozen_immutable(self):
        budget = Budget(max_total_messages=10)
        with pytest.raises(Exception):
            budget.max_total_messages = 20


class TestBudgetExceededError:
    def test_attributes(self):
        err = BudgetExceededError("total_messages", 10, 10)
        assert err.metric == "total_messages"
        assert err.current == 10
        assert err.limit == 10

    def test_message_format(self):
        err = BudgetExceededError("depth", 5, 3)
        assert "depth" in str(err)
        assert "5" in str(err)
        assert "3" in str(err)

    def test_is_exception(self):
        err = BudgetExceededError("x", 1, 1)
        assert isinstance(err, Exception)


class TestBudgetSnapshot:
    def test_fields(self):
        snap = BudgetSnapshot(
            total_messages=5,
            per_agent_messages={"a": 3, "b": 2},
            max_depth_seen=2,
        )
        assert snap.total_messages == 5
        assert snap.per_agent_messages == {"a": 3, "b": 2}
        assert snap.max_depth_seen == 2

    def test_frozen(self):
        snap = BudgetSnapshot(
            total_messages=0, per_agent_messages={}, max_depth_seen=0
        )
        with pytest.raises(Exception):
            snap.total_messages = 1


class TestBudgetTracker:
    def test_no_limits_never_raises(self):
        tracker = BudgetTracker(Budget())
        for i in range(100):
            tracker.check_and_record_message("agent_a", depth=i)

    def test_max_total_messages(self):
        tracker = BudgetTracker(Budget(max_total_messages=3))
        tracker.check_and_record_message("a", depth=0)
        tracker.check_and_record_message("a", depth=0)
        tracker.check_and_record_message("a", depth=0)
        with pytest.raises(BudgetExceededError, match="total_messages"):
            tracker.check_and_record_message("a", depth=0)

    def test_max_agent_messages(self):
        tracker = BudgetTracker(Budget(max_agent_messages=2))
        tracker.check_and_record_message("a", depth=0)
        tracker.check_and_record_message("a", depth=0)
        with pytest.raises(BudgetExceededError, match="agent_messages"):
            tracker.check_and_record_message("a", depth=0)

    def test_max_agent_messages_independent_per_agent(self):
        tracker = BudgetTracker(Budget(max_agent_messages=2))
        tracker.check_and_record_message("a", depth=0)
        tracker.check_and_record_message("a", depth=0)
        # agent "b" has its own counter
        tracker.check_and_record_message("b", depth=0)
        tracker.check_and_record_message("b", depth=0)
        # both should now be at limit
        with pytest.raises(BudgetExceededError):
            tracker.check_and_record_message("a", depth=0)
        with pytest.raises(BudgetExceededError):
            tracker.check_and_record_message("b", depth=0)

    def test_max_depth(self):
        tracker = BudgetTracker(Budget(max_depth=2))
        tracker.check_and_record_message("a", depth=0)
        tracker.check_and_record_message("a", depth=1)
        tracker.check_and_record_message("a", depth=2)
        with pytest.raises(BudgetExceededError, match="depth"):
            tracker.check_and_record_message("a", depth=3)

    def test_snapshot_returns_typed_model(self):
        tracker = BudgetTracker(Budget())
        tracker.check_and_record_message("a", depth=0)
        tracker.check_and_record_message("b", depth=1)
        tracker.check_and_record_message("a", depth=2)

        snap = tracker.snapshot()
        assert isinstance(snap, BudgetSnapshot)
        assert snap.total_messages == 3
        assert snap.per_agent_messages == {"a": 2, "b": 1}
        assert snap.max_depth_seen == 2

    def test_snapshot_empty_tracker(self):
        tracker = BudgetTracker(Budget())
        snap = tracker.snapshot()
        assert snap.total_messages == 0
        assert snap.per_agent_messages == {}
        assert snap.max_depth_seen == 0

    def test_check_before_increment(self):
        """Budget check happens BEFORE incrementing, so limit=3 allows exactly 3 messages."""
        tracker = BudgetTracker(Budget(max_total_messages=3))
        tracker.check_and_record_message("a", depth=0)  # count: 1
        tracker.check_and_record_message("a", depth=0)  # count: 2
        tracker.check_and_record_message("a", depth=0)  # count: 3
        # 4th should fail
        with pytest.raises(BudgetExceededError) as exc_info:
            tracker.check_and_record_message("a", depth=0)
        assert exc_info.value.current == 3
        assert exc_info.value.limit == 3
