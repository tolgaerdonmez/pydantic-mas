"""E2E tests: multiple agents interacting with shared mutable state.

These tests verify that shared state (passed via deps or closures) is
safely mutated by multiple agents within a single MAS run. Since all
agents run as asyncio tasks in a single event loop and all mutations
happen in synchronous (non-awaiting) code, there are no race conditions.
"""

import json

from pydantic_ai import Agent, RunContext, Tool
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_mas._config import AgentConfig
from pydantic_mas._mas import MAS
from pydantic_mas._result import TerminationReason


# ---------------------------------------------------------------------------
# Shared state types
# ---------------------------------------------------------------------------


class SharedReport:
    """Mutable document that multiple agents contribute to."""

    def __init__(self) -> None:
        self.sections: list[str] = []

    def add_section(self, author: str, content: str) -> None:
        self.sections.append(f"[{author}] {content}")


class SharedCounter:
    """Simple counter incremented by multiple agents."""

    def __init__(self) -> None:
        self.value: int = 0
        self.history: list[tuple[str, int]] = []

    def increment(self, by: str) -> int:
        self.value += 1
        self.history.append((by, self.value))
        return self.value


class SharedLog:
    """Append-only log preserving write order."""

    def __init__(self) -> None:
        self.entries: list[str] = []

    def append(self, entry: str) -> None:
        self.entries.append(entry)


# ---------------------------------------------------------------------------
# Model helpers
#
# All handlers are STATELESS — they detect whether a tool was already called
# by inspecting the messages (looking for ToolReturnPart/ToolCallPart) rather
# than using a closure counter. This is critical because MAS reuses the same
# Agent objects across runs, so closure state would leak between runs.
# ---------------------------------------------------------------------------


def _has_tool_interaction(messages: list[ModelMessage]) -> bool:
    """Check if any prior message contains a tool call or return."""
    for msg in messages:
        if hasattr(msg, "parts"):
            for part in msg.parts:
                if hasattr(part, "tool_name"):
                    return True
    return False


def _coordinator_model(targets: list[str]) -> FunctionModel:
    """Model that fans out send_message to all targets, then returns text."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _has_tool_interaction(messages):
            return ModelResponse(parts=[TextPart(content="Delegated")])

        parts = [
            ToolCallPart(
                tool_name="send_message",
                args=json.dumps(
                    {"target_agent": t, "content": f"Task for {t}"}
                ),
                tool_call_id=f"to-{t}",
            )
            for t in targets
        ]
        return ModelResponse(parts=parts)

    return FunctionModel(handler)


def _worker_model(tool_name: str, tool_args: dict) -> FunctionModel:
    """Model that calls a specific tool once, then returns text."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _has_tool_interaction(messages):
            return ModelResponse(parts=[TextPart(content="Done")])

        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=tool_name,
                    args=json.dumps(tool_args),
                    tool_call_id="work-call",
                )
            ]
        )

    return FunctionModel(handler)


def _chain_model(
    tool_name: str, tool_args: dict, send_to: str | None = None
) -> FunctionModel:
    """Model that calls a tool and optionally sends to the next agent."""

    def handler(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        if _has_tool_interaction(messages):
            return ModelResponse(parts=[TextPart(content="Done")])

        parts: list[ToolCallPart] = [
            ToolCallPart(
                tool_name=tool_name,
                args=json.dumps(tool_args),
                tool_call_id="tool-call",
            ),
        ]
        if send_to:
            parts.append(
                ToolCallPart(
                    tool_name="send_message",
                    args=json.dumps(
                        {
                            "target_agent": send_to,
                            "content": "Continue the chain",
                        }
                    ),
                    tool_call_id="chain-call",
                )
            )
        return ModelResponse(parts=parts)

    return FunctionModel(handler)


# ---------------------------------------------------------------------------
# Tool factories (closures capture shared state)
# ---------------------------------------------------------------------------


def _make_report_tool(report: SharedReport, author: str) -> Tool:
    async def add_to_report(content: str) -> str:
        """Add a section to the shared report.

        Args:
            content: The section content to add.
        """
        report.add_section(author, content)
        return "Section added"

    return Tool(add_to_report)


def _make_counter_tool(counter: SharedCounter, agent_id: str) -> Tool:
    async def increment_counter() -> str:
        """Increment the shared counter."""
        new_val = counter.increment(by=agent_id)
        return f"Counter is now {new_val}"

    return Tool(increment_counter)


def _make_log_tool(log: SharedLog, agent_id: str) -> Tool:
    async def log_activity(detail: str) -> str:
        """Log an activity to the shared log.

        Args:
            detail: Description of the activity.
        """
        log.append(f"{agent_id}: {detail}")
        return "Logged"

    return Tool(log_activity)


# ---------------------------------------------------------------------------
# Tests: shared report assembly
# ---------------------------------------------------------------------------


class TestSharedReportAssembly:
    """Coordinator delegates to researcher + writer, both write to shared report."""

    async def test_both_workers_contribute_sections(self):
        report = SharedReport()

        researcher_agent = Agent(
            model=_worker_model(
                "add_to_report", {"content": "Research findings on topic X"}
            ),
            tools=[_make_report_tool(report, "researcher")],
        )
        writer_agent = Agent(
            model=_worker_model(
                "add_to_report", {"content": "Introduction draft for topic X"}
            ),
            tools=[_make_report_tool(report, "writer")],
        )

        mas = MAS(
            agents={
                "coordinator": AgentConfig(
                    agent=Agent(
                        model=_coordinator_model(["researcher", "writer"])
                    )
                ),
                "researcher": AgentConfig(agent=researcher_agent),
                "writer": AgentConfig(agent=writer_agent),
            }
        )

        result = await mas.run(
            entry_agent="coordinator", prompt="Build a report on X"
        )

        assert result.termination_reason == TerminationReason.COMPLETED
        assert len(report.sections) == 2

        authors = [s.split("]")[0].strip("[") for s in report.sections]
        assert "researcher" in authors
        assert "writer" in authors

    async def test_report_content_matches_tool_args(self):
        report = SharedReport()

        mas = MAS(
            agents={
                "coordinator": AgentConfig(
                    agent=Agent(
                        model=_coordinator_model(["researcher", "writer"])
                    )
                ),
                "researcher": AgentConfig(
                    agent=Agent(
                        model=_worker_model(
                            "add_to_report", {"content": "quantum computing"}
                        ),
                        tools=[_make_report_tool(report, "researcher")],
                    )
                ),
                "writer": AgentConfig(
                    agent=Agent(
                        model=_worker_model(
                            "add_to_report", {"content": "executive summary"}
                        ),
                        tools=[_make_report_tool(report, "writer")],
                    )
                ),
            }
        )

        result = await mas.run(entry_agent="coordinator", prompt="Go")

        assert result.termination_reason == TerminationReason.COMPLETED
        contents = " ".join(report.sections)
        assert "quantum computing" in contents
        assert "executive summary" in contents


# ---------------------------------------------------------------------------
# Tests: shared counter fan-out
# ---------------------------------------------------------------------------


class TestSharedCounterFanOut:
    """Coordinator fans out to N workers, each increments a shared counter."""

    async def test_five_workers_increment_counter(self):
        counter = SharedCounter()
        worker_ids = [f"worker_{i}" for i in range(5)]

        agents: dict[str, AgentConfig] = {
            "coordinator": AgentConfig(
                agent=Agent(model=_coordinator_model(worker_ids))
            ),
        }
        for wid in worker_ids:
            agents[wid] = AgentConfig(
                agent=Agent(
                    model=_worker_model("increment_counter", {}),
                    tools=[_make_counter_tool(counter, wid)],
                )
            )

        mas = MAS(agents=agents)
        result = await mas.run(entry_agent="coordinator", prompt="Go")

        assert result.termination_reason == TerminationReason.COMPLETED
        assert counter.value == 5

    async def test_counter_history_has_all_workers(self):
        counter = SharedCounter()
        worker_ids = [f"w{i}" for i in range(3)]

        agents: dict[str, AgentConfig] = {
            "coordinator": AgentConfig(
                agent=Agent(model=_coordinator_model(worker_ids))
            ),
        }
        for wid in worker_ids:
            agents[wid] = AgentConfig(
                agent=Agent(
                    model=_worker_model("increment_counter", {}),
                    tools=[_make_counter_tool(counter, wid)],
                )
            )

        mas = MAS(agents=agents)
        result = await mas.run(entry_agent="coordinator", prompt="Go")

        assert result.termination_reason == TerminationReason.COMPLETED
        who_incremented = {agent_id for agent_id, _ in counter.history}
        assert who_incremented == set(worker_ids)

    async def test_counter_values_are_sequential(self):
        """Each increment produces a monotonically increasing value (no lost updates)."""
        counter = SharedCounter()
        worker_ids = [f"w{i}" for i in range(4)]

        agents: dict[str, AgentConfig] = {
            "coordinator": AgentConfig(
                agent=Agent(model=_coordinator_model(worker_ids))
            ),
        }
        for wid in worker_ids:
            agents[wid] = AgentConfig(
                agent=Agent(
                    model=_worker_model("increment_counter", {}),
                    tools=[_make_counter_tool(counter, wid)],
                )
            )

        mas = MAS(agents=agents)
        await mas.run(entry_agent="coordinator", prompt="Go")

        values = [v for _, v in counter.history]
        assert values == list(range(1, len(worker_ids) + 1))


# ---------------------------------------------------------------------------
# Tests: chain with shared log (ordering)
# ---------------------------------------------------------------------------


class TestChainWithSharedLog:
    """A→B→C chain, each appends to shared log. Verifies causal ordering."""

    async def test_chain_writes_in_causal_order(self):
        log = SharedLog()

        mas = MAS(
            agents={
                "coordinator": AgentConfig(
                    agent=Agent(model=_coordinator_model(["agent_a"]))
                ),
                "agent_a": AgentConfig(
                    agent=Agent(
                        model=_chain_model(
                            "log_activity",
                            {"detail": "step 1"},
                            send_to="agent_b",
                        ),
                        tools=[_make_log_tool(log, "agent_a")],
                    )
                ),
                "agent_b": AgentConfig(
                    agent=Agent(
                        model=_chain_model(
                            "log_activity",
                            {"detail": "step 2"},
                            send_to="agent_c",
                        ),
                        tools=[_make_log_tool(log, "agent_b")],
                    )
                ),
                "agent_c": AgentConfig(
                    agent=Agent(
                        model=_chain_model(
                            "log_activity", {"detail": "step 3"}, send_to=None
                        ),
                        tools=[_make_log_tool(log, "agent_c")],
                    )
                ),
            }
        )

        result = await mas.run(entry_agent="coordinator", prompt="Start chain")

        assert result.termination_reason == TerminationReason.COMPLETED
        assert len(log.entries) == 3

        # Causal order: A processes before B (B depends on A's send_message),
        # B processes before C (C depends on B's send_message).
        assert log.entries[0].startswith("agent_a:")
        assert log.entries[1].startswith("agent_b:")
        assert log.entries[2].startswith("agent_c:")

    async def test_chain_log_content(self):
        log = SharedLog()

        mas = MAS(
            agents={
                "coordinator": AgentConfig(
                    agent=Agent(model=_coordinator_model(["first"]))
                ),
                "first": AgentConfig(
                    agent=Agent(
                        model=_chain_model(
                            "log_activity",
                            {"detail": "gathered data"},
                            send_to="second",
                        ),
                        tools=[_make_log_tool(log, "first")],
                    )
                ),
                "second": AgentConfig(
                    agent=Agent(
                        model=_chain_model(
                            "log_activity",
                            {"detail": "analyzed data"},
                            send_to=None,
                        ),
                        tools=[_make_log_tool(log, "second")],
                    )
                ),
            }
        )

        result = await mas.run(entry_agent="coordinator", prompt="Go")

        assert result.termination_reason == TerminationReason.COMPLETED
        assert log.entries == [
            "first: gathered data",
            "second: analyzed data",
        ]


# ---------------------------------------------------------------------------
# Tests: concurrent fan-out write integrity
# ---------------------------------------------------------------------------


class TestConcurrentWriteIntegrity:
    """Multiple workers write to the same list from a fan-out.
    Verifies no writes are lost and all entries are intact.
    """

    async def test_all_fan_out_writes_present(self):
        report = SharedReport()
        worker_ids = [f"agent_{i}" for i in range(6)]

        agents: dict[str, AgentConfig] = {
            "coordinator": AgentConfig(
                agent=Agent(model=_coordinator_model(worker_ids))
            ),
        }
        for wid in worker_ids:
            agents[wid] = AgentConfig(
                agent=Agent(
                    model=_worker_model(
                        "add_to_report", {"content": f"output from {wid}"}
                    ),
                    tools=[_make_report_tool(report, wid)],
                )
            )

        mas = MAS(agents=agents)
        result = await mas.run(entry_agent="coordinator", prompt="Go")

        assert result.termination_reason == TerminationReason.COMPLETED
        assert len(report.sections) == 6

        # Every worker's output should appear exactly once
        for wid in worker_ids:
            matching = [s for s in report.sections if s.startswith(f"[{wid}]")]
            assert len(matching) == 1, f"Expected exactly 1 section from {wid}"

    async def test_no_corrupted_entries(self):
        """Each entry should be a complete, well-formed string — no interleaving."""
        report = SharedReport()
        worker_ids = [f"w{i}" for i in range(4)]

        agents: dict[str, AgentConfig] = {
            "coordinator": AgentConfig(
                agent=Agent(model=_coordinator_model(worker_ids))
            ),
        }
        for wid in worker_ids:
            agents[wid] = AgentConfig(
                agent=Agent(
                    model=_worker_model(
                        "add_to_report",
                        {"content": f"{'x' * 100}"},  # long string
                    ),
                    tools=[_make_report_tool(report, wid)],
                )
            )

        mas = MAS(agents=agents)
        result = await mas.run(entry_agent="coordinator", prompt="Go")

        assert result.termination_reason == TerminationReason.COMPLETED

        for section in report.sections:
            # Each section should match the pattern [agent_id] content
            assert section.startswith("[w")
            assert "] " in section
            content = section.split("] ", 1)[1]
            assert content == "x" * 100


# ---------------------------------------------------------------------------
# Tests: shared state via deps (RunContext)
# ---------------------------------------------------------------------------


class TestSharedStateViaDeps:
    """Shared state passed through deps/RunContext instead of closures."""

    async def test_deps_shared_across_agents(self):
        counter = SharedCounter()

        async def bump(ctx: RunContext[SharedCounter]) -> str:
            """Increment the shared counter via deps."""
            ctx.deps.increment(by="via-deps")
            return "bumped"

        worker_ids = ["alpha", "beta", "gamma"]

        agents: dict[str, AgentConfig] = {
            "coordinator": AgentConfig(
                agent=Agent(model=_coordinator_model(worker_ids))
            ),
        }
        for wid in worker_ids:
            agent: Agent[SharedCounter] = Agent(
                model=_worker_model("bump", {}),
                tools=[Tool(bump)],
            )
            agents[wid] = AgentConfig(agent=agent, deps=counter)

        mas = MAS(agents=agents)
        result = await mas.run(entry_agent="coordinator", prompt="Go")

        assert result.termination_reason == TerminationReason.COMPLETED
        assert counter.value == 3

    async def test_deps_factory_gives_isolated_state_per_run(self):
        """Each mas.run() gets a fresh counter via deps_factory."""
        counters: list[SharedCounter] = []

        def counter_factory() -> SharedCounter:
            c = SharedCounter()
            counters.append(c)
            return c

        async def bump(ctx: RunContext[SharedCounter]) -> str:
            """Increment."""
            ctx.deps.increment(by="worker")
            return "bumped"

        mas = MAS(
            agents={
                "worker": AgentConfig(
                    agent=Agent(
                        model=_worker_model("bump", {}),
                        tools=[Tool(bump)],
                    ),
                    deps_factory=counter_factory,
                ),
            }
        )

        await mas.run(entry_agent="worker", prompt="run 1")
        await mas.run(entry_agent="worker", prompt="run 2")

        assert len(counters) == 2
        assert counters[0].value == 1
        assert counters[1].value == 1
        assert counters[0] is not counters[1]
