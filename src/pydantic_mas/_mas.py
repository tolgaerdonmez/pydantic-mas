from typing import Any, Callable

from pydantic_mas._agent_node import AgentNode
from pydantic_mas._budget import Budget, BudgetTracker
from pydantic_mas._config import AgentConfig, ReplyStrategy
from pydantic_mas._formatter import default_message_formatter
from pydantic_mas._hooks import MASHooks
from pydantic_mas._instance import MASInstance
from pydantic_mas._message import Message
from pydantic_mas._result import MASResult
from pydantic_mas._router import MessageRouter


class MAS:
    """Top-level MAS blueprint. Holds agent definitions and configuration.

    Each .run() call creates a fresh MASInstance with isolated state.
    """

    def __init__(
        self,
        agents: dict[str, AgentConfig],
        reply_strategy: ReplyStrategy | str = ReplyStrategy.LAST_OUTPUT,
        interrupt_on_send: bool = False,
        budget: Budget | None = None,
        message_formatter: Callable[[Message], str] | None = None,
        hooks: MASHooks | None = None,
    ):
        self.agents = agents
        self.reply_strategy = (
            ReplyStrategy(reply_strategy)
            if isinstance(reply_strategy, str)
            else reply_strategy
        )
        self.interrupt_on_send = interrupt_on_send
        self.budget = budget or Budget()
        self.message_formatter = message_formatter or default_message_formatter
        self.hooks = hooks

    async def run(
        self,
        entry_agent: str,
        prompt: str,
        timeout: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MASResult:
        """Create a new MASInstance and run it to completion.

        Args:
            entry_agent: ID of the agent to receive the initial message.
            prompt: The initial message content.
            timeout: Optional override for budget timeout.
            metadata: Optional metadata stamped onto the initial system->entry
                message envelope. Reachable from `agent_factory` and hooks via
                `incoming_message.metadata` / `ctx.message.metadata`.

        Raises:
            ValueError: If entry_agent is not a registered agent.
        """
        if entry_agent not in self.agents:
            raise ValueError(
                f"Entry agent '{entry_agent}' not found. "
                f"Available agents: {list(self.agents.keys())}"
            )

        budget_tracker = BudgetTracker(self.budget)
        router = MessageRouter(budget_tracker)

        # Shared dict populated below — each node keeps a reference so it
        # can look up peers (needed by the insertion hooks).
        peers: dict[str, AgentNode] = {}

        agent_nodes: list[AgentNode] = []
        for agent_id, config in self.agents.items():
            if config.agent is not None and config.agent.name is None:
                config.agent._name = agent_id
            node = AgentNode(
                agent_id=agent_id,
                agent=config.agent,
                agent_factory=config.agent_factory,
                router=router,
                deps=config.resolve_deps(),
                message_formatter=self.message_formatter,
                interrupt_on_send=self.interrupt_on_send,
                hooks=self.hooks,
                peers=peers,
            )
            router.register(agent_id, node.inbox)
            peers[agent_id] = node
            agent_nodes.append(node)

        instance = MASInstance(
            agent_nodes=agent_nodes,
            router=router,
            budget_tracker=budget_tracker,
        )

        return await instance.run(
            entry_agent=entry_agent,
            prompt=prompt,
            timeout=timeout,
            metadata=metadata,
        )
