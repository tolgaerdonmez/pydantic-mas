import asyncio
import inspect
from enum import StrEnum
from typing import Any, Callable

from pydantic_ai import Agent, Tool
from pydantic_ai._agent_graph import CallToolsNode, ModelRequestNode
from pydantic_ai.agent import AgentRunResult
from pydantic_ai.messages import ModelMessage
from pydantic_ai.toolsets import FunctionToolset
from pydantic_graph.nodes import End

from pydantic_mas._budget import BudgetExceededError
from pydantic_mas._formatter import default_message_formatter
from pydantic_mas._hooks import MASHooks, MASInsertContext
from pydantic_mas._message import Message, MessageType
from pydantic_mas._router import MessageRouter


class _HookRaisedError(BaseException):
    """Private marker wrapping an exception raised from a MAS hook.

    Hook exceptions must propagate out of `mas.run()`, while agent-internal
    crashes stay contained. This wrapper lets the instance tell the two
    apart when draining agent tasks. Inherits from BaseException so it is
    not caught by `except Exception` blocks inside pydantic-ai internals.
    """

    __slots__ = ("original",)

    def __init__(self, original: BaseException):
        super().__init__(str(original))
        self.original = original


class AgentState(StrEnum):
    IDLE = "idle"
    PROCESSING = "processing"


class AgentNode[DepsT]:
    """Runtime wrapper around a single pydantic-ai Agent.

    Each node is an independent actor with its own inbox, processing loop,
    conversation history, and state tracking.

    Type parameter DepsT matches the agent's deps_type so that deps are
    passed through to pydantic-ai with the correct type.
    """

    def __init__(
        self,
        agent_id: str,
        agent: Agent[DepsT],
        router: MessageRouter,
        deps: DepsT = None,
        message_formatter: Callable[[Message], str] | None = None,
        interrupt_on_send: bool = False,
        hooks: MASHooks | None = None,
        peers: "dict[str, AgentNode[Any]] | None" = None,
    ):
        self.agent_id = agent_id
        self.agent = agent
        self.router = router
        self.deps = deps
        self.message_formatter = message_formatter or default_message_formatter
        self._interrupt_on_send = interrupt_on_send
        self._hooks = hooks
        # Shared mutable dict: MAS.run() populates it progressively as each
        # node is constructed, so every node sees the full peer set by the
        # time the run loop starts.
        self._peers = peers

        self.inbox: asyncio.Queue[Message] = asyncio.Queue()
        self.history: list[ModelMessage] = []
        self.state: AgentState = AgentState.IDLE
        self.current_depth: int = 0
        self.current_message: Message | None = None

        self._idle_event: asyncio.Event = asyncio.Event()
        self._idle_event.set()

        self._interrupt_requested: bool = False

    @property
    def idle_event(self) -> asyncio.Event:
        return self._idle_event

    def _make_send_message_tool(self) -> Tool[Any]:
        """Create the send_message tool closure.

        WARNING: This tool MUST remain sequential=True. It calls router.route()
        which mutates shared state (budget counters, message log) via a synchronous
        check-then-act pattern. If pydantic-ai executed multiple send_message calls
        concurrently (its default for tool batches), interleaving at any await point
        could cause race conditions. sequential=True forces pydantic-ai to execute
        tool batches one at a time when this tool is in the batch.
        """
        node = self
        router = self.router

        async def send_message(target_agent: str, content: str) -> str:
            """Send a message to another agent in the system.

            Args:
                target_agent: The ID of the agent to send the message to.
                content: The message content to send.
            """
            try:
                msg = router.route(
                    sender=node.agent_id,
                    receiver=target_agent,
                    content=content,
                    type=MessageType.REQUEST,
                    depth=node.current_depth + 1,
                )

                if node._interrupt_on_send:
                    node._interrupt_requested = True
                return f"Message sent to '{target_agent}' (id: {msg.id})"
            except (ValueError, BudgetExceededError) as e:
                return f"Error: {e}"

        return Tool(send_message, sequential=True)

    def _build_runtime_toolset(self) -> FunctionToolset[DepsT]:
        """Build the toolset of framework-injected tools.

        This is passed via agent.run(toolsets=[...]) which ADDS to the agent's
        existing tools rather than replacing them. Do NOT use agent.override(tools=[...])
        as that REPLACES all tools the developer registered on the agent.
        """
        return FunctionToolset[DepsT]([self._make_send_message_tool()])

    async def run_loop(self) -> None:
        """Main processing loop. Runs until cancelled."""
        try:
            while True:
                self.state = AgentState.IDLE
                self._idle_event.set()

                message = await self.inbox.get()

                self.state = AgentState.PROCESSING
                self._idle_event.clear()
                self.current_message = message
                self.current_depth = message.depth

                await self._process_message(message)
        except asyncio.CancelledError:
            pass
        finally:
            self.state = AgentState.IDLE
            self._idle_event.set()

    async def _process_message(self, message: Message) -> None:
        """Process a single incoming message."""
        message = await self._fire_insertion_hook(message)

        formatted = self.message_formatter(message)
        self._interrupt_requested = False

        if self._interrupt_on_send:
            await self._process_with_interrupt(formatted, message)
        else:
            await self._process_simple(formatted, message)

    async def _fire_insertion_hook(self, message: Message) -> Message:
        """Fire the matching MAS insertion hook for this message.

        Returns the (possibly modified) Message. If no hook is registered,
        returns the message unchanged. Exceptions from the hook propagate.
        """
        if self._hooks is None:
            return message

        if message.type == MessageType.REQUEST:
            hook = self._hooks.on_request_insert
            caller_node = self._peers.get(message.sender) if self._peers else None
            callee_node = self
        elif message.type == MessageType.REPLY:
            hook = self._hooks.on_reply_insert
            caller_node = self
            callee_node = self._peers.get(message.sender) if self._peers else None
        else:
            return message

        if hook is None or callee_node is None:
            return message

        ctx: MASInsertContext[Any, Any] = MASInsertContext(
            caller_id=caller_node.agent_id if caller_node else message.sender,
            caller_deps=caller_node.deps if caller_node else None,
            caller_history=list(caller_node.history) if caller_node else [],
            callee_id=callee_node.agent_id,
            callee_deps=callee_node.deps,
            callee_history=list(callee_node.history),
            message=message,
            depth=message.depth,
        )

        try:
            result = hook(ctx)
            if inspect.isawaitable(result):
                result = await result
        except BaseException as exc:
            raise _HookRaisedError(exc) from exc
        return result

    async def _process_simple(self, formatted: str, message: Message) -> None:
        """Simple processing: run agent to completion."""
        result = await self.agent.run(
            user_prompt=formatted,
            message_history=self.history,
            deps=self.deps,
            toolsets=[self._build_runtime_toolset()],
        )

        self.history = list(result.all_messages())
        self._handle_last_output_reply(message, result)

    async def _process_with_interrupt(self, formatted: str, message: Message) -> None:
        """Processing with interrupt-on-send: use Agent.iter() for turn control.

        Agent.iter() yields nodes BEFORE they execute. When we see a
        CallToolsNode, we call run.next(node) to execute it, THEN check the
        interrupt flag. If set, we break before the next LLM turn.

        After breaking, the tool return results live in next_node.request
        (a ModelRequest) but haven't been appended to run.all_messages() yet.
        We manually append them to preserve complete history.
        """
        interrupted = False

        async with self.agent.iter(
            user_prompt=formatted,
            message_history=self.history,
            deps=self.deps,
            toolsets=[self._build_runtime_toolset()],
        ) as run:
            node = run.next_node
            while not isinstance(node, End):
                next_node = await run.next(node)

                if isinstance(node, CallToolsNode) and self._interrupt_requested:
                    # All tools in this turn have executed.
                    # Capture messages including the pending tool returns.
                    self.history = list(run.all_messages())
                    if isinstance(next_node, ModelRequestNode):
                        self.history.append(next_node.request)
                    interrupted = True
                    break

                node = next_node

            if not interrupted:
                self.history = list(run.all_messages())

        if not interrupted and run.result is not None:
            self._handle_last_output_reply(message, run.result)

    def _handle_last_output_reply(
        self, original_message: Message, result: AgentRunResult[str]
    ) -> None:
        """last_output strategy: auto-reply with the agent's text output."""
        if original_message.type != MessageType.REQUEST:
            return
        if original_message.sender == "system":
            return

        output_text = result.output
        if isinstance(output_text, str) and output_text.strip():
            try:
                self.router.route(
                    sender=self.agent_id,
                    receiver=original_message.sender,
                    content=output_text,
                    type=MessageType.REPLY,
                    in_reply_to=original_message.id,
                    depth=original_message.depth,
                )
            except Exception:
                pass  # budget exceeded during reply — silently drop
