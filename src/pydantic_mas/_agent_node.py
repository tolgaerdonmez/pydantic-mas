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
        enforce_reply_protocol: bool = True,
        hooks: MASHooks | None = None,
        peers: "dict[str, AgentNode[Any]] | None" = None,
    ):
        self.agent_id = agent_id
        self.agent = agent
        self.router = router
        self.deps = deps
        self.message_formatter = message_formatter or default_message_formatter
        self._interrupt_on_send = interrupt_on_send
        self.enforce_reply_protocol = enforce_reply_protocol
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

        # Reply-debt tracking: REQUESTs this agent has accepted but not yet
        # answered. The list has no implicit ordering — debts are removed by
        # id whenever they resolve, in whatever order their sub-conversations
        # finish. `current_debt` points at the entry being served in the
        # current inbox iteration. `_outgoing_scope` records, for each
        # outgoing REQUEST this agent fires, which debt was active at the
        # time, so an incoming REPLY can be re-attributed to the correct
        # debt frame even after the agent has parked and woken.
        self.reply_debt: list[Message] = []
        self.current_debt: Message | None = None
        self._outgoing_scope: dict[str, str] = {}

        self._idle_event: asyncio.Event = asyncio.Event()
        self._idle_event.set()

        self._interrupt_requested: bool = False
        self._reply_intercepted: bool = False

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
                # Reply-protocol intercept: if the model is using send_message
                # to answer the requester whose debt this iteration is serving,
                # route a REPLY directly and terminate the run. The debt may
                # have been resolved on a much earlier inbox cycle (e.g. via a
                # sub-reply chain) — `current_debt` carries that frame across
                # iterations, while `current_message` is just the literal
                # popped envelope.
                debt = node.current_debt
                if (
                    node.enforce_reply_protocol
                    and debt is not None
                    and target_agent == debt.sender
                ):
                    if node._reply_intercepted:
                        return (
                            f"Error: already replied to '{target_agent}' this "
                            "turn; run is terminating."
                        )
                    msg = router.route(
                        sender=node.agent_id,
                        receiver=target_agent,
                        content=content,
                        type=MessageType.REPLY,
                        in_reply_to=debt.id,
                        depth=debt.depth,
                    )
                    # Debt resolved — drop from the list. We deliberately keep
                    # `current_debt` set so a second send to the same target
                    # within the same tool batch hits the `_reply_intercepted`
                    # guard above instead of falling through to a fresh
                    # REQUEST.
                    node.reply_debt = [d for d in node.reply_debt if d.id != debt.id]
                    node._reply_intercepted = True
                    node._interrupt_requested = True
                    return (
                        f"Reply delivered to '{target_agent}' (id: {msg.id}). "
                        "Run terminating."
                    )

                msg = router.route(
                    sender=node.agent_id,
                    receiver=target_agent,
                    content=content,
                    type=MessageType.REQUEST,
                    depth=node.current_depth + 1,
                )

                # Remember which debt frame this outgoing REQUEST was sent
                # under so the eventual REPLY can be routed back to the
                # correct debt even if other inbox messages arrive in between.
                if debt is not None:
                    node._outgoing_scope[msg.id] = debt.id

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
                self._resolve_current_debt(message)

                await self._process_message(message)
        except asyncio.CancelledError:
            pass
        finally:
            self.state = AgentState.IDLE
            self._idle_event.set()

    def _resolve_current_debt(self, message: Message) -> None:
        """Set `current_debt` to the debt frame this iteration is serving.

        - REQUEST from a non-system peer: a brand-new debt — push onto
          `reply_debt` and serve it.
        - REQUEST from "system": no one to repay; debt is None.
        - REPLY: re-attribute via `_outgoing_scope` (recorded when this
          agent fired the originating REQUEST). If the parent debt is
          still outstanding, serve it; otherwise None (sub-reply for an
          already-resolved frame, or for a request we never made).
        - NOTIFICATION (or anything else): no debt frame.
        """
        if message.type == MessageType.REQUEST:
            if message.sender == "system":
                self.current_debt = None
                return
            self.reply_debt.append(message)
            self.current_debt = message
            return

        if message.type == MessageType.REPLY and message.in_reply_to is not None:
            debt_id = self._outgoing_scope.get(message.in_reply_to)
            if debt_id is not None:
                for d in self.reply_debt:
                    if d.id == debt_id:
                        self.current_debt = d
                        return
        self.current_debt = None

    async def _process_message(self, message: Message) -> None:
        """Process a single incoming message."""
        message = await self._fire_insertion_hook(message)

        formatted = self.message_formatter(message)
        self._interrupt_requested = False
        self._reply_intercepted = False

        # The iter() path is required whenever we may need to break the run
        # mid-flight: either explicit interrupt-on-send, or the reply-protocol
        # intercept which terminates B as soon as it tries to send_message
        # back to its current requester.
        if self._interrupt_on_send or self.enforce_reply_protocol:
            await self._process_with_interrupt(formatted)
        else:
            await self._process_simple(formatted)

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

    async def _process_simple(self, formatted: str) -> None:
        """Simple processing: run agent to completion."""
        result = await self.agent.run(
            user_prompt=formatted,
            message_history=self.history,
            deps=self.deps,
            toolsets=[self._build_runtime_toolset()],
        )

        self.history = list(result.all_messages())
        self._handle_last_output_reply(result)

    async def _process_with_interrupt(self, formatted: str) -> None:
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
            self._handle_last_output_reply(run.result)

    def _handle_last_output_reply(self, result: AgentRunResult[str]) -> None:
        """last_output strategy: auto-reply against the active debt frame.

        If `current_debt` is set, the agent's final text settles that debt:
        a REPLY is routed to the debt's original requester (which may be
        several inbox cycles upstream from `current_message`), and the
        debt is removed from the list. If there is no active debt
        (system-entry agent, processing a REPLY whose parent was already
        resolved, NOTIFICATION, etc.) there is nothing to auto-route.
        """
        if self._reply_intercepted:
            return
        debt = self.current_debt
        if debt is None:
            return

        output_text = result.output
        if not (isinstance(output_text, str) and output_text.strip()):
            return

        try:
            self.router.route(
                sender=self.agent_id,
                receiver=debt.sender,
                content=output_text,
                type=MessageType.REPLY,
                in_reply_to=debt.id,
                depth=debt.depth,
            )
            self.reply_debt = [d for d in self.reply_debt if d.id != debt.id]
            self.current_debt = None
        except Exception:
            pass  # budget exceeded during reply — silently drop
