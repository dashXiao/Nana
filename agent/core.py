"""Minimal ReAct loop."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.guardrails import InjectionDetector
from agent.llm import chat as default_chat
from agent.memory.events import MemoryEvent
from agent.memory.long_term import LongTermMemory
from agent.memory.short_term import ShortTermMemory
from agent.tools._contract import ToolRegistry, ToolSpec

ChatFn = Callable[..., Any]

_INJECTION_REFUSAL = (
    "I can only help with NanaShop orders, products, policies, and other support questions. "
    "I cannot help with this request."
)


@dataclass
class ToolInvocation:
    """Complete tool invocation snapshot for evaluation against expected behavior."""

    call_id: str
    name: str
    arguments: dict
    result: dict


@dataclass
class Step:
    """One ReAct step: the assistant message and all dispatched tool calls."""

    assistant_message: dict
    invocations: list[ToolInvocation] = field(default_factory=list)


@dataclass
class RunResult:
    """Terminal state of run(); final=None means max_steps was reached without a final answer."""

    final: str | None
    messages: list[dict]
    steps: list[Step]
    intent: str = "other"
    route_path: str = "default"
    confidence: float = 0.0
    memory_events: list[MemoryEvent] = field(default_factory=list)


class Agent:
    """Minimal ReAct agent with a system prompt, ToolRegistry, and dispatch loop."""

    def __init__(
        self,
        registry: ToolRegistry,
        system_prompt: str,
        max_steps: int = 6,
        chat_fn: ChatFn | None = None,
        router: Any = None,
        short_term: ShortTermMemory | None = None,
        long_term: LongTermMemory | None = None,
        injection_detector: InjectionDetector | None = None,
    ):
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self._chat = chat_fn or default_chat
        self._router = router
        self._short_term = short_term
        self._long_term = long_term
        self._injection_detector = injection_detector or InjectionDetector()
        # Cache long-term memory and reinsert it each turn; rebuilding messages[0]
        # for a new prompt patch would otherwise discard the initial memory block.
        self._lt_cache: dict[str, str] = {}

    def run(
        self,
        user_message: str,
        user_id: str | None = None,
        history: list[dict] | None = None,
        exposed_tools: tuple[str, ...] | None = None,
        prompt_patch: str = "",
        memory_block: str = "",
    ) -> RunResult:
        """Run one ReAct loop.

        - exposed_tools: None exposes all tools; () exposes none; a named subset restricts exposure
        - prompt_patch: a nonempty patch rebuilds the system entry as intent may change each turn
        - memory_block: append a nonempty block to the system prompt for cold-start long-term memory
        """
        messages = self._build_initial_messages(
            history, user_message, prompt_patch, memory_block
        )
        tools_arg = self._tools_for_exposure(exposed_tools)
        steps: list[Step] = []

        for _ in range(self.max_steps):
            resp = self._chat(messages=messages, tools=tools_arg)
            msg = resp.choices[0].message
            assistant_dict = msg.model_dump(exclude_none=True)
            messages.append(assistant_dict)

            step = Step(assistant_message=assistant_dict)
            steps.append(step)

            if not msg.tool_calls:
                return RunResult(final=msg.content, messages=messages, steps=steps)

            for tc in msg.tool_calls:
                args = _parse_arguments(tc.function.arguments)
                # Override with session context: the LLM cannot choose whose data to access.
                if user_id is not None:
                    args["user_id"] = user_id
                result = self.registry.dispatch(tc.function.name, args)
                step.invocations.append(
                    ToolInvocation(
                        call_id=tc.id,
                        name=tc.function.name,
                        arguments=args,
                        result=result,
                    )
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result, ensure_ascii=False),
                    }
                )

            # Terminal tool: stop immediately after escalate_to_human without another LLM call.
            if any(inv.name == "escalate_to_human" for inv in step.invocations):
                return RunResult(
                    final="Transferred to a human agent.", messages=messages, steps=steps
                )

        return RunResult(final=None, messages=messages, steps=steps)

    def run_routed(
        self,
        user_message: str,
        user_id: str | None = None,
        history: list[dict] | None = None,
    ) -> RunResult:
        """Run guardrails → memory → routing → fast path or ReAct with a tool subset.

        A guardrail match returns a fixed refusal; fast_path dispatches a zero-argument terminal tool.
        Attach routing metadata and memory events to RunResult.
        """
        from agent.router import POLICY_MAP, IntentRouter

        if self._router is None:
            self._router = IntentRouter()

        memory_events: list[MemoryEvent] = []

        matched = self._injection_detector.detect(user_message)
        if matched is not None:
            memory_events.append(MemoryEvent("guardrail_injection", matched))
            return RunResult(
                final=_INJECTION_REFUSAL,
                messages=[],
                steps=[],
                intent="injection",
                route_path="guardrails",
                confidence=1.0,
                memory_events=memory_events,
            )

        history, long_term_block = self._apply_memory_before_run(
            history=history, user_id=user_id, events=memory_events,
        )

        decision = self._router.classify(user_message)
        policy = POLICY_MAP.get(decision.intent, POLICY_MAP["other"])

        if policy.fast_path_tool is not None:
            result = self._run_fast_path(
                user_message, user_id, history, decision, policy,
                memory_block=long_term_block,
            )
            result.memory_events = memory_events
            return result

        result = self.run(
            user_message=user_message,
            user_id=user_id,
            history=history,
            exposed_tools=policy.exposed_tools,
            prompt_patch=policy.prompt_patch,
            memory_block=long_term_block,
        )
        result.intent = decision.intent
        result.route_path = decision.route_path
        result.confidence = decision.confidence
        result.memory_events = memory_events
        return result

    def save_long_term(
        self, user_id: str, messages: list[dict]
    ) -> list[MemoryEvent]:
        """Called by the REPL on /reset, /user, or /quit; no-op without long-term memory or messages."""
        if self._long_term is None or not messages:
            return []
        events: list[MemoryEvent] = []
        self._long_term.save_session(user_id, messages, events)
        self._lt_cache.pop(user_id, None)
        return events

    # ---------- Internal helpers ----------

    def _apply_memory_before_run(
        self,
        history: list[dict] | None,
        user_id: str | None,
        events: list[MemoryEvent],
    ) -> tuple[list[dict] | None, str]:
        """Load long-term memory on cold start or use the cache, then compact short-term history."""
        long_term_block = ""
        if self._long_term is not None and user_id is not None:
            if history is None:
                long_term_block = self._long_term.load_block(user_id, events)
                self._lt_cache[user_id] = long_term_block
            else:
                long_term_block = self._lt_cache.get(user_id, "")

        if self._short_term is not None and history:
            history = self._short_term.maybe_compact(history, events)

        return history, long_term_block

    def _build_initial_messages(
        self,
        history: list[dict] | None,
        user_message: str,
        prompt_patch: str,
        memory_block: str = "",
    ) -> list[dict]:
        """Build messages: combine base, patch, and memory into system, then append history and user.

        Append memory_block to the system prompt so rebuilding it for a patch preserves memory.
        """
        system_content = self.system_prompt
        if prompt_patch:
            system_content = f"{system_content}\n\n{prompt_patch}"
        if memory_block:
            system_content = f"{system_content}\n\n{memory_block}"

        rebuild_system = bool(prompt_patch or memory_block)

        if history:
            messages = list(history)
            if rebuild_system:
                if messages and messages[0].get("role") == "system":
                    messages[0] = {"role": "system", "content": system_content}
                else:
                    messages.insert(0, {"role": "system", "content": system_content})
        else:
            messages = [{"role": "system", "content": system_content}]
        messages.append({"role": "user", "content": user_message})
        return messages

    def _tools_for_exposure(
        self, exposed_tools: tuple[str, ...] | None
    ) -> list[dict]:
        """Return tool schemas to expose to the LLM.

        agent.llm.chat omits the tools field for an empty list,
        so the model sees no tools for chitchat.
        """
        if exposed_tools is None:
            return self.registry.openai_tools()
        result: list[dict] = []
        for name in exposed_tools:
            spec: ToolSpec | None = self.registry.get(name)
            if spec is not None:
                result.append(spec.to_openai())
        return result

    def _run_fast_path(
        self,
        user_message: str,
        user_id: str | None,
        history: list[dict] | None,
        decision: Any,
        policy: Any,
        memory_block: str = "",
    ) -> RunResult:
        """Dispatch a zero-argument terminal tool without an LLM call.
        Preserve the normal tool_call/tool message pair for history and evaluation.
        """
        messages = self._build_initial_messages(
            history, user_message, "", memory_block
        )
        result = self.registry.dispatch(policy.fast_path_tool, {})

        call_id = f"fastpath_{policy.fast_path_tool}"
        assistant_msg = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": policy.fast_path_tool,
                        "arguments": "{}",
                    },
                }
            ],
        }
        tool_msg = {
            "role": "tool",
            "tool_call_id": call_id,
            "content": json.dumps(result, ensure_ascii=False),
        }
        messages.extend([assistant_msg, tool_msg])

        invocation = ToolInvocation(
            call_id=call_id,
            name=policy.fast_path_tool,
            arguments={},
            result=result,
        )
        step = Step(assistant_message=assistant_msg, invocations=[invocation])

        return RunResult(
            final=policy.final_template or "",
            messages=messages,
            steps=[step],
            intent=decision.intent,
            route_path=decision.route_path,
            confidence=decision.confidence,
        )


def _parse_arguments(raw: str | dict) -> dict:
    """Safely parse tool_call.arguments, usually a JSON string, into a dict."""
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw}
    return parsed if isinstance(parsed, dict) else {"__raw__": parsed}
