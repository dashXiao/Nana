"""ReAct 循环 · 最小骨架。"""

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
    "抱歉，我只能帮您处理 NanaShop 的订单、商品、政策等客服问题，"
    "这类请求我无法处理。"
)


@dataclass
class ToolInvocation:
    """一次工具调用的完整快照，便于评测阶段比对预期行为。"""

    call_id: str
    name: str
    arguments: dict
    result: dict


@dataclass
class Step:
    """ReAct 一轮：assistant 输出 + 本轮派发的所有工具调用。"""

    assistant_message: dict
    invocations: list[ToolInvocation] = field(default_factory=list)


@dataclass
class RunResult:
    """一次 run() 的终态；final=None 表示达到 max_steps 仍未收敛。"""

    final: str | None
    messages: list[dict]
    steps: list[Step]
    intent: str = "other"
    route_path: str = "default"
    confidence: float = 0.0
    memory_events: list[MemoryEvent] = field(default_factory=list)


class Agent:
    """最小 ReAct Agent：system prompt + ToolRegistry + 循环调度。"""

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
        # 长期块会话内缓存：每轮重注入；否则 patch 非空时 _build_initial_messages
        # 替换 messages[0] 会吞掉首轮嵌入的块。
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
        """执行一次 ReAct 循环。

        - exposed_tools: None → 全工具；() → 零工具（chitchat）；具名子集 → 严格暴露
        - prompt_patch: 非空则覆盖 system 条目（多轮路由各轮意图可能不同）
        - memory_block: 非空则作为第三段追加到 system（冷启动长期记忆）
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
                # session 上下文硬覆盖：LLM 无权决定"查谁的"
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

            # 终止型工具：escalate_to_human 命中即硬切，不再让 LLM 继续
            if any(inv.name == "escalate_to_human" for inv in step.invocations):
                return RunResult(
                    final="已转接人工", messages=messages, steps=steps
                )

        return RunResult(final=None, messages=messages, steps=steps)

    def run_routed(
        self,
        user_message: str,
        user_id: str | None = None,
        history: list[dict] | None = None,
    ) -> RunResult:
        """guardrails → 记忆层 → 路由 → fast_path 或带工具子集的 ReAct。

        guardrails 命中即固定拒绝短路；fast_path 直派零参终止工具（当前仅 escalate）。
        路由元数据与记忆事件回填进 RunResult。
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
        """REPL 在 /reset、/user、/quit 时调用；无 long_term / 空消息则 no-op。"""
        if self._long_term is None or not messages:
            return []
        events: list[MemoryEvent] = []
        self._long_term.save_session(user_id, messages, events)
        self._lt_cache.pop(user_id, None)
        return events

    # ---------- 内部辅助 ----------

    def _apply_memory_before_run(
        self,
        history: list[dict] | None,
        user_id: str | None,
        events: list[MemoryEvent],
    ) -> tuple[list[dict] | None, str]:
        """长期 load（冷启动）/ 取缓存（续轮）→ 短期 compact → 返回给 ReAct。"""
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
        """构造本轮 messages：base + patch + memory_block 合成 system，再续 history + user。

        memory_block 作为 system 后缀，避免 patch 非空替换 messages[0] 时把长期记忆覆盖掉。
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
        """根据 exposed_tools 返回要发给 LLM 的 schema 列表。

        agent.llm.chat 里 `if tools:` 会把空 list 判定为 falsy → 不传 tools 字段，
        恰好实现 chitchat "LLM 看不到任何工具" 的效果。
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
        """绕过 LLM 直接派发零参数终止工具，但仍构造完整的 tool_call/tool 消息对，
        让 messages 形态与正常 ReAct 一致——下游 history/评测不需要特判。
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
    """把 tool_call.arguments（通常是 JSON 字符串）安全地解析为 dict。"""
    if isinstance(raw, dict):
        return dict(raw)
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"__raw__": raw}
    return parsed if isinstance(parsed, dict) else {"__raw__": parsed}
