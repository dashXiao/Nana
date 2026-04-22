"""ReAct 循环 · 最小骨架。

职责：
- 维护 OpenAI chat messages 历史
- 每轮：调 LLM → 若返回 tool_calls 则派发 → 把结果回灌为 tool message → 再调 LLM
  直到模型给出自然语言答复，或达到 `max_steps` 熔断
- 在派发前，把 session 上下文（当前 `user_id`）硬写回 tool_call arguments，
  这样即便 LLM 伪造成别的 user_id，也会被覆盖——工具层鉴权的上游哨卡
- `run()` 接受 exposed_tools / prompt_patch 做工具子集暴露 + 系统提示增强
- `run_routed()` 编排「分类 → 策略 → fast_path 或 ReAct」整条链路

不做的事（留给后续 module）：
- 长/短期记忆（M3·W3）
- Prompt 注入过滤 / 情绪感知（M6/M7·W4）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.llm import chat as default_chat
from agent.tools._contract import ToolRegistry, ToolSpec

ChatFn = Callable[..., Any]


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
    """一次 run() 的终态。"""

    final: str | None           # 模型最终自然语言答复；None 表示达到 max_steps 仍未收敛
    messages: list[dict]        # 完整 message 历史，可作为 history 追加轮次
    steps: list[Step]           # 按轮记录，便于 trace / 评测
    # 路由元数据：未经 run_routed 的调用保持默认值（"other" + "default" + 0.0），
    # 让直接用 run() 的老测试 / 评测 harness 也能安全读取这三个字段。
    intent: str = "other"
    route_path: str = "default"  # "rules" | "llm" | "default"
    confidence: float = 0.0


class Agent:
    """最小 ReAct Agent：system prompt + ToolRegistry + 循环调度。"""

    def __init__(
        self,
        registry: ToolRegistry,
        system_prompt: str,
        max_steps: int = 6,
        chat_fn: ChatFn | None = None,
        router: Any = None,
    ):
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self._chat = chat_fn or default_chat
        # router 懒加载——只有 run_routed 用到；直接 run() 的链路完全不碰它
        self._router = router

    def run(
        self,
        user_message: str,
        user_id: str | None = None,
        history: list[dict] | None = None,
        exposed_tools: tuple[str, ...] | None = None,
        prompt_patch: str = "",
    ) -> RunResult:
        """执行一次 ReAct 循环。

        - `exposed_tools=None` → 暴露全部工具（默认，等同未路由）
        - `exposed_tools=()`   → 零工具（chitchat 场景，强制 LLM 只做自然语言回复）
        - `exposed_tools=("query_orders",)` → 严格子集

        - `prompt_patch` 非空时，会**覆盖**消息序列的 system 条目为
          `self.system_prompt + "\\n\\n" + prompt_patch`。这对多轮路由很关键：
          不同轮可能命中不同意图，patch 必须随轮更新。
          为空时保持旧行为（沿用 history 里的 system），不破坏既有测试。
        """
        messages = self._build_initial_messages(history, user_message, prompt_patch)
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

            # 终止型工具：escalate_to_human 一旦被调，硬切对话，不再让 LLM 继续
            # （当前只有一个终止型工具，按 name 匹配即可；将来多起来再抽 terminates 字段）
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
        """分类 → 策略 → fast_path 或带工具子集的 ReAct。

        路由元数据（intent / route_path / confidence）被回填到 RunResult，
        便于 REPL 打印与 W5 评测分桶统计。
        """
        from agent.router import POLICY_MAP, IntentRouter  # 循环依赖防范

        if self._router is None:
            self._router = IntentRouter()
        decision = self._router.classify(user_message)
        policy = POLICY_MAP.get(decision.intent, POLICY_MAP["other"])

        if policy.fast_path_tool is not None:
            return self._run_fast_path(
                user_message, user_id, history, decision, policy,
            )

        result = self.run(
            user_message=user_message,
            user_id=user_id,
            history=history,
            exposed_tools=policy.exposed_tools,
            prompt_patch=policy.prompt_patch,
        )
        result.intent = decision.intent
        result.route_path = decision.route_path
        result.confidence = decision.confidence
        return result

    # ---------- 内部辅助 ----------

    def _build_initial_messages(
        self,
        history: list[dict] | None,
        user_message: str,
        prompt_patch: str,
    ) -> list[dict]:
        """构造本轮对话 messages 序列。

        - 无 history：新建 system（含 patch）+ user
        - 有 history + 无 patch：沿用 history 原 system（兼容既有测试）
        - 有 history + 有 patch：**替换** messages[0] 为新 system，使 patch 逐轮更新
        """
        system_content = self.system_prompt
        if prompt_patch:
            system_content = f"{system_content}\n\n{prompt_patch}"

        if history:
            messages = list(history)
            if prompt_patch:
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

        注意 `agent.llm.chat` 里 `if tools:` 会把空 list 判定为 falsy → 不传 tools 字段，
        恰好实现 chitchat 意图"LLM 看不到任何工具 → 只能自然语言回复"的效果。
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
    ) -> RunResult:
        """绕过 LLM 直接派发零参数终止工具（当前只有 escalate_to_human）。

        仍构造一条完整的 assistant tool_call + tool result message 序列，
        让 messages 形态与正常 ReAct 一致——下游 history / 评测 harness 不需要特判。
        """
        messages = self._build_initial_messages(history, user_message, "")
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
