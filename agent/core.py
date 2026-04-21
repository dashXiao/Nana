"""ReAct 循环 · 最小骨架。

职责：
- 维护 OpenAI chat messages 历史
- 每轮：调 LLM → 若返回 tool_calls 则派发 → 把结果回灌为 tool message → 再调 LLM
  直到模型给出自然语言答复，或达到 `max_steps` 熔断
- 在派发前，把 session 上下文（当前 `user_id`）硬写回 tool_call arguments，
  这样即便 LLM 伪造成别的 user_id，也会被覆盖——工具层鉴权的上游哨卡

不做的事（留给后续 module）：
- 长/短期记忆（M3·W2）
- 意图路由（M5·W3）
- Prompt 注入过滤 / 情绪感知（M6/M7·W4）
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable

from agent.llm import chat as default_chat
from agent.tools._contract import ToolRegistry

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


class Agent:
    """最小 ReAct Agent：system prompt + ToolRegistry + 循环调度。"""

    def __init__(
        self,
        registry: ToolRegistry,
        system_prompt: str,
        max_steps: int = 6,
        chat_fn: ChatFn | None = None,
    ):
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps
        self._chat = chat_fn or default_chat

    def run(
        self,
        user_message: str,
        user_id: str | None = None,
        history: list[dict] | None = None,
    ) -> RunResult:
        messages: list[dict] = [{"role": "system", "content": self.system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        steps: list[Step] = []

        for _ in range(self.max_steps):
            resp = self._chat(messages=messages, tools=self.registry.openai_tools())
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

        return RunResult(final=None, messages=messages, steps=steps)


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
