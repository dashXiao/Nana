"""Agent 端到端冒烟测试：qwen-plus + query_orders + ReAct 循环。

目的：
- 验证 `agent.core.Agent` 能把真实 LLM 响应正确串起来
- 用户用自然语言发问（不提订单号），Agent 应调 `query_orders` 再给人类可读的回复
- 打印每轮的 tool_call 和最终答复，便于肉眼审阅

跑法：uv run python scripts/smoke_agent.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core import Agent  # noqa: E402
from agent.llm import model_name  # noqa: E402
from agent.tools._contract import ToolRegistry  # noqa: E402
from agent.tools.query_orders import QUERY_ORDERS_SPEC  # noqa: E402

SYSTEM_PROMPT = (
    "你是 NanaShop 的电商客服助手。用户询问自己的订单时，必须调用 query_orders 工具查询，"
    "不要臆造订单号或状态。拿到工具结果后，用简洁中文回复用户。"
)


def run() -> None:
    registry = ToolRegistry([QUERY_ORDERS_SPEC])
    agent = Agent(registry=registry, system_prompt=SYSTEM_PROMPT, max_steps=4)

    print(f"=== model: {model_name()} ===")
    user_id = "U001"
    user_msg = "我最近下的几单都到哪儿了？"
    print(f">>> user_id={user_id}  user: {user_msg}\n")

    result = agent.run(user_msg, user_id=user_id)

    for i, step in enumerate(result.steps, 1):
        print(f"--- step {i} ---")
        tool_calls = step.assistant_message.get("tool_calls")
        if tool_calls:
            for inv in step.invocations:
                print(f"  · tool_call: {inv.name}({json.dumps(inv.arguments, ensure_ascii=False)})")
                ok_flag = inv.result.get("ok")
                if ok_flag:
                    data = inv.result["data"]
                    preview = f"count={data.get('count')}" if "count" in data else "ok"
                    print(f"    → {preview}")
                else:
                    print(f"    → ERR {inv.result.get('error_code')}: {inv.result.get('message')}")
        else:
            print(f"  · assistant: {step.assistant_message.get('content')}")

    print("\n=== final ===")
    print(result.final if result.final else "(未收敛：达到 max_steps)")


if __name__ == "__main__":
    run()
