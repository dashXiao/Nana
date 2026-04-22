"""qwen-plus tool-use 冒烟测试。

目的：
- 验证 DASHSCOPE_API_KEY 生效、OpenAI 兼容端点可通
- 验证 `ToolSpec` → OpenAI tools schema 的转换能被 qwen-plus 正确理解
- 打印完整 function-calling 往返，供后续 ReAct 循环参考

跑法：uv run python scripts/smoke_llm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm import MODEL, chat  # noqa: E402
from agent.tools._contract import ToolRegistry, ToolSpec, ok  # noqa: E402


# --- 定义一个 dummy 工具：两数相加 ---


class AddInput(BaseModel):
    a: float = Field(description="第一个加数")
    b: float = Field(description="第二个加数")


def add_handler(args: AddInput) -> dict:
    return ok({"sum": args.a + args.b})


ADD_SPEC = ToolSpec(
    name="add_numbers",
    description="对两个数字求和并返回结果。只有当用户明确需要计算时才调用。",
    input_model=AddInput,
    handler=add_handler,
)


def run() -> None:
    registry = ToolRegistry([ADD_SPEC])

    messages: list[dict] = [
        {"role": "system", "content": "你是一个助手。需要计算时请调用可用工具，不要自己心算。"},
        {"role": "user", "content": "请用工具帮我算 13 + 29 等于多少？"},
    ]

    print(f"=== model: {MODEL} ===")
    print(">>> 第 1 轮：发送用户消息，期望模型返回 tool_call")

    resp = chat(messages=messages, tools=registry.openai_tools())
    msg = resp.choices[0].message

    if not msg.tool_calls:
        print("⚠️ 模型没有调用工具，直接返回：", msg.content)
        return

    print(f"  · 模型调用工具：{[tc.function.name for tc in msg.tool_calls]}")
    for tc in msg.tool_calls:
        print(f"    - {tc.function.name}({tc.function.arguments})")

    # 把 assistant 的 tool_call 消息追加回历史（OpenAI 协议要求）
    messages.append(msg.model_dump(exclude_none=True))

    # 执行每个 tool_call 并把结果写回 messages
    for tc in msg.tool_calls:
        result = registry.dispatch(tc.function.name, tc.function.arguments)
        print(f"  · 工具返回：{result}")
        messages.append(
            {
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(result, ensure_ascii=False),
            }
        )

    print("\n>>> 第 2 轮：带着 tool_result 回送，期望模型给出自然语言答复")
    resp2 = chat(messages=messages, tools=registry.openai_tools())
    final = resp2.choices[0].message.content
    print(f"  · 模型最终回复：{final}")

    print("\n✅ smoke test passed.")


if __name__ == "__main__":
    run()
