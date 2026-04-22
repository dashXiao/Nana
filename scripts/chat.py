"""Nana Agent 交互式 REPL（人工测试用）。

跑法：uv run python scripts/chat.py
命令：
  /quit        退出
  /reset       清空本轮会话历史（重开一个会话，但不 reseed DB）
  /user U002   切到 U002（同时清空历史，避免跨用户串话）
  /tools       列出当前注册的工具
  /help        显示本帮助

启动时会 seed 一次，保证 DB 是确定的初始状态。
每次 LLM 调工具都会打一行 `[tool] name(args) → 摘要`，方便你肉眼审查路径。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core import Agent, RunResult  # noqa: E402
from agent.llm import model_name  # noqa: E402
from agent.tools._contract import ToolRegistry  # noqa: E402
from agent.tools.initiate_refund import INITIATE_REFUND_SPEC  # noqa: E402
from agent.tools.query_orders import QUERY_ORDERS_SPEC  # noqa: E402
from agent.tools.query_policy import QUERY_POLICY_SPEC  # noqa: E402
from agent.tools.query_product import QUERY_PRODUCT_SPEC  # noqa: E402
from scripts.seed import seed  # noqa: E402

SYSTEM_PROMPT = """\
你是 NanaShop 的电商客服助手。

全站在售商品（固定 3 款）：
- P001：无线蓝牙耳机 X1
- P002：智能手表 Pro
- P003：人体工学办公椅
用户用名称或俗称提问时，对照上表映射到对应 product_id 再调工具。

可用工具：
- query_orders：查询当前用户全部订单
- query_policy：查询全站售后 / 配送 / 退款政策（无参）
- query_product：按 product_ids 批量查商品详情（价格、库存、介绍）
- initiate_refund：对指定订单发起退款（会真改 DB）

必须遵守：
1. 用户询问自己订单时必须调 query_orders，不要臆造订单号或状态。
2. 涉及价格 / 库存 / 商品参数，必须调 query_product，不得凭记忆。
3. 涉及政策（几天内可退、发货时效、退款到账），必须调 query_policy。
4. 发起退款前必须：
   (a) 先通过 query_orders 确认订单存在且属于当前用户；
   (b) 在对话里明确问用户"您确定要对订单 XXX 发起退款吗？"；
   (c) 用户明确确认后才调 initiate_refund。
5. 工具结果拿到后用简洁中文回复。
"""

TOOLS = [
    QUERY_ORDERS_SPEC,
    QUERY_POLICY_SPEC,
    QUERY_PRODUCT_SPEC,
    INITIATE_REFUND_SPEC,
]


def _tool_summary(inv_result: dict) -> str:
    """把工具返回压成一行便于终端阅读。"""
    if not inv_result.get("ok"):
        return f"ERR {inv_result.get('error_code')}: {inv_result.get('message')}"
    data = inv_result["data"]
    if "count" in data:
        ids = [o["order_id"] for o in data.get("orders", [])]
        return f"count={data['count']} ids={ids}"
    if "content" in data:
        return f"content({len(data['content'])} chars)"
    if "products" in data:
        ids = [p["product_id"] for p in data["products"]]
        missing = data.get("missing", [])
        return f"products={ids} missing={missing}"
    return json.dumps(data, ensure_ascii=False)


def _print_trace(result: RunResult) -> None:
    """打印本轮 ReAct 每一步的工具调用（final 文本由主循环单独打）。"""
    for step in result.steps:
        for inv in step.invocations:
            args = json.dumps(inv.arguments, ensure_ascii=False)
            print(f"  [tool] {inv.name}({args}) → {_tool_summary(inv.result)}")


def _print_help() -> None:
    print(
        "可用命令：\n"
        "  /quit        退出\n"
        "  /reset       清空本轮会话历史\n"
        "  /user U00X   切到指定用户（同时清空历史）\n"
        "  /tools       列出注册的工具\n"
        "  /help        显示本帮助\n"
    )


def _print_tools(registry: ToolRegistry) -> None:
    for t in registry.openai_tools():
        fn = t["function"]
        print(f"  · {fn['name']} — {fn['description'][:60]}...")


def run() -> None:
    print(f"=== NanaShop Agent REPL · model: {model_name()} ===")
    print("启动时先 seed 一次，确保初始状态干净...")
    seed()
    print()

    registry = ToolRegistry(TOOLS)
    agent = Agent(registry=registry, system_prompt=SYSTEM_PROMPT, max_steps=6)

    user_id = "U001"
    history: list[dict] | None = None

    print(f"当前用户：{user_id}（张三）。输入 /help 看命令，Ctrl+C 或 /quit 退出。\n")

    while True:
        try:
            line = input(f"[{user_id}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n再见。")
            return

        if not line:
            continue

        # --- 斜杠命令 ---
        if line == "/quit" or line == "/exit":
            print("再见。")
            return
        if line == "/help":
            _print_help()
            continue
        if line == "/reset":
            history = None
            print("(会话历史已清空)\n")
            continue
        if line == "/tools":
            _print_tools(registry)
            continue
        if line.startswith("/user "):
            new_uid = line.split(None, 1)[1].strip()
            if not new_uid:
                print("用法：/user U001")
                continue
            user_id = new_uid
            history = None
            print(f"(切到 {user_id}，历史已清空)\n")
            continue
        if line.startswith("/"):
            print(f"未知命令：{line}（试试 /help）")
            continue

        # --- 正常对话 ---
        try:
            result = agent.run(line, user_id=user_id, history=history)
        except Exception as exc:  # noqa: BLE001 — REPL 里兜底保活
            print(f"  [error] {type(exc).__name__}: {exc}\n")
            continue

        _print_trace(result)

        final = result.final or "(未收敛：达到 max_steps，请 /reset 后换种问法)"
        print(f"\n[agent] {final}\n")

        # 把本轮结果作为下一轮的 history，维持多轮上下文
        history = result.messages


if __name__ == "__main__":
    run()
