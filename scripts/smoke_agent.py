"""Agent 端到端冒烟：ReAct 循环 + 已实现的 3 个工具联动。

覆盖的 scenario：
- S1 · 查订单：LLM 应调 query_orders
- S2 · 查政策：LLM 应调 query_policy，不得凭记忆答
- S3 · 多轮退款（带确认）：turn 1 先确认意图（通常会先 query_orders），
  turn 2 用户确认后才 initiate_refund；最后从 DB 读 O005 验证确实被改成 refunded

每次跑前都 reseed，避免上一次跑留下的 refunded 状态干扰。
跑法：uv run python scripts/smoke_agent.py
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
from db import db_session  # noqa: E402
from scripts.seed import seed  # noqa: E402

SYSTEM_PROMPT = """\
你是 NanaShop 的电商客服助手。可用工具：
- query_orders：查询当前用户全部订单
- query_policy：查询全站售后 / 配送 / 退款政策（无参）
- initiate_refund：对指定订单发起退款（会真改 DB！慎用）

必须遵守：
1. 用户询问自己订单时必须调 query_orders，不要臆造订单号或状态。
2. 涉及政策（几天内可退、发货时效、退款到账等）必须调 query_policy，不得凭记忆。
3. 发起退款前必须：
   (a) 先通过 query_orders 确认订单存在且属于当前用户；
   (b) 在对话里明确问用户"您确定要对订单 XXX 发起退款吗？"；
   (c) 用户明确确认后才调 initiate_refund。
4. 工具结果拿到后用简洁中文回复。
"""


def _fmt_result(data: dict) -> str:
    """工具返回值的压缩显示，避免终端刷屏。"""
    if "count" in data:
        ids = [o["order_id"] for o in data.get("orders", [])]
        return f"count={data['count']} ids={ids}"
    if "content" in data:
        return f"content({len(data['content'])} chars)"
    return json.dumps(data, ensure_ascii=False)


def _print_run(result: RunResult) -> None:
    for i, step in enumerate(result.steps, 1):
        if step.invocations:
            for inv in step.invocations:
                args = json.dumps(inv.arguments, ensure_ascii=False)
                if inv.result.get("ok"):
                    print(f"  [step {i}] TOOL {inv.name}({args}) → {_fmt_result(inv.result['data'])}")
                else:
                    print(
                        f"  [step {i}] TOOL {inv.name}({args}) → "
                        f"ERR {inv.result['error_code']}: {inv.result['message']}"
                    )
        else:
            print(f"  [step {i}] ASST: {step.assistant_message.get('content')}")
    print(f"  >>> final: {result.final}\n")


def _status(order_id: str) -> str:
    with db_session() as conn:
        row = conn.execute(
            "SELECT status FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
    return row["status"] if row else "(missing)"


def run() -> None:
    print(f"=== model: {model_name()} ===\n")
    seed()
    print()

    registry = ToolRegistry(
        [QUERY_ORDERS_SPEC, INITIATE_REFUND_SPEC, QUERY_POLICY_SPEC]
    )
    agent = Agent(registry=registry, system_prompt=SYSTEM_PROMPT, max_steps=6)
    user_id = "U001"

    # ============ S1 · 查订单 ============
    print("--- S1 · 查订单 ---")
    msg1 = "我最近下的几单都到哪儿了？"
    print(f"USER: {msg1}")
    r1 = agent.run(msg1, user_id=user_id)
    _print_run(r1)
    s1_tools = [inv.name for step in r1.steps for inv in step.invocations]
    assert "query_orders" in s1_tools, f"S1 应该调 query_orders，实际调了：{s1_tools}"
    print("  ✓ S1 PASS: 调用了 query_orders\n")

    # ============ S2 · 查政策 ============
    print("--- S2 · 查政策 ---")
    msg2 = "签收后几天可以无理由退货？运费谁承担？"
    print(f"USER: {msg2}")
    r2 = agent.run(msg2, user_id=user_id)
    _print_run(r2)
    s2_tools = [inv.name for step in r2.steps for inv in step.invocations]
    assert "query_policy" in s2_tools, f"S2 应该调 query_policy，实际调了：{s2_tools}"
    print("  ✓ S2 PASS: 调用了 query_policy\n")

    # ============ S3 · 多轮退款（带确认） ============
    # 用 O001 —— U001 的 pending 订单（2 小时前刚下、尚未发货），
    # 取消未发货订单无政策争议，能干净地测 prompt 的"先确认再执行"约束
    print("--- S3 · 多轮退款（turn 1 应仅确认，不得直接 refund） ---")
    order_id = "O001"
    before = _status(order_id)
    print(f"  ({order_id} before: {before})")

    msg3a = f"{order_id} 那张单我不想要了，帮我退掉"
    print(f"USER: {msg3a}")
    r3a = agent.run(msg3a, user_id=user_id)
    _print_run(r3a)
    s3a_tools = [inv.name for step in r3a.steps for inv in step.invocations]
    assert "initiate_refund" not in s3a_tools, (
        f"S3 turn 1 未经确认就调了 initiate_refund，违反 prompt 规则！工具序列：{s3a_tools}"
    )
    assert _status(order_id) == before, "S3 turn 1 结束后 DB 不应被改动"
    print("  ✓ S3 turn 1 PASS: 未擅自退款\n")

    msg3b = "确认，退这单"
    print(f"USER: {msg3b}")
    r3b = agent.run(msg3b, user_id=user_id, history=r3a.messages)
    _print_run(r3b)
    s3b_tools = [inv.name for step in r3b.steps for inv in step.invocations]
    assert "initiate_refund" in s3b_tools, (
        f"S3 turn 2 用户已确认，应调 initiate_refund，实际调了：{s3b_tools}"
    )
    after = _status(order_id)
    print(f"  ({order_id} after: {after})")
    assert after == "refunded", f"S3 退款应生效，{order_id} 实际状态：{after}"
    print("  ✓ S3 turn 2 PASS: DB 状态已改为 refunded\n")

    print("=== all scenarios passed. ===")


if __name__ == "__main__":
    run()
