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

# 某些 IDE 终端（PyCharm / VSCode 的默认配置）LANG 不是 utf-8，
# input() 会用 surrogateescape 把中文读成 \udcXX 孤立代理，
# 后续拼 JSON 发给 LLM 时就崩 UnicodeEncodeError。这里强制 UTF-8。
for _stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core import Agent, RunResult  # noqa: E402
from agent.llm import model_name  # noqa: E402
from agent.router import BASE_SYSTEM_PROMPT, POLICY_MAP  # noqa: E402
from agent.tools._contract import ToolRegistry  # noqa: E402
from agent.tools.escalate_to_human import ESCALATE_TO_HUMAN_SPEC  # noqa: E402
from agent.tools.initiate_refund import INITIATE_REFUND_SPEC  # noqa: E402
from agent.tools.query_orders import QUERY_ORDERS_SPEC  # noqa: E402
from agent.tools.query_policy import QUERY_POLICY_SPEC  # noqa: E402
from agent.tools.query_product import QUERY_PRODUCT_SPEC  # noqa: E402
from scripts.seed import seed  # noqa: E402

# REPL 走 run_routed，base 只含身份 + 格式两行；
# 连商品 catalog 都下沉到 product_inquiry 的 patch（只有该意图需要名称→ID 映射）。
# 各意图的行为规则与必要资料由 POLICY_MAP[intent].prompt_patch 在每轮注入到 system 末尾。
# 这样 LLM 每轮只看到"与当前意图相关的规则 + 必要资料"，patch 真实承担路由价值。
SYSTEM_PROMPT = BASE_SYSTEM_PROMPT

TOOLS = [
    QUERY_ORDERS_SPEC,
    QUERY_POLICY_SPEC,
    QUERY_PRODUCT_SPEC,
    INITIATE_REFUND_SPEC,
    ESCALATE_TO_HUMAN_SPEC,
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
    """打印本轮路由决策 + LLM 真正看到的 system + 工具子集 + 工具调用。

    每轮都把 messages[0] 的 system 完整打出来，肉眼确认：
    - 意图命中后 prompt_patch 是否真的被追加到 system 末尾
    - 不同意图下 system 是否随之变化（不是一直用兜底）
    """
    print(
        f"  [route] {result.route_path}:{result.intent} "
        f"(conf={result.confidence:.2f})"
    )

    policy = POLICY_MAP.get(result.intent)

    # 工具子集：肉眼确认暴露给 LLM 的工具数量真的被裁了
    if policy is None:
        tools_display = "(unknown intent)"
    elif policy.exposed_tools is None:
        tools_display = "ALL（兜底：暴露全部 5 工具）"
    elif len(policy.exposed_tools) == 0:
        tools_display = "() （零工具，LLM 只能自然语言回复）"
    else:
        tools_display = ", ".join(policy.exposed_tools)
    print(f"  [exposed_tools] {tools_display}")

    # fast_path 绕 LLM，没有 system 调用可打；其他路径打完整 system
    if policy is not None and policy.fast_path_tool is not None:
        print(f"  [fast_path] {policy.fast_path_tool} → 绕过 LLM，无 chat.completions 调用")
    elif result.messages and result.messages[0].get("role") == "system":
        sys_content = result.messages[0]["content"]
        print("  ┌─ [system prompt 本轮实际送给 LLM 的完整内容] " + "─" * 20)
        for line in sys_content.splitlines():
            print(f"  │ {line}")
        print("  └─" + "─" * 68)

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

        # --- 正常对话（走意图路由） ---
        try:
            result = agent.run_routed(line, user_id=user_id, history=history)
        except Exception as exc:  # noqa: BLE001 — REPL 里兜底保活
            print(f"  [error] {type(exc).__name__}: {exc}\n")
            continue

        _print_trace(result)

        final = result.final or "(未收敛：达到 max_steps，请 /reset 后换种问法)"
        print(f"\n[agent] {final}\n")

        # 转人工即终态：下一轮不能继续挂在已结束的对话上，强制清空历史
        if result.final == "已转接人工":
            history = None
            print("(会话已转人工，历史已清空；输入新问题可开启下一轮)\n")
            continue

        # 把本轮结果作为下一轮的 history，维持多轮上下文
        history = result.messages


if __name__ == "__main__":
    run()
