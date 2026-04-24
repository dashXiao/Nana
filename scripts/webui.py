"""Nana Agent Gradio Web UI（主流 AI 问答界面风格：消息气泡 + 输入框 + 用户切换）。

跑法：uv run python scripts/webui.py
默认监听 http://127.0.0.1:7860 ；`--share` 开公网隧道（gradio 托管）。

设计要点：
- **UI 展示 vs Agent history 分两条**：gr.Chatbot 只接收 user/assistant 两类气泡，
  agent 内部 history 含 system / tool_call / tool 三类，分开维护避免把工具流水泄给前端
- **gr.State 按会话隔离**：每个浏览器 tab 一份状态，不会串台（取代 REPL 的全局变量）
- **新对话 / 切换用户**：与 REPL 语义一致——把当前 history 写回长期记忆再清空
- **启动 seed 一次**：demo 从确定初始态开；持久化到 `db/nana.db` + `memory/sessions/`
- **调试轨迹开关**：侧栏勾上「显示调试轨迹」后，聊天框下方挂 per-turn markdown 面板，
  展示路由 / 暴露工具 / 记忆事件 / system / 工具调用；与 scripts/chat.py 打的 trace 同构
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Callable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core import Agent, RunResult  # noqa: E402
from agent.memory.long_term import LongTermMemory  # noqa: E402
from agent.memory.short_term import ShortTermMemory  # noqa: E402
from agent.router import BASE_SYSTEM_PROMPT, POLICY_MAP  # noqa: E402
from agent.tools._contract import ToolRegistry  # noqa: E402
from agent.tools.escalate_to_human import ESCALATE_TO_HUMAN_SPEC  # noqa: E402
from agent.tools.initiate_refund import INITIATE_REFUND_SPEC  # noqa: E402
from agent.tools.query_orders import QUERY_ORDERS_SPEC  # noqa: E402
from agent.tools.query_policy import QUERY_POLICY_SPEC  # noqa: E402
from agent.tools.query_product import QUERY_PRODUCT_SPEC  # noqa: E402
from scripts.seed import seed  # noqa: E402


USERS = [
    ("U001（张三）", "U001"),
    ("U002（李四）", "U002"),
]
_USER_VALUE_TO_ID = dict(USERS)
_DEFAULT_USER_LABEL = USERS[0][0]


def build_agent() -> Agent:
    """与 REPL 同构的 agent 装配。"""
    registry = ToolRegistry([
        QUERY_ORDERS_SPEC,
        QUERY_POLICY_SPEC,
        QUERY_PRODUCT_SPEC,
        INITIATE_REFUND_SPEC,
        ESCALATE_TO_HUMAN_SPEC,
    ])
    return Agent(
        registry=registry,
        system_prompt=BASE_SYSTEM_PROMPT,
        max_steps=6,
        short_term=ShortTermMemory(max_rounds=4, keep_recent=1),
        long_term=LongTermMemory(),
    )


# ---------- 纯函数：事件处理（单测全部在这里） ----------


FALLBACK_REPLY = "抱歉，未收敛到答复（可能达到最大步数），请换个说法重试。"
EMPTY_TRACE_MD = "_(尚无调用轨迹)_"


def _tool_summary(inv_result: dict) -> str:
    """把工具返回压成一行便于阅读（与 scripts/chat.py 同构）。"""
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


def format_turn_trace(result: RunResult) -> str:
    """把一轮 RunResult 渲染成 markdown，供调试面板展示。

    与 `scripts/chat.py::_print_trace` 同源，顺序：route → exposed_tools →
    system / fast_path → 工具调用。记忆事件不单列——长期记忆已经嵌在 system
    prompt 里，单独一栏会重复。system 走 blockquote（避免视觉上突兀的代码块）。
    """
    lines: list[str] = [
        f"- **[route]** {result.route_path}:{result.intent} "
        f"(conf={result.confidence:.2f})"
    ]

    policy = POLICY_MAP.get(result.intent)
    if policy is None:
        tools_display = "(unknown intent)"
    elif policy.exposed_tools is None:
        tools_display = "ALL（兜底：全 5 工具）"
    elif len(policy.exposed_tools) == 0:
        tools_display = "()（零工具，只能自然语言回复）"
    else:
        tools_display = ", ".join(policy.exposed_tools)
    lines.append(f"- **[exposed_tools]** {tools_display}")

    if policy is not None and policy.fast_path_tool is not None:
        lines.append(
            f"- **[fast_path]** {policy.fast_path_tool} → 绕过 LLM，无 chat.completions 调用"
        )
    elif result.messages and result.messages[0].get("role") == "system":
        sys_content = result.messages[0]["content"]
        lines.append("- **[system prompt]** 本轮实际送给 LLM 的完整内容：")
        # 用 code fence **阻止 markdown 解析**——否则 system prompt 里内嵌的
        # `## 2026-04-23` 会被渲成 h2 巨字，空行会把 blockquote 切断造成远距
        # 间隙。视觉上的"代码块外观"由 CSS 压平（见 custom_css）。
        lines.append("```")
        lines.append(sys_content)
        lines.append("```")

    for step in result.steps:
        for inv in step.invocations:
            args = json.dumps(inv.arguments, ensure_ascii=False)
            lines.append(
                f"- **[tool]** `{inv.name}({args})` → `{_tool_summary(inv.result)}`"
            )

    return "\n".join(lines)


def format_error_trace(exc: Exception) -> str:
    """agent 抛异常时降级展示的 trace 片段。"""
    return f"- **[error]** `{type(exc).__name__}: {exc}`"


def render_trace_log(log: list[str]) -> str:
    """把累计的 per-turn trace 渲染成一整块 markdown（带 Turn N 分节）。"""
    if not log:
        return EMPTY_TRACE_MD
    sections = [f"### Turn {i}\n\n{t}" for i, t in enumerate(log, 1)]
    return "\n\n---\n\n".join(sections)


def respond(
    user_msg: str,
    chat_display: list[dict] | None,
    session_state: dict | None,
    user_label: str,
    agent: Agent,
) -> tuple[str, list[dict], dict, str]:
    """处理一条用户消息：更新 UI 气泡 + agent history state + 本轮 trace 文本。

    返回 `(清空输入框, 新 chat_display, 新 state, 本轮 trace_text)`。
    trace_text 供 UI 层拼到调试面板；空消息时返回空串，调试面板不追加。
    """
    chat_display = chat_display or []
    state = dict(session_state or {})
    user_id = _USER_VALUE_TO_ID[user_label]

    text = (user_msg or "").strip()
    if not text:
        return "", chat_display, state, ""

    prev_history = state.get("history")
    try:
        result = agent.run_routed(text, user_id=user_id, history=prev_history)
        final = result.final or FALLBACK_REPLY
        new_agent_history = result.messages
        trace_text = format_turn_trace(result)
    except Exception as exc:  # noqa: BLE001 — UI 边界兜底，agent 异常不能崩页
        final = f"[内部异常] {type(exc).__name__}: {exc}"
        new_agent_history = prev_history
        trace_text = format_error_trace(exc)

    new_display = chat_display + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": final},
    ]
    state["history"] = new_agent_history
    state["user_id"] = user_id
    return "", new_display, state, trace_text


def new_conversation(
    session_state: dict | None,
    agent: Agent,
) -> tuple[list[dict], dict]:
    """新对话：把当前 session history 写回长期记忆，清空 UI 与 state。"""
    state = dict(session_state or {})
    uid = state.get("user_id")
    hist = state.get("history")
    if uid and hist:
        agent.save_long_term(uid, hist)
    state["history"] = None
    # user_id 保留：user 没切换
    return [], state


def switch_user(
    new_label: str,
    session_state: dict | None,
    agent: Agent,
) -> tuple[list[dict], dict]:
    """切换用户：同新对话 + 换 user_id。"""
    state = dict(session_state or {})
    old_uid = state.get("user_id")
    hist = state.get("history")
    if old_uid and hist:
        agent.save_long_term(old_uid, hist)
    state["history"] = None
    state["user_id"] = _USER_VALUE_TO_ID[new_label]
    return [], state


# ---------- Gradio 组装 ----------


def build_demo(agent: Agent):
    import gradio as gr

    def _on_msg(user_msg, chat_display, session_state, user_label, trace_log):
        out, display, state, trace_text = respond(
            user_msg, chat_display, session_state, user_label, agent=agent,
        )
        new_log = list(trace_log or [])
        if trace_text:
            new_log.append(trace_text)
        return out, display, state, new_log, render_trace_log(new_log)

    def _on_new_conv(session_state):
        display, state = new_conversation(session_state, agent=agent)
        return display, state, [], render_trace_log([])

    def _on_user_change(new_label, session_state):
        display, state = switch_user(new_label, session_state, agent=agent)
        return display, state, [], render_trace_log([])

    with gr.Blocks(title="NanaShop 客服") as demo:
        gr.Markdown("## NanaShop 客服\n电商客服多轮对话 Agent")

        session_state = gr.State(value={"history": None, "user_id": "U001"})
        trace_state = gr.State(value=[])

        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### 会话设置")
                user_picker = gr.Radio(
                    choices=[label for label, _ in USERS],
                    value=_DEFAULT_USER_LABEL,
                    label="当前用户",
                )
                new_btn = gr.Button("新对话", variant="secondary")
                show_debug = gr.Checkbox(
                    label="显示调试轨迹",
                    value=False,
                )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    show_label=False,
                    height=560,
                )
                msg_box = gr.Textbox(
                    placeholder="问点什么？例如：我要退 O001、你们耳机多少钱、转人工……",
                    show_label=False,
                    submit_btn=True,
                    autofocus=True,
                )

            with gr.Column(scale=2, visible=False, elem_id="trace-col") as trace_col:
                gr.Markdown("### 调试轨迹（每轮累计）")
                trace_panel = gr.Markdown(
                    value=EMPTY_TRACE_MD,
                    elem_id="trace-panel",
                )

        msg_box.submit(
            _on_msg,
            inputs=[msg_box, chatbot, session_state, user_picker, trace_state],
            outputs=[msg_box, chatbot, session_state, trace_state, trace_panel],
        )
        new_btn.click(
            _on_new_conv,
            inputs=[session_state],
            outputs=[chatbot, session_state, trace_state, trace_panel],
        )
        user_picker.change(
            _on_user_change,
            inputs=[user_picker, session_state],
            outputs=[chatbot, session_state, trace_state, trace_panel],
        )
        show_debug.change(
            lambda on: gr.update(visible=bool(on)),
            inputs=show_debug,
            outputs=trace_col,
        )

    return demo


# ---------- CLI ----------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="NanaShop Agent Web UI")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="开 gradio 公网隧道")
    args = parser.parse_args(argv)

    print("启动前 seed 一次，DB 回到初始态…")
    seed()

    import gradio as gr

    agent = build_agent()
    demo = build_demo(agent)
    theme = gr.themes.Soft(
        primary_hue="slate",
        radius_size="lg",
    )
    # Gradio 6 默认底栏有 "通过 API 使用 / 使用 Gradio 构建 / 设置" 三项——
    # 全都挂在 <footer> 里，一条 CSS 整体隐藏。
    # trace 面板走柔和底色 + 小字号；代码 / 引用不再额外加视觉框，保持自然。
    custom_css = """
footer { display: none !important; }

#trace-col {
    background: #f7f7f8;
    border: 1px solid rgba(0, 0, 0, 0.06);
    border-radius: 10px;
    padding: 12px 16px;
}
#trace-panel {
    font-size: 0.85rem;
    line-height: 1.55;
    color: #374151;
    max-height: 560px;
    overflow-y: auto;
}
#trace-panel code {
    background: transparent;
    padding: 0;
    font-size: inherit;
}
/* system prompt 走 code fence（防 markdown 解析），CSS 把"代码块"外观压平：
   无背景、继承字体与字号、保留换行空白、仿 blockquote 左描边。 */
#trace-panel pre {
    background: transparent;
    border: none;
    border-left: 3px solid rgba(0, 0, 0, 0.12);
    padding: 2px 10px;
    margin: 6px 0;
    color: #4b5563;
    white-space: pre-wrap;
    font-size: inherit;
    font-family: inherit;
}
#trace-panel pre code {
    background: transparent;
    padding: 0;
    font-size: inherit;
    font-family: inherit;
}
#trace-panel h3 { font-size: 0.9rem; margin-top: 14px; }
#trace-panel hr { margin: 12px 0; opacity: 0.4; }
""".strip()

    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=theme,
        css=custom_css,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
