"""Gradio Web UI for Nana Agent with chat bubbles, input, and user switching.

Run: uv run python scripts/webui.py
Listens on http://127.0.0.1:7860; --share enables a Gradio-hosted public tunnel.

Design:
- Separate user-facing bubbles from internal system, tool-call, and tool history.
- gr.State isolates state per browser session.
- New chats and user switches save long-term memory before clearing history.
- Seed once at startup; persist to db/nana.db and memory/sessions/.
- An optional trace panel shows routing, exposed tools, prompts, and tool calls.
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
    ("U001 (Alex Chen)", "U001"),
    ("U002 (Nina Li)", "U002"),
]
_USER_VALUE_TO_ID = dict(USERS)
_DEFAULT_USER_LABEL = USERS[0][0]


def build_agent() -> Agent:
    """Assemble the agent with the same components as the REPL."""
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


# ---------- Pure event handlers, covered by unit tests ----------


FALLBACK_REPLY = "I could not complete a response within the step limit. Please rephrase and try again."
EMPTY_TRACE_MD = "_(No trace yet)_"


def _tool_summary(inv_result: dict) -> str:
    """Compact tool results into one readable line, as in scripts/chat.py."""
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
    """Render one RunResult as Markdown for the debug panel.

    Show routing, guardrail matches, loaded memory, and executed tools first.
    Keep the complete system prompt in a collapsed disclosure so supporting
    evidence stays visible. Render prompt and memory text inside code fences.
    """
    lines: list[str] = [
        f"- **[route]** {result.route_path}:{result.intent} "
        f"(conf={result.confidence:.2f})"
    ]

    policy = POLICY_MAP.get(result.intent)
    if result.route_path == "guardrails":
        tools_display = "() (blocked before routing)"
    elif policy is None:
        tools_display = "(unknown intent)"
    elif policy.exposed_tools is None:
        tools_display = "ALL (fallback: all 5 tools)"
    elif len(policy.exposed_tools) == 0:
        tools_display = "() (no tools; natural-language replies only)"
    else:
        tools_display = ", ".join(policy.exposed_tools)
    lines.append(f"- **[exposed_tools]** {tools_display}")

    if result.route_path == "guardrails":
        for event in result.memory_events:
            if event.kind == "guardrail_injection":
                lines.append(f"- **[matched rule]** `{event.detail}`")
        tool_count = sum(len(step.invocations) for step in result.steps)
        lines.append(
            f"- **[execution]** {len(result.steps)} model calls · {tool_count} tool calls"
        )

    sys_content = ""
    if policy is not None and policy.fast_path_tool is not None:
        lines.append(
            f"- **[fast_path]** {policy.fast_path_tool} → bypasses the LLM; no chat.completions call"
        )
    elif result.messages and result.messages[0].get("role") == "system":
        sys_content = result.messages[0]["content"]
        memory_start = sys_content.find("[Long-term memory:")
        if memory_start != -1:
            lines.extend([
                "\n**[loaded memory]** Context included in this turn:",
                "```text",
                sys_content[memory_start:],
                "```",
            ])

    for step in result.steps:
        for inv in step.invocations:
            args = json.dumps(inv.arguments, ensure_ascii=False)
            lines.extend([
                f"- **[tool]** `{inv.name}({args})`",
                f"  - Result: `{_tool_summary(inv.result)}`",
            ])

    if sys_content:
        lines.extend([
            "\n<details><summary>[system prompt] Full prompt</summary>\n",
            "```text",
            sys_content,
            "```",
            "\n</details>",
        ])

    return "\n".join(lines)


def format_error_trace(exc: Exception) -> str:
    """Render a fallback trace when the agent raises an exception."""
    return f"- **[error]** `{type(exc).__name__}: {exc}`"


def render_trace_log(log: list[str]) -> str:
    """Keep the latest trace open and earlier turns available as disclosures."""
    if not log:
        return EMPTY_TRACE_MD
    sections = []
    for i, trace in enumerate(log, 1):
        if i == len(log):
            sections.append(f"### Turn {i}\n\n{trace}")
        else:
            sections.append(
                f"<details><summary>Turn {i} · Previous turn</summary>\n\n"
                f"{trace}\n\n</details>"
            )
    return "\n\n---\n\n".join(sections)


def respond(
    user_msg: str,
    chat_display: list[dict] | None,
    session_state: dict | None,
    user_label: str,
    agent: Agent,
) -> tuple[str, list[dict], dict, str]:
    """Handle a user message, updating visible bubbles, agent history, and the turn trace.

    Return (cleared input, new chat display, new state, trace text).
    An empty input returns an empty trace so the panel does not append a turn.
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
    except Exception as exc:  # noqa: BLE001 — Keep agent errors from breaking the UI.
        final = f"[Internal error] {type(exc).__name__}: {exc}"
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
    """Start a new chat: save current history to long-term memory, then clear UI and state."""
    state = dict(session_state or {})
    uid = state.get("user_id")
    hist = state.get("history")
    if uid and hist:
        agent.save_long_term(uid, hist)
    state["history"] = None
    # Preserve user_id because the selected user has not changed.
    return [], state


def switch_user(
    new_label: str,
    session_state: dict | None,
    agent: Agent,
) -> tuple[list[dict], dict]:
    """Switch users: start a new chat and replace user_id."""
    state = dict(session_state or {})
    old_uid = state.get("user_id")
    hist = state.get("history")
    if old_uid and hist:
        agent.save_long_term(old_uid, hist)
    state["history"] = None
    state["user_id"] = _USER_VALUE_TO_ID[new_label]
    return [], state


# ---------- Gradio assembly ----------


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

    with gr.Blocks(title="NanaShop Support") as demo:
        gr.Markdown("## NanaShop Support\nMulti-turn e-commerce customer support agent")

        session_state = gr.State(value={"history": None, "user_id": "U001"})
        trace_state = gr.State(value=[])

        with gr.Row():
            with gr.Column(scale=1, min_width=220):
                gr.Markdown("### Session settings")
                user_picker = gr.Radio(
                    choices=[label for label, _ in USERS],
                    value=_DEFAULT_USER_LABEL,
                    label="Current user",
                )
                new_btn = gr.Button("New chat", variant="secondary")
                show_debug = gr.Checkbox(
                    label="Show debug trace",
                    value=False,
                )

            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    show_label=False,
                    height=640,
                )
                msg_box = gr.Textbox(
                    placeholder="Ask a question: refund O001, earbud prices, or talk to a human...",
                    show_label=False,
                    submit_btn=True,
                    autofocus=True,
                )

            with gr.Column(scale=2, visible=False, elem_id="trace-col") as trace_col:
                gr.Markdown("### Debug trace (all turns)")
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
    parser.add_argument("--share", action="store_true", help="Enable a public Gradio tunnel")
    args = parser.parse_args(argv)

    print("Seeding the database before startup...")
    seed()

    import gradio as gr

    agent = build_agent()
    demo = build_demo(agent)
    theme = gr.themes.Soft(
        primary_hue="slate",
        radius_size="lg",
        font=["system-ui", "-apple-system", "Segoe UI", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "monospace"],
    )
    # Gradio 6 puts API access, branding, and settings links in the footer.
    # Hide that footer with a single CSS rule.
    # Use a soft background and small type for traces without extra code or quote boxes.
    custom_css = """
footer { display: none !important; }
.gradio-container { max-width: 1700px !important; }

#trace-col {
    background: #f7f7f8;
    border: 1px solid rgba(0, 0, 0, 0.06);
    border-radius: 10px;
    padding: 12px 16px;
}
#trace-panel {
    font-size: 0.95rem;
    line-height: 1.55;
    color: #374151;
    max-height: 640px;
    overflow-y: auto;
}
#trace-panel code {
    background: transparent;
    padding: 0;
    font-size: inherit;
    overflow-wrap: anywhere;
}
#trace-panel summary { cursor: pointer; font-weight: 500; padding: 6px 0; }
/* System prompts use fences to prevent Markdown parsing. Soften code blocks:
   no background, inherited typography, preserved whitespace, and a left border. */
#trace-panel pre {
    background: transparent;
    border: none;
    border-left: 3px solid rgba(0, 0, 0, 0.12);
    padding: 2px 10px;
    margin: 6px 0;
    color: #4b5563;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
    font-size: inherit;
    font-family: inherit;
}
#trace-panel pre code {
    background: transparent;
    padding: 0;
    font-size: inherit;
    font-family: inherit;
    white-space: pre-wrap;
    overflow-wrap: anywhere;
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
