"""Interactive Nana Agent REPL for manual testing.

Run: uv run python scripts/chat.py
Commands:
  /quit        Save the session to long-term memory and exit.
  /reset       Save and clear history without reseeding the database.
  /user U002   Save history and switch users, clearing history to isolate accounts.
  /tools       List registered tools.
  /help        Show command help.

The database is seeded once at startup for a deterministic initial state.
Each turn prints routing, exposed tools, memory events, system prompt, and tool calls.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Some IDE terminals use a non-UTF-8 locale.
# input() may read non-ASCII text as surrogateescape code points,
# causing UnicodeEncodeError in JSON requests. Force UTF-8 here.
for _stream in (sys.stdin, sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.core import Agent, RunResult  # noqa: E402
from agent.llm import LIGHT_MODEL, MODEL  # noqa: E402
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

TOOLS = [
    QUERY_ORDERS_SPEC,
    QUERY_POLICY_SPEC,
    QUERY_PRODUCT_SPEC,
    INITIATE_REFUND_SPEC,
    ESCALATE_TO_HUMAN_SPEC,
]


def _tool_summary(inv_result: dict) -> str:
    """Compact tool results into one line for terminal display."""
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
    """Print the trace in data-flow order: route, exposed tools, memory, system, then tool calls."""
    print(
        f"  [route] {result.route_path}:{result.intent} "
        f"(conf={result.confidence:.2f})"
    )

    policy = POLICY_MAP.get(result.intent)

    # Display the subset to verify that tool exposure is restricted.
    if policy is None:
        tools_display = "(unknown intent)"
    elif policy.exposed_tools is None:
        tools_display = "ALL (fallback: all 5 tools exposed)"
    elif len(policy.exposed_tools) == 0:
        tools_display = "() (no tools; natural-language replies only)"
    else:
        tools_display = ", ".join(policy.exposed_tools)
    print(f"  [exposed_tools] {tools_display}")

    # Show long-term loads and short-term compression events for inspection.
    for ev in result.memory_events:
        # Flatten multiline details, including full compression summaries, for readability.
        detail = ev.detail.replace("\n", " ")
        print(f"  [memory] {ev.kind}: {detail}")

    # The fast path skips the LLM; other paths print the complete system prompt.
    if policy is not None and policy.fast_path_tool is not None:
        print(f"  [fast_path] {policy.fast_path_tool} → bypasses the LLM; no chat.completions call")
    elif result.messages and result.messages[0].get("role") == "system":
        sys_content = result.messages[0]["content"]
        print("  ┌─ [complete system prompt sent to the LLM this turn] " + "─" * 20)
        for line in sys_content.splitlines():
            print(f"  │ {line}")
        print("  └─" + "─" * 68)

    # Tool calls follow routing and prompt inputs in data-flow order.
    for step in result.steps:
        for inv in step.invocations:
            args = json.dumps(inv.arguments, ensure_ascii=False)
            print(f"  [tool] {inv.name}({args}) → {_tool_summary(inv.result)}")


def _print_help() -> None:
    print(
        "Available commands:\n"
        "  /quit        Exit (save session to long-term memory)\n"
        "  /reset       Clear history (save session to long-term memory)\n"
        "  /user U00X   Switch users (save session and clear history)\n"
        "  /tools       List registered tools\n"
        "  /help        Show this help\n"
    )


def _print_tools(registry: ToolRegistry) -> None:
    for t in registry.openai_tools():
        fn = t["function"]
        print(f"  · {fn['name']} — {fn['description'][:60]}...")


def _flush_session_to_long_term(
    agent: Agent, user_id: str, history: list[dict] | None
) -> None:
    """Save the current history to long-term memory on session changes or exit, and print events."""
    if not history:
        return
    events = agent.save_long_term(user_id, history)
    for ev in events:
        detail = ev.detail.replace("\n", " ")
        print(f"  [memory] {ev.kind}: {detail}")


def run() -> None:
    print(f"=== NanaShop Agent REPL · main {MODEL} / light {LIGHT_MODEL} ===")
    print("Seeding the database for a clean initial state...")
    seed()
    print()

    registry = ToolRegistry(TOOLS)
    # Short-term: compact history above four user turns and retain the latest turn.
    # Long-term: load the latest three per-user summaries at cold start; append on session end.
    short_term = ShortTermMemory(max_rounds=4, keep_recent=1)
    long_term = LongTermMemory()
    agent = Agent(
        registry=registry,
        system_prompt=BASE_SYSTEM_PROMPT,
        max_steps=6,
        short_term=short_term,
        long_term=long_term,
    )

    user_id = "U001"
    history: list[dict] | None = None

    print(f"Current user: {user_id} (Alex Chen). Use /help for commands, Ctrl+C or /quit to exit.\n")

    while True:
        try:
            line = input(f"[{user_id}] > ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            _flush_session_to_long_term(agent, user_id, history)
            print("Goodbye.")
            return

        if not line:
            continue

        # --- Slash commands ---
        if line == "/quit" or line == "/exit":
            _flush_session_to_long_term(agent, user_id, history)
            print("Goodbye.")
            return
        if line == "/help":
            _print_help()
            continue
        if line == "/reset":
            _flush_session_to_long_term(agent, user_id, history)
            history = None
            print("(History cleared; previous session saved to long-term memory)\n")
            continue
        if line == "/tools":
            _print_tools(registry)
            continue
        if line.startswith("/user "):
            new_uid = line.split(None, 1)[1].strip()
            if not new_uid:
                print("Usage: /user U001")
                continue
            _flush_session_to_long_term(agent, user_id, history)
            user_id = new_uid
            history = None
            print(f"(Switched to {user_id}; history cleared and previous session saved to long-term memory)\n")
            continue
        if line.startswith("/"):
            print(f"Unknown command: {line} (try /help)")
            continue

        # --- Regular conversation through intent routing ---
        try:
            result = agent.run_routed(line, user_id=user_id, history=history)
        except Exception as exc:  # noqa: BLE001 — Keep the REPL alive on agent errors.
            print(f"  [error] {type(exc).__name__}: {exc}\n")
            continue

        _print_trace(result)

        final = result.final or "(No final answer: max_steps reached. Use /reset and rephrase your request.)"
        print(f"\n[agent] {final}\n")

        # A handoff ends the session: save and clear history before the next conversation.
        if result.final == "Transferred to a human agent.":
            _flush_session_to_long_term(agent, user_id, result.messages)
            history = None
            print("(Session handed off; history saved and cleared. Enter a new question to start again.)\n")
            continue

        # Use this turn's messages as history to preserve multi-turn context.
        history = result.messages


if __name__ == "__main__":
    run()
