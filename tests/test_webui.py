"""Pure-function tests for Gradio event handlers.

No server or live model calls. Use a stub agent to check state transitions for
respond, new_conversation, switch_user, and the trace renderers.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent.core import RunResult, Step, ToolInvocation
from agent.memory.events import MemoryEvent
from scripts.webui import (
    EMPTY_TRACE_MD,
    FALLBACK_REPLY,
    USERS,
    format_error_trace,
    format_turn_trace,
    new_conversation,
    render_trace_log,
    respond,
    switch_user,
)


# ---------- stub agent ----------


@dataclass
class _FakeRunResult:
    """Minimal run_routed return value matching the fields consumed from RunResult."""

    final: str | None
    messages: list[dict]
    # Trace fields have defaults so unrelated tests need not set them.
    intent: str = "other"
    route_path: str = "default"
    confidence: float = 0.0
    memory_events: list = field(default_factory=list)
    steps: list = field(default_factory=list)


@dataclass
class _StubAgent:
    """Record run_routed and save_long_term calls without contacting an LLM."""

    reply: str | None = "ok"
    raise_exc: Exception | None = None
    run_calls: list[dict] = field(default_factory=list)
    save_calls: list[tuple[str, list[dict]]] = field(default_factory=list)

    def run_routed(
        self, msg: str, user_id: str, history: list[dict] | None = None,
    ) -> _FakeRunResult:
        self.run_calls.append({"msg": msg, "user_id": user_id, "history": history})
        if self.raise_exc is not None:
            raise self.raise_exc
        # Mimic internal agent history by appending this turn to prev_history.
        new_hist = list(history or []) + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": self.reply or ""},
        ]
        return _FakeRunResult(final=self.reply, messages=new_hist)

    def save_long_term(self, user_id: str, history: list[dict]) -> list[Any]:
        self.save_calls.append((user_id, list(history)))
        return []


# ---------- respond ----------


def test_respond_appends_user_and_assistant_bubbles():
    agent = _StubAgent(reply="I found your order O001...")
    out, display, state, _trace = respond(
        "Check my orders", [], None, "U001 (Alex Chen)", agent=agent,
    )
    assert out == ""  # Input box cleared
    assert len(display) == 2
    assert display[0] == {"role": "user", "content": "Check my orders"}
    assert display[1] == {"role": "assistant", "content": "I found your order O001..."}
    assert state["user_id"] == "U001"
    assert state["history"] is not None
    # The agent receives the correct user_id.
    assert agent.run_calls[0]["user_id"] == "U001"
    assert agent.run_calls[0]["msg"] == "Check my orders"


def test_respond_passes_existing_history_to_agent():
    """Forward previous session history into the next run_routed turn."""
    agent = _StubAgent(reply="Okay")
    seed_hist = [{"role": "system", "content": "sys"}]
    initial_state = {"history": seed_hist, "user_id": "U001"}
    out, display, state, _trace = respond(
        "Second turn", [], initial_state, "U001 (Alex Chen)", agent=agent,
    )
    assert agent.run_calls[0]["history"] == seed_hist
    # The stub returns previous history plus two messages.
    assert len(state["history"]) == 3


def test_respond_maps_user_label_to_id():
    agent = _StubAgent()
    respond("hi", [], None, "U002 (Nina Li)", agent=agent)
    assert agent.run_calls[0]["user_id"] == "U002"


def test_respond_empty_message_no_call():
    agent = _StubAgent()
    out, display, state, trace = respond("   ", [], None, "U001 (Alex Chen)", agent=agent)
    assert agent.run_calls == []
    assert display == []
    assert state == {}  # Unchanged
    assert trace == ""  # An empty message does not append a trace.


def test_respond_agent_exception_becomes_error_bubble():
    agent = _StubAgent(raise_exc=RuntimeError("boom"))
    out, display, state, trace = respond("x", [], None, "U001 (Alex Chen)", agent=agent)
    assert "[Internal error]" in display[-1]["content"]
    assert "RuntimeError" in display[-1]["content"]
    # The trace uses the error branch.
    assert "[error]" in trace
    assert "RuntimeError" in trace and "boom" in trace


def test_respond_fallback_when_final_none():
    agent = _StubAgent(reply=None)  # The agent did not reach a final answer.
    out, display, state, _trace = respond("x", [], None, "U001 (Alex Chen)", agent=agent)
    assert display[-1]["content"] == FALLBACK_REPLY


def test_respond_preserves_previous_display_turns():
    agent = _StubAgent(reply="r2")
    prev_display = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "r1"},
    ]
    _, display, _, _ = respond("q2", prev_display, None, "U001 (Alex Chen)", agent=agent)
    assert [m["content"] for m in display] == ["q1", "r1", "q2", "r2"]


def test_respond_returns_nonempty_trace_on_success():
    """A normal response includes a nonempty trace with routing information."""
    agent = _StubAgent(reply="ok")
    _, _, _, trace = respond("hi", [], None, "U001 (Alex Chen)", agent=agent)
    assert trace  # Nonempty
    assert "[route]" in trace


# ---------- new_conversation ----------


def test_new_conversation_flushes_to_long_term_and_clears_display():
    agent = _StubAgent()
    state = {"history": [{"role": "user", "content": "old"}], "user_id": "U001"}
    display, new_state = new_conversation(state, agent=agent)
    assert display == []
    assert new_state["history"] is None
    assert new_state["user_id"] == "U001"  # Preserve the user.
    # Save long-term memory.
    assert agent.save_calls == [("U001", [{"role": "user", "content": "old"}])]


def test_new_conversation_noop_if_no_history():
    agent = _StubAgent()
    state = {"history": None, "user_id": "U001"}
    display, new_state = new_conversation(state, agent=agent)
    assert display == []
    assert new_state["history"] is None
    assert agent.save_calls == []


def test_new_conversation_noop_if_state_missing():
    agent = _StubAgent()
    display, new_state = new_conversation(None, agent=agent)
    assert display == []
    assert agent.save_calls == []


# ---------- switch_user ----------


def test_switch_user_flushes_and_changes_user():
    agent = _StubAgent()
    state = {"history": [{"role": "user", "content": "x"}], "user_id": "U001"}
    display, new_state = switch_user("U002 (Nina Li)", state, agent=agent)
    assert display == []
    assert new_state["user_id"] == "U002"
    assert new_state["history"] is None
    assert agent.save_calls == [("U001", [{"role": "user", "content": "x"}])]


def test_switch_user_without_prior_history_no_save():
    agent = _StubAgent()
    state = {"history": None, "user_id": "U001"}
    display, new_state = switch_user("U002 (Nina Li)", state, agent=agent)
    assert new_state["user_id"] == "U002"
    assert agent.save_calls == []


# ---------- USERS mapping ----------


def test_users_constant_shape():
    """User labels must map to valid Uxxx IDs and include parentheses."""
    for label, uid in USERS:
        assert uid.startswith("U")
        assert "(" in label and ")" in label


@pytest.mark.parametrize("label,uid", USERS)
def test_respond_for_each_user(label, uid):
    agent = _StubAgent()
    respond("hi", [], None, label, agent=agent)
    assert agent.run_calls[-1]["user_id"] == uid


# ---------- format_turn_trace ----------


def _make_result(
    *,
    intent: str = "other",
    route_path: str = "llm",
    confidence: float = 0.8,
    memory_events: list | None = None,
    steps: list | None = None,
    system_content: str | None = "SYS PROMPT BODY",
) -> RunResult:
    """Build a minimal RunResult for trace rendering tests."""
    messages: list[dict] = []
    if system_content is not None:
        messages.append({"role": "system", "content": system_content})
    return RunResult(
        final="ok",
        messages=messages,
        steps=steps or [],
        intent=intent,
        route_path=route_path,
        confidence=confidence,
        memory_events=memory_events or [],
    )


def test_format_turn_trace_contains_route_line():
    md = format_turn_trace(_make_result(intent="order_query", confidence=0.92))
    assert "[route]" in md
    assert "order_query" in md
    assert "0.92" in md


def test_format_turn_trace_exposed_tools_for_known_intent():
    """order_query exposes a named tool subset that must appear in the trace."""
    md = format_turn_trace(_make_result(intent="order_query"))
    assert "[exposed_tools]" in md
    assert "query_orders" in md  # order_query exposes this tool.


def test_format_turn_trace_exposed_tools_other_fallback():
    """The other fallback uses exposed_tools=None and renders ALL."""
    md = format_turn_trace(_make_result(intent="other"))
    assert "ALL" in md


def test_format_turn_trace_fast_path_skips_system_dump():
    """Escalation displays the fast path rather than a system prompt."""
    md = format_turn_trace(
        _make_result(intent="escalation", system_content="some-sys"),
    )
    assert "[fast_path]" in md
    assert "escalate_to_human" in md
    assert "some-sys" not in md  # The fast path does not echo the system prompt.


def test_format_turn_trace_omits_unloaded_memory_events():
    """Raw memory events alone must not be presented as loaded prompt context."""
    evs = [
        MemoryEvent(kind="load_long_term", detail="Loaded 2 history sections"),
        MemoryEvent(kind="compress", detail="summary"),
    ]
    md = format_turn_trace(_make_result(memory_events=evs))
    assert "[memory]" not in md
    assert "load_long_term" not in md


def test_format_turn_trace_shows_actual_loaded_memory():
    memory = "[Long-term memory: recent session summaries (count: 1)]\nThe user asked about P002."
    md = format_turn_trace(_make_result(system_content=f"Base prompt\n\n{memory}"))
    assert "[loaded memory]" in md
    assert memory in md
    assert "<details><summary>[system prompt]" in md
    assert "<details open" not in md


def test_format_turn_trace_explains_guardrail_short_circuit():
    md = format_turn_trace(_make_result(
        intent="injection",
        route_path="guardrails",
        system_content=None,
        memory_events=[MemoryEvent("guardrail_injection", "prompt_leak")],
    ))
    assert "prompt_leak" in md
    assert "0 model calls" in md
    assert "0 tool calls" in md
    assert "blocked before routing" in md
    assert "unknown intent" not in md


def test_format_turn_trace_renders_tool_calls_with_args_and_result():
    step = Step(
        assistant_message={"role": "assistant", "content": ""},
        invocations=[
            ToolInvocation(
                call_id="c1",
                name="query_orders",
                arguments={"status": "shipped"},
                result={"ok": True, "data": {"count": 2, "orders": [
                    {"order_id": "O001"}, {"order_id": "O002"},
                ]}},
            ),
        ],
    )
    md = format_turn_trace(_make_result(steps=[step]))
    assert "[tool]" in md
    assert "query_orders" in md
    assert "shipped" in md
    assert "count=2" in md
    assert "O001" in md and "O002" in md


def test_format_turn_trace_tool_error_summary():
    step = Step(
        assistant_message={"role": "assistant", "content": ""},
        invocations=[
            ToolInvocation(
                call_id="c1",
                name="initiate_refund",
                arguments={"order_id": "O999"},
                result={
                    "ok": False,
                    "error_code": "NOT_FOUND",
                    "message": "Order does not exist",
                },
            ),
        ],
    )
    md = format_turn_trace(_make_result(steps=[step]))
    assert "ERR NOT_FOUND" in md


def test_format_turn_trace_system_dump_as_code_fence():
    """Fence system prompts to preserve embedded Markdown without headings or layout gaps; CSS softens the code styling."""
    md = format_turn_trace(_make_result(
        intent="order_query", system_content="Line1\n## Heading inside\n\nLine2",
    ))
    assert "[system prompt]" in md
    assert "```" in md  # Use fences to prevent Markdown parsing.
    assert "Line1" in md and "Line2" in md
    assert "## Heading inside" in md  # Preserve raw text rather than rendering a heading.


# ---------- format_error_trace ----------


def test_format_error_trace_contains_type_and_message():
    md = format_error_trace(RuntimeError("boom"))
    assert "[error]" in md
    assert "RuntimeError" in md
    assert "boom" in md


# ---------- render_trace_log ----------


def test_render_trace_log_empty_placeholder():
    assert render_trace_log([]) == EMPTY_TRACE_MD


def test_render_trace_log_multiple_turns_numbered():
    md = render_trace_log(["alpha", "beta", "gamma"])
    assert "Turn 1" in md
    assert "Turn 2" in md
    assert "Turn 3" in md
    assert "alpha" in md and "beta" in md and "gamma" in md
    # Separate sections with ---.
    assert md.count("---") == 2
