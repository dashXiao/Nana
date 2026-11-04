"""ShortTermMemory tests with scripted responses and no live requests.

Check the strict turn threshold, retention of recent assistant/tool tails, the system
summary structure, and one compression event containing the unprefixed summary.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.llm import LIGHT_MODEL
from agent.memory.events import MemoryEvent
from agent.memory.short_term import ShortTermMemory


def _fake_chat(content: str):
    def _fn(messages, tools=None, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    return _fn


def _u(text: str) -> dict:
    return {"role": "user", "content": text}


def _a(text: str) -> dict:
    return {"role": "assistant", "content": text}


def _build_history(n_rounds: int) -> list[dict]:
    """Generate n user/assistant turns with one leading system message."""
    msgs: list[dict] = [{"role": "system", "content": "You are a support assistant"}]
    for i in range(n_rounds):
        msgs.append(_u(f"Turn {i + 1} question"))
        msgs.append(_a(f"Turn {i + 1} answer"))
    return msgs


# ============ No compression ============


def test_no_compact_when_under_threshold():
    mem = ShortTermMemory(max_rounds=4, keep_recent=1, chat_fn=_fake_chat("Summary"))
    msgs = _build_history(4)  # Exactly at the threshold
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    assert out == msgs and out is not msgs  # Same content, distinct list
    assert events == []


def test_no_compact_when_only_one_user_turn():
    mem = ShortTermMemory(max_rounds=4, keep_recent=1, chat_fn=_fake_chat("Summary"))
    msgs = [{"role": "system", "content": "sys"}, _u("hi")]
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    assert out == msgs
    assert events == []


# ============ Compression ============


def test_compact_triggers_above_threshold():
    """Five turns exceed max_rounds=4; retain the last turn."""
    mem = ShortTermMemory(
        max_rounds=4, keep_recent=1, chat_fn=_fake_chat("The first four turns covered order status, product prices, and refund policy"),
    )
    msgs = _build_history(5)
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    # Original system + summary system + final user and assistant turn
    assert len(out) == 4
    assert out[0]["role"] == "system" and out[0]["content"] == "You are a support assistant"
    assert out[1]["role"] == "system"
    assert out[1]["content"].startswith("[Earlier conversation summary]")
    assert "The first four turns covered order status" in out[1]["content"]
    # Keep the latest turn unchanged.
    assert out[2] == _u("Turn 5 question")
    assert out[3] == _a("Turn 5 answer")

    # One compress event containing the summary without its prefix
    assert len(events) == 1
    assert events[0].kind == "compress"
    assert events[0].detail == "The first four turns covered order status, product prices, and refund policy"


def test_keep_recent_two_retains_last_two_rounds():
    mem = ShortTermMemory(max_rounds=3, keep_recent=2, chat_fn=_fake_chat("First two turns"))
    msgs = _build_history(4)
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    # System + summary + last two user/assistant pairs
    assert len(out) == 6
    user_contents = [m["content"] for m in out if m["role"] == "user"]
    assert user_contents == ["Turn 3 question", "Turn 4 question"]


def test_compact_preserves_tool_messages_in_tail():
    """Retain the complete assistant/tool/result chain within the kept turn."""
    mem = ShortTermMemory(max_rounds=2, keep_recent=1, chat_fn=_fake_chat("Earlier summary"))
    msgs = [
        {"role": "system", "content": "sys"},
        _u("Old question 1"), _a("Old answer 1"),
        _u("Old question 2"), _a("Old answer 2"),
        _u("New question"),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "query_orders", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
        _a("New answer"),
    ]
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    # Three turns exceed two; retain the full latest assistant/tool chain.
    tail_roles = [m["role"] for m in out[-4:]]
    assert tail_roles == ["user", "assistant", "tool", "assistant"]
    assert out[0]["role"] == "system" and out[0]["content"] == "sys"
    assert out[1]["role"] == "system" and out[1]["content"].startswith("[Earlier conversation summary]")


# ============ Fallback ============


def test_compression_uses_light_model():
    """Summarization must use LIGHT_MODEL under the model-tier contract."""
    captured: dict = {}

    def capturing_chat(messages, tools=None, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Summary"))]
        )

    mem = ShortTermMemory(max_rounds=2, keep_recent=1, chat_fn=capturing_chat)
    msgs = _build_history(3)
    mem.maybe_compact(msgs, [])
    assert captured.get("model") == LIGHT_MODEL


def test_llm_failure_degrades_to_placeholder_summary():
    """If chat_fn fails, preserve the compacted structure with a fallback summary."""

    def _boom(messages, tools=None, **_kw):
        raise RuntimeError("network down")

    mem = ShortTermMemory(max_rounds=2, keep_recent=1, chat_fn=_boom)
    msgs = _build_history(3)
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    # Compression still occurs.
    assert len(out) == 4
    assert out[1]["content"].startswith("[Earlier conversation summary]")
    assert "Summary failed" in out[1]["content"]
    assert events[0].kind == "compress"


