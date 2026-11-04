"""LongTermMemory tests with isolated temporary files and mocked chat responses.

Check missing-file cold starts, loading only recent sections, minimum-turn thresholds,
events for skipped saves, and frontmatter written only on the first save.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

from agent.llm import LIGHT_MODEL
from agent.memory.events import MemoryEvent
from agent.memory.long_term import LongTermMemory


def _fake_chat(content: str):
    def _fn(messages, tools=None, **_kw):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    return _fn


def _conv(n_rounds: int) -> list[dict]:
    msgs: list[dict] = [{"role": "system", "content": "sys"}]
    for i in range(n_rounds):
        msgs.append({"role": "user", "content": f"Question {i + 1}"})
        msgs.append({"role": "assistant", "content": f"Answer {i + 1}"})
    return msgs


# ============ load_block ============


def test_load_block_cold_start(tmp_path: Path):
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat(""))
    events: list[MemoryEvent] = []

    block = mem.load_block("U001", events)

    assert block == ""
    assert len(events) == 1
    assert events[0].kind == "load_long_term"
    assert "First conversation" in events[0].detail


def test_load_block_returns_recent_three_sections(tmp_path: Path):
    """Load only the latest three sections from a file containing five."""
    path = tmp_path / "U001.md"
    path.write_text(
        "---\nuser_id: U001\n---\n\n"
        "## 2026-01-01 10:00\nSession 1 summary\n\n"
        "## 2026-02-01 10:00\nSession 2 summary\n\n"
        "## 2026-03-01 10:00\nSession 3 summary\n\n"
        "## 2026-04-01 10:00\nSession 4 summary\n\n"
        "## 2026-04-15 10:00\nSession 5 summary\n",
        encoding="utf-8",
    )
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat(""))
    events: list[MemoryEvent] = []

    block = mem.load_block("U001", events)

    assert block.startswith("[Long-term memory: recent session summaries (count: 3)]")
    assert "Session 1 summary" not in block
    assert "Session 2 summary" not in block
    assert "Session 3 summary" in block
    assert "Session 4 summary" in block
    assert "Session 5 summary" in block
    assert events[0].detail == "Session summaries loaded: 3 (available: 5)"


def test_load_block_fewer_than_three_sections(tmp_path: Path):
    path = tmp_path / "U002.md"
    path.write_text(
        "---\nuser_id: U002\n---\n\n## 2026-04-22 10:00\nOnly one section\n",
        encoding="utf-8",
    )
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat(""))
    events: list[MemoryEvent] = []

    block = mem.load_block("U002", events)

    assert "Only one section" in block
    assert "count: 1" in block


# ============ save_session ============


def test_save_session_below_min_turns_is_skipped(tmp_path: Path):
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat("Must not be called"), min_turns=2)
    msgs = _conv(1)  # Only one turn
    events: list[MemoryEvent] = []

    mem.save_session("U001", msgs, events)

    assert not (tmp_path / "U001.md").exists()
    assert events[0].kind == "save_long_term"
    assert "skipped" in events[0].detail


def test_save_session_writes_frontmatter_on_first_save(tmp_path: Path):
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat("The user discussed order O001"))
    msgs = _conv(3)
    events: list[MemoryEvent] = []
    now = datetime(2026, 4, 22, 15, 30)

    mem.save_session("U001", msgs, events, now=now)

    path = tmp_path / "U001.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\nuser_id: U001\n---")
    assert "## 2026-04-22 15:30" in text
    assert "The user discussed order O001" in text
    assert events[0].kind == "save_long_term"
    assert "The user discussed order O001" in events[0].detail


def test_save_session_appends_without_duplicating_frontmatter(tmp_path: Path):
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat("Second summary"))
    path = tmp_path / "U001.md"
    path.write_text(
        "---\nuser_id: U001\n---\n\n## 2026-04-20 10:00\nFirst summary\n",
        encoding="utf-8",
    )
    events: list[MemoryEvent] = []

    mem.save_session("U001", _conv(2), events, now=datetime(2026, 4, 22, 16, 0))

    text = path.read_text(encoding="utf-8")
    assert text.count("---\nuser_id: U001\n---") == 1
    assert "First summary" in text
    assert "Second summary" in text
    assert "## 2026-04-22 16:00" in text


def test_save_session_empty_messages_skipped(tmp_path: Path):
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat(""), min_turns=0)
    events: list[MemoryEvent] = []

    mem.save_session("U001", [{"role": "system", "content": "sys"}], events)

    assert not (tmp_path / "U001.md").exists()
    assert events[0].kind == "save_long_term"


# ============ Model-tier contract ============


def test_summary_uses_light_model(tmp_path: Path):
    """Summarization must use LIGHT_MODEL under the model-tier contract."""
    captured: dict = {}

    def capturing_chat(messages, tools=None, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Summary"))]
        )

    mem = LongTermMemory(base_dir=tmp_path, chat_fn=capturing_chat)
    mem.save_session("U001", _conv(2), [])
    assert captured.get("model") == LIGHT_MODEL


# ============ Event contract: every public call emits an event ============


def test_every_public_call_emits_exactly_one_event(tmp_path: Path):
    """Every public operation must emit an event for REPL observability."""
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat("Summary X"))
    events: list[MemoryEvent] = []

    mem.load_block("U999", events)
    assert len(events) == 1
    mem.save_session("U999", _conv(1), events)  # Skipped saves still emit an event.
    assert len(events) == 2
    mem.save_session("U999", _conv(2), events)  # Write a section.
    assert len(events) == 3
    mem.load_block("U999", events)  # The file now exists.
    assert len(events) == 4
