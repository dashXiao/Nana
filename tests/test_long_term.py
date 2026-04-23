"""LongTermMemory 单元测试。

用 tmp_path 隔离文件系统，chat_fn 替身避免真实 LLM。重点验证：
- 冷启动：文件不存在 → load_block 返回 ""，记 "首次对话，无历史记忆"
- 仅取最近 N 段：文件有 >3 段时 load_block 只拼最近 3 段
- 写入门槛：user 轮数 < min_turns → 跳过，仍记事件
- 文件形态：首次建立含 frontmatter；后续 append 不重复 frontmatter
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

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
        msgs.append({"role": "user", "content": f"问{i + 1}"})
        msgs.append({"role": "assistant", "content": f"答{i + 1}"})
    return msgs


# ============ load_block ============


def test_load_block_cold_start(tmp_path: Path):
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat(""))
    events: list[MemoryEvent] = []

    block = mem.load_block("U001", events)

    assert block == ""
    assert len(events) == 1
    assert events[0].kind == "load_long_term"
    assert "首次对话" in events[0].detail


def test_load_block_returns_recent_three_sections(tmp_path: Path):
    """文件里有 5 段，load_block 只拼最近 3 段。"""
    path = tmp_path / "U001.md"
    path.write_text(
        "---\nuser_id: U001\n---\n\n"
        "## 2026-01-01 10:00\n第1次摘要\n\n"
        "## 2026-02-01 10:00\n第2次摘要\n\n"
        "## 2026-03-01 10:00\n第3次摘要\n\n"
        "## 2026-04-01 10:00\n第4次摘要\n\n"
        "## 2026-04-15 10:00\n第5次摘要\n",
        encoding="utf-8",
    )
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat(""))
    events: list[MemoryEvent] = []

    block = mem.load_block("U001", events)

    assert block.startswith("[长期记忆 · 最近 3 次会话摘要]")
    assert "第1次摘要" not in block
    assert "第2次摘要" not in block
    assert "第3次摘要" in block
    assert "第4次摘要" in block
    assert "第5次摘要" in block
    assert events[0].detail == "载入 3 段（文件共 5 段）"


def test_load_block_fewer_than_three_sections(tmp_path: Path):
    path = tmp_path / "U002.md"
    path.write_text(
        "---\nuser_id: U002\n---\n\n## 2026-04-22 10:00\n仅一段\n",
        encoding="utf-8",
    )
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat(""))
    events: list[MemoryEvent] = []

    block = mem.load_block("U002", events)

    assert "仅一段" in block
    assert "最近 1 次" in block


# ============ save_session ============


def test_save_session_below_min_turns_is_skipped(tmp_path: Path):
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat("不该被调"), min_turns=2)
    msgs = _conv(1)  # 只有 1 轮
    events: list[MemoryEvent] = []

    mem.save_session("U001", msgs, events)

    assert not (tmp_path / "U001.md").exists()
    assert events[0].kind == "save_long_term"
    assert "跳过" in events[0].detail


def test_save_session_writes_frontmatter_on_first_save(tmp_path: Path):
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat("本次用户聊了订单 O001"))
    msgs = _conv(3)
    events: list[MemoryEvent] = []
    now = datetime(2026, 4, 22, 15, 30)

    mem.save_session("U001", msgs, events, now=now)

    path = tmp_path / "U001.md"
    assert path.exists()
    text = path.read_text(encoding="utf-8")
    assert text.startswith("---\nuser_id: U001\n---")
    assert "## 2026-04-22 15:30" in text
    assert "本次用户聊了订单 O001" in text
    assert events[0].kind == "save_long_term"
    assert "本次用户聊了订单 O001" in events[0].detail


def test_save_session_appends_without_duplicating_frontmatter(tmp_path: Path):
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat("第二次摘要"))
    path = tmp_path / "U001.md"
    path.write_text(
        "---\nuser_id: U001\n---\n\n## 2026-04-20 10:00\n第一次摘要\n",
        encoding="utf-8",
    )
    events: list[MemoryEvent] = []

    mem.save_session("U001", _conv(2), events, now=datetime(2026, 4, 22, 16, 0))

    text = path.read_text(encoding="utf-8")
    assert text.count("---\nuser_id: U001\n---") == 1
    assert "第一次摘要" in text
    assert "第二次摘要" in text
    assert "## 2026-04-22 16:00" in text


def test_save_session_empty_messages_skipped(tmp_path: Path):
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat(""), min_turns=0)
    events: list[MemoryEvent] = []

    mem.save_session("U001", [{"role": "system", "content": "sys"}], events)

    assert not (tmp_path / "U001.md").exists()
    assert events[0].kind == "save_long_term"


# ============ 事件契约：所有公开方法至少记一条事件 ============


def test_every_public_call_emits_exactly_one_event(tmp_path: Path):
    """REPL 依赖"每次调用必有事件"来打印轨迹；回归锁死。"""
    mem = LongTermMemory(base_dir=tmp_path, chat_fn=_fake_chat("摘要X"))
    events: list[MemoryEvent] = []

    mem.load_block("U999", events)
    assert len(events) == 1
    mem.save_session("U999", _conv(1), events)  # 跳过仍记事件
    assert len(events) == 2
    mem.save_session("U999", _conv(2), events)  # 真写入
    assert len(events) == 3
    mem.load_block("U999", events)  # 这次有文件
    assert len(events) == 4
