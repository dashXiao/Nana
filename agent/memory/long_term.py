"""长期记忆 · 用户级 append-only markdown：每段 `## YYYY-MM-DD HH:MM` 一次会话摘要。"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agent.llm import LIGHT_MODEL, chat as default_chat
from agent.memory._render import render_for_summary
from agent.memory.events import MemoryEvent

ChatFn = Callable[..., Any]

_DEFAULT_BASE_DIR = Path("memory/sessions")
_SECTION_HEADER_RE = re.compile(r"^## ", re.MULTILINE)


_SUMMARY_SYSTEM = """\
你是跨会话记忆摘要器。把一次客服会话压成一段极简摘要，用于下次该用户再访问时加载。
要求：
- 中文、120 字以内
- 只保留对"下次对话有用"的事实：用户关心过什么订单 / 商品，做过什么决定，有什么未决事项
- 不编造、不加寒暄、不加建议、不包含敏感隐私
- 输出纯文本一段，无 markdown、无前缀
"""


class LongTermMemory:
    def __init__(
        self,
        base_dir: Path | None = None,
        chat_fn: ChatFn | None = None,
        min_turns: int = 2,
        recent_sections: int = 3,
    ):
        self._base_dir = Path(base_dir) if base_dir is not None else _DEFAULT_BASE_DIR
        self._chat = chat_fn or default_chat
        self._min_turns = min_turns
        self._recent = recent_sections

    def load_block(self, user_id: str, events: list[MemoryEvent]) -> str:
        """读取该用户最近 N 段摘要，拼成可直接嵌入 system 的 markdown 块。"""
        path = self._path_for(user_id)
        if not path.exists():
            events.append(MemoryEvent("load_long_term", "首次对话，无历史记忆"))
            return ""

        sections = _split_sections(path.read_text(encoding="utf-8"))
        if not sections:
            events.append(MemoryEvent("load_long_term", "首次对话，无历史记忆"))
            return ""

        recent = sections[-self._recent :]
        events.append(MemoryEvent(
            "load_long_term",
            f"载入 {len(recent)} 段（文件共 {len(sections)} 段）",
        ))
        return "[长期记忆 · 最近 {} 次会话摘要]\n{}".format(
            len(recent), "\n".join(recent)
        )

    def save_session(
        self,
        user_id: str,
        messages: list[dict],
        events: list[MemoryEvent],
        now: datetime | None = None,
    ) -> None:
        """对本会话做摘要并追加到用户文件。轮数 < min_turns 或摘要为空则跳过。"""
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if user_turns < self._min_turns:
            events.append(MemoryEvent(
                "save_long_term",
                f"turns={user_turns} < min_turns={self._min_turns}，跳过",
            ))
            return

        summary = self._summarize(messages)
        if not summary:
            events.append(MemoryEvent("save_long_term", "摘要为空，跳过"))
            return

        ts = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
        path = self._path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            new_text = path.read_text(encoding="utf-8").rstrip() + f"\n\n## {ts}\n{summary}\n"
        else:
            new_text = f"---\nuser_id: {user_id}\n---\n\n## {ts}\n{summary}\n"
        path.write_text(new_text, encoding="utf-8")

        events.append(MemoryEvent(
            "save_long_term",
            f"写入 1 段（turns={user_turns}）：{summary}",
        ))

    def _path_for(self, user_id: str) -> Path:
        return self._base_dir / f"{user_id}.md"

    def _summarize(self, messages: list[dict]) -> str:
        dialogue = [m for m in messages if m.get("role") != "system"]
        if not dialogue:
            return ""
        try:
            resp = self._chat(
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": render_for_summary(dialogue)},
                ],
                tools=None,
                model=LIGHT_MODEL,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            return f"（本会话摘要失败：{type(exc).__name__}）"


def _split_sections(text: str) -> list[str]:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    parts = _SECTION_HEADER_RE.split(text)
    return [f"## {p.rstrip()}" for p in parts[1:] if p.strip()]
