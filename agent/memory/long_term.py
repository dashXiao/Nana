"""长期记忆 · 用户级 append-only markdown。

谱系：ChatGPT Memory / MemGPT / Letta（跨会话的用户级状态）。
与 Claude Code "会话级无状态" 的设计哲学**不同**——电商客服天然有"老客回头"
场景，认得用户能显著改观体验；故本项目采用 per-user 持久化。

存储形态：
- 路径：`memory/sessions/{user_id}.md`（每用户单文件，append-only）
- 内容：YAML frontmatter + 多段 `## YYYY-MM-DD HH:MM` 会话摘要
- 载入：**只取最近 3 段**拼成一个 system 块，避免历史越堆越重
- 写回：对话轮数 < `min_turns` 时跳过（避免 1 轮闲聊就污染记忆）

设计原则：
- 文件系统足矣，不上向量库（见 PLAN §2 约束）
- 读/写均生成 `MemoryEvent`，REPL 肉眼可见
- 首次会话读不到文件 ≠ 错误，记一条 `detail="首次对话，无历史记忆"`
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agent.llm import chat as default_chat
from agent.memory.events import MemoryEvent

ChatFn = Callable[..., Any]

_DEFAULT_BASE_DIR = Path("memory/sessions")
_SECTION_HEADER_RE = re.compile(r"^## ", re.MULTILINE)
_RECENT_SECTIONS = 3


_SUMMARY_SYSTEM = """\
你是跨会话记忆摘要器。把一次客服会话压成一段极简摘要，用于下次该用户再访问时加载。
要求：
- 中文、120 字以内
- 只保留对"下次对话有用"的事实：用户关心过什么订单 / 商品，做过什么决定，有什么未决事项
- 不编造、不加寒暄、不加建议、不包含敏感隐私
- 输出纯文本一段，无 markdown、无前缀
"""


class LongTermMemory:
    """每用户一个 md 文件的长期记忆。

    `base_dir`: 默认 `memory/sessions/`；测试可注入 `tmp_path`
    `chat_fn`:  摘要走 qwen-plus，可测试注入
    `min_turns`: 用户轮数低于该值时 `save_session` 跳过（默认 2）
    `recent_sections`: `load_block` 取最近多少段（默认 3）
    """

    def __init__(
        self,
        base_dir: Path | None = None,
        chat_fn: ChatFn | None = None,
        min_turns: int = 2,
        recent_sections: int = _RECENT_SECTIONS,
    ):
        self._base_dir = Path(base_dir) if base_dir is not None else _DEFAULT_BASE_DIR
        self._chat = chat_fn or default_chat
        self._min_turns = min_turns
        self._recent = recent_sections

    # ---------- 读 ----------

    def load_block(self, user_id: str, events: list[MemoryEvent]) -> str:
        """读取该用户最近 N 段摘要，拼成可直接嵌入 system 的 markdown 块。

        文件不存在 / 内容为空 → 返回 `""`，并记一条 `load_long_term` 事件
        detail='首次对话，无历史记忆'，方便 REPL 观察到冷启动路径。
        """
        path = self._path_for(user_id)
        if not path.exists():
            events.append(
                MemoryEvent(kind="load_long_term", detail="首次对话，无历史记忆")
            )
            return ""

        text = path.read_text(encoding="utf-8")
        sections = _split_sections(text)
        if not sections:
            events.append(
                MemoryEvent(kind="load_long_term", detail="历史文件存在但无有效段落")
            )
            return ""

        recent = sections[-self._recent :]
        block = "[长期记忆 · 最近 {} 次会话摘要]\n{}".format(
            len(recent), "\n".join(recent)
        )
        events.append(
            MemoryEvent(
                kind="load_long_term",
                detail=f"载入 {len(recent)} 段（文件共 {len(sections)} 段）",
            )
        )
        return block

    # ---------- 写 ----------

    def save_session(
        self,
        user_id: str,
        messages: list[dict],
        events: list[MemoryEvent],
        now: datetime | None = None,
    ) -> None:
        """对本会话做摘要并追加到用户文件。轮数不足 `min_turns` 则跳过。"""
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if user_turns < self._min_turns:
            events.append(
                MemoryEvent(
                    kind="save_long_term",
                    detail=f"turns={user_turns} < min_turns={self._min_turns}，跳过",
                )
            )
            return

        summary = self._summarize(messages)
        if not summary:
            events.append(
                MemoryEvent(kind="save_long_term", detail="摘要为空，跳过写入")
            )
            return

        ts = (now or datetime.now()).strftime("%Y-%m-%d %H:%M")
        path = self._path_for(user_id)
        path.parent.mkdir(parents=True, exist_ok=True)

        if path.exists():
            existing = path.read_text(encoding="utf-8")
            new_text = existing.rstrip() + f"\n\n## {ts}\n{summary}\n"
        else:
            frontmatter = f"---\nuser_id: {user_id}\n---\n\n"
            new_text = frontmatter + f"## {ts}\n{summary}\n"
        path.write_text(new_text, encoding="utf-8")

        events.append(
            MemoryEvent(
                kind="save_long_term",
                detail=f"写入 1 段（turns={user_turns}）：{summary}",
            )
        )

    # ---------- 内部 ----------

    def _path_for(self, user_id: str) -> Path:
        return self._base_dir / f"{user_id}.md"

    def _summarize(self, messages: list[dict]) -> str:
        dialogue = [m for m in messages if m.get("role") != "system"]
        if not dialogue:
            return ""
        rendered = _render_for_summary(dialogue)
        try:
            resp = self._chat(
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": rendered},
                ],
                tools=None,
            )
            return (resp.choices[0].message.content or "").strip()
        except Exception as exc:  # noqa: BLE001
            return f"（本会话摘要失败：{type(exc).__name__}）"


def _split_sections(text: str) -> list[str]:
    """把 md 文本按 `^## ` 切成段，frontmatter 之前的内容丢弃。"""
    # 去 frontmatter
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]

    parts = _SECTION_HEADER_RE.split(text)
    # split 后 parts[0] 是标题前的杂项，丢弃；其余每段前补回 "## "
    sections = [f"## {p.rstrip()}" for p in parts[1:] if p.strip()]
    return sections


def _render_for_summary(dialogue: list[dict]) -> str:
    """和 short_term 同构——保持两处压缩风格一致，便于观察效果。"""
    lines: list[str] = []
    for m in dialogue:
        role = m.get("role", "?")
        content = m.get("content") or ""
        if role == "assistant" and m.get("tool_calls"):
            names = [
                tc.get("function", {}).get("name", "?")
                for tc in m["tool_calls"]
            ]
            lines.append(f"[assistant→tool_calls] {', '.join(names)}")
            if content:
                lines.append(f"[assistant] {content}")
        elif role == "tool":
            lines.append(f"[tool_result] {content[:200]}")
        else:
            lines.append(f"[{role}] {content}")
    return "\n".join(lines)
