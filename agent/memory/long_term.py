"""Long-term memory: append-only per-user Markdown with timestamped session summaries."""

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
Summarize a customer support session into a very brief memory for the user's next visit.
Requirements:
- English, at most 60 words.
- Keep only facts useful next time: relevant orders or products, decisions, and unresolved issues.
- Focus on the user's questions and decisions. Do not retain catalog prices or stock counts; look them up again next session.
- Preserve IDs, currencies, and units exactly. Never infer a currency or add a currency symbol absent from the dialogue.
- Do not invent facts, add greetings or advice, or include sensitive personal information.
- Output one plain-text paragraph without Markdown or a prefix.
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
        """Load the latest N user summaries as a Markdown block for the system prompt."""
        path = self._path_for(user_id)
        if not path.exists():
            events.append(MemoryEvent("load_long_term", "First conversation; no saved memory"))
            return ""

        sections = _split_sections(path.read_text(encoding="utf-8"))
        if not sections:
            events.append(MemoryEvent("load_long_term", "First conversation; no saved memory"))
            return ""

        recent = sections[-self._recent :]
        events.append(MemoryEvent(
            "load_long_term",
            f"Session summaries loaded: {len(recent)} (available: {len(sections)})",
        ))
        return "[Long-term memory: recent session summaries (count: {})]\n{}".format(
            len(recent), "\n".join(recent)
        )

    def save_session(
        self,
        user_id: str,
        messages: list[dict],
        events: list[MemoryEvent],
        now: datetime | None = None,
    ) -> None:
        """Summarize and append this session; skip short sessions or empty summaries."""
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if user_turns < self._min_turns:
            events.append(MemoryEvent(
                "save_long_term",
                f"turns={user_turns} < min_turns={self._min_turns}; skipped",
            ))
            return

        summary = self._summarize(messages)
        if not summary:
            events.append(MemoryEvent("save_long_term", "Empty summary; skipped"))
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
            f"Saved 1 section (turns={user_turns}): {summary}",
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
            return f"(Session summary failed: {type(exc).__name__})"


def _split_sections(text: str) -> list[str]:
    if text.startswith("---"):
        end = text.find("\n---", 3)
        if end != -1:
            text = text[end + 4 :]
    parts = _SECTION_HEADER_RE.split(text)
    return [f"## {p.rstrip()}" for p in parts[1:] if p.strip()]
