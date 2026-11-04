"""Short-term summary buffer: compact older turns after max_rounds and keep recent turns."""

from __future__ import annotations

from typing import Any, Callable

from agent.llm import LIGHT_MODEL, chat as default_chat
from agent.memory._render import render_for_summary
from agent.memory.events import MemoryEvent

ChatFn = Callable[..., Any]


_SUMMARY_SYSTEM = """\
Compress the supplied customer support conversation into a very brief summary
that preserves context for later turns. Requirements:
- English, at most 40 words.
- Keep only business facts: user requests, agent actions, tool calls, and key conclusions.
- Preserve IDs, currencies, and units exactly; never infer a currency or add a currency symbol absent from the dialogue.
- Do not invent facts or add greetings or advice.
- Output one plain-text paragraph without Markdown or a prefix such as "Summary:".
"""


class ShortTermMemory:
    def __init__(
        self,
        max_rounds: int = 4,
        keep_recent: int = 1,
        chat_fn: ChatFn | None = None,
    ):
        self._max_rounds = max_rounds
        self._keep_recent = keep_recent
        self._chat = chat_fn or default_chat

    def maybe_compact(
        self, messages: list[dict], events: list[MemoryEvent],
    ) -> list[dict]:
        """Compact earlier messages above the turn threshold; otherwise return a shallow copy."""
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if user_turns <= self._max_rounds:
            return list(messages)

        user_positions = [i for i, m in enumerate(messages) if m.get("role") == "user"]
        cut_idx = user_positions[-self._keep_recent]
        head, tail = messages[:cut_idx], messages[cut_idx:]

        preserved_system = [m for m in head if m.get("role") == "system"]
        dialogue = [m for m in head if m.get("role") != "system"]
        summary = self._summarize(dialogue)

        events.append(MemoryEvent(kind="compress", detail=summary))
        return preserved_system + [
            {"role": "system", "content": f"[Earlier conversation summary]\n{summary}"}
        ] + tail

    def _summarize(self, dialogue: list[dict]) -> str:
        try:
            resp = self._chat(
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": render_for_summary(dialogue)},
                ],
                tools=None,
                model=LIGHT_MODEL,
            )
            return (resp.choices[0].message.content or "").strip() or "(LLM returned an empty summary)"
        except Exception as exc:  # noqa: BLE001
            return f"(Summary failed: {type(exc).__name__})"
