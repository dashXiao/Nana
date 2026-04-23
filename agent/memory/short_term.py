"""短期记忆 · Summary-Buffer 压缩。

谱系：LangChain `ConversationSummaryBufferMemory` / Claude Code auto-compact。
按用户轮数触发：超过 `max_rounds` 时，把早期对话交给 LLM 摘成一段 system，
保留末 `keep_recent` 轮原始细节。纯函数——不改 self、不持久化。
"""

from __future__ import annotations

from typing import Any, Callable

from agent.llm import chat as default_chat
from agent.memory._render import render_for_summary
from agent.memory.events import MemoryEvent

ChatFn = Callable[..., Any]


_SUMMARY_SYSTEM = """\
你是对话历史压缩器。把给定的多轮客服对话片段压成一段极简摘要，
以便后续轮次仍能理解上下文。要求：
- 中文、80 字以内
- 只保留业务事实（用户问过什么、agent 做了什么、调了哪些工具、关键结论）
- 不编造信息、不加寒暄、不加建议
- 输出纯文本一段，无 markdown、无前缀"摘要："之类
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
        """用户轮数超阈值时压缩早期消息；否则原样（新列表）返回。"""
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
            {"role": "system", "content": f"[早期对话摘要]\n{summary}"}
        ] + tail

    def _summarize(self, dialogue: list[dict]) -> str:
        try:
            resp = self._chat(
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": render_for_summary(dialogue)},
                ],
                tools=None,
            )
            return (resp.choices[0].message.content or "").strip() or "（LLM 返回空摘要）"
        except Exception as exc:  # noqa: BLE001 — LLM 不稳定时不应炸掉对话
            return f"（摘要失败：{type(exc).__name__}）"
