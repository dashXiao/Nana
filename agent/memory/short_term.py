"""短期记忆 · Summary-Buffer 压缩。

谱系：LangChain `ConversationSummaryBufferMemory` / Claude Code auto-compact。
思路：**按用户轮数触发**，当 messages 里 `role == "user"` 的条数超过 `max_rounds`，
把"早期"那部分消息交给 LLM 压成一段摘要，**原地**替换成一条
`{"role": "system", "content": "[早期对话摘要]\\n..."}`，保留末尾
`keep_recent` 轮不动，维持最近轮次的原始细节。

设计原则：
- **入口先压后跑**：在 `run_routed` 入口调用，让 LLM 本轮只看压缩后的历史
- **纯函数化**：不改 `self`，不持久化，只算出新的 messages 列表并回传
- **可观测**：触发压缩时追加一条 `MemoryEvent("compress", 完整摘要文本)`
- 展示性为主：参数写死默认（4 / 1），够看出效果即可；真实场景再调
"""

from __future__ import annotations

from typing import Any, Callable

from agent.llm import chat as default_chat
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
    """Summary-Buffer 短期记忆。

    `max_rounds`: 用户轮数阈值（含 system / assistant / tool 消息不计数）
    `keep_recent`: 压缩时末尾保留的用户轮数（连同该轮之后的所有消息）
    `chat_fn`: 允许测试注入 fake LLM；默认复用 `agent.llm.chat`
    """

    def __init__(
        self,
        max_rounds: int = 4,
        keep_recent: int = 1,
        chat_fn: ChatFn | None = None,
    ):
        if keep_recent >= max_rounds:
            raise ValueError("keep_recent 必须 < max_rounds，否则永不触发压缩")
        self._max_rounds = max_rounds
        self._keep_recent = keep_recent
        self._chat = chat_fn or default_chat

    def maybe_compact(
        self,
        messages: list[dict],
        events: list[MemoryEvent],
    ) -> list[dict]:
        """若用户轮数超阈值，压缩早期消息；否则原样返回。

        返回的是**新列表**（不原地修改入参），调用方负责替换引用。
        压缩成功时 `events` 末尾会被追加一条 `compress` 事件。
        """
        user_turns = sum(1 for m in messages if m.get("role") == "user")
        if user_turns <= self._max_rounds:
            return list(messages)

        cut_idx = self._find_cut_index(messages)
        if cut_idx is None:
            # 理论不会到这——user_turns > max_rounds 一定找得到切点
            return list(messages)

        head, tail = messages[:cut_idx], messages[cut_idx:]
        # 头部里的 system 条目原样保留，只摘要"对话肉"
        preserved_system = [m for m in head if m.get("role") == "system"]
        dialogue = [m for m in head if m.get("role") != "system"]

        summary = self._summarize(dialogue)
        summary_msg = {
            "role": "system",
            "content": f"[早期对话摘要]\n{summary}",
        }

        events.append(MemoryEvent(kind="compress", detail=summary))
        return preserved_system + [summary_msg] + tail

    # ---------- 内部 ----------

    def _find_cut_index(self, messages: list[dict]) -> int | None:
        """找到倒数第 `keep_recent` 个 user 消息的位置——该位置及之后保留原样。"""
        user_positions = [
            i for i, m in enumerate(messages) if m.get("role") == "user"
        ]
        if len(user_positions) <= self._keep_recent:
            return None
        return user_positions[-self._keep_recent]

    def _summarize(self, dialogue: list[dict]) -> str:
        """把早期对话肉扔给 LLM 压成一段话。失败则降级为朴素拼接兜底。"""
        if not dialogue:
            return "（无内容）"
        rendered = _render_for_summary(dialogue)
        try:
            resp = self._chat(
                messages=[
                    {"role": "system", "content": _SUMMARY_SYSTEM},
                    {"role": "user", "content": rendered},
                ],
                tools=None,
            )
            text = (resp.choices[0].message.content or "").strip()
            return text or "（LLM 返回空摘要）"
        except Exception as exc:  # noqa: BLE001 — 摘要失败不应炸掉对话
            return f"（摘要失败：{type(exc).__name__}；保留原始对话前若干轮略）"


def _render_for_summary(dialogue: list[dict]) -> str:
    """把 messages 序列拍扁成文本喂给摘要 LLM。"""
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
