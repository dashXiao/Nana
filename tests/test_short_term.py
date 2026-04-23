"""ShortTermMemory 单元测试。

用脚本化 chat_fn 替身，不发真实请求。重点验证：
- 触发阈值：用户轮数 ≤ max_rounds 不压；> max_rounds 才压
- 保留策略：keep_recent 轮原样保留（含其 assistant/tool 尾巴）
- 原地替换：压缩后 messages 头部是 system + 摘要；`[早期对话摘要]` 前缀
- 可观测：成功压缩一次记一条 MemoryEvent("compress", 摘要文本)
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

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
    """生成 n 轮 user+assistant 对话（含一条 system 头）。"""
    msgs: list[dict] = [{"role": "system", "content": "你是客服"}]
    for i in range(n_rounds):
        msgs.append(_u(f"第{i + 1}轮问题"))
        msgs.append(_a(f"第{i + 1}轮回答"))
    return msgs


# ============ 不触发 ============


def test_no_compact_when_under_threshold():
    mem = ShortTermMemory(max_rounds=4, keep_recent=1, chat_fn=_fake_chat("摘要"))
    msgs = _build_history(4)  # 恰好等于阈值
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    assert out == msgs and out is not msgs  # 内容相同、对象不同
    assert events == []


def test_no_compact_when_only_one_user_turn():
    mem = ShortTermMemory(max_rounds=4, keep_recent=1, chat_fn=_fake_chat("摘要"))
    msgs = [{"role": "system", "content": "sys"}, _u("hi")]
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    assert out == msgs
    assert events == []


# ============ 触发 ============


def test_compact_triggers_above_threshold():
    """5 轮 > max_rounds=4 → 触发，保留最后 1 轮。"""
    mem = ShortTermMemory(
        max_rounds=4, keep_recent=1, chat_fn=_fake_chat("前四轮用户问订单状态、商品价格和退款政策"),
    )
    msgs = _build_history(5)
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    # 结构：原 system + 摘要 system + 最后一轮 user + 最后一轮 assistant
    assert len(out) == 4
    assert out[0]["role"] == "system" and out[0]["content"] == "你是客服"
    assert out[1]["role"] == "system"
    assert out[1]["content"].startswith("[早期对话摘要]")
    assert "前四轮用户问订单状态" in out[1]["content"]
    # 尾巴：最新 1 轮原样
    assert out[2] == _u("第5轮问题")
    assert out[3] == _a("第5轮回答")

    # 事件：一条 compress，detail 是摘要正文（不带前缀）
    assert len(events) == 1
    assert events[0].kind == "compress"
    assert events[0].detail == "前四轮用户问订单状态、商品价格和退款政策"


def test_keep_recent_two_retains_last_two_rounds():
    mem = ShortTermMemory(max_rounds=3, keep_recent=2, chat_fn=_fake_chat("前两轮"))
    msgs = _build_history(4)
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    # system + 摘要 + 最后 2 轮（user+assistant ×2 = 4）
    assert len(out) == 6
    user_contents = [m["content"] for m in out if m["role"] == "user"]
    assert user_contents == ["第3轮问题", "第4轮问题"]


def test_compact_preserves_tool_messages_in_tail():
    """保留段内若带 assistant-with-tool_calls + tool result，应整块保留。"""
    mem = ShortTermMemory(max_rounds=2, keep_recent=1, chat_fn=_fake_chat("早期摘要"))
    msgs = [
        {"role": "system", "content": "sys"},
        _u("旧问题1"), _a("旧回答1"),
        _u("旧问题2"), _a("旧回答2"),
        _u("新问题"),
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "query_orders", "arguments": "{}"}}
            ],
        },
        {"role": "tool", "tool_call_id": "c1", "content": '{"ok": true}'},
        _a("新回答"),
    ]
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    # 3 轮 > 2 触发；保留末尾 1 轮：新问题 + 整个 assistant→tool→assistant 链
    tail_roles = [m["role"] for m in out[-4:]]
    assert tail_roles == ["user", "assistant", "tool", "assistant"]
    assert out[0]["role"] == "system" and out[0]["content"] == "sys"
    assert out[1]["role"] == "system" and out[1]["content"].startswith("[早期对话摘要]")


# ============ 降级 ============


def test_llm_failure_degrades_to_placeholder_summary():
    """chat_fn 抛异常时 maybe_compact 仍要返回压缩后的结构，摘要用兜底文本。"""

    def _boom(messages, tools=None, **_kw):
        raise RuntimeError("network down")

    mem = ShortTermMemory(max_rounds=2, keep_recent=1, chat_fn=_boom)
    msgs = _build_history(3)
    events: list[MemoryEvent] = []

    out = mem.maybe_compact(msgs, events)

    # 仍触发压缩
    assert len(out) == 4
    assert out[1]["content"].startswith("[早期对话摘要]")
    assert "摘要失败" in out[1]["content"]
    assert events[0].kind == "compress"


# ============ 构造参数合法性 ============


def test_invalid_keep_recent_raises():
    with pytest.raises(ValueError):
        ShortTermMemory(max_rounds=2, keep_recent=2)
    with pytest.raises(ValueError):
        ShortTermMemory(max_rounds=2, keep_recent=5)
