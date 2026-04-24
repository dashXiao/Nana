"""Gradio Web UI 事件处理器纯函数单测。

不起 gradio 服务；用 stub agent 验 respond / new_conversation / switch_user /
format_turn_trace / format_error_trace / render_trace_log 这些纯函数把"输入
状态 → 新状态"映射对。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest

from agent.core import RunResult, Step, ToolInvocation
from agent.memory.events import MemoryEvent
from scripts.webui import (
    EMPTY_TRACE_MD,
    FALLBACK_REPLY,
    USERS,
    format_error_trace,
    format_turn_trace,
    new_conversation,
    render_trace_log,
    respond,
    switch_user,
)


# ---------- stub agent ----------


@dataclass
class _FakeRunResult:
    """respond 路径里 agent.run_routed 的返回；与 RunResult 对齐最小字段。"""

    final: str | None
    messages: list[dict]
    # trace 所需字段，默认值让既有测试无需关心
    intent: str = "other"
    route_path: str = "default"
    confidence: float = 0.0
    memory_events: list = field(default_factory=list)
    steps: list = field(default_factory=list)


@dataclass
class _StubAgent:
    """记录 run_routed / save_long_term 的调用；不连 LLM。"""

    reply: str | None = "ok"
    raise_exc: Exception | None = None
    run_calls: list[dict] = field(default_factory=list)
    save_calls: list[tuple[str, list[dict]]] = field(default_factory=list)

    def run_routed(
        self, msg: str, user_id: str, history: list[dict] | None = None,
    ) -> _FakeRunResult:
        self.run_calls.append({"msg": msg, "user_id": user_id, "history": history})
        if self.raise_exc is not None:
            raise self.raise_exc
        # 模拟 agent 内部维护：在 prev_history 基础上把本轮拼进去
        new_hist = list(history or []) + [
            {"role": "user", "content": msg},
            {"role": "assistant", "content": self.reply or ""},
        ]
        return _FakeRunResult(final=self.reply, messages=new_hist)

    def save_long_term(self, user_id: str, history: list[dict]) -> list[Any]:
        self.save_calls.append((user_id, list(history)))
        return []


# ---------- respond ----------


def test_respond_appends_user_and_assistant_bubbles():
    agent = _StubAgent(reply="已为您查到订单 O001…")
    out, display, state, _trace = respond(
        "查我的订单", [], None, "U001（张三）", agent=agent,
    )
    assert out == ""  # 输入框清空
    assert len(display) == 2
    assert display[0] == {"role": "user", "content": "查我的订单"}
    assert display[1] == {"role": "assistant", "content": "已为您查到订单 O001…"}
    assert state["user_id"] == "U001"
    assert state["history"] is not None
    # agent 被调用且 user_id 映射正确
    assert agent.run_calls[0]["user_id"] == "U001"
    assert agent.run_calls[0]["msg"] == "查我的订单"


def test_respond_passes_existing_history_to_agent():
    """多轮：prev session_state.history 必须被带到下一轮 run_routed 里。"""
    agent = _StubAgent(reply="好的")
    seed_hist = [{"role": "system", "content": "sys"}]
    initial_state = {"history": seed_hist, "user_id": "U001"}
    out, display, state, _trace = respond(
        "第二轮", [], initial_state, "U001（张三）", agent=agent,
    )
    assert agent.run_calls[0]["history"] == seed_hist
    # 更新后的 history 已替换（本测试里 stub 实现返回 prev+2）
    assert len(state["history"]) == 3


def test_respond_maps_user_label_to_id():
    agent = _StubAgent()
    respond("hi", [], None, "U002（李四）", agent=agent)
    assert agent.run_calls[0]["user_id"] == "U002"


def test_respond_empty_message_no_call():
    agent = _StubAgent()
    out, display, state, trace = respond("   ", [], None, "U001（张三）", agent=agent)
    assert agent.run_calls == []
    assert display == []
    assert state == {}  # 不变
    assert trace == ""  # 空消息不追加 trace


def test_respond_agent_exception_becomes_error_bubble():
    agent = _StubAgent(raise_exc=RuntimeError("boom"))
    out, display, state, trace = respond("x", [], None, "U001（张三）", agent=agent)
    assert "[内部异常]" in display[-1]["content"]
    assert "RuntimeError" in display[-1]["content"]
    # trace 走 error 分支
    assert "[error]" in trace
    assert "RuntimeError" in trace and "boom" in trace


def test_respond_fallback_when_final_none():
    agent = _StubAgent(reply=None)  # agent 未收敛
    out, display, state, _trace = respond("x", [], None, "U001（张三）", agent=agent)
    assert display[-1]["content"] == FALLBACK_REPLY


def test_respond_preserves_previous_display_turns():
    agent = _StubAgent(reply="r2")
    prev_display = [
        {"role": "user", "content": "q1"},
        {"role": "assistant", "content": "r1"},
    ]
    _, display, _, _ = respond("q2", prev_display, None, "U001（张三）", agent=agent)
    assert [m["content"] for m in display] == ["q1", "r1", "q2", "r2"]


def test_respond_returns_nonempty_trace_on_success():
    """正常路径：trace_text 非空、含 route 行（更细断言留给 format_turn_trace 测试）。"""
    agent = _StubAgent(reply="ok")
    _, _, _, trace = respond("hi", [], None, "U001（张三）", agent=agent)
    assert trace  # 非空
    assert "[route]" in trace


# ---------- new_conversation ----------


def test_new_conversation_flushes_to_long_term_and_clears_display():
    agent = _StubAgent()
    state = {"history": [{"role": "user", "content": "old"}], "user_id": "U001"}
    display, new_state = new_conversation(state, agent=agent)
    assert display == []
    assert new_state["history"] is None
    assert new_state["user_id"] == "U001"  # user 保留
    # 长期记忆被写入
    assert agent.save_calls == [("U001", [{"role": "user", "content": "old"}])]


def test_new_conversation_noop_if_no_history():
    agent = _StubAgent()
    state = {"history": None, "user_id": "U001"}
    display, new_state = new_conversation(state, agent=agent)
    assert display == []
    assert new_state["history"] is None
    assert agent.save_calls == []


def test_new_conversation_noop_if_state_missing():
    agent = _StubAgent()
    display, new_state = new_conversation(None, agent=agent)
    assert display == []
    assert agent.save_calls == []


# ---------- switch_user ----------


def test_switch_user_flushes_and_changes_user():
    agent = _StubAgent()
    state = {"history": [{"role": "user", "content": "x"}], "user_id": "U001"}
    display, new_state = switch_user("U002（李四）", state, agent=agent)
    assert display == []
    assert new_state["user_id"] == "U002"
    assert new_state["history"] is None
    assert agent.save_calls == [("U001", [{"role": "user", "content": "x"}])]


def test_switch_user_without_prior_history_no_save():
    agent = _StubAgent()
    state = {"history": None, "user_id": "U001"}
    display, new_state = switch_user("U002（李四）", state, agent=agent)
    assert new_state["user_id"] == "U002"
    assert agent.save_calls == []


# ---------- USERS 映射 ----------


def test_users_constant_shape():
    """UI 依赖的 label→id 映射不能写错：都有 Uxxx id、label 包含括号。"""
    for label, uid in USERS:
        assert uid.startswith("U")
        assert "（" in label and "）" in label


@pytest.mark.parametrize("label,uid", USERS)
def test_respond_for_each_user(label, uid):
    agent = _StubAgent()
    respond("hi", [], None, label, agent=agent)
    assert agent.run_calls[-1]["user_id"] == uid


# ---------- format_turn_trace ----------


def _make_result(
    *,
    intent: str = "other",
    route_path: str = "llm",
    confidence: float = 0.8,
    memory_events: list | None = None,
    steps: list | None = None,
    system_content: str | None = "SYS PROMPT BODY",
) -> RunResult:
    """构造一个最小 RunResult 用于 trace 渲染测试。"""
    messages: list[dict] = []
    if system_content is not None:
        messages.append({"role": "system", "content": system_content})
    return RunResult(
        final="ok",
        messages=messages,
        steps=steps or [],
        intent=intent,
        route_path=route_path,
        confidence=confidence,
        memory_events=memory_events or [],
    )


def test_format_turn_trace_contains_route_line():
    md = format_turn_trace(_make_result(intent="order_query", confidence=0.92))
    assert "[route]" in md
    assert "order_query" in md
    assert "0.92" in md


def test_format_turn_trace_exposed_tools_for_known_intent():
    """order_query 有具体 exposed_tools 子集，渲染应列出名字。"""
    md = format_turn_trace(_make_result(intent="order_query"))
    assert "[exposed_tools]" in md
    assert "query_orders" in md  # order_query 暴露这个工具


def test_format_turn_trace_exposed_tools_other_fallback():
    """other 兜底：POLICY_MAP[other].exposed_tools is None → 渲染 "ALL"。"""
    md = format_turn_trace(_make_result(intent="other"))
    assert "ALL" in md


def test_format_turn_trace_fast_path_skips_system_dump():
    """escalation 走 fast_path：不应把 system prompt 打出来，应显示 fast_path 行。"""
    md = format_turn_trace(
        _make_result(intent="escalation", system_content="some-sys"),
    )
    assert "[fast_path]" in md
    assert "escalate_to_human" in md
    assert "some-sys" not in md  # fast_path 不回显 system


def test_format_turn_trace_omits_memory_section():
    """记忆事件不单列——长期记忆已嵌在 system prompt 里，单独一栏会视觉重复。"""
    evs = [
        MemoryEvent(kind="load_long_term", detail="命中 2 段历史"),
        MemoryEvent(kind="compress", detail="summary"),
    ]
    md = format_turn_trace(_make_result(memory_events=evs))
    assert "[memory]" not in md
    assert "load_long_term" not in md


def test_format_turn_trace_renders_tool_calls_with_args_and_result():
    step = Step(
        assistant_message={"role": "assistant", "content": ""},
        invocations=[
            ToolInvocation(
                call_id="c1",
                name="query_orders",
                arguments={"status": "shipped"},
                result={"ok": True, "data": {"count": 2, "orders": [
                    {"order_id": "O001"}, {"order_id": "O002"},
                ]}},
            ),
        ],
    )
    md = format_turn_trace(_make_result(steps=[step]))
    assert "[tool]" in md
    assert "query_orders" in md
    assert "shipped" in md
    assert "count=2" in md
    assert "O001" in md and "O002" in md


def test_format_turn_trace_tool_error_summary():
    step = Step(
        assistant_message={"role": "assistant", "content": ""},
        invocations=[
            ToolInvocation(
                call_id="c1",
                name="initiate_refund",
                arguments={"order_id": "O999"},
                result={
                    "ok": False,
                    "error_code": "NOT_FOUND",
                    "message": "订单不存在",
                },
            ),
        ],
    )
    md = format_turn_trace(_make_result(steps=[step]))
    assert "ERR NOT_FOUND" in md


def test_format_turn_trace_system_dump_as_code_fence():
    """system prompt 用 code fence 包裹——防止内嵌的 markdown（如 `## 2026-..`
    被渲成 h2 巨字、空行被吞出大缝）。code 外观由 CSS 压平为"无背景 + 同字号"。"""
    md = format_turn_trace(_make_result(
        intent="order_query", system_content="Line1\n## Heading inside\n\nLine2",
    ))
    assert "[system prompt]" in md
    assert "```" in md  # 用 fence 隔离 markdown 解析
    assert "Line1" in md and "Line2" in md
    assert "## Heading inside" in md  # 原文保留，不被解析成 heading


# ---------- format_error_trace ----------


def test_format_error_trace_contains_type_and_message():
    md = format_error_trace(RuntimeError("boom"))
    assert "[error]" in md
    assert "RuntimeError" in md
    assert "boom" in md


# ---------- render_trace_log ----------


def test_render_trace_log_empty_placeholder():
    assert render_trace_log([]) == EMPTY_TRACE_MD


def test_render_trace_log_multiple_turns_numbered():
    md = render_trace_log(["alpha", "beta", "gamma"])
    assert "Turn 1" in md
    assert "Turn 2" in md
    assert "Turn 3" in md
    assert "alpha" in md and "beta" in md and "gamma" in md
    # 分节用 --- 分隔
    assert md.count("---") == 2
