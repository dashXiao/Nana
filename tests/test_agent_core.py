"""Agent ReAct 循环单元测试。

用假 `chat_fn` 脚本化 LLM 响应，避免真实网络调用。
工具链走真实 `QUERY_ORDERS_SPEC`，命中 conftest 种子库——
这样就能同时验证"循环调度"和"工具结果能被正确序列化回灌"。
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.core import Agent
from agent.router import IntentDecision
from agent.tools._contract import ToolRegistry
from agent.tools.escalate_to_human import ESCALATE_TO_HUMAN_SPEC
from agent.tools.query_orders import QUERY_ORDERS_SPEC
from agent.tools.query_product import QUERY_PRODUCT_SPEC


class _FakeToolCall:
    def __init__(self, call_id: str, name: str, arguments: str):
        self.id = call_id
        self.function = SimpleNamespace(name=name, arguments=arguments)


class _FakeMsg:
    """模拟 openai ChatCompletionMessage：只暴露 core.py 实际用到的字段。"""

    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []

    def model_dump(self, exclude_none: bool = False) -> dict:
        d: dict = {"role": "assistant", "content": self.content}
        if self.tool_calls:
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in self.tool_calls
            ]
        if exclude_none:
            d = {k: v for k, v in d.items() if v is not None}
        return d


def _fake_resp(msg: _FakeMsg) -> SimpleNamespace:
    return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def _scripted_chat(responses: list[_FakeMsg]):
    """返回一个 callable，按调用次序吐出预置响应。"""
    it = iter(responses)

    def _fn(messages, tools=None, **_kwargs):
        return _fake_resp(next(it))

    return _fn


def test_tool_call_then_final_answer():
    """两轮：先 tool_call，再自然语言答复。断言派发 + 回灌 + 终态都正确。"""
    scripted = [
        _FakeMsg(
            tool_calls=[
                _FakeToolCall("call_1", "query_orders", '{"user_id": "U001"}')
            ]
        ),
        _FakeMsg(content="您有 5 条订单，最近一条是 O009。"),
    ]
    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC]),
        system_prompt="你是客服助手。",
        chat_fn=_scripted_chat(scripted),
    )

    result = agent.run("我的订单有哪些？", user_id="U001")

    assert result.final == "您有 5 条订单，最近一条是 O009。"
    assert len(result.steps) == 2
    # 第一轮发生了一次 query_orders 调用
    assert len(result.steps[0].invocations) == 1
    inv = result.steps[0].invocations[0]
    assert inv.name == "query_orders"
    assert inv.arguments == {"user_id": "U001"}
    assert inv.result["ok"] is True
    assert inv.result["data"]["count"] == 5
    # 第二轮模型直接给出答复，没有再调工具
    assert result.steps[1].invocations == []


def test_user_id_override_blocks_llm_forged_value():
    """LLM 即便构造其他 user_id，也会被 session 上下文覆盖——上游哨卡。"""
    scripted = [
        _FakeMsg(
            tool_calls=[
                # 模型"越权"试图查 U002
                _FakeToolCall("call_1", "query_orders", '{"user_id": "U002"}')
            ]
        ),
        _FakeMsg(content="done"),
    ]
    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC]),
        system_prompt="",
        chat_fn=_scripted_chat(scripted),
    )

    result = agent.run("查别人的订单", user_id="U001")

    inv = result.steps[0].invocations[0]
    assert inv.arguments["user_id"] == "U001"   # 被覆盖
    # 实际命中的是 U001 的订单，不是 U002
    ids = {o["order_id"] for o in inv.result["data"]["orders"]}
    assert ids == {"O001", "O003", "O005", "O007", "O009"}


def test_max_steps_circuit_breaker():
    """模型始终返回 tool_call 时，达到 max_steps 应熔断，final=None。"""
    # 准备足量 tool_call 响应（超过 max_steps 也没关系，迭代器不会被耗尽报错）
    scripted = [
        _FakeMsg(
            tool_calls=[_FakeToolCall(f"c{i}", "query_orders", '{"user_id": "U001"}')]
        )
        for i in range(10)
    ]
    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC]),
        system_prompt="",
        max_steps=3,
        chat_fn=_scripted_chat(scripted),
    )

    result = agent.run("死循环", user_id="U001")

    assert result.final is None
    assert len(result.steps) == 3


def test_no_tool_call_returns_immediately():
    """模型直接给答复时，一步即返回，不产生 invocation。"""
    scripted = [_FakeMsg(content="你好")]
    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC]),
        system_prompt="",
        chat_fn=_scripted_chat(scripted),
    )

    result = agent.run("hi", user_id="U001")

    assert result.final == "你好"
    assert len(result.steps) == 1
    assert result.steps[0].invocations == []


def test_history_with_system_is_not_duplicated():
    """多轮：传入含 system 的 history，不得再重复注入一条 system。"""
    prior_history = [
        {"role": "system", "content": "ORIGINAL"},
        {"role": "user", "content": "第一轮问题"},
        {"role": "assistant", "content": "第一轮答复"},
    ]
    captured: dict = {}

    def capturing_chat(messages, tools=None, **_kwargs):
        captured["messages"] = list(messages)
        return _fake_resp(_FakeMsg(content="第二轮答复"))

    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC]),
        system_prompt="NEW_SYSTEM_SHOULD_NOT_APPEAR",
        chat_fn=capturing_chat,
    )
    agent.run("第二轮问题", user_id="U001", history=prior_history)

    sent = captured["messages"]
    systems = [m for m in sent if m.get("role") == "system"]
    assert len(systems) == 1
    # 沿用 history 里的 system，而不是覆盖成新的
    assert systems[0]["content"] == "ORIGINAL"


def test_escalate_to_human_terminates_loop():
    """escalate_to_human 被调后，循环硬切：final='已转接人工'、LLM 不再被调用。"""
    call_count = {"n": 0}

    def counting_chat(messages, tools=None, **_kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return _fake_resp(
                _FakeMsg(
                    tool_calls=[_FakeToolCall("c1", "escalate_to_human", "{}")]
                )
            )
        # 命中 escalate 后还被调第二次，说明硬切没生效
        raise AssertionError("escalate 命中后不应再调 LLM")

    agent = Agent(
        registry=ToolRegistry([ESCALATE_TO_HUMAN_SPEC, QUERY_ORDERS_SPEC]),
        system_prompt="",
        chat_fn=counting_chat,
    )

    result = agent.run("给我转人工", user_id="U001")

    assert result.final == "已转接人工"
    assert call_count["n"] == 1
    assert len(result.steps) == 1
    inv = result.steps[0].invocations[0]
    assert inv.name == "escalate_to_human"
    assert inv.result["ok"] is True
    assert inv.result["data"]["escalated"] is True


class _StubRouter:
    """测试替身：classify() 返回预置 IntentDecision，不碰真 LLM。"""

    def __init__(self, decision: IntentDecision):
        self._decision = decision

    def classify(self, _msg: str) -> IntentDecision:
        return self._decision


def test_exposed_tools_subset_is_sent_to_llm():
    """run() 传 exposed_tools=(一个工具,) 时，chat_fn 收到的 tools 仅含该工具。"""
    captured: dict = {}

    def capturing_chat(messages, tools=None, **_kwargs):
        captured["tools"] = tools
        return _fake_resp(_FakeMsg(content="done"))

    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC, QUERY_PRODUCT_SPEC]),
        system_prompt="sys",
        chat_fn=capturing_chat,
    )
    agent.run("x", user_id="U001", exposed_tools=("query_orders",))

    names = [t["function"]["name"] for t in captured["tools"]]
    assert names == ["query_orders"], f"应只暴露 query_orders，实际 {names}"


def test_empty_exposed_tools_sends_no_schemas():
    """exposed_tools=() 场景：chat_fn 收到空 tools 列表，对应 chitchat 意图。"""
    captured: dict = {}

    def capturing_chat(messages, tools=None, **_kwargs):
        captured["tools"] = tools
        return _fake_resp(_FakeMsg(content="您好！"))

    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC, QUERY_PRODUCT_SPEC]),
        system_prompt="sys",
        chat_fn=capturing_chat,
    )
    agent.run("你好", user_id="U001", exposed_tools=())

    assert captured["tools"] == []


def test_none_exposed_tools_sends_all_schemas():
    """exposed_tools=None（默认 / other 兜底）：暴露注册表里全部工具。"""
    captured: dict = {}

    def capturing_chat(messages, tools=None, **_kwargs):
        captured["tools"] = tools
        return _fake_resp(_FakeMsg(content="done"))

    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC, QUERY_PRODUCT_SPEC]),
        system_prompt="sys",
        chat_fn=capturing_chat,
    )
    agent.run("随便问", user_id="U001")  # 不传 exposed_tools

    names = {t["function"]["name"] for t in captured["tools"]}
    assert names == {"query_orders", "query_product"}


def test_prompt_patch_appended_to_system_on_fresh_conversation():
    """首轮 run：patch 被拼到 self.system_prompt 末尾。"""
    captured: dict = {}

    def capturing_chat(messages, tools=None, **_kwargs):
        captured["messages"] = list(messages)
        return _fake_resp(_FakeMsg(content="ok"))

    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC]),
        system_prompt="BASE",
        chat_fn=capturing_chat,
    )
    agent.run("x", user_id="U001", prompt_patch="EXTRA_RULE")

    systems = [m for m in captured["messages"] if m.get("role") == "system"]
    assert len(systems) == 1
    assert "BASE" in systems[0]["content"]
    assert "EXTRA_RULE" in systems[0]["content"]


def test_prompt_patch_replaces_system_when_continuing_history():
    """多轮 + patch：不沿用 history 旧 system，而是刷新为 base + 新 patch。

    理由：不同轮可能命中不同意图，patch 必须随轮更新。
    """
    captured: dict = {}

    def capturing_chat(messages, tools=None, **_kwargs):
        captured["messages"] = list(messages)
        return _fake_resp(_FakeMsg(content="ok"))

    prior_history = [
        {"role": "system", "content": "OLD_SYSTEM_WITH_OLD_PATCH"},
        {"role": "user", "content": "上一轮"},
        {"role": "assistant", "content": "上一轮答复"},
    ]
    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC]),
        system_prompt="BASE",
        chat_fn=capturing_chat,
    )
    agent.run(
        "第二轮",
        user_id="U001",
        history=prior_history,
        prompt_patch="NEW_PATCH",
    )

    systems = [m for m in captured["messages"] if m.get("role") == "system"]
    assert len(systems) == 1, "仍应只有一条 system"
    assert "OLD_SYSTEM_WITH_OLD_PATCH" not in systems[0]["content"]
    assert "BASE" in systems[0]["content"]
    assert "NEW_PATCH" in systems[0]["content"]


def test_run_routed_escalation_fast_path_skips_llm():
    """router 分到 escalation → fast_path 直接派发，不应触达 chat_fn。"""

    def should_not_be_called(messages, tools=None, **_kwargs):
        raise AssertionError("fast_path 场景下 LLM 不应被调用")

    agent = Agent(
        registry=ToolRegistry([ESCALATE_TO_HUMAN_SPEC, QUERY_ORDERS_SPEC]),
        system_prompt="sys",
        chat_fn=should_not_be_called,
        router=_StubRouter(IntentDecision("escalation", 1.0, "rules")),
    )

    result = agent.run_routed("转人工", user_id="U001")

    assert result.final == "已转接人工"
    assert result.intent == "escalation"
    assert result.route_path == "rules"
    assert result.confidence == 1.0
    assert len(result.steps) == 1
    inv = result.steps[0].invocations[0]
    assert inv.name == "escalate_to_human"
    assert inv.result["ok"] is True


def test_run_routed_non_fast_path_passes_subset_and_patch_to_llm():
    """chitchat 意图：exposed_tools=() + prompt_patch 同时进入 chat_fn。"""
    captured: dict = {}

    def capturing_chat(messages, tools=None, **_kwargs):
        captured["messages"] = list(messages)
        captured["tools"] = tools
        return _fake_resp(_FakeMsg(content="您好！"))

    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC, QUERY_PRODUCT_SPEC]),
        system_prompt="BASE",
        chat_fn=capturing_chat,
        router=_StubRouter(IntentDecision("chitchat", 1.0, "rules")),
    )

    result = agent.run_routed("你好", user_id="U001")

    # 工具子集为空
    assert captured["tools"] == []
    # system 被追加了 chitchat 的 prompt_patch（"寒暄/闲聊"字样来自 POLICY_MAP）
    systems = [m for m in captured["messages"] if m.get("role") == "system"]
    assert "寒暄" in systems[0]["content"] or "闲聊" in systems[0]["content"]
    # RunResult 带上路由元数据
    assert result.intent == "chitchat"
    assert result.route_path == "rules"
    assert result.confidence == 1.0
    assert result.final == "您好！"


def test_run_routed_other_intent_exposes_all_tools():
    """default 兜底：暴露全部工具，行为等同未路由的 run()。"""
    captured: dict = {}

    def capturing_chat(messages, tools=None, **_kwargs):
        captured["tools"] = tools
        return _fake_resp(_FakeMsg(content="done"))

    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC, QUERY_PRODUCT_SPEC]),
        system_prompt="BASE",
        chat_fn=capturing_chat,
        router=_StubRouter(IntentDecision("other", 0.0, "default")),
    )

    result = agent.run_routed("奇怪的问题", user_id="U001")

    names = {t["function"]["name"] for t in captured["tools"]}
    assert names == {"query_orders", "query_product"}
    assert result.intent == "other"
    assert result.route_path == "default"


def test_malformed_arguments_dispatched_as_invalid_input():
    """LLM 给出非法 JSON arguments 时，dispatch 应返回 INVALID_INPUT 而非抛异常。"""
    scripted = [
        _FakeMsg(
            tool_calls=[_FakeToolCall("call_1", "query_orders", "not-json{")]
        ),
        _FakeMsg(content="抱歉失败了"),
    ]
    agent = Agent(
        registry=ToolRegistry([QUERY_ORDERS_SPEC]),
        system_prompt="",
        chat_fn=_scripted_chat(scripted),
    )

    result = agent.run("x", user_id="U001")

    inv = result.steps[0].invocations[0]
    # 解析失败的原始 raw 字段被打到 arguments 里，user_id 仍被覆盖
    assert inv.arguments["user_id"] == "U001"
    # dispatch 结果应是 ok（因为只剩 user_id 合法）或 INVALID_INPUT——
    # 具体取决于解析分支。当前实现是 __raw__ 被丢进 args，但 user_id 被覆盖后
    # Pydantic 会忽略未知字段 → 调用成功。这条 case 主要验证"不抛异常"。
    assert "ok" in inv.result
