"""IntentRouter 单元测试。

规则层走真实正则，LLM 分类层用脚本化 chat_fn 替身——不发真实请求。
既覆盖成功路径，也覆盖所有能想到的降级路径（格式错 / 网络错 / 低置信度 / 未知标签）。
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.router import (
    POLICY_MAP,
    VALID_INTENTS,
    IntentDecision,
    IntentRouter,
    _parse_intent_json,
)


def _fake_chat(content: str):
    """返回一个 chat_fn：无论怎么调，都吐出预置 content。"""

    def _fn(messages, tools=None, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    return _fn


def _raising_chat(exc: Exception):
    def _fn(messages, tools=None, **_kwargs):
        raise exc

    return _fn


# ============ 规则层 ============


def test_rule_matches_escalation_variants():
    router = IntentRouter(chat_fn=_fake_chat(""))
    for msg in ["转人工", "人工", "转客服", "找真人", "转人工！", "人工?"]:
        d = router.classify(msg)
        assert d.intent == "escalation", f"{msg!r} 应命中 escalation，实际 {d}"
        assert d.route_path == "rules"
        assert d.confidence == 1.0


def test_rule_matches_chitchat_variants():
    router = IntentRouter(chat_fn=_fake_chat(""))
    for msg in ["你好", "您好", "hi", "Hello", "哈喽", "在吗", "谢谢", "早上好"]:
        d = router.classify(msg)
        assert d.intent == "chitchat", f"{msg!r} 应命中 chitchat，实际 {d}"
        assert d.route_path == "rules"


def test_rule_matches_negative_emotion_variants():
    """强负面情绪关键词内容匹配即命中，无需整串。"""
    router = IntentRouter(chat_fn=_fake_chat(""))
    cases = [
        "你们就是骗子！",
        "这什么破服务，气死我了",
        "再不发货我就投诉到消协",
        "我要起诉你们",
        "差评！退款都不给退",
        "滚，别烦我",
        "你们这店简直垃圾",
    ]
    for msg in cases:
        d = router.classify(msg)
        assert d.intent == "negative_emotion", f"{msg!r} 应命中 negative_emotion，实际 {d}"
        assert d.route_path == "rules"
        assert d.confidence == 1.0


def test_rule_escalation_takes_precedence_over_emotion():
    """'转人工' 整串精确命中优先于情绪模糊匹配——保留 intent 来源语义。"""
    router = IntentRouter(chat_fn=_fake_chat(""))
    d = router.classify("转人工")
    assert d.intent == "escalation"


def test_rule_does_not_greedily_match_compound_input():
    """整串匹配哨卡：带业务诉求的句子不能被 chitchat/escalation 吞掉。"""
    # 需要 LLM 返回一个合法结果，否则会降级到 default
    router = IntentRouter(
        chat_fn=_fake_chat('{"intent": "order_query", "confidence": 0.9}')
    )
    for msg in ["你好我想查 O001", "转人工帮我看下订单", "hello 有货吗"]:
        d = router.classify(msg)
        assert d.route_path != "rules", f"{msg!r} 不应命中规则，实际 {d}"


def test_empty_input_falls_back_to_default():
    router = IntentRouter(chat_fn=_fake_chat(""))
    d = router.classify("")
    assert d == IntentDecision("other", 0.0, "default")
    d = router.classify("   ")
    assert d.intent == "other" and d.route_path == "default"


# ============ LLM 分类层 ============


def test_llm_classification_success():
    router = IntentRouter(
        chat_fn=_fake_chat('{"intent": "refund_request", "confidence": 0.88}')
    )
    d = router.classify("O001 帮我退了")
    assert d.intent == "refund_request"
    assert d.route_path == "llm"
    assert d.confidence == 0.88


def test_llm_classifies_negative_emotion_when_rules_miss():
    """规则词表未覆盖的情绪表述，LLM 层仍能归 negative_emotion。"""
    router = IntentRouter(
        chat_fn=_fake_chat('{"intent": "negative_emotion", "confidence": 0.92}')
    )
    d = router.classify("你们的服务实在太让我失望了，不想再说什么了")
    assert d.intent == "negative_emotion"
    assert d.route_path == "llm"


def test_llm_low_confidence_degrades_to_default():
    """即便 LLM 给出了合法 intent，confidence < 阈值也必须降级。"""
    router = IntentRouter(
        chat_fn=_fake_chat('{"intent": "order_query", "confidence": 0.3}'),
        confidence_threshold=0.6,
    )
    d = router.classify("呃呃呃")
    assert d.intent == "other"
    assert d.route_path == "default"
    assert d.confidence == 0.0


def test_llm_unknown_intent_label_falls_back():
    router = IntentRouter(
        chat_fn=_fake_chat('{"intent": "ufo_sighting", "confidence": 0.95}')
    )
    d = router.classify("我要退货")
    assert d.intent == "other"
    assert d.route_path == "default"


def test_llm_malformed_json_falls_back():
    router = IntentRouter(chat_fn=_fake_chat("not-json-at-all{"))
    d = router.classify("我要退货")
    assert d.intent == "other"
    assert d.route_path == "default"


def test_llm_markdown_wrapped_json_still_parses():
    """LLM 偶尔不听话把 JSON 套进 ```json ... ```，容错应拆包裹。"""
    router = IntentRouter(
        chat_fn=_fake_chat('```json\n{"intent": "policy_question", "confidence": 0.9}\n```')
    )
    d = router.classify("几天能退")
    assert d.intent == "policy_question"
    assert d.route_path == "llm"


def test_llm_raises_exception_degrades_to_default():
    """网络 / 配额等异常不得抛到调用方。"""
    router = IntentRouter(chat_fn=_raising_chat(RuntimeError("boom")))
    d = router.classify("随便什么话")
    assert d.intent == "other"
    assert d.route_path == "default"


# ============ POLICY_MAP 形态断言 ============


def test_policy_map_covers_all_valid_intents():
    """不允许 classify 出的意图在 POLICY_MAP 里查无此值。"""
    missing = VALID_INTENTS - set(POLICY_MAP.keys())
    assert not missing, f"POLICY_MAP 缺意图：{missing}"


def test_escalation_policy_is_fast_path():
    p = POLICY_MAP["escalation"]
    assert p.fast_path_tool == "escalate_to_human"
    assert p.final_template == "已转接人工"
    assert p.exposed_tools == ()


def test_negative_emotion_policy_is_fast_path():
    """和 escalation 行为同构，intent 独立供 W5 分桶。"""
    p = POLICY_MAP["negative_emotion"]
    assert p.fast_path_tool == "escalate_to_human"
    assert p.final_template == "已转接人工"
    assert p.exposed_tools == ()


def test_chitchat_policy_has_empty_tool_subset():
    """chitchat 必须暴露零工具——防止 LLM 对打招呼意外调 query_orders。"""
    p = POLICY_MAP["chitchat"]
    assert p.exposed_tools == ()
    assert p.fast_path_tool is None
    assert p.prompt_patch  # 非空


def test_other_policy_is_fallback_all_tools():
    """other 必须暴露全部工具（用 None 表达），等同未路由时的行为。"""
    p = POLICY_MAP["other"]
    assert p.exposed_tools is None
    assert p.fast_path_tool is None


# ============ _parse_intent_json 边界 ============


def test_parse_intent_json_accepts_int_confidence():
    # 模型可能写成 1 或 0，需要能转 float
    assert _parse_intent_json('{"intent": "chitchat", "confidence": 1}') == ("chitchat", 1.0)


def test_parse_intent_json_clamps_out_of_range():
    assert _parse_intent_json('{"intent": "x", "confidence": 1.5}')[1] == 1.0
    assert _parse_intent_json('{"intent": "x", "confidence": -0.3}')[1] == 0.0


def test_parse_intent_json_rejects_missing_fields():
    assert _parse_intent_json('{"intent": "chitchat"}') is None
    assert _parse_intent_json('{"confidence": 0.9}') is None


def test_parse_intent_json_rejects_non_object():
    assert _parse_intent_json("[1,2,3]") is None
    assert _parse_intent_json('"hello"') is None
