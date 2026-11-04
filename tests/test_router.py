"""IntentRouter tests using real regex rules and a scripted classifier stub.

Chinese rule fixtures retain multilingual routing coverage.
No live requests. Cover successful classification and fallbacks for malformed output,
network errors, low confidence, and unknown labels.
"""

from __future__ import annotations

from types import SimpleNamespace

from agent.llm import LIGHT_MODEL
from agent.router import (
    POLICY_MAP,
    VALID_INTENTS,
    IntentDecision,
    IntentRouter,
    _parse_intent_json,
)


def _fake_chat(content: str):
    """Return a chat_fn that always produces the predefined content."""

    def _fn(messages, tools=None, **_kwargs):
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    return _fn


def _raising_chat(exc: Exception):
    def _fn(messages, tools=None, **_kwargs):
        raise exc

    return _fn


# ============ Rule layer ============


def test_rule_matches_escalation_variants():
    router = IntentRouter(chat_fn=_fake_chat(""))
    for msg in [
        "转人工", "人工", "转客服", "找真人", "转人工！", "人工?",
        "Talk to a human.", "Speak to a person", "human",
        "TRANSFER ME TO AN AGENT!", "talk to a human agent",
    ]:
        d = router.classify(msg)
        assert d.intent == "escalation", f"{msg!r} should match escalation; got {d}"
        assert d.route_path == "rules"
        assert d.confidence == 1.0


def test_rule_matches_chitchat_variants():
    router = IntentRouter(chat_fn=_fake_chat(""))
    for msg in [
        "你好", "您好", "hi", "Hello", "哈喽", "在吗", "谢谢", "早上好",
        "Hey!", "Good morning", "Thank you.",
    ]:
        d = router.classify(msg)
        assert d.intent == "chitchat", f"{msg!r} should match chitchat; got {d}"
        assert d.route_path == "rules"


def test_rule_matches_negative_emotion_variants():
    """Strong negative-emotion keywords match within a message rather than requiring a full match."""
    router = IntentRouter(chat_fn=_fake_chat(""))
    cases = [
        "你们就是骗子！",
        "这什么破服务，气死我了",
        "再不发货我就投诉到消协",
        "我要起诉你们",
        "差评！退款都不给退",
        "滚，别烦我",
        "你们这店简直垃圾",
        "You are scammers!",
        "What a garbage store",
        "I am furious about this order",
        "I want to file a complaint",
        "This is a rip-off",
    ]
    for msg in cases:
        d = router.classify(msg)
        assert d.intent == "negative_emotion", f"{msg!r} should match negative_emotion; got {d}"
        assert d.route_path == "rules"
        assert d.confidence == 1.0


def test_rule_escalation_takes_precedence_over_emotion():
    """Exact handoff requests take precedence over fuzzy emotion matching to retain intent provenance."""
    router = IntentRouter(chat_fn=_fake_chat(""))
    d = router.classify("转人工")
    assert d.intent == "escalation"


def test_rule_does_not_greedily_match_compound_input():
    """Business requests following greetings or handoff words must not be swallowed by anchored rules."""
    # Return a valid LLM result to avoid fallback to the default path.
    router = IntentRouter(
        chat_fn=_fake_chat('{"intent": "order_query", "confidence": 0.9}')
    )
    for msg in [
        "你好我想查 O001", "转人工帮我看下订单", "hello 有货吗",
        "Hello, check O001", "Talk to a human about my order",
        "Is there an issue with my order?",
    ]:
        d = router.classify(msg)
        assert d.route_path != "rules", f"{msg!r} should not match a rule; got {d}"


def test_empty_input_falls_back_to_default():
    router = IntentRouter(chat_fn=_fake_chat(""))
    d = router.classify("")
    assert d == IntentDecision("other", 0.0, "default")
    d = router.classify("   ")
    assert d.intent == "other" and d.route_path == "default"


# ============ LLM classification ============


def test_llm_classification_success():
    router = IntentRouter(
        chat_fn=_fake_chat('{"intent": "refund_request", "confidence": 0.88}')
    )
    d = router.classify("Please refund O001")
    assert d.intent == "refund_request"
    assert d.route_path == "llm"
    assert d.confidence == 0.88


def test_llm_classifies_negative_emotion_when_rules_miss():
    """The LLM can detect negative emotion absent from the rule vocabulary."""
    router = IntentRouter(
        chat_fn=_fake_chat('{"intent": "negative_emotion", "confidence": 0.92}')
    )
    d = router.classify("Your service has been deeply disappointing. I have nothing more to say.")
    assert d.intent == "negative_emotion"
    assert d.route_path == "llm"


def test_llm_low_confidence_degrades_to_default():
    """A valid intent below the confidence threshold must still fall back."""
    router = IntentRouter(
        chat_fn=_fake_chat('{"intent": "order_query", "confidence": 0.3}'),
        confidence_threshold=0.6,
    )
    d = router.classify("Um, um, um")
    assert d.intent == "other"
    assert d.route_path == "default"
    assert d.confidence == 0.0


def test_llm_unknown_intent_label_falls_back():
    router = IntentRouter(
        chat_fn=_fake_chat('{"intent": "ufo_sighting", "confidence": 0.95}')
    )
    d = router.classify("I want to return an item")
    assert d.intent == "other"
    assert d.route_path == "default"


def test_llm_malformed_json_falls_back():
    router = IntentRouter(chat_fn=_fake_chat("not-json-at-all{"))
    d = router.classify("I want to return an item")
    assert d.intent == "other"
    assert d.route_path == "default"


def test_llm_markdown_wrapped_json_still_parses():
    """Allow classifier JSON wrapped in Markdown code fences."""
    router = IntentRouter(
        chat_fn=_fake_chat('```json\n{"intent": "policy_question", "confidence": 0.9}\n```')
    )
    d = router.classify("How many days do I have to return it?")
    assert d.intent == "policy_question"
    assert d.route_path == "llm"


def test_llm_classifier_uses_light_model():
    """Classification must use LIGHT_MODEL under the two-tier model contract."""
    captured: dict = {}

    def capturing_chat(messages, tools=None, **kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content='{"intent": "order_query", "confidence": 0.9}'
            ))]
        )

    router = IntentRouter(chat_fn=capturing_chat)
    router.classify("O001 status")
    assert captured.get("model") == LIGHT_MODEL


def test_llm_raises_exception_degrades_to_default():
    """Network or quota errors must not escape to the caller."""
    router = IntentRouter(chat_fn=_raising_chat(RuntimeError("boom")))
    d = router.classify("Any message")
    assert d.intent == "other"
    assert d.route_path == "default"


# ============ POLICY_MAP contract ============


def test_policy_map_covers_all_valid_intents():
    """Every classifiable intent must exist in POLICY_MAP."""
    missing = VALID_INTENTS - set(POLICY_MAP.keys())
    assert not missing, f"POLICY_MAP is missing intents: {missing}"


def test_escalation_policy_is_fast_path():
    p = POLICY_MAP["escalation"]
    assert p.fast_path_tool == "escalate_to_human"
    assert p.final_template == "Transferred to a human agent."
    assert p.exposed_tools == ()


def test_negative_emotion_policy_is_fast_path():
    """Negative emotion shares handoff behavior but keeps its own intent for metrics."""
    p = POLICY_MAP["negative_emotion"]
    assert p.fast_path_tool == "escalate_to_human"
    assert p.final_template == "Transferred to a human agent."
    assert p.exposed_tools == ()


def test_chitchat_policy_has_empty_tool_subset():
    """Chitchat must expose no tools to prevent accidental order queries for greetings."""
    p = POLICY_MAP["chitchat"]
    assert p.exposed_tools == ()
    assert p.fast_path_tool is None
    assert p.prompt_patch  # Nonempty


def test_other_policy_is_fallback_all_tools():
    """Other exposes all tools via None, matching unrouted behavior."""
    p = POLICY_MAP["other"]
    assert p.exposed_tools is None
    assert p.fast_path_tool is None


# ============ _parse_intent_json boundaries ============


def test_parse_intent_json_accepts_int_confidence():
    # Integer confidence values must coerce to float.
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
