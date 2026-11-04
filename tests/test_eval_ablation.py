"""Pure E4 tests for configuration selection, ForcedOtherRouter, comparison tables, and usage wrappers; no LLM calls."""

from __future__ import annotations

import pytest

from agent.llm import LIGHT_MODEL, MODEL
from eval.ablation import (
    CONFIGS,
    AblationConfig,
    ConfigResult,
    _select_configs,
    format_comparison_table,
)
from eval.harness import (
    ScenarioAgg,
    SampleResult,
    TokenUsage,
    _ForcedOtherRouter,
    _make_capturing_chat,
)
from eval.judge import DimAgg, DimScore, JudgeResult, JudgeRun


# ---------- _ForcedOtherRouter ----------


def test_forced_other_router_always_returns_other():
    router = _ForcedOtherRouter(chat_fn=lambda **_: None)
    # Return other for every input, including ordinary rule matches.
    for msg in ("Talk to a human", "Hello", "Please refund O001", "", "Garbage store"):
        decision = router.classify(msg)
        assert decision.intent == "other"
        assert decision.confidence == 1.0
        assert decision.route_path == "forced"


def test_forced_other_router_never_calls_llm():
    """ForcedOtherRouter must skip the LLM; verify with a chat stub that raises."""

    def boom(**_):
        raise AssertionError("ForcedOtherRouter must not call chat")

    router = _ForcedOtherRouter(chat_fn=boom)
    # boom() fails if an LLM branch is accidentally used.
    router.classify("This compound query normally requires LLM classification")


# ---------- _make_capturing_chat force_model ----------


class _FakeUsage:
    def __init__(self, p, c, t):
        self.prompt_tokens = p
        self.completion_tokens = c
        self.total_tokens = t


class _FakeResp:
    def __init__(self, usage=None):
        self.usage = usage
        self.choices = []


def test_capturing_chat_force_model_overrides_downstream(monkeypatch):
    """force_model must override the model passed to real_chat."""
    import agent.llm

    seen: list[str] = []

    def fake_chat(messages, tools=None, model=None, **kwargs):
        seen.append(model)
        return _FakeResp(_FakeUsage(10, 5, 15))

    monkeypatch.setattr(agent.llm, "chat", fake_chat)

    usage: dict[str, TokenUsage] = {}
    chat = _make_capturing_chat(usage, force_model=MODEL)
    chat(messages=[], model=LIGHT_MODEL)  # Request LIGHT_MODEL, but force the main model.
    assert seen == [MODEL]
    # Attribute usage to the main-model bucket too.
    assert MODEL in usage
    assert usage[MODEL].total == 15
    assert LIGHT_MODEL not in usage


def test_capturing_chat_no_force_preserves_original_model(monkeypatch):
    import agent.llm

    seen: list[str] = []

    def fake_chat(messages, tools=None, model=None, **kwargs):
        seen.append(model)
        return _FakeResp(_FakeUsage(3, 2, 5))

    monkeypatch.setattr(agent.llm, "chat", fake_chat)

    usage: dict[str, TokenUsage] = {}
    chat = _make_capturing_chat(usage)  # force_model=None
    chat(messages=[], model=LIGHT_MODEL)
    chat(messages=[], model=MODEL)
    assert seen == [LIGHT_MODEL, MODEL]
    # Count usage independently for both models.
    assert usage[LIGHT_MODEL].total == 5
    assert usage[MODEL].total == 5


# ---------- _select_configs ----------


def test_select_configs_default_returns_all():
    got = _select_configs([])
    assert [c.name for c in got] == [c.name for c in CONFIGS]


def test_select_configs_by_name():
    got = _select_configs(["baseline", "force-main"])
    assert [c.name for c in got] == ["baseline", "force-main"]


def test_select_configs_unknown_raises():
    with pytest.raises(SystemExit, match="Unknown configurations"):
        _select_configs(["not-a-config"])


def test_configs_build_kwargs_match_design():
    """The three configurations select tiered models, a single model, and full context respectively."""
    by_name = {c.name: c for c in CONFIGS}
    assert by_name["baseline"].build_kwargs == {}
    assert by_name["force-main"].build_kwargs == {"force_model": MODEL}
    assert by_name["full-context"].build_kwargs == {"force_intent_other": True}


# ---------- format_comparison_table ----------


def _pass_sample(sid: str) -> SampleResult:
    return SampleResult(
        sample_id=sid, scenario="x", turns_count=1,
        intent_expected="refund_request", intent_observed="refund_request",
        intent_match=True,
        tools_expected=(), tools_forbidden=(), tools_observed=(),
        tools_expected_ok=True, tools_ordered_ok=True, tools_forbidden_ok=True,
        final_answer="ok",
        phrases_expected_missing=(), phrases_forbidden_hit=(),
        phrases_expected_ok=True, phrases_forbidden_ok=True,
        latency_s=1.0,
        tokens_by_model={MODEL: TokenUsage(100, 50, 150)},
    )


def _mk_config_result(
    name: str,
    *,
    passed: int = 1,
    samples: int = 1,
    faith: float = 5.0,
    help_: float = 4.0,
    tone: float = 5.0,
    main_total: int = 150,
    light_total: int = 0,
) -> ConfigResult:
    cfg = AblationConfig(name=name, build_kwargs={}, description=f"{name} desc")
    overall = ScenarioAgg()
    overall.samples = samples
    overall.passed = passed
    overall.intent_hits = passed
    overall.tools_expected_hits = passed
    overall.tools_forbidden_clean = passed
    overall.phrases_expected_hits = passed
    overall.phrases_forbidden_clean = passed
    overall.latency_total_s = 1.0 * samples
    if main_total:
        overall.tokens[MODEL] = TokenUsage(main_total // 2, main_total // 2, main_total)
    if light_total:
        overall.tokens[LIGHT_MODEL] = TokenUsage(light_total // 2, light_total // 2, light_total)

    dims = {
        "faithfulness": DimAgg(mean=faith, n=samples),
        "helpfulness": DimAgg(mean=help_, n=samples),
        "safety_tone": DimAgg(mean=tone, n=samples),
    }
    return ConfigResult(
        config=cfg,
        sample_results=[_pass_sample(f"{name}-s{i}") for i in range(samples)],
        agent_agg={"_overall": overall},
        judge_results=[],
        judge_agg={"_overall": dims},
        judge_tokens=TokenUsage(10, 10, 20),
    )


def test_format_comparison_table_contains_all_configs():
    results = [
        _mk_config_result("baseline", passed=3, samples=3,
                          main_total=300, light_total=100, faith=4.8),
        _mk_config_result("force-main", passed=3, samples=3,
                          main_total=500, light_total=0, faith=4.8),
        _mk_config_result("full-context", passed=2, samples=3,
                          main_total=800, light_total=100, faith=4.2),
    ]
    table = format_comparison_table(results)
    # All three configuration names appear.
    for n in ("baseline", "force-main", "full-context"):
        assert n in table
    # Both table headers appear.
    assert "E2" in table and "E3" in table
    # Compare key columns.
    assert "pass" in table
    assert "faith" in table
    assert "tok_main" in table
    # Check for a cost column.
    assert "¥" in table
    # Judge overhead section
    assert "Judge overhead" in table


def test_format_comparison_table_handles_missing_judge_bucket():
    """A failed configuration with no overall judge dimensions must still render."""
    r = _mk_config_result("baseline", passed=1, samples=1)
    r.judge_agg = {}  # Clear manually.
    table = format_comparison_table([r])
    assert "no judge runs" in table


# ---------- ConfigResult data shape ----------


def test_config_result_defaults():
    """ConfigResult initializes judge_tokens as an empty TokenUsage."""
    cfg = AblationConfig(name="t", build_kwargs={}, description="")
    r = ConfigResult(
        config=cfg, sample_results=[], agent_agg={},
        judge_results=[], judge_agg={},
    )
    assert r.judge_tokens.total == 0
