"""Pure tests for E2 scoring, filtering, aggregation, and cost estimates.

No end-to-end model calls; live runs belong to smoke scripts or explicit --run.
These tests protect metric definitions when scoring changes.
"""

from __future__ import annotations

from agent.llm import LIGHT_MODEL, MODEL
from eval.harness import (
    ScenarioAgg,
    SampleResult,
    TokenUsage,
    _estimate_cost,
    _filter_samples,
    _PRICING_PER_1M,
    _score_phrases,
    _score_tools,
    aggregate,
    format_table,
    score_run,
)
from eval.loader import Sample


def _make_sample(
    *,
    sample_id: str = "s1",
    scenario: str = "refund_flow",
    intent: str = "refund_request",
    user_id: str = "U001",
    turns: tuple[str, ...] = ("help",),
    expected_tools: tuple[str, ...] = (),
    expected_phrases: tuple[str, ...] = (),
    forbidden_tools: tuple[str, ...] = (),
    forbidden_phrases: tuple[str, ...] = (),
) -> Sample:
    return Sample(
        id=sample_id,
        scenario=scenario,
        intent=intent,
        user_id=user_id,
        turns=turns,
        expected_tools=expected_tools,
        expected_phrases=expected_phrases,
        forbidden_tools=forbidden_tools,
        forbidden_phrases=forbidden_phrases,
    )


# ---------- _score_tools ----------


def test_score_tools_empty_expected_always_passes():
    sample = _make_sample()
    exp_ok, ord_ok, forb_ok = _score_tools(sample, ("anything",))
    assert exp_ok and ord_ok is False and forb_ok


def test_score_tools_set_hit_but_ordered_mismatch():
    sample = _make_sample(expected_tools=("query_orders", "initiate_refund"))
    exp_ok, ord_ok, forb_ok = _score_tools(
        sample, ("initiate_refund", "query_orders")
    )
    assert exp_ok is True
    assert ord_ok is False
    assert forb_ok is True


def test_score_tools_ordered_exact_match():
    sample = _make_sample(expected_tools=("query_orders", "initiate_refund"))
    exp_ok, ord_ok, _ = _score_tools(
        sample, ("query_orders", "initiate_refund")
    )
    assert exp_ok and ord_ok


def test_score_tools_missing_expected():
    sample = _make_sample(expected_tools=("query_orders", "query_policy"))
    exp_ok, ord_ok, _ = _score_tools(sample, ("query_orders",))
    assert exp_ok is False
    assert ord_ok is False


def test_score_tools_forbidden_hit():
    sample = _make_sample(
        expected_tools=("query_orders",), forbidden_tools=("initiate_refund",),
    )
    _, _, forb_ok = _score_tools(sample, ("query_orders", "initiate_refund"))
    assert forb_ok is False


def test_score_tools_forbidden_clean():
    sample = _make_sample(forbidden_tools=("initiate_refund",))
    _, _, forb_ok = _score_tools(sample, ("query_orders",))
    assert forb_ok is True


# ---------- _score_phrases ----------


def test_score_phrases_all_found():
    sample = _make_sample(expected_phrases=("299", "50"))
    missing, hit, exp_ok, forb_ok = _score_phrases(
        sample, "Price: CNY 299; stock: 50 units"
    )
    assert missing == ()
    assert hit == ()
    assert exp_ok and forb_ok


def test_score_phrases_some_missing():
    sample = _make_sample(expected_phrases=("299", "50"))
    missing, hit, exp_ok, _ = _score_phrases(sample, "Price: CNY 299")
    assert missing == ("50",)
    assert exp_ok is False
    assert hit == ()


def test_score_phrases_forbidden_hit():
    sample = _make_sample(forbidden_phrases=("mAh", "milliamp"))
    missing, hit, _, forb_ok = _score_phrases(
        sample, "The battery is 3000 mAh with good battery life"
    )
    assert "mAh" in hit
    assert forb_ok is False


def test_score_phrases_empty_both_sides_passes():
    sample = _make_sample()
    missing, hit, exp_ok, forb_ok = _score_phrases(sample, "Any content")
    assert missing == () and hit == ()
    assert exp_ok and forb_ok


def test_score_phrases_ignores_english_capitalization():
    sample = _make_sample(
        expected_phrases=("refund",),
        forbidden_phrases=("PASSWORD",),
    )
    missing, hit, expected_ok, forbidden_ok = _score_phrases(
        sample, "Refund issued. The password is secret."
    )
    assert missing == ()
    assert hit == ("PASSWORD",)
    assert expected_ok is True
    assert forbidden_ok is False


# ---------- score_run ----------


def test_score_run_all_green():
    sample = _make_sample(
        expected_tools=("query_product",),
        expected_phrases=("299",),
    )
    r = score_run(
        sample=sample,
        observed_intent="refund_request",
        observed_tools=("query_product",),
        final_answer="Price: 299",
        latency_s=1.23,
        tokens_by_model={MODEL: TokenUsage(100, 50, 150)},
        turns_count=1,
    )
    assert r.passed is True
    assert r.intent_match is True
    assert r.tools_ordered_ok is True
    assert r.phrases_expected_ok is True
    assert r.latency_s == 1.23
    # Copy token usage deeply so results do not share mutable usage objects.
    assert r.tokens_by_model[MODEL].total == 150


def test_score_run_error_short_circuits_passed():
    sample = _make_sample()
    r = score_run(
        sample=sample,
        observed_intent="refund_request",
        observed_tools=(),
        final_answer="",
        latency_s=0.0,
        tokens_by_model={},
        turns_count=1,
        error="RuntimeError: boom",
    )
    assert r.passed is False
    assert r.error == "RuntimeError: boom"


def test_score_run_intent_mismatch_fails():
    sample = _make_sample(intent="refund_request")
    r = score_run(
        sample=sample,
        observed_intent="other",
        observed_tools=(),
        final_answer="",
        latency_s=0.0,
        tokens_by_model={},
        turns_count=1,
    )
    assert r.intent_match is False
    assert r.passed is False


# ---------- _filter_samples ----------


def test_filter_by_ids():
    samples = [
        _make_sample(sample_id="a", scenario="x"),
        _make_sample(sample_id="b", scenario="y"),
        _make_sample(sample_id="c", scenario="x"),
    ]
    got = _filter_samples(samples, ids=["a", "c"], scenarios=[])
    assert [s.id for s in got] == ["a", "c"]


def test_filter_by_scenarios():
    samples = [
        _make_sample(sample_id="a", scenario="x"),
        _make_sample(sample_id="b", scenario="y"),
        _make_sample(sample_id="c", scenario="x"),
    ]
    got = _filter_samples(samples, ids=[], scenarios=["y"])
    assert [s.id for s in got] == ["b"]


def test_filter_ids_and_scenarios_intersect():
    samples = [
        _make_sample(sample_id="a", scenario="x"),
        _make_sample(sample_id="b", scenario="y"),
        _make_sample(sample_id="c", scenario="x"),
    ]
    got = _filter_samples(samples, ids=["a", "b"], scenarios=["x"])
    assert [s.id for s in got] == ["a"]


def test_filter_no_args_returns_all():
    samples = [_make_sample(sample_id="a"), _make_sample(sample_id="b")]
    assert _filter_samples(samples, ids=[], scenarios=[]) == samples


# ---------- aggregate / format_table ----------


def _pass_result(sid: str, scenario: str = "x") -> SampleResult:
    return SampleResult(
        sample_id=sid, scenario=scenario, turns_count=1,
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


def _fail_result(sid: str, scenario: str = "x") -> SampleResult:
    r = _pass_result(sid, scenario)
    r.intent_match = False
    r.tokens_by_model = {LIGHT_MODEL: TokenUsage(80, 20, 100)}
    return r


def test_aggregate_buckets_overall_and_per_scenario():
    results = [
        _pass_result("a", "x"),
        _fail_result("b", "x"),
        _pass_result("c", "y"),
    ]
    agg = aggregate(results)
    assert set(agg.keys()) == {"_overall", "x", "y"}
    assert agg["_overall"].samples == 3
    assert agg["_overall"].passed == 2
    assert agg["x"].samples == 2
    assert agg["x"].passed == 1
    assert agg["y"].samples == 1


def test_aggregate_sums_tokens_across_models():
    results = [_pass_result("a"), _fail_result("b")]
    agg = aggregate(results)
    tokens = agg["_overall"].tokens
    assert tokens[MODEL].total == 150
    assert tokens[LIGHT_MODEL].total == 100


def test_format_table_contains_scenarios_and_overall():
    results = [_pass_result("a", "x"), _fail_result("b", "y")]
    table = format_table(aggregate(results))
    assert "OVERALL" in table
    assert "x" in table
    assert "y" in table
    assert MODEL in table


# ---------- _estimate_cost ----------


def test_estimate_cost_sums_per_model():
    """Cost sums input and output tokens times configured prices per million tokens; derive expectations from the current pricing map."""
    main_in, main_out = _PRICING_PER_1M[MODEL]
    light_in, light_out = _PRICING_PER_1M[LIGHT_MODEL]
    tokens = {
        MODEL: TokenUsage(prompt=1_000, completion=500, total=1_500),
        LIGHT_MODEL: TokenUsage(prompt=2_000, completion=1_000, total=3_000),
    }
    cost = _estimate_cost(tokens)
    expected = (
        (1_000 * main_in + 500 * main_out) / 1_000_000
        + (2_000 * light_in + 1_000 * light_out) / 1_000_000
    )
    assert abs(cost - expected) < 1e-9


def test_estimate_cost_ignores_unknown_model():
    """Unknown models contribute zero cost without raising errors."""
    tokens = {"unknown-model-xyz": TokenUsage(1_000, 500, 1_500)}
    assert _estimate_cost(tokens) == 0.0


# ---------- TokenUsage ----------


def test_token_usage_add_and_merge():
    a = TokenUsage()
    a.add(10, 5, 15)
    a.add(20, 10, 30)
    assert (a.prompt, a.completion, a.total) == (30, 15, 45)

    b = TokenUsage(5, 5, 10)
    a.merge(b)
    assert (a.prompt, a.completion, a.total) == (35, 20, 55)


def test_scenario_agg_ingest_counts_booleans():
    agg = ScenarioAgg()
    agg.ingest(_pass_result("a"))
    agg.ingest(_fail_result("b"))
    assert agg.samples == 2
    assert agg.passed == 1
    assert agg.intent_hits == 1
    assert agg.tools_expected_hits == 2
    assert agg.latency_total_s == 2.0
