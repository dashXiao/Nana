"""E3 Judge 纯函数单测：prompt 构造 / JSON 解析 / 聚合 / 边界。

通过 mock chat_fn 完全绕开 LLM，零成本。真实端到端留给 `eval/judge.py` CLI。
"""

from __future__ import annotations

import json

import pytest

from eval.judge import (
    DimAgg,
    DimScore,
    JudgeResult,
    JudgeRun,
    _filter_samples,
    aggregate,
    build_user_prompt,
    format_table,
    judge_sample,
    parse_judge_response,
)


# ---------- parse_judge_response ----------


def _raw(faith: int = 5, helpful: int = 4, tone: int = 3) -> str:
    return json.dumps({
        "faithfulness": {"score": faith, "reasoning": "all cited"},
        "helpfulness": {"score": helpful, "reasoning": "mostly answered"},
        "safety_tone": {"score": tone, "reasoning": "neutral"},
    })


def test_parse_clean_json():
    run = parse_judge_response(_raw(5, 4, 3))
    assert run.faithfulness.score == 5
    assert run.helpfulness.score == 4
    assert run.safety_tone.score == 3
    assert run.faithfulness.reasoning == "all cited"


def test_parse_code_fence_stripped():
    text = f"```json\n{_raw(4, 4, 4)}\n```"
    run = parse_judge_response(text)
    assert run.faithfulness.score == 4


def test_parse_extra_preamble_ignored():
    text = f"我认为如下：\n{_raw(3, 3, 3)}\n希望有帮助。"
    run = parse_judge_response(text)
    assert run.helpfulness.score == 3


def test_parse_missing_dim_raises():
    bad = json.dumps({"faithfulness": {"score": 5, "reasoning": "x"}})
    with pytest.raises(ValueError, match="helpfulness"):
        parse_judge_response(bad)


def test_parse_score_out_of_range_raises():
    bad = json.dumps({
        "faithfulness": {"score": 6, "reasoning": "x"},
        "helpfulness": {"score": 4, "reasoning": "x"},
        "safety_tone": {"score": 3, "reasoning": "x"},
    })
    with pytest.raises(ValueError, match="faithfulness"):
        parse_judge_response(bad)


def test_parse_non_integer_score_raises():
    bad = json.dumps({
        "faithfulness": {"score": "five", "reasoning": "x"},
        "helpfulness": {"score": 4, "reasoning": "x"},
        "safety_tone": {"score": 3, "reasoning": "x"},
    })
    with pytest.raises(ValueError):
        parse_judge_response(bad)


def test_parse_no_braces_raises():
    with pytest.raises(ValueError, match="没找到 JSON"):
        parse_judge_response("just prose no json")


# ---------- build_user_prompt ----------


def _fake_sample(
    sample_id: str = "s1",
    scenario: str = "refund_flow",
    final: str = "已退款",
    tool_calls: list[dict] | None = None,
    turns: list[str] | None = None,
) -> dict:
    s = {
        "sample_id": sample_id,
        "scenario": scenario,
        "final_answer": final,
        "tool_calls": tool_calls or [],
    }
    if turns is not None:
        s["_turns"] = turns
    return s


def test_prompt_includes_all_sections():
    sample = _fake_sample(
        final="您好，已退款。",
        tool_calls=[{"name": "query_orders",
                     "arguments": {"user_id": "U001"},
                     "result": {"ok": True, "data": {"orders": []}}}],
        turns=["帮我退款"],
    )
    text = build_user_prompt(sample)
    assert "refund_flow" in text
    assert "帮我退款" in text
    assert "query_orders" in text
    assert "U001" in text
    assert "您好，已退款。" in text


def test_prompt_no_tool_calls_branch():
    sample = _fake_sample(tool_calls=[], turns=["hi"])
    text = build_user_prompt(sample)
    assert "(无工具调用)" in text


def test_prompt_no_turns_fallback():
    sample = _fake_sample(tool_calls=[])  # no _turns key
    text = build_user_prompt(sample)
    assert "turns 未附带" in text


def test_prompt_truncates_long_results():
    sample = _fake_sample(
        tool_calls=[{"name": "query_policy",
                     "arguments": {},
                     "result": {"ok": True, "data": {"content": "x" * 5000}}}],
        turns=["q"],
    )
    text = build_user_prompt(sample)
    assert "已截断" in text


# ---------- judge_sample（mock chat） ----------


class _MockResp:
    def __init__(self, text: str):
        self.choices = [type("C", (), {"message": type("M", (), {"content": text})})]


def _chat_returning(*texts: str):
    """构造一个按 call-count 轮流返回不同文本的 chat_fn。"""
    calls = iter(texts)

    def chat(messages, model=None, **kwargs):  # noqa: ARG001
        return _MockResp(next(calls))

    return chat


def test_judge_sample_k1():
    sample = _fake_sample(turns=["hi"])
    chat = _chat_returning(_raw(5, 5, 5))
    r = judge_sample(sample, chat_fn=chat, k=1)
    assert r.error is None
    assert len(r.runs) == 1
    assert r.runs[0].faithfulness.score == 5


def test_judge_sample_k3_averages():
    sample = _fake_sample(turns=["hi"])
    chat = _chat_returning(_raw(5, 5, 5), _raw(3, 3, 3), _raw(4, 4, 4))
    r = judge_sample(sample, chat_fn=chat, k=3)
    assert r.mean("faithfulness") == pytest.approx(4.0)
    assert r.stdev("faithfulness") > 0


def test_judge_sample_records_error_on_bad_response():
    sample = _fake_sample(turns=["hi"])
    chat = _chat_returning("definitely not json")
    r = judge_sample(sample, chat_fn=chat, k=1)
    assert r.error is not None
    assert r.runs == []


def test_judge_sample_stops_at_first_parse_error():
    """K=3 时若第二次解析失败，不吞前面已成功的 runs。"""
    sample = _fake_sample(turns=["hi"])
    chat = _chat_returning(_raw(5, 5, 5), "not json", _raw(4, 4, 4))
    r = judge_sample(sample, chat_fn=chat, k=3)
    assert r.error is not None
    # 中断前已解析成功的那一轮应该还在
    assert len(r.runs) == 1


# ---------- 聚合 ----------


def _result(sid: str, scenario: str, *scores: tuple[int, int, int]) -> JudgeResult:
    r = JudgeResult(sample_id=sid, scenario=scenario)
    for f, h, t in scores:
        r.runs.append(JudgeRun(
            faithfulness=DimScore(f, ""),
            helpfulness=DimScore(h, ""),
            safety_tone=DimScore(t, ""),
        ))
    return r


def test_aggregate_overall_and_per_scenario_means():
    results = [
        _result("a", "x", (5, 5, 5)),
        _result("b", "x", (3, 3, 3)),
        _result("c", "y", (1, 1, 1)),
    ]
    agg = aggregate(results)
    assert agg["x"]["faithfulness"].mean == pytest.approx(4.0)
    assert agg["x"]["faithfulness"].n == 2
    assert agg["y"]["faithfulness"].mean == pytest.approx(1.0)
    assert agg["_overall"]["faithfulness"].mean == pytest.approx(3.0)
    assert agg["_overall"]["faithfulness"].n == 3


def test_aggregate_multisample_flattens_runs():
    r = _result("a", "x", (5, 5, 5), (3, 3, 3))  # K=2
    agg = aggregate([r])
    # 两个 run 都进桶
    assert agg["x"]["faithfulness"].n == 2
    assert agg["x"]["faithfulness"].mean == pytest.approx(4.0)


def test_aggregate_skips_errored():
    r = JudgeResult(sample_id="x", scenario="s", error="oops")
    agg = aggregate([r])
    # 空桶也应存在，但 n=0
    assert agg["_overall"]["faithfulness"].n == 0


# ---------- format_table ----------


def test_format_table_contains_scenarios_and_overall():
    results = [_result("a", "x", (5, 5, 5)), _result("b", "y", (1, 1, 1))]
    table = format_table(aggregate(results))
    assert "OVERALL" in table
    assert "x" in table
    assert "y" in table


# ---------- _filter_samples ----------


def test_filter_by_ids():
    samples = [
        {"sample_id": "a", "scenario": "x"},
        {"sample_id": "b", "scenario": "y"},
    ]
    got = _filter_samples(samples, ids=["b"], scenarios=[])
    assert got == [{"sample_id": "b", "scenario": "y"}]


def test_filter_by_scenarios():
    samples = [
        {"sample_id": "a", "scenario": "x"},
        {"sample_id": "b", "scenario": "y"},
    ]
    got = _filter_samples(samples, ids=[], scenarios=["x"])
    assert got == [{"sample_id": "a", "scenario": "x"}]


def test_filter_no_match_returns_empty():
    samples = [{"sample_id": "a", "scenario": "x"}]
    assert _filter_samples(samples, ids=["z"], scenarios=[]) == []


# ---------- JudgeResult 均值 / 方差 ----------


def test_result_mean_stdev_single_run():
    r = _result("a", "x", (5, 5, 5))
    assert r.mean("faithfulness") == 5
    assert r.stdev("faithfulness") == 0


def test_result_mean_stdev_none_when_no_runs():
    r = JudgeResult(sample_id="a", scenario="x")
    assert r.mean("faithfulness") is None
    assert r.stdev("faithfulness") is None
