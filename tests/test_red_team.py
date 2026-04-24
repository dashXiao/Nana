"""E5 红队聚合 · 纯函数测试。

不跑 harness、不起 LLM，只喂入"假 SampleResult 字典"测三分类 + 族级聚合 + 表格渲染。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from eval.red_team import (
    FamilyAgg,
    aggregate_by_family,
    classify_verdict,
    format_red_team_table,
    main,
)


def _make_result(
    sample_id: str,
    *,
    intent_observed: str,
    passed: bool,
) -> dict[str, Any]:
    """最小构造：classify_verdict 只看这两个字段。"""
    return {"sample_id": sample_id, "intent_observed": intent_observed, "passed": passed}


# ---------- classify_verdict ----------


def test_classify_guardrails_block_when_intent_injection():
    r = _make_result("rt-rh-01", intent_observed="injection", passed=True)
    assert classify_verdict(r) == "guardrails_block"


def test_classify_guardrails_wins_over_downstream_even_if_not_passed():
    """guardrails 命中时 intent 就是 injection；哪怕断言其他字段没过，归到 guard。

    这避免"规则层拦下来了但评测 yaml 写得有点瑕疵" 误入 downstream/bypass 栏——
    intent=injection 本身是 agent.core 里 guardrails 短路才会打的硬标签。"""
    r = _make_result("rt-rh-01", intent_observed="injection", passed=False)
    assert classify_verdict(r) == "guardrails_block"


def test_classify_downstream_block_when_passed_other_intent():
    r = _make_result("rt-rh-03", intent_observed="other", passed=True)
    assert classify_verdict(r) == "downstream_block"


def test_classify_bypassed_when_not_passed_non_injection():
    r = _make_result("rt-pe-01", intent_observed="refund_request", passed=False)
    assert classify_verdict(r) == "bypassed"


# ---------- aggregate_by_family ----------


def test_aggregate_ignores_samples_without_family():
    """没打 attack_family 的样本（比如跨 scenario 复用报告）不入红队表。"""
    family_map = {"rt-rh-01": "role_hijack"}
    results = [
        _make_result("rt-rh-01", intent_observed="injection", passed=True),
        _make_result("refund-01", intent_observed="refund_request", passed=True),  # 非红队
    ]
    buckets = aggregate_by_family(results, family_map)
    assert buckets["_overall"].total == 1
    assert buckets["role_hijack"].total == 1
    assert "refund_flow" not in buckets


def test_aggregate_counts_three_verdicts_correctly():
    family_map = {
        "a-guard": "role_hijack",
        "a-down": "role_hijack",
        "a-bypass": "role_hijack",
    }
    results = [
        _make_result("a-guard", intent_observed="injection", passed=True),
        _make_result("a-down", intent_observed="other", passed=True),
        _make_result("a-bypass", intent_observed="other", passed=False),
    ]
    buckets = aggregate_by_family(results, family_map)
    rh = buckets["role_hijack"]
    assert rh.total == 3
    assert rh.guardrails_block == 1
    assert rh.downstream_block == 1
    assert rh.bypassed == 1
    assert rh.bypassed_ids == ["a-bypass"]
    assert rh.intercept_rate == pytest.approx(2 / 3)


def test_aggregate_overall_sums_across_families():
    family_map = {"x": "role_hijack", "y": "privilege_escalation"}
    results = [
        _make_result("x", intent_observed="injection", passed=True),
        _make_result("y", intent_observed="refund_request", passed=False),
    ]
    buckets = aggregate_by_family(results, family_map)
    assert buckets["_overall"].total == 2
    assert buckets["_overall"].guardrails_block == 1
    assert buckets["_overall"].bypassed == 1
    assert buckets["_overall"].intercept_rate == 0.5


def test_family_agg_empty_intercept_rate_zero():
    """空族的 intercept_rate 不能除零。"""
    agg = FamilyAgg(family="empty")
    assert agg.intercept_rate == 0.0


# ---------- format_red_team_table ----------


def test_format_table_contains_headers_and_all_families():
    family_map = {
        "a": "role_hijack",
        "b": "privilege_escalation",
    }
    results = [
        _make_result("a", intent_observed="injection", passed=True),
        _make_result("b", intent_observed="refund_request", passed=True),
    ]
    buckets = aggregate_by_family(results, family_map)
    table = format_red_team_table(buckets)
    assert "family" in table and "intercept" in table
    assert "role_hijack" in table
    assert "privilege_escalation" in table
    assert "OVERALL" not in table  # 用小写 _overall 的 family 名字渲染
    # overall 行存在（通过"_overall" family 名字）
    assert "_overall" in table


def test_format_table_lists_bypassed_ids_section():
    family_map = {"x": "role_hijack", "y": "role_hijack"}
    results = [
        _make_result("x", intent_observed="other", passed=False),
        _make_result("y", intent_observed="injection", passed=True),
    ]
    buckets = aggregate_by_family(results, family_map)
    table = format_red_team_table(buckets)
    assert "绕过的样本" in table
    assert "x" in table  # bypassed id 出现
    # y 没被绕过，但 id 出现在表体里也是可能的；重点是 bypassed 区
    lines = table.splitlines()
    bypass_idx = next(i for i, line in enumerate(lines) if "绕过的样本" in line)
    bypass_block = "\n".join(lines[bypass_idx:])
    assert "x" in bypass_block
    assert "y" not in bypass_block


def test_format_table_omits_bypass_section_when_clean():
    """全部拦住时不应该冒出空标题"绕过的样本："。"""
    family_map = {"x": "role_hijack"}
    results = [_make_result("x", intent_observed="injection", passed=True)]
    buckets = aggregate_by_family(results, family_map)
    table = format_red_team_table(buckets)
    assert "绕过的样本" not in table


# ---------- main CLI ----------


def test_main_errors_on_missing_report(tmp_path: Path, capsys):
    rc = main(["--report", str(tmp_path / "nope.json")])
    assert rc == 1
    assert "报告不存在" in capsys.readouterr().err


def test_main_errors_when_report_has_no_red_team_samples(tmp_path: Path, capsys):
    """喂一份只有非红队样本的报告，应退出 1 并提示用户跑 red_team 剧本。"""
    report = tmp_path / "r.json"
    report.write_text(
        json.dumps({
            "samples": [{
                "sample_id": "refund-01",  # 真实数据集里确实没 attack_family
                "intent_observed": "refund_request",
                "passed": True,
            }]
        }),
        encoding="utf-8",
    )
    rc = main(["--report", str(report)])
    assert rc == 1
    assert "red_team" in capsys.readouterr().err


def test_main_prints_table_on_real_red_team_report(tmp_path: Path, capsys):
    """手搓一份"假装跑过"的报告（用 red_team.yaml 里真实 id）→ 应成功出表。"""
    report = tmp_path / "r.json"
    report.write_text(
        json.dumps({
            "samples": [
                {"sample_id": "rt-rh-01", "intent_observed": "injection", "passed": True},
                {"sample_id": "rt-rh-03", "intent_observed": "other", "passed": True},
                {"sample_id": "rt-pe-01", "intent_observed": "refund_request", "passed": False},
            ]
        }),
        encoding="utf-8",
    )
    rc = main(["--report", str(report)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "role_hijack" in out
    assert "privilege_escalation" in out
    assert "rt-pe-01" in out  # 被绕过的样本要被点名
