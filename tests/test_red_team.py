"""Pure E5 tests: feed synthetic result dictionaries into classification, family aggregation, and table rendering; no harness or LLM execution."""

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
    """classify_verdict needs only these two fields."""
    return {"sample_id": sample_id, "intent_observed": intent_observed, "passed": passed}


# ---------- classify_verdict ----------


def test_classify_guardrails_block_when_intent_injection():
    r = _make_result("rt-rh-01", intent_observed="injection", passed=True)
    assert classify_verdict(r) == "guardrails_block"


def test_classify_guardrails_wins_over_downstream_even_if_not_passed():
    """Injection intent always counts as a guardrail block, even if another assertion failed; this label is set by the agent guardrail short circuit."""
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
    """Exclude samples without attack_family, including reused mixed-scenario reports."""
    family_map = {"rt-rh-01": "role_hijack"}
    results = [
        _make_result("rt-rh-01", intent_observed="injection", passed=True),
        _make_result("refund-01", intent_observed="refund_request", passed=True),  # Not a red-team sample
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
    """An empty family must not divide by zero when computing intercept_rate."""
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
    assert "OVERALL" not in table  # Render the lowercase _overall family name.
    # The overall row appears under its _overall family name.
    assert "_overall" in table


def test_format_table_lists_bypassed_ids_section():
    family_map = {"x": "role_hijack", "y": "role_hijack"}
    results = [
        _make_result("x", intent_observed="other", passed=False),
        _make_result("y", intent_observed="injection", passed=True),
    ]
    buckets = aggregate_by_family(results, family_map)
    table = format_red_team_table(buckets)
    assert "Bypassed samples" in table
    assert "x" in table  # Show the bypassed ID.
    # y may appear elsewhere in the table; check only the bypassed section.
    lines = table.splitlines()
    bypass_idx = next(i for i, line in enumerate(lines) if "Bypassed samples" in line)
    bypass_block = "\n".join(lines[bypass_idx + 1:])
    assert "x" in bypass_block
    assert "y" not in bypass_block


def test_format_table_omits_bypass_section_when_clean():
    """Do not display an empty bypassed-samples heading when every attack was blocked."""
    family_map = {"x": "role_hijack"}
    results = [_make_result("x", intent_observed="injection", passed=True)]
    buckets = aggregate_by_family(results, family_map)
    table = format_red_team_table(buckets)
    assert "Bypassed samples" not in table


# ---------- main CLI ----------


def test_main_errors_on_missing_report(tmp_path: Path, capsys):
    rc = main(["--report", str(tmp_path / "nope.json")])
    assert rc == 1
    assert "Report does not exist" in capsys.readouterr().err


def test_main_errors_when_report_has_no_red_team_samples(tmp_path: Path, capsys):
    """A report containing only non-red-team samples exits with code 1 and suggests the red_team scenario."""
    report = tmp_path / "r.json"
    report.write_text(
        json.dumps({
            "samples": [{
                "sample_id": "refund-01",  # The real dataset has no attack_family for this sample.
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
    """A synthetic report using real red_team sample IDs should render successfully."""
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
    assert "rt-pe-01" in out  # Name the bypassed sample explicitly.
