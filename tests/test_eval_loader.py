"""评测数据集 loader 测试：schema 完整性 + 真实 yaml 能加载。

不测语义正确性（那是 E2–E5 harness 要覆盖的事），只保证：
- 六个剧本 yaml 都能解析
- 全局 id 无冲突
- intent / user_id / 工具名的值域校验能在非法值上正确抛错
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from eval.loader import (
    EvalDatasetError,
    Sample,
    group_by_scenario,
    load_all,
    load_dataset_file,
)


def test_all_six_scenarios_loadable():
    samples = load_all()
    groups = group_by_scenario(samples)
    assert set(groups.keys()) == {
        "angry_chase",
        "refund_flow",
        "policy_inquiry",
        "product_inquiry",
        "returning_customer",
        "adversarial",
    }
    # v1 首批：每剧本 ≥ 3 条
    for scenario, bucket in groups.items():
        assert len(bucket) >= 3, f"{scenario} 样本数 {len(bucket)} < 3"


def test_global_ids_unique():
    samples = load_all()
    ids = [s.id for s in samples]
    assert len(ids) == len(set(ids))


def test_sample_fields_are_frozen_tuples():
    """契约不变式：Sample 字段全是 tuple / str，避免调用方无意修改。"""
    samples = load_all()
    s = samples[0]
    assert isinstance(s, Sample)
    assert isinstance(s.turns, tuple)
    assert isinstance(s.expected_tools, tuple)
    assert isinstance(s.forbidden_tools, tuple)


def test_rejects_unknown_intent(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({
            "scenario": "bad",
            "samples": [{
                "id": "bad-01",
                "intent": "fly_to_moon",
                "user_id": "U001",
                "turns": ["hi"],
            }],
        }, allow_unicode=True),
        encoding="utf-8",
    )
    with pytest.raises(EvalDatasetError, match="intent"):
        load_dataset_file(bad)


def test_rejects_unknown_user(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({
            "scenario": "bad",
            "samples": [{
                "id": "bad-01",
                "intent": "other",
                "user_id": "U999",
                "turns": ["hi"],
            }],
        }, allow_unicode=True),
        encoding="utf-8",
    )
    with pytest.raises(EvalDatasetError, match="user_id"):
        load_dataset_file(bad)


def test_rejects_unknown_tool(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({
            "scenario": "bad",
            "samples": [{
                "id": "bad-01",
                "intent": "refund_request",
                "user_id": "U001",
                "turns": ["hi"],
                "expected_tools": ["hack_the_planet"],
            }],
        }, allow_unicode=True),
        encoding="utf-8",
    )
    with pytest.raises(EvalDatasetError, match="工具名"):
        load_dataset_file(bad)


def test_rejects_empty_turns(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({
            "scenario": "bad",
            "samples": [{
                "id": "bad-01",
                "intent": "other",
                "user_id": "U001",
                "turns": [],
            }],
        }, allow_unicode=True),
        encoding="utf-8",
    )
    with pytest.raises(EvalDatasetError, match="turns"):
        load_dataset_file(bad)


def test_duplicate_ids_across_files_rejected(tmp_path: Path):
    for name in ("a.yaml", "b.yaml"):
        (tmp_path / name).write_text(
            yaml.safe_dump({
                "scenario": name.removesuffix(".yaml"),
                "samples": [{
                    "id": "dup-01",
                    "intent": "other",
                    "user_id": "U001",
                    "turns": ["hi"],
                }],
            }, allow_unicode=True),
            encoding="utf-8",
        )
    with pytest.raises(EvalDatasetError, match="id 冲突"):
        load_all(tmp_path)


def test_preconditions_long_term_blocks_parsed():
    """老客复购剧本应当带 preconditions.long_term_blocks。"""
    samples = load_all()
    returning = [s for s in samples if s.scenario == "returning_customer"]
    assert all(len(s.preconditions.long_term_blocks) >= 1 for s in returning), \
        "returning_customer 每条样本都应声明长期记忆前置"


def test_refund_flow_covers_positive_and_negative():
    """退货剧本既要有走 initiate_refund 的正例，也要有不调 initiate_refund 的负例。"""
    samples = load_all()
    refund = [s for s in samples if s.scenario == "refund_flow"]
    positives = [s for s in refund if "initiate_refund" in s.expected_tools]
    negatives = [s for s in refund if "initiate_refund" in s.forbidden_tools]
    assert positives, "缺少退款正例"
    assert negatives, "缺少退款负例（超期 / 已退 等）"


def test_adversarial_covers_injection_and_cross_user():
    """恶意剧本要同时覆盖 guardrails 注入 + 工具层越权两条防御。"""
    samples = load_all()
    adv = [s for s in samples if s.scenario == "adversarial"]
    assert any(s.intent == "injection" for s in adv)
    assert any(s.intent == "refund_request" and s.user_id == "U001"
               and any("O002" in t for t in s.turns)
               for s in adv), "缺少 U001 尝试退 O002 的越权样本"
