"""Dataset loader tests for schema integrity and real YAML loading.

Check parsing, global ID uniqueness, and invalid intent/user/tool values.
Semantic correctness is evaluated by the E2–E5 harnesses.
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


def test_all_scenarios_loadable():
    samples = load_all()
    groups = group_by_scenario(samples)
    assert set(groups.keys()) == {
        "angry_chase",
        "refund_flow",
        "policy_inquiry",
        "product_inquiry",
        "returning_customer",
        "adversarial",
        "red_team",
        "chitchat_other",
        "abnormal_orders",
    }
    # At least three samples per scenario
    for scenario, bucket in groups.items():
        assert len(bucket) >= 3, f"{scenario} sample count {len(bucket)} < 3"


def test_global_ids_unique():
    samples = load_all()
    ids = [s.id for s in samples]
    assert len(ids) == len(set(ids))


def test_sample_fields_are_frozen_tuples():
    """Sample fields use immutable tuples and strings to prevent accidental mutation."""
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
    with pytest.raises(EvalDatasetError, match="tool name"):
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
    with pytest.raises(EvalDatasetError, match="Duplicate sample id"):
        load_all(tmp_path)


def test_preconditions_long_term_blocks_parsed():
    """Returning-customer samples must declare long-term memory preconditions."""
    samples = load_all()
    returning = [s for s in samples if s.scenario == "returning_customer"]
    assert all(len(s.preconditions.long_term_blocks) >= 1 for s in returning), \
        "Every returning_customer sample must declare long-term memory preconditions"


def test_refund_flow_covers_positive_and_negative():
    """Refund scenarios must include both successful refunds and cases that must not invoke the refund tool."""
    samples = load_all()
    refund = [s for s in samples if s.scenario == "refund_flow"]
    positives = [s for s in refund if "initiate_refund" in s.expected_tools]
    negatives = [s for s in refund if "initiate_refund" in s.forbidden_tools]
    assert positives, "Missing a successful refund case"
    assert negatives, "Missing a rejected refund case (expired or already refunded)"


def test_rejects_unknown_attack_family(tmp_path: Path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(
        yaml.safe_dump({
            "scenario": "bad",
            "samples": [{
                "id": "bad-01",
                "intent": "injection",
                "user_id": "U001",
                "turns": ["hi"],
                "attack_family": "mind_control",
            }],
        }, allow_unicode=True),
        encoding="utf-8",
    )
    with pytest.raises(EvalDatasetError, match="attack_family"):
        load_dataset_file(bad)


def test_attack_family_optional_for_non_red_team():
    """Non-red-team samples may omit attack_family."""
    samples = load_all()
    # Historical adversarial samples are unlabeled and should retain None.
    adv = [s for s in samples if s.scenario == "adversarial"]
    assert adv and all(s.attack_family is None for s in adv)
    # Every red_team sample must have an attack-family label.
    rt = [s for s in samples if s.scenario == "red_team"]
    assert rt and all(s.attack_family is not None for s in rt)


def test_red_team_covers_five_families_six_each():
    """Red-team contract: five attack families with six variants each, 30 total."""
    samples = load_all()
    rt = [s for s in samples if s.scenario == "red_team"]
    assert len(rt) == 30
    families: dict[str, int] = {}
    for s in rt:
        families[s.attack_family] = families.get(s.attack_family, 0) + 1
    assert set(families.keys()) == {
        "role_hijack",
        "prompt_leak",
        "jailbreak_token",
        "privilege_escalation",
        "policy_fabrication",
    }
    assert all(n == 6 for n in families.values())


def test_adversarial_covers_injection_and_cross_user():
    """Adversarial scenarios cover both injection filtering and cross-user authorization."""
    samples = load_all()
    adv = [s for s in samples if s.scenario == "adversarial"]
    assert any(s.intent == "injection" for s in adv)
    assert any(s.intent == "refund_request" and s.user_id == "U001"
               and any("O002" in t for t in s.turns)
               for s in adv), "Missing a cross-user case where U001 attempts to refund O002"
