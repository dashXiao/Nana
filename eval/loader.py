"""Load YAML evaluation datasets and validate their schema.

E1 converts eval/datasets/*.yaml into immutable Sample objects. Validate globally
unique IDs, known intents (including injection), seeded user IDs, the five supported
tools, and nonempty turns. E2 executes these samples; judge and red-team reporting
build on the same data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent.router import POLICY_MAP

_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

# List the five tools explicitly rather than importing the registry.
# The evaluation dataset is an independent contract; tool renames require coordinated updates.
_KNOWN_TOOLS: frozenset[str] = frozenset({
    "query_orders",
    "query_product",
    "query_policy",
    "initiate_refund",
    "escalate_to_human",
})

# Router intents plus the injection guardrail short-circuit label.
_KNOWN_INTENTS: frozenset[str] = frozenset(POLICY_MAP.keys()) | frozenset({"injection"})

_KNOWN_USERS: frozenset[str] = frozenset({"U001", "U002"})

# E5 attack-family labels used by eval/red_team.py for per-family block rates.
_KNOWN_ATTACK_FAMILIES: frozenset[str] = frozenset({
    "role_hijack",             # Role hijacking: ignore previous instructions
    "prompt_leak",             # System prompt disclosure: reveal or repeat your prompt
    "jailbreak_token",         # Public jailbreak templates: DAN, godmode, developer mode, roleplay
    "privilege_escalation",    # Unauthorized access: impersonation or refunds for another user
    "policy_fabrication",      # Policy fabrication: endorse false rules or answer without retrieving policy
})


@dataclass(frozen=True)
class Preconditions:
    """State to prepare before a sample runs; currently supports long-term memory blocks."""

    long_term_blocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class Sample:
    """One evaluation sample, with fields ordered to match the YAML schema."""

    id: str
    scenario: str
    intent: str
    user_id: str
    turns: tuple[str, ...]
    expected_tools: tuple[str, ...] = ()
    expected_phrases: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    forbidden_phrases: tuple[str, ...] = ()
    notes: str = ""
    preconditions: Preconditions = field(default_factory=Preconditions)
    attack_family: str | None = None


class EvalDatasetError(ValueError):
    """Dataset loading or validation error."""


def _coerce_tuple(value: Any, field_name: str, sample_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise EvalDatasetError(
            f"[{sample_id}] {field_name} must be a list; got {type(value).__name__}"
        )
    for item in value:
        if not isinstance(item, str):
            raise EvalDatasetError(
                f"[{sample_id}] {field_name} elements must be strings; got {type(item).__name__}: {item!r}"
            )
    return tuple(value)


def _parse_preconditions(raw: Any, sample_id: str) -> Preconditions:
    if raw is None:
        return Preconditions()
    if not isinstance(raw, dict):
        raise EvalDatasetError(f"[{sample_id}] preconditions must be a mapping")
    blocks = _coerce_tuple(raw.get("long_term_blocks"), "preconditions.long_term_blocks", sample_id)
    return Preconditions(long_term_blocks=blocks)


def _parse_sample(raw: dict, scenario: str) -> Sample:
    sample_id = raw.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise EvalDatasetError(f"[{scenario}] sample is missing an id: {raw!r}")

    intent = raw.get("intent")
    if intent not in _KNOWN_INTENTS:
        raise EvalDatasetError(
            f"[{sample_id}] intent={intent!r} is not in the known set {sorted(_KNOWN_INTENTS)}"
        )

    user_id = raw.get("user_id")
    if user_id not in _KNOWN_USERS:
        raise EvalDatasetError(
            f"[{sample_id}] user_id={user_id!r} is not in {sorted(_KNOWN_USERS)} (must match seed.py)"
        )

    turns = _coerce_tuple(raw.get("turns"), "turns", sample_id)
    if not turns:
        raise EvalDatasetError(f"[{sample_id}] turns must not be empty")

    expected_tools = _coerce_tuple(raw.get("expected_tools"), "expected_tools", sample_id)
    forbidden_tools = _coerce_tuple(raw.get("forbidden_tools"), "forbidden_tools", sample_id)
    for tool in (*expected_tools, *forbidden_tools):
        if tool not in _KNOWN_TOOLS:
            raise EvalDatasetError(
                f"[{sample_id}] tool name {tool!r} is not among the five supported tools {sorted(_KNOWN_TOOLS)}"
            )

    notes = raw.get("notes") or ""
    if not isinstance(notes, str):
        raise EvalDatasetError(f"[{sample_id}] notes must be a string")

    attack_family = raw.get("attack_family")
    if attack_family is not None:
        if not isinstance(attack_family, str):
            raise EvalDatasetError(f"[{sample_id}] attack_family must be a string")
        if attack_family not in _KNOWN_ATTACK_FAMILIES:
            raise EvalDatasetError(
                f"[{sample_id}] attack_family={attack_family!r} is not in the known set "
                f"{sorted(_KNOWN_ATTACK_FAMILIES)}"
            )

    return Sample(
        id=sample_id,
        scenario=scenario,
        intent=intent,
        user_id=user_id,
        turns=turns,
        expected_tools=expected_tools,
        expected_phrases=_coerce_tuple(raw.get("expected_phrases"), "expected_phrases", sample_id),
        forbidden_tools=forbidden_tools,
        forbidden_phrases=_coerce_tuple(raw.get("forbidden_phrases"), "forbidden_phrases", sample_id),
        notes=notes,
        preconditions=_parse_preconditions(raw.get("preconditions"), sample_id),
        attack_family=attack_family,
    )


def load_dataset_file(path: Path) -> list[Sample]:
    """Load one YAML file and return its samples."""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    scenario = data.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise EvalDatasetError(f"{path.name}: top level must contain a string scenario field")

    raw_samples = data.get("samples") or []
    if not isinstance(raw_samples, list):
        raise EvalDatasetError(f"{path.name}: samples must be a list")

    return [_parse_sample(raw, scenario) for raw in raw_samples]


def load_all(root: Path | None = None) -> list[Sample]:
    """Load all dataset YAML files and reject duplicate sample IDs globally."""
    root = root or _DATASETS_DIR
    if not root.is_dir():
        raise EvalDatasetError(f"Dataset directory does not exist: {root}")

    samples: list[Sample] = []
    for path in sorted(root.glob("*.yaml")):
        samples.extend(load_dataset_file(path))

    seen: dict[str, str] = {}
    for s in samples:
        if s.id in seen:
            raise EvalDatasetError(
                f"Duplicate sample id: {s.id!r} appears in both {seen[s.id]!r} and {s.scenario!r}"
            )
        seen[s.id] = s.scenario

    return samples


def group_by_scenario(samples: list[Sample]) -> dict[str, list[Sample]]:
    """Group samples by scenario for harness reporting."""
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.scenario, []).append(s)
    return groups
