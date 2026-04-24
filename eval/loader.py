"""评测样本加载器：yaml → 校验 → 内存 Sample 结构。

E1 的职责只是把 `eval/datasets/*.yaml` 规整成 Python 对象并做完整性校验：
- id 全局唯一
- intent ∈ 已知意图集合（POLICY_MAP ∪ {"injection"}）
- user_id ∈ {"U001", "U002"}（与 seed 数据对齐）
- expected_tools / forbidden_tools 中出现的工具 ∈ 5 工具预算
- turns 非空

E2 harness 再在本模块基础上跑样本；Judge / 红队指标稍后各自扩。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agent.router import POLICY_MAP

_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"

# 5 工具预算（CLAUDE.md §5）。在此显式列出而不是从 registry 导入——
# 评测集是独立事实源，工具名变更需要同步更新样本、这里的硬列表正是告警开关。
_KNOWN_TOOLS: frozenset[str] = frozenset({
    "query_orders",
    "query_product",
    "query_policy",
    "initiate_refund",
    "escalate_to_human",
})

# 路由可能的意图结果 + guardrails 短路的 "injection"
_KNOWN_INTENTS: frozenset[str] = frozenset(POLICY_MAP.keys()) | frozenset({"injection"})

_KNOWN_USERS: frozenset[str] = frozenset({"U001", "U002"})


@dataclass(frozen=True)
class Preconditions:
    """样本运行前需要打桩的状态。v1 只支持长期记忆块注入。"""

    long_term_blocks: tuple[str, ...] = ()


@dataclass(frozen=True)
class Sample:
    """单条评测样本。字段顺序与 yaml 对齐，方便 side-by-side 阅读。"""

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


class EvalDatasetError(ValueError):
    """样本集加载/校验异常。"""


def _coerce_tuple(value: Any, field_name: str, sample_id: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise EvalDatasetError(
            f"[{sample_id}] {field_name} 必须是 list，实际 {type(value).__name__}"
        )
    for item in value:
        if not isinstance(item, str):
            raise EvalDatasetError(
                f"[{sample_id}] {field_name} 元素必须是 str，出现 {type(item).__name__}: {item!r}"
            )
    return tuple(value)


def _parse_preconditions(raw: Any, sample_id: str) -> Preconditions:
    if raw is None:
        return Preconditions()
    if not isinstance(raw, dict):
        raise EvalDatasetError(f"[{sample_id}] preconditions 必须是 mapping")
    blocks = _coerce_tuple(raw.get("long_term_blocks"), "preconditions.long_term_blocks", sample_id)
    return Preconditions(long_term_blocks=blocks)


def _parse_sample(raw: dict, scenario: str) -> Sample:
    sample_id = raw.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise EvalDatasetError(f"[{scenario}] 发现缺失 id 的样本: {raw!r}")

    intent = raw.get("intent")
    if intent not in _KNOWN_INTENTS:
        raise EvalDatasetError(
            f"[{sample_id}] intent={intent!r} 不在已知集合 {sorted(_KNOWN_INTENTS)}"
        )

    user_id = raw.get("user_id")
    if user_id not in _KNOWN_USERS:
        raise EvalDatasetError(
            f"[{sample_id}] user_id={user_id!r} 不在 {sorted(_KNOWN_USERS)}（与 seed.py 对齐）"
        )

    turns = _coerce_tuple(raw.get("turns"), "turns", sample_id)
    if not turns:
        raise EvalDatasetError(f"[{sample_id}] turns 不能为空")

    expected_tools = _coerce_tuple(raw.get("expected_tools"), "expected_tools", sample_id)
    forbidden_tools = _coerce_tuple(raw.get("forbidden_tools"), "forbidden_tools", sample_id)
    for tool in (*expected_tools, *forbidden_tools):
        if tool not in _KNOWN_TOOLS:
            raise EvalDatasetError(
                f"[{sample_id}] 工具名 {tool!r} 不在 5 工具预算 {sorted(_KNOWN_TOOLS)}"
            )

    notes = raw.get("notes") or ""
    if not isinstance(notes, str):
        raise EvalDatasetError(f"[{sample_id}] notes 必须是 str")

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
    )


def load_dataset_file(path: Path) -> list[Sample]:
    """加载单个 yaml 文件并返回样本列表。"""
    with path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    scenario = data.get("scenario")
    if not isinstance(scenario, str) or not scenario:
        raise EvalDatasetError(f"{path.name}: 顶层必须有字符串字段 scenario")

    raw_samples = data.get("samples") or []
    if not isinstance(raw_samples, list):
        raise EvalDatasetError(f"{path.name}: samples 必须是 list")

    return [_parse_sample(raw, scenario) for raw in raw_samples]


def load_all(root: Path | None = None) -> list[Sample]:
    """加载 `eval/datasets/` 下全部 yaml；全局 id 去重。"""
    root = root or _DATASETS_DIR
    if not root.is_dir():
        raise EvalDatasetError(f"数据集目录不存在: {root}")

    samples: list[Sample] = []
    for path in sorted(root.glob("*.yaml")):
        samples.extend(load_dataset_file(path))

    seen: dict[str, str] = {}
    for s in samples:
        if s.id in seen:
            raise EvalDatasetError(
                f"样本 id 冲突: {s.id!r} 同时出现在 {seen[s.id]!r} 和 {s.scenario!r}"
            )
        seen[s.id] = s.scenario

    return samples


def group_by_scenario(samples: list[Sample]) -> dict[str, list[Sample]]:
    """按 scenario 分桶，便于 harness 按剧本报表。"""
    groups: dict[str, list[Sample]] = {}
    for s in samples:
        groups.setdefault(s.scenario, []).append(s)
    return groups
