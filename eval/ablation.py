"""E4 消融实验：三配置对比，佐证"双模型分层 + 按需加载"两项架构决策。

设计要点：
- **三配置**：
  - `baseline`：分层模型（ReAct=主模型 / 路由+记忆=轻量模型） + 路由按需加载 patch
  - `force-main`：所有 chat 强制走主模型，关掉分层；预期 token 成本升、质量不升
  - `full-context`：始终判定 intent=other，走 `other` 兜底 patch（全量通用规则 + 全 5 工具）；
    预期 per-turn prompt 变长（多挂了规则/catalog），token 上升，质量不升
- **轻改 harness**：`build_agent` 新增 `force_model` / `force_intent_other`，默认不变
- **复用 E2+E3**：每配置跑 `harness.run_all` → `judge.run_batch`，对比表并列
- **成本可控**：judge 只跑 K=1；`--smoke` 用 1 条样本验装配

CLI:
    uv run python -m eval.ablation --smoke              # 冒烟：3 config × 1 sample
    uv run python -m eval.ablation --run                # 全量：3 config × 24 samples
    uv run python -m eval.ablation --run --configs baseline force-main
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from agent.llm import LIGHT_MODEL, MODEL

from eval import harness, judge
from eval.harness import SampleResult, ScenarioAgg, TokenUsage, _estimate_cost
from eval.judge import DimAgg, JudgeResult
from eval.loader import Sample, load_all

ChatFn = Callable[..., Any]


# ---------- 配置矩阵 ----------


@dataclass(frozen=True)
class AblationConfig:
    name: str
    build_kwargs: dict[str, Any]
    description: str


CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig(
        name="baseline",
        build_kwargs={},
        description="分层模型 + 路由按需加载（本方案）",
    ),
    AblationConfig(
        name="force-main",
        build_kwargs={"force_model": MODEL},
        description="所有 chat 强制走主模型，关掉双模型分层",
    ),
    AblationConfig(
        name="full-context",
        build_kwargs={"force_intent_other": True},
        description="始终 intent=other：走兜底 patch（全规则 + 全工具），关掉按需加载",
    ),
)


# ---------- Judge chat 包装（独立 token 计数） ----------


def _make_judge_chat(tokens_sink: TokenUsage) -> ChatFn:
    """Judge 的 chat 独立于 agent 预算统计；本函数只管把 judge 的用量挤到 sink。

    不切模型（judge 一律 MODEL，即主模型）。
    """
    from agent.llm import chat as real_chat

    def wrapped(messages, tools=None, model=None, **kwargs):
        resp = real_chat(messages, tools=tools, model=model or MODEL, **kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            tokens_sink.add(
                prompt=getattr(usage, "prompt_tokens", 0) or 0,
                completion=getattr(usage, "completion_tokens", 0) or 0,
                total=getattr(usage, "total_tokens", 0) or 0,
            )
        return resp

    return wrapped


# ---------- 单配置跑完的快照 ----------


@dataclass
class ConfigResult:
    config: AblationConfig
    sample_results: list[SampleResult]
    agent_agg: dict[str, ScenarioAgg]
    judge_results: list[JudgeResult]
    judge_agg: dict[str, dict[str, DimAgg]]
    judge_tokens: TokenUsage = field(default_factory=TokenUsage)


def run_config(
    config: AblationConfig,
    samples: list[Sample],
    judge_k: int = 1,
) -> ConfigResult:
    print(f"\n=== Config: {config.name} — {config.description} ===")
    sample_results = harness.run_all(samples, build_kwargs=config.build_kwargs)
    agent_agg = harness.aggregate(sample_results)

    # harness 报告的 sample dict 就是 judge 的输入格式
    sample_dicts = [harness._result_to_dict(r) for r in sample_results]

    judge_tokens = TokenUsage()
    judge_chat = _make_judge_chat(judge_tokens)

    def _progress(sid: str, r: JudgeResult) -> None:
        if r.error:
            print(f"  · judge {sid}: ERROR {r.error}")
        else:
            print(
                f"  · judge {sid}: "
                f"faith={r.mean('faithfulness'):.1f} "
                f"help={r.mean('helpfulness'):.1f} "
                f"tone={r.mean('safety_tone'):.1f}"
            )

    judge_results = judge.run_batch(
        sample_dicts, chat_fn=judge_chat, k=judge_k, progress=_progress,
    )
    judge_agg = judge.aggregate(judge_results)

    return ConfigResult(
        config=config,
        sample_results=sample_results,
        agent_agg=agent_agg,
        judge_results=judge_results,
        judge_agg=judge_agg,
        judge_tokens=judge_tokens,
    )


# ---------- 对比表 ----------


def format_comparison_table(results: list[ConfigResult]) -> str:
    """两张并排表：能力+成本（E2） / 质量（E3）。行是 config，列是指标。"""
    lines: list[str] = []

    # --- E2 能力 + 成本 ---
    lines.append("## E2 指标（能力 / 成本 / 延迟）")
    header = (
        f"{'config':<14} "
        f"{'N':>3} "
        f"{'pass':>5} "
        f"{'intent':>6} "
        f"{'tool ✓':>6} "
        f"{'tok_main':>9} "
        f"{'tok_light':>9} "
        f"{'cost':>8} "
        f"{'lat_avg':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))
    for r in results:
        overall = r.agent_agg["_overall"]
        n = overall.samples
        main_tok = overall.tokens.get(MODEL, TokenUsage()).total
        light_tok = overall.tokens.get(LIGHT_MODEL, TokenUsage()).total
        cost = _estimate_cost(overall.tokens)
        lat = overall.latency_total_s / n if n else 0.0
        lines.append(
            f"{r.config.name:<14} "
            f"{n:>3} "
            f"{_pct(overall.passed, n):>5} "
            f"{_pct(overall.intent_hits, n):>6} "
            f"{_pct(overall.tools_expected_hits, n):>6} "
            f"{main_tok:>9} "
            f"{light_tok:>9} "
            f"{'¥' + format(cost, '.4f'):>8} "
            f"{lat:6.1f}s"
        )

    # --- E3 质量 ---
    lines.append("")
    lines.append("## E3 Judge 维度（faith / help / tone）")
    header2 = (
        f"{'config':<14} "
        f"{'N':>3} "
        f"{'faith':>6} "
        f"{'help':>6} "
        f"{'tone':>6}"
    )
    lines.append(header2)
    lines.append("-" * len(header2))
    for r in results:
        dims = r.judge_agg.get("_overall", {})
        if not dims:
            lines.append(f"{r.config.name:<14} (no judge runs)")
            continue
        n = dims["faithfulness"].n
        lines.append(
            f"{r.config.name:<14} "
            f"{n:>3} "
            f"{dims['faithfulness'].mean:6.2f} "
            f"{dims['helpfulness'].mean:6.2f} "
            f"{dims['safety_tone'].mean:6.2f}"
        )

    # --- 评估自身开销 ---
    lines.append("")
    lines.append("## Judge 自身开销（不计入 agent 预算）")
    total_judge = TokenUsage()
    for r in results:
        total_judge.merge(r.judge_tokens)
    judge_cost = _estimate_cost({MODEL: total_judge})
    lines.append(
        f"judge total tokens = {total_judge.total} "
        f"(prompt={total_judge.prompt} / completion={total_judge.completion}) "
        f"≈ ¥{judge_cost:.4f}"
    )

    return "\n".join(lines)


def _pct(num: int, denom: int) -> str:
    return f"{100 * num / denom:4.0f}%" if denom else "  - "


# ---------- 报告落盘 ----------


def _config_to_dict(r: ConfigResult) -> dict[str, Any]:
    overall = r.agent_agg["_overall"]
    dims = r.judge_agg.get("_overall", {})
    return {
        "name": r.config.name,
        "description": r.config.description,
        "build_kwargs": r.config.build_kwargs,
        "e2": {
            "samples": overall.samples,
            "passed": overall.passed,
            "intent_hits": overall.intent_hits,
            "tools_expected_hits": overall.tools_expected_hits,
            "tools_forbidden_clean": overall.tools_forbidden_clean,
            "phrases_expected_hits": overall.phrases_expected_hits,
            "phrases_forbidden_clean": overall.phrases_forbidden_clean,
            "latency_total_s": overall.latency_total_s,
            "tokens": {m: dataclasses.asdict(u) for m, u in overall.tokens.items()},
            "cost": _estimate_cost(overall.tokens),
        },
        "e3": {
            "dims": {d: dataclasses.asdict(v) for d, v in dims.items()},
        },
        "judge_tokens": dataclasses.asdict(r.judge_tokens),
    }


def write_report(results: list[ConfigResult], out_path: Path, k: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "k": k,
        "configs": [_config_to_dict(r) for r in results],
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------- CLI ----------


def _select_configs(names: list[str]) -> list[AblationConfig]:
    if not names:
        return list(CONFIGS)
    known = {c.name: c for c in CONFIGS}
    missing = [n for n in names if n not in known]
    if missing:
        raise SystemExit(f"未知配置：{missing}，可选：{list(known)}")
    return [known[n] for n in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E4 消融实验")
    parser.add_argument("--run", action="store_true",
                        help="真跑（默认 dry-run，仅列计划）")
    parser.add_argument("--smoke", action="store_true",
                        help="冒烟：只用 1 条样本（快速验 3 配置的装配正确）")
    parser.add_argument("--configs", nargs="*", default=[],
                        help=f"只跑指定 config；可选：{[c.name for c in CONFIGS]}")
    parser.add_argument("--k", type=int, default=1,
                        help="Judge 每样本采样次数（默认 1）")
    parser.add_argument("--out", type=Path, default=None,
                        help="报告输出路径；默认 eval/reports/{ts}-ablation.json")
    args = parser.parse_args(argv)

    samples = load_all()
    if args.smoke:
        samples = samples[:1]
    selected = _select_configs(args.configs)

    print(f"消融配置：{len(selected)} 个 × 样本 {len(samples)} 条 × Judge K={args.k}")
    for c in selected:
        print(f"  - {c.name:<14} {c.description}")
    for s in samples:
        print(f"  sample: {s.id:<16} {s.scenario}")

    if not args.run and not args.smoke:
        print("\n[dry-run] --run 真跑（会调 LLM 产生成本） / --smoke 跑 1 条冒烟")
        return 0

    results: list[ConfigResult] = []
    for cfg in selected:
        results.append(run_config(cfg, samples, judge_k=args.k))

    print("\n" + format_comparison_table(results))

    out_path = args.out or (
        Path(__file__).resolve().parent / "reports"
        / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-ablation.json"
    )
    write_report(results, out_path, k=args.k)
    print(f"\n报告已落盘：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
