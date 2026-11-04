"""E4 ablation study of model tiers and intent-specific context loading.

Configurations:
- baseline: main-model ReAct plus lightweight routing/memory and focused prompt patches.
- force-main: use the main model for all calls to measure the effect of model tiers.
- full-context: force intent=other with all rules and tools to measure focused loading.

Reuse harness.run_all and judge.run_batch for metrics and quality. Judge defaults to K=1;
--smoke runs one sample per configuration to check assembly at low cost.

CLI:
    uv run python -m eval.ablation --smoke
    uv run python -m eval.ablation --run
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


# ---------- Configuration matrix ----------


@dataclass(frozen=True)
class AblationConfig:
    name: str
    build_kwargs: dict[str, Any]
    description: str


CONFIGS: tuple[AblationConfig, ...] = (
    AblationConfig(
        name="baseline",
        build_kwargs={},
        description="Tiered models with intent-specific context loading (baseline)",
    ),
    AblationConfig(
        name="force-main",
        build_kwargs={"force_model": MODEL},
        description="Main model for all chat calls; disable model tiers",
    ),
    AblationConfig(
        name="full-context",
        build_kwargs={"force_intent_other": True},
        description="Always intent=other: full fallback rules and tools; disable focused loading",
    ),
)


# ---------- Judge chat wrapper with separate token accounting ----------


def _make_judge_chat(tokens_sink: TokenUsage) -> ChatFn:
    """Accumulate judge usage separately from the agent budget; always use the main model."""
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


# ---------- Per-configuration snapshot ----------


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

    # Harness sample dictionaries are the input format expected by the judge.
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


# ---------- Comparison tables ----------


def format_comparison_table(results: list[ConfigResult]) -> str:
    """Render configuration comparisons for E2 capability/cost and E3 quality."""
    lines: list[str] = []

    # --- E2 capability and cost ---
    lines.append("## E2 metrics (capability / cost / latency)")
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

    # --- E3 quality ---
    lines.append("")
    lines.append("## E3 judge dimensions (faith / help / tone)")
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

    # --- Evaluation overhead ---
    lines.append("")
    lines.append("## Judge overhead (excluded from agent budget)")
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


# ---------- Save reports ----------


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
        raise SystemExit(f"Unknown configurations: {missing}; available: {list(known)}")
    return [known[n] for n in names]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E4 ablation study")
    parser.add_argument("--run", action="store_true",
                        help="Execute live calls (default: dry run, list the plan only)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke check: one sample to verify all three configurations")
    parser.add_argument("--configs", nargs="*", default=[],
                        help=f"Run only selected configurations; available: {[c.name for c in CONFIGS]}")
    parser.add_argument("--scenarios", nargs="*", default=[],
                        help="Filter by scenarios; accepts multiple values; default: all samples")
    parser.add_argument("--k", type=int, default=1,
                        help="Judge runs per sample (default: 1)")
    parser.add_argument("--out", type=Path, default=None,
                        help="Report output path; default: eval/reports/{ts}-ablation.json")
    args = parser.parse_args(argv)

    samples = load_all()
    if args.scenarios:
        sc_set = set(args.scenarios)
        samples = [s for s in samples if s.scenario in sc_set]
    if args.smoke:
        samples = samples[:1]
    if not samples:
        print("No samples match the filters", file=sys.stderr)
        return 1
    selected = _select_configs(args.configs)

    print(f"Ablation: {len(selected)} configurations × {len(samples)} samples × judge K={args.k}")
    for c in selected:
        print(f"  - {c.name:<14} {c.description}")
    for s in samples:
        print(f"  sample: {s.id:<16} {s.scenario}")

    if not args.run and not args.smoke:
        print("\n[dry-run] --run executes live calls (incurs model usage costs); --smoke runs one sample")
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
    print(f"\nReport saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
