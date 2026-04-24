"""E2 自动指标 harness：跑 E1 评测集 → 采集确定性指标 → 聚合报表。

设计要点：
- **只测确定性指标**：intent accuracy / tool call accuracy / phrase match / latency / tokens。
  Faithfulness / Helpfulness 等主观维度留给 E3 LLM-as-Judge。
- **成本保护**：默认 --dry-run 只列要跑的样本；--run 才真的打 qwen-plus。
- **双模型 tokens 分桶**：包装 `agent.llm.chat` 抓 `response.usage`，按实际 model 累计，
  这是 §9 #11 "双模型分层省多少钱" 叙事的硬证据来源。
- **preconditions**：long_term_blocks 落到临时目录的 md 文件，跑完整体清理，
  不污染真实 memory/sessions/。

CLI：
    uv run python -m eval.harness                 # dry-run，仅列样本
    uv run python -m eval.harness --run           # 真跑全部 24 条
    uv run python -m eval.harness --run --scenarios refund_flow
    uv run python -m eval.harness --run --ids refund-01 refund-02 --out /tmp/r.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from agent.core import Agent, RunResult
from agent.guardrails.injection import InjectionDetector
from agent.llm import LIGHT_MODEL, MODEL
from agent.memory.long_term import LongTermMemory
from agent.memory.short_term import ShortTermMemory
from agent.router import BASE_SYSTEM_PROMPT, IntentRouter
from agent.tools._contract import ToolRegistry
from agent.tools.escalate_to_human import ESCALATE_TO_HUMAN_SPEC
from agent.tools.initiate_refund import INITIATE_REFUND_SPEC
from agent.tools.query_orders import QUERY_ORDERS_SPEC
from agent.tools.query_policy import QUERY_POLICY_SPEC
from agent.tools.query_product import QUERY_PRODUCT_SPEC

from eval.loader import Sample, group_by_scenario, load_all
from scripts.seed import seed as _seed_db


def _reseed_db_silently() -> None:
    """样本间硬重置 SQLite 到种子态：避免 refund 类样本互相污染。
    seed() 会打印摘要，这里吞掉 stdout 让 harness 输出保持整洁。"""
    with contextlib.redirect_stdout(io.StringIO()):
        _seed_db()

# qwen 系列计价（¥ / 百万 tokens，input / output）。
# 数据为 portfolio 叙事级精度，真实 billing 以 DashScope 为准。
_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    "qwen-plus": (4.0, 12.0),
    "qwen-turbo": (0.6, 2.0),
}


# ---------- 指标数据结构 ----------


@dataclass
class TokenUsage:
    prompt: int = 0
    completion: int = 0
    total: int = 0

    def add(self, prompt: int, completion: int, total: int) -> None:
        self.prompt += prompt
        self.completion += completion
        self.total += total

    def merge(self, other: "TokenUsage") -> None:
        self.prompt += other.prompt
        self.completion += other.completion
        self.total += other.total


@dataclass
class SampleResult:
    """单条样本跑完后的完整快照。"""

    sample_id: str
    scenario: str
    turns_count: int

    # 路由与工具观测
    intent_expected: str
    intent_observed: str
    intent_match: bool

    tools_expected: tuple[str, ...]
    tools_forbidden: tuple[str, ...]
    tools_observed: tuple[str, ...]
    tools_expected_ok: bool      # expected 是否全部出现（set-wise）
    tools_ordered_ok: bool       # 观测序列是否严格等于 expected 序列
    tools_forbidden_ok: bool     # forbidden 是否无一出现

    # 答复与短语命中
    final_answer: str
    phrases_expected_missing: tuple[str, ...]
    phrases_forbidden_hit: tuple[str, ...]
    phrases_expected_ok: bool
    phrases_forbidden_ok: bool

    # 工程指标
    latency_s: float
    tokens_by_model: dict[str, TokenUsage]

    # 错误兜底：异常发生时填，其他字段尽量填可读默认
    error: str | None = None

    # 完整工具调用流水：E3 Judge 判 faithfulness 时作为"agent 可见证据"
    tool_calls: tuple[dict, ...] = ()

    @property
    def passed(self) -> bool:
        """样本级 pass 判定：路由对 + 工具覆盖对 + 禁用工具干净 + 短语两侧干净 + 无异常。"""
        return (
            self.error is None
            and self.intent_match
            and self.tools_expected_ok
            and self.tools_forbidden_ok
            and self.phrases_expected_ok
            and self.phrases_forbidden_ok
        )


# ---------- Agent / chat 装配 ----------


def _make_capturing_chat(usage_by_model: dict[str, TokenUsage]):
    """包装 agent.llm.chat：保持行为不变，只把 response.usage 按 model 累加到字典。"""
    from agent.llm import chat as real_chat  # 延迟 import，避免环境变量未就绪报错

    def wrapped(messages, tools=None, model=None, **kwargs):
        resolved_model = model or MODEL
        response = real_chat(messages, tools=tools, model=model, **kwargs)
        usage = getattr(response, "usage", None)
        if usage is not None:
            bucket = usage_by_model.setdefault(resolved_model, TokenUsage())
            bucket.add(
                prompt=getattr(usage, "prompt_tokens", 0) or 0,
                completion=getattr(usage, "completion_tokens", 0) or 0,
                total=getattr(usage, "total_tokens", 0) or 0,
            )
        return response

    return wrapped


def build_agent(
    memory_dir: Path,
    usage_by_model: dict[str, TokenUsage],
) -> Agent:
    """装配评测用 Agent：5 工具 + 路由 + 双记忆 + 注入闸 + 捕获用量的 chat。"""
    chat_fn = _make_capturing_chat(usage_by_model)
    registry = ToolRegistry([
        QUERY_ORDERS_SPEC,
        QUERY_POLICY_SPEC,
        QUERY_PRODUCT_SPEC,
        INITIATE_REFUND_SPEC,
        ESCALATE_TO_HUMAN_SPEC,
    ])
    return Agent(
        registry=registry,
        system_prompt=BASE_SYSTEM_PROMPT,
        chat_fn=chat_fn,
        router=IntentRouter(chat_fn=chat_fn),
        short_term=ShortTermMemory(chat_fn=chat_fn),
        long_term=LongTermMemory(base_dir=memory_dir, chat_fn=chat_fn),
        injection_detector=InjectionDetector(),
    )


# ---------- preconditions ----------


def _seed_long_term(memory_dir: Path, user_id: str, blocks: tuple[str, ...]) -> None:
    """把样本的 long_term_blocks 写入临时 memory 文件，格式对齐 LongTermMemory.save_session。"""
    memory_dir.mkdir(parents=True, exist_ok=True)
    path = memory_dir / f"{user_id}.md"
    body = "\n".join(b.strip() for b in blocks)
    path.write_text(
        f"---\nuser_id: {user_id}\n---\n\n{body}\n",
        encoding="utf-8",
    )


def _clear_long_term(memory_dir: Path, user_id: str) -> None:
    path = memory_dir / f"{user_id}.md"
    if path.exists():
        path.unlink()


# ---------- 打分 ----------


def _score_tools(
    sample: Sample, observed: tuple[str, ...]
) -> tuple[bool, bool, bool]:
    expected_set = set(sample.expected_tools)
    forbidden_set = set(sample.forbidden_tools)
    observed_set = set(observed)

    expected_ok = expected_set.issubset(observed_set)
    ordered_ok = tuple(observed) == tuple(sample.expected_tools)
    forbidden_ok = observed_set.isdisjoint(forbidden_set)
    return expected_ok, ordered_ok, forbidden_ok


def _score_phrases(
    sample: Sample, final_answer: str
) -> tuple[tuple[str, ...], tuple[str, ...], bool, bool]:
    missing = tuple(p for p in sample.expected_phrases if p not in final_answer)
    hit = tuple(p for p in sample.forbidden_phrases if p in final_answer)
    return missing, hit, not missing, not hit


def score_run(
    sample: Sample,
    observed_intent: str,
    observed_tools: tuple[str, ...],
    final_answer: str,
    latency_s: float,
    tokens_by_model: dict[str, TokenUsage],
    turns_count: int,
    tool_calls: tuple[dict, ...] = (),
    error: str | None = None,
) -> SampleResult:
    """把一次 run 的观测聚合成 SampleResult；纯函数，便于单测。"""
    expected_ok, ordered_ok, forbidden_ok = _score_tools(sample, observed_tools)
    missing, hit, phrases_exp_ok, phrases_forb_ok = _score_phrases(sample, final_answer)
    return SampleResult(
        sample_id=sample.id,
        scenario=sample.scenario,
        turns_count=turns_count,
        intent_expected=sample.intent,
        intent_observed=observed_intent,
        intent_match=observed_intent == sample.intent,
        tools_expected=sample.expected_tools,
        tools_forbidden=sample.forbidden_tools,
        tools_observed=observed_tools,
        tools_expected_ok=expected_ok,
        tools_ordered_ok=ordered_ok,
        tools_forbidden_ok=forbidden_ok,
        final_answer=final_answer,
        phrases_expected_missing=missing,
        phrases_forbidden_hit=hit,
        phrases_expected_ok=phrases_exp_ok,
        phrases_forbidden_ok=phrases_forb_ok,
        latency_s=latency_s,
        tokens_by_model={m: TokenUsage(**dataclasses.asdict(u))
                         for m, u in tokens_by_model.items()},
        tool_calls=tool_calls,
        error=error,
    )


# ---------- 跑样本 ----------


def run_sample(sample: Sample, agent: Agent, memory_dir: Path) -> SampleResult:
    """跑单条样本：铺 preconditions → 多轮跑 → 聚合 → 清理。"""
    blocks = sample.preconditions.long_term_blocks
    if blocks:
        _seed_long_term(memory_dir, sample.user_id, blocks)
    else:
        _clear_long_term(memory_dir, sample.user_id)

    usage_this_sample: dict[str, TokenUsage] = {}
    # 换一份新的累加器用于本样本；agent.chat 闭包里的 dict 是 build_agent 时绑的那个，
    # 这里我们直接复用 agent 的内部闭包 —— 为了隔离，每条样本前后做一次差值
    # 实际实现：build_agent 传的 usage_by_model 是全局，每样本开始时 snapshot，结束时 diff
    # 为简洁，本函数不处理 usage 分摊，由调用方 run_all 负责 snapshot / diff
    error: str | None = None
    history: list[dict] | None = None
    observed_tools: list[str] = []
    tool_calls: list[dict] = []
    final_answer = ""
    observed_intent = "other"

    start = time.perf_counter()
    try:
        for i, turn in enumerate(sample.turns):
            result: RunResult = agent.run_routed(
                turn, user_id=sample.user_id, history=history,
            )
            if i == 0:
                observed_intent = result.intent
            for step in result.steps:
                for inv in step.invocations:
                    observed_tools.append(inv.name)
                    tool_calls.append({
                        "name": inv.name,
                        "arguments": inv.arguments,
                        "result": inv.result,
                    })
            if result.final is not None:
                final_answer = result.final
            history = result.messages
    except Exception as exc:  # noqa: BLE001 — harness 边界兜底
        error = f"{type(exc).__name__}: {exc}"
    latency_s = time.perf_counter() - start

    if blocks:
        _clear_long_term(memory_dir, sample.user_id)

    return score_run(
        sample=sample,
        observed_intent=observed_intent,
        observed_tools=tuple(observed_tools),
        final_answer=final_answer,
        latency_s=latency_s,
        tokens_by_model=usage_this_sample,
        turns_count=len(sample.turns),
        tool_calls=tuple(tool_calls),
        error=error,
    )


def run_all(samples: list[Sample]) -> list[SampleResult]:
    """串行跑全部样本。每样本独立 diff 出本轮的 tokens 用量。"""
    results: list[SampleResult] = []
    with tempfile.TemporaryDirectory(prefix="nana-eval-") as tmp:
        memory_dir = Path(tmp)
        usage_by_model: dict[str, TokenUsage] = {}
        agent = build_agent(memory_dir=memory_dir, usage_by_model=usage_by_model)

        for sample in samples:
            # 每条样本前重置 DB 到种子态；否则 refund-02 退掉 O003 后，
            # refund-05 再点 O003 就只能看到 refunded。样本应彼此独立。
            _reseed_db_silently()
            before = {m: TokenUsage(**dataclasses.asdict(u))
                      for m, u in usage_by_model.items()}
            res = run_sample(sample, agent, memory_dir)
            # diff：after - before
            sample_usage: dict[str, TokenUsage] = {}
            for m, after in usage_by_model.items():
                b = before.get(m, TokenUsage())
                d = TokenUsage(
                    prompt=after.prompt - b.prompt,
                    completion=after.completion - b.completion,
                    total=after.total - b.total,
                )
                if d.total > 0:
                    sample_usage[m] = d
            res.tokens_by_model = sample_usage
            results.append(res)
            print(f"  · {res.sample_id}: "
                  f"{'PASS' if res.passed else 'FAIL'} "
                  f"({res.latency_s:.1f}s)")
    return results


# ---------- 聚合 & 输出 ----------


@dataclass
class ScenarioAgg:
    samples: int = 0
    passed: int = 0
    intent_hits: int = 0
    tools_expected_hits: int = 0
    tools_ordered_hits: int = 0
    tools_forbidden_clean: int = 0
    phrases_expected_hits: int = 0
    phrases_forbidden_clean: int = 0
    latency_total_s: float = 0.0
    tokens: dict[str, TokenUsage] = field(default_factory=dict)

    def ingest(self, r: SampleResult) -> None:
        self.samples += 1
        self.passed += r.passed
        self.intent_hits += r.intent_match
        self.tools_expected_hits += r.tools_expected_ok
        self.tools_ordered_hits += r.tools_ordered_ok
        self.tools_forbidden_clean += r.tools_forbidden_ok
        self.phrases_expected_hits += r.phrases_expected_ok
        self.phrases_forbidden_clean += r.phrases_forbidden_ok
        self.latency_total_s += r.latency_s
        for m, u in r.tokens_by_model.items():
            self.tokens.setdefault(m, TokenUsage()).merge(u)


def aggregate(results: list[SampleResult]) -> dict[str, ScenarioAgg]:
    """按剧本 + overall 分桶；返回 dict（key="_overall" 为总盘）。"""
    buckets: dict[str, ScenarioAgg] = {"_overall": ScenarioAgg()}
    for r in results:
        buckets["_overall"].ingest(r)
        buckets.setdefault(r.scenario, ScenarioAgg()).ingest(r)
    return buckets


def _pct(num: int, denom: int) -> str:
    return f"{100 * num / denom:4.0f}%" if denom else "  - "


def _estimate_cost(tokens: dict[str, TokenUsage]) -> float:
    total = 0.0
    for model, u in tokens.items():
        if model not in _PRICING_PER_1M:
            continue
        in_rate, out_rate = _PRICING_PER_1M[model]
        total += (u.prompt * in_rate + u.completion * out_rate) / 1_000_000
    return total


def format_table(agg: dict[str, ScenarioAgg]) -> str:
    """人类可读的聚合表。按"剧本行 + overall 行"输出关键比例 + 延迟 + tokens。"""
    lines: list[str] = []
    header = (
        f"{'scenario':<22} "
        f"{'N':>3} "
        f"{'pass':>5} "
        f"{'intent':>6} "
        f"{'tool ✓':>6} "
        f"{'tool ×':>6} "
        f"{'phr ✓':>5} "
        f"{'phr ×':>5} "
        f"{'lat_avg':>7}"
    )
    lines.append(header)
    lines.append("-" * len(header))

    def row(name: str, a: ScenarioAgg) -> str:
        n = a.samples
        lat = a.latency_total_s / n if n else 0.0
        return (
            f"{name:<22} "
            f"{n:>3} "
            f"{_pct(a.passed, n):>5} "
            f"{_pct(a.intent_hits, n):>6} "
            f"{_pct(a.tools_expected_hits, n):>6} "
            f"{_pct(a.tools_forbidden_clean, n):>6} "
            f"{_pct(a.phrases_expected_hits, n):>5} "
            f"{_pct(a.phrases_forbidden_clean, n):>5} "
            f"{lat:6.1f}s"
        )

    # 先打各剧本，再打 overall
    for name in sorted(k for k in agg if k != "_overall"):
        lines.append(row(name, agg[name]))
    lines.append("-" * len(header))
    lines.append(row("OVERALL", agg["_overall"]))

    # tokens & cost
    lines.append("")
    lines.append("Tokens & 估算成本（_PRICING_PER_1M 为 portfolio 精度）：")
    overall_tokens = agg["_overall"].tokens
    for m, u in sorted(overall_tokens.items()):
        lines.append(
            f"  {m:<12} prompt={u.prompt:>6} "
            f"completion={u.completion:>6} total={u.total:>6}"
        )
    cost = _estimate_cost(overall_tokens)
    lines.append(f"  估算总成本 ≈ ¥{cost:.4f}")

    return "\n".join(lines)


def _result_to_dict(r: SampleResult) -> dict[str, Any]:
    d = dataclasses.asdict(r)
    # TokenUsage 转纯 dict
    d["tokens_by_model"] = {m: dataclasses.asdict(u) for m, u in r.tokens_by_model.items()}
    d["passed"] = r.passed
    return d


def write_report(
    results: list[SampleResult], agg: dict[str, ScenarioAgg], out_path: Path
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "primary_model": MODEL,
        "light_model": LIGHT_MODEL,
        "samples": [_result_to_dict(r) for r in results],
        "aggregate": {
            name: {
                **{k: v for k, v in dataclasses.asdict(a).items() if k != "tokens"},
                "tokens": {m: dataclasses.asdict(u) for m, u in a.tokens.items()},
                "cost_estimate": _estimate_cost(a.tokens),
            }
            for name, a in agg.items()
        },
    }
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


# ---------- CLI ----------


def _filter_samples(
    samples: list[Sample], ids: list[str], scenarios: list[str]
) -> list[Sample]:
    if ids:
        id_set = set(ids)
        samples = [s for s in samples if s.id in id_set]
    if scenarios:
        sc_set = set(scenarios)
        samples = [s for s in samples if s.scenario in sc_set]
    return samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="E2 自动指标 harness")
    parser.add_argument("--run", action="store_true",
                        help="真的调 LLM 跑样本。默认 --dry-run，仅列出要跑的样本。")
    parser.add_argument("--ids", nargs="*", default=[],
                        help="按样本 id 过滤，可多个")
    parser.add_argument("--scenarios", nargs="*", default=[],
                        help="按剧本名过滤，可多个")
    parser.add_argument("--out", type=Path, default=None,
                        help="报告输出路径；默认 eval/reports/{timestamp}.json")
    args = parser.parse_args(argv)

    samples = _filter_samples(load_all(), args.ids, args.scenarios)
    if not samples:
        print("没有样本匹配过滤条件", file=sys.stderr)
        return 1

    print(f"待跑样本数：{len(samples)}")
    for s in samples:
        print(f"  - {s.id:<20} {s.scenario:<20} turns={len(s.turns)}")

    if not args.run:
        print("\n[dry-run] 想真跑请加 --run（会调 qwen-plus，产生成本）")
        return 0

    print("\n开始跑样本…")
    results = run_all(samples)
    agg = aggregate(results)

    print("\n" + format_table(agg))

    out_path = args.out or (
        Path(__file__).resolve().parent / "reports"
        / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    write_report(results, agg, out_path)
    print(f"\n报告已落盘：{out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
