"""E2 evaluation harness: run datasets and report intent, tool, phrase, latency, and token metrics.

CLI:
    uv run python -m eval.harness                 # dry run
    uv run python -m eval.harness --run           # full run
    uv run python -m eval.harness --run --scenarios refund_flow
    uv run python -m eval.harness --run --ids refund-01 --out /tmp/r.json
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import io
import json
import os
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
from agent.router import BASE_SYSTEM_PROMPT, IntentDecision, IntentRouter
from agent.tools._contract import ToolRegistry
from agent.tools.escalate_to_human import ESCALATE_TO_HUMAN_SPEC
from agent.tools.initiate_refund import INITIATE_REFUND_SPEC
from agent.tools.query_orders import QUERY_ORDERS_SPEC
from agent.tools.query_policy import QUERY_POLICY_SPEC
from agent.tools.query_product import QUERY_PRODUCT_SPEC

from eval.loader import Sample, group_by_scenario, load_all
from scripts.seed import seed as _seed_db


def _reseed_db_silently() -> None:
    """Reset SQLite to seeded state between samples, suppressing seed output to keep reports readable."""
    with contextlib.redirect_stdout(io.StringIO()):
        _seed_db()

# Pricing (CNY per million input/output tokens) is configured in .env.
def _parse_pricing(env_var: str) -> tuple[float, float]:
    raw = os.getenv(env_var, "")
    parts = raw.split(",")
    return (float(parts[0]), float(parts[1]))


_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    MODEL: _parse_pricing("LLM_MODEL_PRICING"),
    LIGHT_MODEL: _parse_pricing("LLM_LIGHT_MODEL_PRICING"),
}


# ---------- Metric data structures ----------


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
    """Complete snapshot after one sample finishes."""

    sample_id: str
    scenario: str
    turns_count: int

    # Observed routing and tool calls
    intent_expected: str
    intent_observed: str
    intent_match: bool

    tools_expected: tuple[str, ...]
    tools_forbidden: tuple[str, ...]
    tools_observed: tuple[str, ...]
    tools_expected_ok: bool      # All expected tools appeared (set-wise).
    tools_ordered_ok: bool       # The observed sequence exactly matches the expected sequence.
    tools_forbidden_ok: bool     # No forbidden tools appeared.

    # Answer and phrase matches
    final_answer: str
    phrases_expected_missing: tuple[str, ...]
    phrases_forbidden_hit: tuple[str, ...]
    phrases_expected_ok: bool
    phrases_forbidden_ok: bool

    # Operational metrics
    latency_s: float
    tokens_by_model: dict[str, TokenUsage]

    # Capture exceptions here; use readable defaults for the other fields.
    error: str | None = None

    # Full tool trace serves as agent-visible evidence for the E3 faithfulness judge.
    tool_calls: tuple[dict, ...] = ()

    @property
    def passed(self) -> bool:
        """Pass requires matching intent, tool coverage, no forbidden tools or phrases, and no errors."""
        return (
            self.error is None
            and self.intent_match
            and self.tools_expected_ok
            and self.tools_forbidden_ok
            and self.phrases_expected_ok
            and self.phrases_forbidden_ok
        )


# ---------- Agent / chat assembly ----------


def _make_capturing_chat(
    usage_by_model: dict[str, TokenUsage],
    force_model: str | None = None,
):
    """Wrap agent.llm.chat and accumulate response.usage by model without changing behavior.

    force_model overrides every downstream model choice, allowing E4 to run classification,
    compression, and summarization on the main model for the single-model ablation.
    """
    from agent.llm import chat as real_chat  # Import lazily so environment variables can be configured first.

    def wrapped(messages, tools=None, model=None, **kwargs):
        effective_model = force_model or model or MODEL
        response = real_chat(
            messages, tools=tools, model=effective_model, **kwargs,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            bucket = usage_by_model.setdefault(effective_model, TokenUsage())
            bucket.add(
                prompt=getattr(usage, "prompt_tokens", 0) or 0,
                completion=getattr(usage, "completion_tokens", 0) or 0,
                total=getattr(usage, "total_tokens", 0) or 0,
            )
        return response

    return wrapped


class _ForcedOtherRouter(IntentRouter):
    """E4 ablation router: skip classification and always return intent="other".

    Expose all five tools and the full fallback patch on every turn to measure the token
    and cost impact of intent-specific loading. route_path="forced" marks this configuration.
    """

    def classify(self, msg: str) -> IntentDecision:  # noqa: ARG002 — Intentionally ignore the message.
        return IntentDecision("other", 1.0, "forced")


def build_agent(
    memory_dir: Path,
    usage_by_model: dict[str, TokenUsage],
    force_model: str | None = None,
    force_intent_other: bool = False,
) -> Agent:
    """Build an evaluation agent with five tools, routing, both memory layers, injection filtering,
    and usage tracking. force_model selects one model for all calls; force_intent_other
    replaces the router for the full-context ablation. Defaults reproduce baseline behavior.
    """
    chat_fn = _make_capturing_chat(usage_by_model, force_model=force_model)
    registry = ToolRegistry([
        QUERY_ORDERS_SPEC,
        QUERY_POLICY_SPEC,
        QUERY_PRODUCT_SPEC,
        INITIATE_REFUND_SPEC,
        ESCALATE_TO_HUMAN_SPEC,
    ])
    router_cls = _ForcedOtherRouter if force_intent_other else IntentRouter
    return Agent(
        registry=registry,
        system_prompt=BASE_SYSTEM_PROMPT,
        chat_fn=chat_fn,
        router=router_cls(chat_fn=chat_fn),
        short_term=ShortTermMemory(chat_fn=chat_fn),
        long_term=LongTermMemory(base_dir=memory_dir, chat_fn=chat_fn),
        injection_detector=InjectionDetector(),
    )


# ---------- preconditions ----------


def _seed_long_term(memory_dir: Path, user_id: str, blocks: tuple[str, ...]) -> None:
    """Write sample memory blocks to temporary files using the LongTermMemory session format."""
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


# ---------- Scoring ----------


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
    # Ignore English capitalization while retaining original phrases in reports.
    answer = final_answer.casefold()
    missing = tuple(p for p in sample.expected_phrases if p.casefold() not in answer)
    hit = tuple(p for p in sample.forbidden_phrases if p.casefold() in answer)
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
    """Aggregate observations into SampleResult; a pure function for unit testing."""
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


# ---------- Sample execution ----------


def run_sample(sample: Sample, agent: Agent, memory_dir: Path) -> SampleResult:
    """Run one sample: prepare preconditions, execute turns, aggregate results, and clean up."""
    blocks = sample.preconditions.long_term_blocks
    if blocks:
        _seed_long_term(memory_dir, sample.user_id, blocks)
    else:
        _clear_long_term(memory_dir, sample.user_id)

    # run_all fills tokens_by_model using usage snapshots before and after each sample.
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
    except Exception as exc:  # noqa: BLE001 — Harness boundary error handling.
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
        tokens_by_model={},
        turns_count=len(sample.turns),
        tool_calls=tuple(tool_calls),
        error=error,
    )


def run_all(
    samples: list[Sample],
    build_kwargs: dict[str, Any] | None = None,
) -> list[SampleResult]:
    """Run samples serially, attributing token usage with per-sample snapshots.

    Pass build_kwargs to build_agent for force_model and force_intent_other ablations.
    None preserves baseline behavior.
    """
    results: list[SampleResult] = []
    build_kwargs = build_kwargs or {}
    with tempfile.TemporaryDirectory(prefix="nana-eval-") as tmp:
        memory_dir = Path(tmp)
        usage_by_model: dict[str, TokenUsage] = {}
        agent = build_agent(
            memory_dir=memory_dir,
            usage_by_model=usage_by_model,
            **build_kwargs,
        )

        for sample in samples:
            # Reseed before each sample so a refund of O003 in refund-02
            # does not change what refund-05 observes. Samples must remain independent.
            _reseed_db_silently()
            before = {m: TokenUsage(**dataclasses.asdict(u))
                      for m, u in usage_by_model.items()}
            res = run_sample(sample, agent, memory_dir)
            # Usage delta: after - before
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


# ---------- Aggregation and output ----------


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
    """Aggregate by scenario and overall; the _overall key contains totals."""
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
    """Render readable scenario and overall metrics, latency, and token usage."""
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

    # Render individual scenarios before overall totals.
    for name in sorted(k for k in agg if k != "_overall"):
        lines.append(row(name, agg[name]))
    lines.append("-" * len(header))
    lines.append(row("OVERALL", agg["_overall"]))

    # tokens & cost
    lines.append("")
    lines.append("Tokens and estimated cost (approximate portfolio-level pricing):")
    overall_tokens = agg["_overall"].tokens
    for m, u in sorted(overall_tokens.items()):
        lines.append(
            f"  {m:<12} prompt={u.prompt:>6} "
            f"completion={u.completion:>6} total={u.total:>6}"
        )
    cost = _estimate_cost(overall_tokens)
    lines.append(f"  Estimated total cost ≈ ¥{cost:.4f}")

    return "\n".join(lines)


def _result_to_dict(r: SampleResult) -> dict[str, Any]:
    d = dataclasses.asdict(r)
    # Convert TokenUsage instances to plain dictionaries.
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
    parser = argparse.ArgumentParser(description="E2 automated evaluation harness")
    parser.add_argument("--run", action="store_true",
                        help="Run samples with live LLM calls. By default, only list planned samples.")
    parser.add_argument("--ids", nargs="*", default=[],
                        help="Filter by sample IDs; accepts multiple values")
    parser.add_argument("--scenarios", nargs="*", default=[],
                        help="Filter by scenarios; accepts multiple values")
    parser.add_argument("--out", type=Path, default=None,
                        help="Report output path; default: eval/reports/{timestamp}.json")
    args = parser.parse_args(argv)

    samples = _filter_samples(load_all(), args.ids, args.scenarios)
    if not samples:
        print("No samples match the filters", file=sys.stderr)
        return 1

    print(f"Samples to run: {len(samples)}")
    for s in samples:
        print(f"  - {s.id:<20} {s.scenario:<20} turns={len(s.turns)}")

    if not args.run:
        print("\n[dry-run] Add --run for live execution (incurs model usage costs)")
        return 0

    print("\nRunning samples...")
    results = run_all(samples)
    agg = aggregate(results)

    print("\n" + format_table(agg))

    out_path = args.out or (
        Path(__file__).resolve().parent / "reports"
        / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}.json"
    )
    write_report(results, agg, out_path)
    print(f"\nReport saved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
