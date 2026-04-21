"""query_policy · 返回 `knowledge/policies.md` 全文。

设计要点（PLAN §9 决策）：
- **无参**、**不切分**：文件体积 < 2K tokens，整文件入上下文信息完整度最高
- 切分会人为制造 chunking 选择问题，复杂度不值
- `policies.md` 是政策的**唯一事实源**；所有涉及退换货/配送/退款时效的回答都必须走此工具，禁止 LLM 臆造
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ._contract import ErrorCode, ToolSpec, err, ok

_POLICIES_PATH = Path(__file__).resolve().parent.parent.parent / "knowledge" / "policies.md"


class QueryPolicyInput(BaseModel):
    """无参工具——LLM 不需要传任何字段。"""


def query_policy_handler(_args: QueryPolicyInput) -> dict:
    """读全文返回；文件缺失时明确报错而非 INTERNAL 兜底。"""
    if not _POLICIES_PATH.exists():
        return err(ErrorCode.INTERNAL, f"政策文件不存在：{_POLICIES_PATH}")
    content = _POLICIES_PATH.read_text(encoding="utf-8")
    return ok({"content": content})


QUERY_POLICY_SPEC = ToolSpec(
    name="query_policy",
    description=(
        "查询 NanaShop 的售后与配送政策全文（退换货规则、发货/物流时效、退款到账、常见 FAQ）。"
        "无需任何参数。所有涉及政策的回答都必须先调此工具，不得凭记忆臆造。"
    ),
    input_model=QueryPolicyInput,
    handler=query_policy_handler,
)
