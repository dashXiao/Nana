"""query_policy: return the full knowledge/policies.md document.

Design:
- No arguments or chunking: the small document fits in context as a whole.
- Loading the full file avoids unnecessary chunk selection complexity.
- policies.md is the single source of truth for returns, shipping, and refund timing.
  Answers about these policies must use this tool rather than model recollection.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from ._contract import ErrorCode, ToolSpec, err, ok

_POLICIES_PATH = Path(__file__).resolve().parent.parent.parent / "knowledge" / "policies.md"


class QueryPolicyInput(BaseModel):
    """Zero-argument tool; the LLM supplies no fields."""


def query_policy_handler(_args: QueryPolicyInput) -> dict:
    """Return the full text, with an explicit error if the file is missing."""
    if not _POLICIES_PATH.exists():
        return err(ErrorCode.INTERNAL, f"Policy file does not exist: {_POLICIES_PATH}")
    content = _POLICIES_PATH.read_text(encoding="utf-8")
    return ok({"content": content})


QUERY_POLICY_SPEC = ToolSpec(
    name="query_policy",
    description=(
        "Retrieve NanaShop policies on returns, shipping, delivery, refund processing, and FAQs. "
        "No arguments required."
    ),
    input_model=QueryPolicyInput,
    handler=query_policy_handler,
)
