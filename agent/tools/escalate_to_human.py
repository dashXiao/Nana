"""escalate_to_human: terminal signal for a human support handoff.

Design:
- No arguments: invoking the tool signals that a handoff is needed.
- No side effects: this portfolio demo creates no ticket, database write, or notification.
- The loop in agent/core.py detects the marker, stops the conversation, and returns
  the handoff response. The tool itself only returns a marker.
"""

from __future__ import annotations

from pydantic import BaseModel

from ._contract import ToolSpec, ok


class EscalateInput(BaseModel):
    """Zero-argument tool; the LLM supplies no fields."""


def escalate_to_human_handler(_args: EscalateInput) -> dict:
    """Return only a signal; the agent loop handles termination."""
    return ok({"escalated": True})


ESCALATE_TO_HUMAN_SPEC = ToolSpec(
    name="escalate_to_human",
    description=(
        "Transfer the conversation to a human agent. The current session ends immediately after this call."
    ),
    input_model=EscalateInput,
    handler=escalate_to_human_handler,
)
