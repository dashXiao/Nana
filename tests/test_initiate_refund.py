"""initiate_refund tests for pending, shipped, and delivered orders, repeat refunds,
unauthorized access, missing orders, invalid inputs, and required schema fields.

The tool checks existence, ownership, and state. The model evaluates the return
window using policies.md; conversation-level smoke/evaluation tests cover that policy.
"""

from __future__ import annotations

import pytest

from agent.tools._contract import ErrorCode
from agent.tools.initiate_refund import INITIATE_REFUND_SPEC
from db import db_session
from scripts.seed import seed


@pytest.fixture(autouse=True)
def _reseed_between_tests():
    """Reseed before and after each mutating test to isolate cases and later modules that depend on original statuses."""
    seed()
    yield


def _status(order_id: str) -> str | None:
    with db_session() as conn:
        row = conn.execute(
            "SELECT status FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
    return row["status"] if row else None


def test_pending_can_be_refunded():
    """O001 (U001, pending) → refunded."""
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O001"})
    assert result["ok"] is True
    assert result["data"] == {"order_id": "O001", "status": "refunded"}
    assert _status("O001") == "refunded"


def test_shipped_can_be_refunded():
    """O003 (U001, shipped) can be refunded; policy timing is evaluated by the model."""
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O003"})
    assert result["ok"] is True
    assert _status("O003") == "refunded"


def test_delivered_can_be_refunded_regardless_of_days():
    """O005 shipped nine days ago but the tool allows it: policy evaluation belongs to the model, while the tool enforces hard facts."""
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O005"})
    assert result["ok"] is True
    assert _status("O005") == "refunded"


def test_refunded_cannot_refund_again():
    """O007 is already refunded; another refund returns INVALID_STATE."""
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O007"})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_STATE


def test_cross_user_refund_is_unauthorized():
    """U001 cannot refund U002's O002; return UNAUTHORIZED and leave its status unchanged."""
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O002"})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.UNAUTHORIZED
    # Verify that the database was not modified.
    assert _status("O002") == "pending"


def test_unknown_order_returns_not_found():
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O999"})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.NOT_FOUND


def test_missing_order_id_returns_invalid_input():
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001"})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_INPUT


def test_missing_user_id_returns_invalid_input():
    result = INITIATE_REFUND_SPEC.invoke({"order_id": "O001"})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_INPUT


def test_openai_schema_requires_both_ids():
    """The input schema requires user_id and order_id."""
    schema = INITIATE_REFUND_SPEC.to_openai()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "initiate_refund"
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"user_id", "order_id"}
