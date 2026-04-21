"""initiate_refund 工具单元测试。

覆盖：
- 正常路径：pending / shipped / delivered 三态各一条 → refunded
- 幂等：已 refunded 再退 → INVALID_STATE
- 越权：U001 尝试退 U002 的订单 → UNAUTHORIZED，DB 不变
- 订单不存在 → NOT_FOUND
- 入参缺失 → INVALID_INPUT
- OpenAI schema 不变式：order_id 必需（与 query_orders 对照）
"""

from __future__ import annotations

from agent.tools._contract import ErrorCode
from agent.tools.initiate_refund import INITIATE_REFUND_SPEC
from db import db_session


def _status(order_id: str) -> str | None:
    with db_session() as conn:
        row = conn.execute(
            "SELECT status FROM orders WHERE order_id = ?", (order_id,)
        ).fetchone()
    return row["status"] if row else None


def test_pending_can_be_refunded():
    """O001 (U001, pending) → refunded。"""
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O001"})
    assert result["ok"] is True
    assert result["data"] == {"order_id": "O001", "status": "refunded"}
    assert _status("O001") == "refunded"


def test_shipped_can_be_refunded():
    """O003 (U001, shipped) → refunded。"""
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O003"})
    assert result["ok"] is True
    assert _status("O003") == "refunded"


def test_delivered_can_be_refunded():
    """O005 (U001, delivered) → refunded。"""
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O005"})
    assert result["ok"] is True
    assert _status("O005") == "refunded"


def test_refunded_cannot_refund_again():
    """O007 已是 refunded，再退 → INVALID_STATE。"""
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O007"})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_STATE


def test_cross_user_refund_is_unauthorized():
    """核心鉴权示范：U001 退 U002 的 O002 → UNAUTHORIZED，状态不变。"""
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O002"})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.UNAUTHORIZED
    # 事后校验：DB 没被篡改
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
    """关键设计不变性：入参必含 user_id 和 order_id。"""
    schema = INITIATE_REFUND_SPEC.to_openai()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "initiate_refund"
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"user_id", "order_id"}
