"""initiate_refund 工具单元测试。

覆盖：
- 正常路径：pending / shipped / delivered 订单 → refunded（工具层不校验天数）
- 幂等：已 refunded 再退 → INVALID_STATE
- 越权：U001 尝试退 U002 的订单 → UNAUTHORIZED，DB 不变
- 订单不存在 → NOT_FOUND
- 入参缺失 → INVALID_INPUT
- OpenAI schema 不变式：user_id 与 order_id 都必需

注意：退货时效政策不在工具层校验，由 LLM 结合 policies.md 自判。
因此本文件不测政策违规；超期订单应在对话层被拦住（见 smoke/eval）。
"""

from __future__ import annotations

import pytest

from agent.tools._contract import ErrorCode
from agent.tools.initiate_refund import INITIATE_REFUND_SPEC
from db import db_session
from scripts.seed import seed


@pytest.fixture(autouse=True)
def _reseed_between_tests():
    """本模块每个用例都会改写订单状态；每条用例前重置到种子态，避免相互污染
    以及污染后续模块（如 test_query_orders 依赖 O009 仍为 delivered）。"""
    seed()
    yield


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
    """O003 (U001, shipped) → refunded。天数由 LLM 判断，工具层放行。"""
    result = INITIATE_REFUND_SPEC.invoke({"user_id": "U001", "order_id": "O003"})
    assert result["ok"] is True
    assert _status("O003") == "refunded"


def test_delivered_can_be_refunded_regardless_of_days():
    """O005 (U001, delivered, 发货 9 天) 工具层也应放行——天数判断不在这一层。

    这是对"工具层只管硬事实，政策交给 LLM"这一架构决策的不变式锁定。
    """
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
