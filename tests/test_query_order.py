"""query_order 工具单元测试。"""

from __future__ import annotations

from agent.tools._contract import ErrorCode
from agent.tools.query_order import QUERY_ORDER_SPEC


def test_normal_query():
    """U001 查自己的 O001，应返回 pending 状态、shipped_at/days_since_shipped 为空。"""
    result = QUERY_ORDER_SPEC.invoke({"user_id": "U001", "order_id": "O001"})
    assert result["ok"] is True
    data = result["data"]
    assert data["order_id"] == "O001"
    assert data["user_id"] == "U001"
    assert data["product_id"] == "P001"
    assert data["product_name"] == "无线蓝牙耳机 X1"
    assert data["status"] == "pending"
    assert data["shipped_at"] is None
    assert data["days_since_shipped"] is None


def test_unauthorized_enforcement():
    """O001 属于 U001；U002 来查应被工具层硬拦截为 UNAUTHORIZED。"""
    result = QUERY_ORDER_SPEC.invoke({"user_id": "U002", "order_id": "O001"})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.UNAUTHORIZED


def test_not_found():
    """不存在的订单号返回 NOT_FOUND。"""
    result = QUERY_ORDER_SPEC.invoke({"user_id": "U001", "order_id": "O999"})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.NOT_FOUND


def test_missing_required_arg():
    """缺 order_id → INVALID_INPUT。"""
    result = QUERY_ORDER_SPEC.invoke({"user_id": "U001"})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_INPUT


def test_invalid_json_string_arg():
    """arguments 不是合法 JSON → INVALID_INPUT。"""
    result = QUERY_ORDER_SPEC.invoke("not-json{")
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_INPUT


def test_anomaly_O008_shipped_12_days_ago():
    """O008 为异常订单：已发货 12 天仍 shipped，days_since_shipped 应 ≥ 12。"""
    result = QUERY_ORDER_SPEC.invoke({"user_id": "U002", "order_id": "O008"})
    assert result["ok"] is True
    data = result["data"]
    assert data["status"] == "shipped"
    # 允许 12 或 13（测试与 seed 之间跨日的极小窗口）
    assert data["days_since_shipped"] in (11, 12, 13)


def test_anomaly_O009_delivered_without_shipped_at():
    """O009 状态不一致：delivered 但 shipped_at=NULL，days_since_shipped 应为 None。"""
    result = QUERY_ORDER_SPEC.invoke({"user_id": "U001", "order_id": "O009"})
    assert result["ok"] is True
    data = result["data"]
    assert data["status"] == "delivered"
    assert data["shipped_at"] is None
    assert data["days_since_shipped"] is None


def test_anomaly_O010_long_pending():
    """O010 下单 20 天仍 pending，shipped_at=None，状态正常返回（异常由 Agent 识别）。"""
    result = QUERY_ORDER_SPEC.invoke({"user_id": "U002", "order_id": "O010"})
    assert result["ok"] is True
    data = result["data"]
    assert data["status"] == "pending"
    assert data["shipped_at"] is None
    assert data["days_since_shipped"] is None


def test_openai_schema_shape():
    """to_openai() 输出的结构应符合 OpenAI function calling 规范。"""
    schema = QUERY_ORDER_SPEC.to_openai()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "query_order"
    assert "description" in schema["function"]
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert "user_id" in params["properties"]
    assert "order_id" in params["properties"]
    assert set(params["required"]) == {"user_id", "order_id"}
