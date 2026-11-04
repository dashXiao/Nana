"""Unit tests for the query_orders tool."""

from __future__ import annotations

from agent.tools._contract import ErrorCode
from agent.tools.query_orders import QUERY_ORDERS_SPEC


def test_u001_returns_all_orders():
    """U001 has five orders returned in descending created_at order."""
    result = QUERY_ORDERS_SPEC.invoke({"user_id": "U001"})
    assert result["ok"] is True
    data = result["data"]
    assert data["count"] == 5
    ids = [o["order_id"] for o in data["orders"]]
    assert set(ids) == {"O001", "O003", "O005", "O007", "O009"}
    # Verify descending order.
    times = [o["created_at"] for o in data["orders"]]
    assert times == sorted(times, reverse=True)


def test_u002_returns_all_orders():
    result = QUERY_ORDERS_SPEC.invoke({"user_id": "U002"})
    assert result["ok"] is True
    data = result["data"]
    assert data["count"] == 5
    ids = [o["order_id"] for o in data["orders"]]
    assert set(ids) == {"O002", "O004", "O006", "O008", "O010"}


def test_each_order_includes_product_name_and_days_shipped():
    """Each order includes product_name and days_since_shipped."""
    result = QUERY_ORDERS_SPEC.invoke({"user_id": "U001"})
    assert result["ok"] is True
    for o in result["data"]["orders"]:
        assert "product_name" in o
        assert "days_since_shipped" in o


def test_unknown_user_returns_not_found():
    result = QUERY_ORDERS_SPEC.invoke({"user_id": "U999"})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.NOT_FOUND


def test_missing_user_id_returns_invalid_input():
    result = QUERY_ORDERS_SPEC.invoke({})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_INPUT


def test_invalid_json_string_arg():
    result = QUERY_ORDERS_SPEC.invoke("not-json{")
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_INPUT


def test_anomaly_O008_shipped_days():
    """O008 has been shipped for 12 days; days_since_shipped must be at least 11."""
    result = QUERY_ORDERS_SPEC.invoke({"user_id": "U002"})
    o008 = next(o for o in result["data"]["orders"] if o["order_id"] == "O008")
    assert o008["status"] == "shipped"
    assert o008["days_since_shipped"] in (11, 12, 13)


def test_anomaly_O009_delivered_without_shipped_at():
    """O009 is delivered but has shipped_at=None."""
    result = QUERY_ORDERS_SPEC.invoke({"user_id": "U001"})
    o009 = next(o for o in result["data"]["orders"] if o["order_id"] == "O009")
    assert o009["status"] == "delivered"
    assert o009["shipped_at"] is None
    assert o009["days_since_shipped"] is None


def test_anomaly_O010_long_pending():
    """O010 has been pending for 20 days; the agent decides whether that is abnormal."""
    result = QUERY_ORDERS_SPEC.invoke({"user_id": "U002"})
    o010 = next(o for o in result["data"]["orders"] if o["order_id"] == "O010")
    assert o010["status"] == "pending"
    assert o010["shipped_at"] is None


def test_openai_schema_has_only_user_id():
    """The tool accepts only user_id, never order_id."""
    schema = QUERY_ORDERS_SPEC.to_openai()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "query_orders"
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert set(params["required"]) == {"user_id"}
    assert "order_id" not in params["properties"]
