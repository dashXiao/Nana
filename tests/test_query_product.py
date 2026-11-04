"""query_product tests covering single and multiple IDs, stable input ordering,
deduplication, missing IDs, empty or missing inputs, frontmatter stripping,
and the OpenAI tool schema.
"""

from __future__ import annotations

from agent.tools._contract import ErrorCode
from agent.tools.query_product import QUERY_PRODUCT_SPEC


def test_single_id_returns_db_plus_description():
    result = QUERY_PRODUCT_SPEC.invoke({"product_ids": ["P001"]})
    assert result["ok"] is True
    data = result["data"]
    assert data["missing"] == []
    assert len(data["products"]) == 1
    p = data["products"][0]
    assert p["product_id"] == "P001"
    assert p["name"] == "X1 Wireless Earbuds"
    assert p["price"] == 299.0
    assert p["stock"] == 50
    assert isinstance(p["description"], str)
    assert "Description" in p["description"]


def test_multi_ids_preserve_order():
    """Return P002 and P001 in the requested order for comparisons."""
    result = QUERY_PRODUCT_SPEC.invoke({"product_ids": ["P002", "P001"]})
    assert result["ok"] is True
    ids = [p["product_id"] for p in result["data"]["products"]]
    assert ids == ["P002", "P001"]


def test_dedup_preserves_first_occurrence():
    result = QUERY_PRODUCT_SPEC.invoke({"product_ids": ["P001", "P002", "P001"]})
    assert result["ok"] is True
    ids = [p["product_id"] for p in result["data"]["products"]]
    assert ids == ["P001", "P002"]


def test_partial_missing_split_correctly():
    result = QUERY_PRODUCT_SPEC.invoke({"product_ids": ["P001", "P999", "P002"]})
    assert result["ok"] is True
    data = result["data"]
    assert [p["product_id"] for p in data["products"]] == ["P001", "P002"]
    assert data["missing"] == ["P999"]


def test_all_missing_returns_empty_products():
    result = QUERY_PRODUCT_SPEC.invoke({"product_ids": ["P998", "P999"]})
    assert result["ok"] is True
    assert result["data"]["products"] == []
    assert result["data"]["missing"] == ["P998", "P999"]


def test_empty_list_is_invalid_input():
    """Pydantic min_length=1 rejects an empty product list."""
    result = QUERY_PRODUCT_SPEC.invoke({"product_ids": []})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_INPUT


def test_missing_product_ids_is_invalid_input():
    result = QUERY_PRODUCT_SPEC.invoke({})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_INPUT


def test_description_strips_frontmatter():
    """Strip YAML frontmatter so product_id and name are not repeated in the description."""
    result = QUERY_PRODUCT_SPEC.invoke({"product_ids": ["P001"]})
    desc = result["data"]["products"][0]["description"]
    assert not desc.startswith("---")
    # Frontmatter fields are absent from the description body.
    assert "product_id: P001" not in desc
    # Markdown section headings remain in the description.
    assert desc.lstrip().startswith("##")


def test_openai_schema_requires_product_ids_list():
    schema = QUERY_PRODUCT_SPEC.to_openai()
    assert schema["function"]["name"] == "query_product"
    params = schema["function"]["parameters"]
    assert set(params["required"]) == {"product_ids"}
    prop = params["properties"]["product_ids"]
    assert prop["type"] == "array"
    assert prop["items"] == {"type": "string"}
    assert prop.get("minItems") == 1
