"""query_product 工具单元测试。

覆盖：
- 单 ID / 多 ID 基本路径
- 输入顺序保留（对比场景"A 和 B 哪个好"需要）
- 去重
- 部分 ID 不存在 → products + missing 并存
- 全部 ID 不存在 → products=[]，missing 全部
- 空 list / 缺 product_ids → INVALID_INPUT
- md frontmatter 被剥掉，不与 DB 字段重复
- OpenAI schema 不变式
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
    assert p["name"] == "无线蓝牙耳机 X1"
    assert p["price"] == 299.0
    assert p["stock"] == 50
    assert isinstance(p["description"], str)
    assert "产品描述" in p["description"]


def test_multi_ids_preserve_order():
    """["P002","P001"] 必须按此顺序返回，对比场景需要。"""
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
    """min_length=1 由 Pydantic 拦住——空 list 对工具无意义。"""
    result = QUERY_PRODUCT_SPEC.invoke({"product_ids": []})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_INPUT


def test_missing_product_ids_is_invalid_input():
    result = QUERY_PRODUCT_SPEC.invoke({})
    assert result["ok"] is False
    assert result["error_code"] == ErrorCode.INVALID_INPUT


def test_description_strips_frontmatter():
    """md 开头的 --- yaml --- 块应被剥掉，不重复暴露 product_id/name。"""
    result = QUERY_PRODUCT_SPEC.invoke({"product_ids": ["P001"]})
    desc = result["data"]["products"][0]["description"]
    assert not desc.startswith("---")
    # frontmatter 字段不在描述正文里
    assert "product_id: P001" not in desc
    # 但描述正文的 markdown 标题还在
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
