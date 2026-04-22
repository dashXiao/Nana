"""query_policy 工具单元测试。

覆盖：
- 正常返回 + 内容类型 / 关键段落存在
- 无参 / 空 dict / 空 JSON 字符串 三种入参形态都能成功
- OpenAI schema 确实是"零参数"——LLM 不需要也不能传字段
"""

from __future__ import annotations

from agent.tools.query_policy import QUERY_POLICY_SPEC


def test_returns_full_policy_markdown():
    """返回 ok 且 content 是 str，含关键段落。"""
    result = QUERY_POLICY_SPEC.invoke({})
    assert result["ok"] is True
    content = result["data"]["content"]
    assert isinstance(content, str)
    # 关键政策段落都应该在返回里——Agent 凭这些才能回答
    assert "退货" in content
    assert "发货" in content
    assert "退款" in content
    assert "配送" in content


def test_accepts_no_args_forms():
    """无参工具对三种"空"入参形态都应等价成功。"""
    for args in ({}, "", "{}"):
        result = QUERY_POLICY_SPEC.invoke(args)
        assert result["ok"] is True, f"入参形态 {args!r} 未通过"


def test_openai_schema_has_no_required_fields():
    """关键设计不变性：工具不向 LLM 暴露任何入参字段。"""
    schema = QUERY_POLICY_SPEC.to_openai()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "query_policy"
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    # Pydantic 空 model → properties 空；required 可能不存在或为空
    assert params.get("properties", {}) == {}
    assert params.get("required", []) == []
