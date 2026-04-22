"""escalate_to_human 工具单元测试。

覆盖：
- 基本调用返回 ok + escalated=True（Agent 循环靠 name 识别，marker 字段仅作内容标识）
- 无参 / 空 dict / 空 JSON 字符串 三种入参形态都等价成功（与 query_policy 对齐）
- OpenAI schema 不向 LLM 暴露任何必填字段

循环终止逻辑由 tests/test_agent_core.py 覆盖，不在本文件重复。
"""

from __future__ import annotations

from agent.tools.escalate_to_human import ESCALATE_TO_HUMAN_SPEC


def test_returns_escalated_marker():
    result = ESCALATE_TO_HUMAN_SPEC.invoke({})
    assert result["ok"] is True
    assert result["data"] == {"escalated": True}


def test_accepts_no_args_forms():
    for args in ({}, "", "{}"):
        result = ESCALATE_TO_HUMAN_SPEC.invoke(args)
        assert result["ok"] is True, f"入参形态 {args!r} 未通过"


def test_openai_schema_has_no_required_fields():
    schema = ESCALATE_TO_HUMAN_SPEC.to_openai()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "escalate_to_human"
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert params.get("properties", {}) == {}
    assert params.get("required", []) == []
