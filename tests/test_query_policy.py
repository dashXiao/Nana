"""query_policy tests: successful full-text retrieval and key sections, equivalent
empty argument forms, and a zero-argument OpenAI schema.
"""

from __future__ import annotations

from agent.tools.query_policy import QUERY_POLICY_SPEC


def test_returns_full_policy_markdown():
    """Return ok with string content containing key policy sections."""
    result = QUERY_POLICY_SPEC.invoke({})
    assert result["ok"] is True
    content = result["data"]["content"]
    assert isinstance(content, str)
    # Include all key policy sections needed for grounded answers.
    assert "Returns" in content
    assert "Dispatch" in content
    assert "Refunds" in content
    assert "Shipping" in content


def test_accepts_no_args_forms():
    """All three empty argument forms succeed equivalently."""
    for args in ({}, "", "{}"):
        result = QUERY_POLICY_SPEC.invoke(args)
        assert result["ok"] is True, f"Argument form {args!r} failed"


def test_openai_schema_has_no_required_fields():
    """The tool schema exposes no input fields."""
    schema = QUERY_POLICY_SPEC.to_openai()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "query_policy"
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    # An empty Pydantic model has no properties; required may be absent or empty.
    assert params.get("properties", {}) == {}
    assert params.get("required", []) == []
