"""escalate_to_human tests: success with escalated=True, equivalent empty argument
forms, and no required schema fields. The loop detects the tool by name; the marker
identifies the result. Loop termination is covered by test_agent_core.py.
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
        assert result["ok"] is True, f"Argument form {args!r} failed"


def test_openai_schema_has_no_required_fields():
    schema = ESCALATE_TO_HUMAN_SPEC.to_openai()
    assert schema["type"] == "function"
    assert schema["function"]["name"] == "escalate_to_human"
    params = schema["function"]["parameters"]
    assert params["type"] == "object"
    assert params.get("properties", {}) == {}
    assert params.get("required", []) == []
