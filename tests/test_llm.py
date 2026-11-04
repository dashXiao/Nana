"""Request compatibility tests for the Qwen chat-completions adapter."""

from types import SimpleNamespace

import pytest

from agent import llm


@pytest.fixture
def captured_request(monkeypatch):
    captured = {}
    response = object()

    def create(**payload):
        captured.update(payload)
        return response

    monkeypatch.setattr(
        llm,
        "client",
        SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create))),
    )
    monkeypatch.setattr(llm, "ENABLE_THINKING", False)
    return captured, response


@pytest.mark.parametrize("tools", [None, []])
def test_default_request_is_non_thinking_and_omits_empty_tools(captured_request, tools):
    captured, response = captured_request
    messages = [{"role": "user", "content": "Hello"}]

    assert llm.chat(messages, tools=tools) is response

    assert captured["model"] == llm.MODEL
    assert captured["messages"] == messages
    assert captured["extra_body"] == {"enable_thinking": False}
    assert "tools" not in captured
    assert "stream" not in captured


def test_light_model_and_tool_schema_are_forwarded(captured_request):
    captured, _ = captured_request
    tools = [{"type": "function", "function": {"name": "query_orders"}}]

    llm.chat([], tools=tools, model=llm.LIGHT_MODEL, temperature=0)

    assert captured["model"] == llm.LIGHT_MODEL
    assert captured["tools"] == tools
    assert captured["temperature"] == 0


def test_thinking_setting_is_applied(captured_request, monkeypatch):
    captured, _ = captured_request
    monkeypatch.setattr(llm, "ENABLE_THINKING", True)

    llm.chat([])

    assert captured["extra_body"]["enable_thinking"] is True


def test_explicit_extra_body_overrides_defaults_without_mutation(captured_request):
    captured, _ = captured_request
    extra_body = {"enable_thinking": True, "thinking_budget": 128}

    llm.chat([], extra_body=extra_body)

    assert captured["extra_body"] == extra_body
    assert captured["extra_body"] is not extra_body
    assert extra_body == {"enable_thinking": True, "thinking_budget": 128}
