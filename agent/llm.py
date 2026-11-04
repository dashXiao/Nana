"""Qwen client using OpenAI-compatible, non-streaming chat completions."""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


MODEL = os.getenv("LLM_MODEL", "qwen3.7-plus")
LIGHT_MODEL = os.getenv("LLM_LIGHT_MODEL", "qwen3.8-flash")
ENABLE_THINKING = os.getenv("LLM_ENABLE_THINKING", "false").strip().lower() == "true"
client = OpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL"),
)


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    model: str | None = None,
    **kwargs: Any,
):
    """Send a synchronous completion, omitting empty tools for chitchat.

    Explicitly select thinking mode because newer Qwen models enable it by
    default. Preserve caller-supplied extra_body options without mutating them.
    """
    extra_body = {"enable_thinking": ENABLE_THINKING}
    extra_body.update(kwargs.pop("extra_body", None) or {})
    payload: dict[str, Any] = {
        "model": model or MODEL,
        "messages": messages,
        "extra_body": extra_body,
        **kwargs,
    }
    if tools:
        payload["tools"] = tools
    return client.chat.completions.create(**payload)
