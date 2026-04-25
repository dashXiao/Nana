"""LLM 客户端（OpenAI 兼容端点）：MODEL 主模型 + LIGHT_MODEL 轻量模型，model= 参数切换。"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


MODEL = os.getenv("LLM_MODEL")
LIGHT_MODEL = os.getenv("LLM_LIGHT_MODEL")
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
    """chat.completions.create 薄封装；tools 为空时不传该字段（chitchat 依赖此行为）。"""
    payload: dict[str, Any] = {
        "model": model or MODEL,
        "messages": messages,
        **kwargs,
    }
    if tools:
        payload["tools"] = tools
    return client.chat.completions.create(**payload)
