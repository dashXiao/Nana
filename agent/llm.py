"""LLM 客户端（走 OpenAI 兼容端点，具体型号见 .env）。

薄封装：env 读取 + 模块级 client + 单 chat 函数，通过 `model=` 参数切换模型。

**双模型分层**（PLAN §9 #11）：
- `MODEL` 主模型 —— ReAct 主循环用，强 function calling + 最终答复
- `LIGHT_MODEL` 轻量模型 —— 路由分类 / 记忆摘要用，便宜 + 快
"""

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
    """最小封装：chat.completions.create。

    - `tools` 为 None/空时不传该字段（chitchat 意图依赖这个行为）
    - `model=None` 默认用 `MODEL`（主模型）；轻量任务显式传 `model=LIGHT_MODEL`
    """
    payload: dict[str, Any] = {
        "model": model or MODEL,
        "messages": messages,
        **kwargs,
    }
    if tools:
        payload["tools"] = tools
    return client.chat.completions.create(**payload)
