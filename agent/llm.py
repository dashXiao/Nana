"""qwen-plus 客户端（走阿里云 DashScope 的 OpenAI 兼容端点）。

薄封装：env 读取 + 模块级 client + 单 chat 函数。ReAct 循环由 `agent/core.py` 负责。
env 三件套必填（DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL / LLM_MODEL），缺任何一个
import 即报错——拒绝静默兜底成默认端点/默认模型，避免".env 没配好但跑起来不知道"。
"""

from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def _required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} 未设置，请在 .env 填入（参考 .env.example）。")
    return value


MODEL = _required_env("LLM_MODEL")
client = OpenAI(
    api_key=_required_env("DASHSCOPE_API_KEY"),
    base_url=_required_env("DASHSCOPE_BASE_URL"),
)


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs: Any,
):
    """最小封装：chat.completions.create；tools 为 None/空时不传该字段。"""
    payload: dict[str, Any] = {"model": MODEL, "messages": messages, **kwargs}
    if tools:
        payload["tools"] = tools
    return client.chat.completions.create(**payload)
