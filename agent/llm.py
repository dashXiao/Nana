"""qwen-plus 客户端（走阿里云 DashScope 的 OpenAI 兼容端点）。

设计目标：薄封装——只做 env 读取 + 默认值填充，把原生 openai SDK 对象直接暴露出去，
不在这里做 ReAct 逻辑。ReAct 循环由上层 `agent/core.py` 负责。
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

_DEFAULT_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
_DEFAULT_MODEL = "qwen-plus"


@lru_cache(maxsize=1)
def get_client() -> OpenAI:
    """返回单例 OpenAI client，指向 DashScope 兼容端点。"""
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DASHSCOPE_API_KEY 未设置。请在 .env 填入从 https://bailian.console.aliyun.com/ 获取的 key。"
        )
    base_url = os.getenv("DASHSCOPE_BASE_URL", _DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url)


def model_name() -> str:
    """返回当前使用的模型名（默认 qwen-plus）。"""
    return os.getenv("LLM_MODEL", _DEFAULT_MODEL)


def chat(
    messages: list[dict],
    tools: list[dict] | None = None,
    **kwargs: Any,
):
    """最小封装：chat.completions.create，带默认 model；tools 为 None 时不传该字段。"""
    client = get_client()
    payload: dict[str, Any] = {"model": model_name(), "messages": messages, **kwargs}
    if tools:
        payload["tools"] = tools
    return client.chat.completions.create(**payload)
