"""qwen 系列客户端（走阿里云 DashScope 的 OpenAI 兼容端点）。

薄封装：env 读取 + 模块级 client + 单 chat 函数，通过 `model=` 参数切换模型。

**双模型分层**（PLAN §9 #11）：
- `MODEL` 主模型（默认 qwen-plus）—— ReAct 主循环用，强 function calling + 最终答复
- `LIGHT_MODEL` 轻量模型（默认 qwen-turbo）—— 路由分类 / 记忆摘要用，便宜 + 快

env 必填（DASHSCOPE_API_KEY / DASHSCOPE_BASE_URL / LLM_MODEL / LLM_LIGHT_MODEL），
缺任何一个 import 即报错——拒绝静默兜底，避免 ".env 没配好但跑起来不知道"。
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
LIGHT_MODEL = _required_env("LLM_LIGHT_MODEL")
client = OpenAI(
    api_key=_required_env("DASHSCOPE_API_KEY"),
    base_url=_required_env("DASHSCOPE_BASE_URL"),
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
