"""工具通用契约：OpenAI function calling schema + 统一返回形态。

约定：
- 每个工具用 `ToolSpec` 声明（name / description / Pydantic 入参模型 / handler）。
- 入参校验由 Pydantic 负责；对 LLM 暴露的 schema 由 Pydantic 的 JSON Schema 生成。
- 返回统一为：
    成功 → {"ok": True,  "data": <any>}
    失败 → {"ok": False, "error_code": <str>, "message": <str>}
  LLM 只看到这个 dict 的 JSON 序列化形态，能明确区分"调用成功但无数据"与"调用失败"。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ValidationError


class ErrorCode:
    """工具层标准错误码。保持小而稳定——新增需求时优先复用。"""

    INVALID_INPUT = "INVALID_INPUT"    # 入参不符 schema / 字段缺失
    NOT_FOUND = "NOT_FOUND"            # 资源不存在（订单/商品等）
    UNAUTHORIZED = "UNAUTHORIZED"      # 工具层鉴权失败（越权访问他人数据）
    INVALID_STATE = "INVALID_STATE"    # 当前状态不允许此操作（如已退款订单不可再退）
    INTERNAL = "INTERNAL"              # 其他内部异常（handler 抛出未捕获）


def ok(data: Any) -> dict:
    """包装成功返回。"""
    return {"ok": True, "data": data}


def err(code: str, message: str) -> dict:
    """包装失败返回。"""
    return {"ok": False, "error_code": code, "message": message}


@dataclass
class ToolSpec:
    """单个工具的契约声明。"""

    name: str
    description: str
    input_model: type[BaseModel]
    handler: Callable[[BaseModel], dict]

    def to_openai(self) -> dict:
        """转为 OpenAI chat.completions `tools` 数组中的一条 entry。"""
        schema = self.input_model.model_json_schema()
        # Pydantic 生成的 JSON Schema 会带 title，OpenAI 侧不需要
        schema.pop("title", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": schema,
            },
        }

    def invoke(self, arguments: str | dict) -> dict:
        """统一入口：接 LLM 返回的 arguments（JSON str 或 dict）→ 校验 → 调 handler。

        任何异常都被兜成 err(...)，不会抛到调用方。
        """
        try:
            if isinstance(arguments, str):
                arguments = json.loads(arguments) if arguments else {}
        except json.JSONDecodeError as e:
            return err(ErrorCode.INVALID_INPUT, f"arguments 不是合法 JSON: {e}")

        try:
            args = self.input_model(**arguments)
        except ValidationError as e:
            first = e.errors()[0] if e.errors() else {}
            field = ".".join(str(x) for x in first.get("loc", [])) or "<root>"
            msg = first.get("msg", str(e))
            return err(ErrorCode.INVALID_INPUT, f"{field}: {msg}")

        try:
            return self.handler(args)
        except Exception as exc:  # noqa: BLE001 — 边界兜底
            return err(ErrorCode.INTERNAL, f"{type(exc).__name__}: {exc}")


class ToolRegistry:
    """以 name 查找 ToolSpec，供 Agent 循环拿到 tool_call 后派发。"""

    def __init__(self, specs: list[ToolSpec]):
        self._by_name: dict[str, ToolSpec] = {s.name: s for s in specs}
        if len(self._by_name) != len(specs):
            raise ValueError("存在同名工具，name 必须唯一")

    def get(self, name: str) -> ToolSpec | None:
        return self._by_name.get(name)

    def openai_tools(self) -> list[dict]:
        return [s.to_openai() for s in self._by_name.values()]

    def dispatch(self, name: str, arguments: str | dict) -> dict:
        """按 name 查工具并执行；不存在时返回 NOT_FOUND 错误。"""
        spec = self.get(name)
        if spec is None:
            return err(ErrorCode.NOT_FOUND, f"未知工具: {name}")
        return spec.invoke(arguments)
