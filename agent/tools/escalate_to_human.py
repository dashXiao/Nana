"""escalate_to_human · 终止型工具：将对话转接给人工客服。

设计要点：
- **零参数**：调用此工具的动作本身就是"应当转人工"的判断信号，无需 LLM 附带理由
- **无副作用**：不持久化工单、不写 DB、不发通知——portfolio 场景到此为止
- **终止语义由循环捕获**：Agent 在 `agent/core.py` 识别到本工具被调用后，硬切对话
  并返回 "已转接人工"；工具本身只返回一个 marker，不知道终止这回事
"""

from __future__ import annotations

from pydantic import BaseModel

from ._contract import ToolSpec, ok


class EscalateInput(BaseModel):
    """无参工具——LLM 不需要传任何字段。"""


def escalate_to_human_handler(_args: EscalateInput) -> dict:
    """纯信号返回；终止由 Agent 循环统一处理。"""
    return ok({"escalated": True})


ESCALATE_TO_HUMAN_SPEC = ToolSpec(
    name="escalate_to_human",
    description=(
        "将当前对话转接给人工客服，调用后本次会话立刻结束、不再响应用户。"
    ),
    input_model=EscalateInput,
    handler=escalate_to_human_handler,
)
