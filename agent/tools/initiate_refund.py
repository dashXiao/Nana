"""initiate_refund · 对订单发起退款。

设计要点：
- 入参 `user_id` 由 Agent session 注入，`order_id` 由 LLM 从 `query_orders` 结果里挑
- 工具层硬拦截越权：`order.user_id != args.user_id` → UNAUTHORIZED，
  无论 prompt 写得多宽松都挡得住（分层防御的"硬规则"层）
- 已退款订单再退 → INVALID_STATE，天然幂等
- 不校验退货期：期限规则在 `knowledge/policies.md`，
  不硬编码到工具，避免两处事实源
- 对话层"必须确认才执行"由 system prompt 引导，不进工具
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from db import db_session

from ._contract import ErrorCode, ToolSpec, err, ok


class InitiateRefundInput(BaseModel):
    user_id: str = Field(
        description="当前会话用户 ID。由 Agent 从 session context 注入，不得由 LLM 构造。"
    )
    order_id: str = Field(
        description="待退款订单 ID，例如 O005。由 LLM 从 query_orders 返回结果中挑选。"
    )


def initiate_refund_handler(args: InitiateRefundInput) -> dict:
    """存在性 → 归属 → 状态，三道校验顺序决定错误码优先级。"""
    with db_session() as conn:
        row = conn.execute(
            "SELECT user_id, status FROM orders WHERE order_id = ?",
            (args.order_id,),
        ).fetchone()

        if row is None:
            return err(ErrorCode.NOT_FOUND, f"订单 {args.order_id} 不存在")

        if row["user_id"] != args.user_id:
            return err(
                ErrorCode.UNAUTHORIZED,
                f"订单 {args.order_id} 不属于当前用户",
            )

        if row["status"] == "refunded":
            return err(
                ErrorCode.INVALID_STATE,
                f"订单 {args.order_id} 已退款，无法重复操作",
            )

        conn.execute(
            "UPDATE orders SET status = 'refunded' WHERE order_id = ?",
            (args.order_id,),
        )

    return ok({"order_id": args.order_id, "status": "refunded"})


INITIATE_REFUND_SPEC = ToolSpec(
    name="initiate_refund",
    description=(
        "对指定订单发起退款，执行后状态置为 refunded。"
        "调用前必须满足：(1) 订单属于当前用户；(2) 订单尚未退款；"
        "(3) 已在对话中明确向用户确认退款意图（必须先问'您确定要对订单 XXX 发起退款吗'）。"
        "工具层会硬校验 (1)(2)，(3) 是客服协议要求。"
    ),
    input_model=InitiateRefundInput,
    handler=initiate_refund_handler,
)
