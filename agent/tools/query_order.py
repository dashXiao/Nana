"""query_order · 按订单号查询订单详情。

工具层鉴权：`user_id` 必须与订单归属一致，否则返回 UNAUTHORIZED。
无论对话路径如何（Prompt 注入 / 模型幻觉 / 越权引导），都在此处硬拦截。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from db import db_session

from ._contract import ErrorCode, ToolSpec, err, ok


class QueryOrderInput(BaseModel):
    user_id: str = Field(description="当前会话用户 ID，例如 U001。用于鉴权，不得查他人订单。")
    order_id: str = Field(description="要查询的订单号，例如 O001。")


def _days_since(dt_str: str | None, now: datetime) -> int | None:
    """从 ISO 字符串到现在的整日数；None → None。"""
    if not dt_str:
        return None
    dt = datetime.fromisoformat(dt_str)
    delta = now - dt
    return max(0, delta.days)


def query_order_handler(args: QueryOrderInput) -> dict:
    """查订单 → 鉴权 → 组装带 days_since_shipped 的响应。"""
    with db_session() as conn:
        row = conn.execute(
            """
            SELECT o.order_id, o.user_id, o.product_id, p.name AS product_name,
                   o.status, o.created_at, o.shipped_at
            FROM orders o
            JOIN products p ON p.product_id = o.product_id
            WHERE o.order_id = ?
            """,
            (args.order_id,),
        ).fetchone()

    if row is None:
        return err(ErrorCode.NOT_FOUND, f"订单 {args.order_id} 不存在")

    if row["user_id"] != args.user_id:
        return err(ErrorCode.UNAUTHORIZED, f"无权查询订单 {args.order_id}")

    now = datetime.now()
    data = {
        "order_id": row["order_id"],
        "user_id": row["user_id"],
        "product_id": row["product_id"],
        "product_name": row["product_name"],
        "status": row["status"],
        "created_at": row["created_at"],
        "shipped_at": row["shipped_at"],
        "days_since_shipped": _days_since(row["shipped_at"], now),
    }
    return ok(data)


QUERY_ORDER_SPEC = ToolSpec(
    name="query_order",
    description=(
        "按订单号查询订单详情，返回状态、商品、下单时间、发货时间与已发货天数。"
        "仅允许查询当前用户自己的订单；对越权请求将返回 UNAUTHORIZED 错误。"
    ),
    input_model=QueryOrderInput,
    handler=query_order_handler,
)
