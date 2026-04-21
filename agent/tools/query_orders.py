"""query_orders · 查询当前用户的全部订单（按下单时间倒序）。

设计要点：
- 入参只含 `user_id`，不含 `order_id` —— 从架构上消除跨用户枚举路径。
- `user_id` 应由 Agent 框架从 session context 注入，不由 LLM 或用户输入决定。
- 真实客服场景里用户往往不记得订单号，先拿到全貌再定位才贴合人。
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from db import db_session

from ._contract import ErrorCode, ToolSpec, err, ok


class QueryOrdersInput(BaseModel):
    user_id: str = Field(
        description="当前会话用户 ID，例如 U001。由 Agent 从 session context 注入，不得由 LLM 构造。"
    )


def _days_since(dt_str: str | None, now: datetime) -> int | None:
    """从 ISO 字符串到现在的整日数；None → None。"""
    if not dt_str:
        return None
    dt = datetime.fromisoformat(dt_str)
    return max(0, (now - dt).days)


def query_orders_handler(args: QueryOrdersInput) -> dict:
    """先校验用户存在，再拉全部订单并组装 days_since_shipped。"""
    with db_session() as conn:
        user = conn.execute(
            "SELECT 1 FROM users WHERE user_id = ?", (args.user_id,)
        ).fetchone()
        if user is None:
            return err(ErrorCode.NOT_FOUND, f"用户 {args.user_id} 不存在")

        rows = conn.execute(
            """
            SELECT o.order_id, o.product_id, p.name AS product_name,
                   o.status, o.created_at, o.shipped_at
            FROM orders o
            JOIN products p ON p.product_id = o.product_id
            WHERE o.user_id = ?
            ORDER BY o.created_at DESC
            """,
            (args.user_id,),
        ).fetchall()

    now = datetime.now()
    orders = [
        {
            "order_id": r["order_id"],
            "product_id": r["product_id"],
            "product_name": r["product_name"],
            "status": r["status"],
            "created_at": r["created_at"],
            "shipped_at": r["shipped_at"],
            "days_since_shipped": _days_since(r["shipped_at"], now),
        }
        for r in rows
    ]
    return ok({"orders": orders, "count": len(orders)})


QUERY_ORDERS_SPEC = ToolSpec(
    name="query_orders",
    description=(
        "查询当前用户的全部订单列表，按下单时间倒序返回（最新在前）。"
        "每条含订单号、商品、状态、下单/发货时间、已发货天数。"
        "用户无需提供订单号——先调此工具看全貌，再在响应中定位具体订单。"
    ),
    input_model=QueryOrdersInput,
    handler=query_orders_handler,
)
