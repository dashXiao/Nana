"""
NanaShop 种子数据
"""

from __future__ import annotations

from datetime import datetime, timedelta


USERS: list[tuple[str, str]] = [
    ("U001", "张三"),
    ("U002", "李娜"),
]

# 来源：knowledge/products/P001.md / P002.md / P003.md
PRODUCTS: list[tuple[str, str, float, int]] = [
    ("P001", "无线蓝牙耳机", 299.0, 50),
    ("P002", "智能手表 Pro", 899.0, 30),
    ("P003", "人体工学办公椅", 1299.0, 20),
]


def iso(dt: datetime) -> str:
    """ datetime 格式转换 """
    return dt.isoformat(timespec="seconds")


def build_orders(now: datetime) -> list[tuple]:
    """
    构造固定订单
    字段顺序：(order_id, user_id, product_id, status, created_at, shipped_at)
    """
    def ago(days: int, hours: int = 0) -> datetime:
        return now - timedelta(days=days, hours=hours)

    return [
        # --- 正常单 ---
        ("O001", "U001", "P001", "pending",   iso(ago(0, 2)),  None),
        ("O002", "U002", "P002", "pending",   iso(ago(1)),     None),
        ("O003", "U001", "P003", "shipped",   iso(ago(3)),     iso(ago(2))),
        ("O004", "U002", "P001", "shipped",   iso(ago(5)),     iso(ago(4))),
        ("O005", "U001", "P002", "delivered", iso(ago(10)),    iso(ago(9))),
        ("O006", "U002", "P003", "delivered", iso(ago(15)),    iso(ago(13))),
        ("O007", "U001", "P001", "refunded",  iso(ago(20)),    iso(ago(19))),
        # --- 异常单（考验 Agent 鲁棒性）---
        # O008：已发货 12 天仍停留在 shipped → 疑似丢件 / 物流异常
        ("O008", "U002", "P002", "shipped",   iso(ago(13)),    iso(ago(12))),
        # O009：status=delivered 但 shipped_at 为 NULL → 数据不一致
        ("O009", "U001", "P003", "delivered", iso(ago(7)),     None),
        # O010：下单 20 天仍 pending → 商家久未发货
        ("O010", "U002", "P001", "pending",   iso(ago(20)),    None),
    ]
