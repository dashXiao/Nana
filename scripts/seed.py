"""NanaShop SQLite 种子数据脚本。

极简原则：写死 2 用户 / 3 商品 / 10 订单，无随机。
商品 (product_id, name) 的唯一事实源在 knowledge/products/P00X.md 的 frontmatter，
这里为了开发便利直接写死价格和库存（不频繁改动，冗余可接受）。
"""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import db_session, init_schema  # noqa: E402


USERS: list[tuple[str, str]] = [
    ("U001", "张三"),
    ("U002", "李娜"),
]

# 来源：knowledge/products/P001.md / P002.md / P003.md
PRODUCTS: list[tuple[str, str, float, int]] = [
    ("P001", "无线蓝牙耳机 X1", 299.0, 50),
    ("P002", "智能手表 Pro", 899.0, 30),
    ("P003", "人体工学办公椅", 1299.0, 20),
]


def iso(dt: datetime) -> str:
    """把 datetime 转成秒级 ISO 字符串（SQLite 存储格式）。"""
    return dt.isoformat(timespec="seconds")


def build_orders(now: datetime) -> list[tuple]:
    """构造 10 条固定订单：7 条正常 + 3 条异常（O008/O009/O010），用于压测 Agent。

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


def seed() -> None:
    """入口：重置 schema → 写入 2 用户 / 3 商品 / 10 订单，并打印摘要。"""
    print("· resetting schema...")
    init_schema(drop_existing=True)

    now = datetime.now().replace(microsecond=0)
    order_rows = build_orders(now)

    with db_session() as conn:
        # 用户
        conn.executemany("INSERT INTO users(user_id, name) VALUES (?, ?)", USERS)
        # 商品
        conn.executemany(
            "INSERT INTO products(product_id, name, price, stock) VALUES (?, ?, ?, ?)",
            PRODUCTS,
        )
        # 订单
        conn.executemany(
            "INSERT INTO orders(order_id, user_id, product_id, status, created_at, shipped_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            order_rows,
        )

    print(f"· users: {len(USERS)}")
    print(f"· products: {len(PRODUCTS)}")
    print(f"· orders: {len(order_rows)} (含 3 条异常: O008 疑似丢件 / O009 状态不一致 / O010 久未发货)")
    print("done.")


if __name__ == "__main__":
    seed()
