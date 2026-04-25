"""NanaShop SQLite 种子数据脚本：重置 schema → 写入固定数据。"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from db import db_session, init_schema  # noqa: E402
from scripts.seed_data import PRODUCTS, USERS, build_orders  # noqa: E402


def seed() -> None:
    """入口：重置 schema → 写入  用户 / 商品 / 订单，并打印摘要。"""
    print("· resetting schema...")
    init_schema(drop_existing=True)

    now = datetime.now().replace(microsecond=0)
    order_rows = build_orders(now)

    with db_session() as conn:
        conn.executemany("INSERT INTO users(user_id, name) VALUES (?, ?)", USERS)
        conn.executemany(
            "INSERT INTO products(product_id, name, price, stock) VALUES (?, ?, ?, ?)",
            PRODUCTS,
        )
        conn.executemany(
            "INSERT INTO orders(order_id, user_id, product_id, status, created_at, shipped_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            order_rows,
        )

    print(f"· users: {len(USERS)}")
    print(f"· products: {len(PRODUCTS)}")
    print(f"· orders: {len(order_rows)}")
    print("done.")


if __name__ == "__main__":
    seed()
