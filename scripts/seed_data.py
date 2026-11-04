"""NanaShop seed data

"""

from __future__ import annotations

from datetime import datetime, timedelta


USERS: list[tuple[str, str]] = [
    ("U001", "Alex Chen"),
    ("U002", "Nina Li"),
]

# Source: knowledge/products/P001.md / P002.md / P003.md
PRODUCTS: list[tuple[str, str, float, int]] = [
    ("P001", "X1 Wireless Earbuds", 299.0, 50),
    ("P002", "Smartwatch Pro", 899.0, 30),
    ("P003", "Ergonomic Office Chair", 1299.0, 20),
]


def iso(dt: datetime) -> str:
    """ Format a datetime as an ISO timestamp. """
    return dt.isoformat(timespec="seconds")


def build_orders(now: datetime) -> list[tuple]:
    """Build deterministic demo orders.
    Field order: (order_id, user_id, product_id, status, created_at, shipped_at)
    """
    def ago(days: int, hours: int = 0) -> datetime:
        return now - timedelta(days=days, hours=hours)

    return [
        # --- Standard orders ---
        ("O001", "U001", "P001", "pending",   iso(ago(0, 2)),  None),
        ("O002", "U002", "P002", "pending",   iso(ago(1)),     None),
        ("O003", "U001", "P003", "shipped",   iso(ago(3)),     iso(ago(2))),
        ("O004", "U002", "P001", "shipped",   iso(ago(5)),     iso(ago(4))),
        ("O005", "U001", "P002", "delivered", iso(ago(10)),    iso(ago(9))),
        ("O006", "U002", "P003", "delivered", iso(ago(15)),    iso(ago(13))),
        ("O007", "U001", "P001", "refunded",  iso(ago(20)),    iso(ago(19))),
        # --- Abnormal orders for agent robustness checks---
        # O008: shipped 12 days ago but still in transit; possible loss or delivery issue
        ("O008", "U002", "P002", "shipped",   iso(ago(13)),    iso(ago(12))),
        # O009: delivered with a NULL shipped_at; inconsistent data
        ("O009", "U001", "P003", "delivered", iso(ago(7)),     None),
        # O010: pending for 20 days; extended fulfillment delay
        ("O010", "U002", "P001", "pending",   iso(ago(20)),    None),
    ]
