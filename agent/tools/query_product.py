"""query_product · 按 product_ids 批量返回商品（DB 字段 + md 完整描述 + missing 列表）。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from db import db_session

from ._contract import ToolSpec, ok

_PRODUCTS_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge" / "products"


class QueryProductInput(BaseModel):
    product_ids: list[str] = Field(
        min_length=1,
        description="要查询的商品 ID 列表，例如 ['P001','P002']。支持一次传多个便于对比。",
    )


def _strip_frontmatter(md: str) -> str:
    """剥掉 md 文件头部的 YAML frontmatter（`---` 包围块），避免与结构化字段重复。"""
    if not md.startswith("---\n"):
        return md
    end = md.find("\n---\n", 4)
    if end == -1:
        return md
    return md[end + 5:].lstrip()


def _read_description(product_id: str) -> str | None:
    """返回商品 md 正文；文件缺失时 None（DB 有但 md 无属数据不一致，不静默隐藏）。"""
    path = _PRODUCTS_DIR / f"{product_id}.md"
    if not path.exists():
        return None
    return _strip_frontmatter(path.read_text(encoding="utf-8"))


def query_product_handler(args: QueryProductInput) -> dict:
    """去重保序 → 批量查 DB → 逐个拼接 md 描述。"""
    seen: set[str] = set()
    ordered: list[str] = []
    for pid in args.product_ids:
        if pid not in seen:
            seen.add(pid)
            ordered.append(pid)

    placeholders = ",".join("?" * len(ordered))
    with db_session() as conn:
        rows = conn.execute(
            f"SELECT product_id, name, price, stock FROM products "
            f"WHERE product_id IN ({placeholders})",
            ordered,
        ).fetchall()

    by_id = {r["product_id"]: r for r in rows}

    products: list[dict] = []
    missing: list[str] = []
    for pid in ordered:
        row = by_id.get(pid)
        if row is None:
            missing.append(pid)
            continue
        products.append(
            {
                "product_id": pid,
                "name": row["name"],
                "price": row["price"],
                "stock": row["stock"],
                "description": _read_description(pid),
            }
        )

    return ok({"products": products, "missing": missing})


QUERY_PRODUCT_SPEC = ToolSpec(
    name="query_product",
    description=(
        "按商品 ID 批量查询商品详情，返回价格、库存和完整介绍。"
        "支持一次传多个 ID 做对比。"
    ),
    input_model=QueryProductInput,
    handler=query_product_handler,
)
