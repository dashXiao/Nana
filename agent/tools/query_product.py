"""query_product: batch retrieval by product_ids, including DB fields, full descriptions, and missing IDs."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, Field

from db import db_session

from ._contract import ToolSpec, ok

_PRODUCTS_DIR = Path(__file__).resolve().parent.parent.parent / "knowledge" / "products"


class QueryProductInput(BaseModel):
    product_ids: list[str] = Field(
        min_length=1,
        description="Product IDs to retrieve, such as ['P001', 'P002']; pass multiple IDs for comparison.",
    )


def _strip_frontmatter(md: str) -> str:
    """Strip YAML frontmatter delimited by --- to avoid duplicating structured fields."""
    if not md.startswith("---\n"):
        return md
    end = md.find("\n---\n", 4)
    if end == -1:
        return md
    return md[end + 5:].lstrip()


def _read_description(product_id: str) -> str | None:
    """Return the product Markdown body, or None to expose a missing description file."""
    path = _PRODUCTS_DIR / f"{product_id}.md"
    if not path.exists():
        return None
    return _strip_frontmatter(path.read_text(encoding="utf-8"))


def query_product_handler(args: QueryProductInput) -> dict:
    """Deduplicate in input order, query the DB in a batch, then attach Markdown descriptions."""
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
        "Retrieve product details by ID, including prices in CNY, stock, and full descriptions. "
        "Pass multiple IDs for comparison."
    ),
    input_model=QueryProductInput,
    handler=query_product_handler,
)
