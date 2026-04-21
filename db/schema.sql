-- NanaShop Agent · SQLite schema
-- 极简三表：users / products / orders

DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS users;

CREATE TABLE users (
    user_id TEXT PRIMARY KEY,
    name    TEXT NOT NULL
);

CREATE TABLE products (
    product_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    price      REAL NOT NULL CHECK (price >= 0),
    stock      INTEGER NOT NULL CHECK (stock >= 0)
);

-- status 枚举：pending / shipped / delivered / refunded
-- shipped_at 仅在 status >= shipped 时非空；具体由工具层现算 days_since_shipped
CREATE TABLE orders (
    order_id   TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL REFERENCES users(user_id),
    product_id TEXT NOT NULL REFERENCES products(product_id),
    status     TEXT NOT NULL CHECK (status IN ('pending','shipped','delivered','refunded')),
    created_at TEXT NOT NULL,
    shipped_at TEXT
);

CREATE INDEX idx_orders_user ON orders(user_id);
CREATE INDEX idx_orders_status ON orders(status);
