from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / os.getenv("NANA_DB_PATH", "db/nana.db")
SCHEMA_PATH = PROJECT_ROOT / "db" / "schema.sql"


def get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def db_session():
    conn = get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_schema(drop_existing: bool = False) -> None:
    if drop_existing and DB_PATH.exists():
        DB_PATH.unlink()
    ddl = SCHEMA_PATH.read_text(encoding="utf-8")
    with db_session() as conn:
        conn.executescript(ddl)
