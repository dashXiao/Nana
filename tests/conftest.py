"""Pytest 全局配置：隔离测试 DB、session 级 seed。

在任何 test 模块导入 db 之前，先把 NANA_DB_PATH 指到专用测试 DB，
避免污染开发库。seed() 每次测试 session 跑一次，写入 2 用户 / 3 商品 / 10 订单。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 必须在 `from db import ...` 之前设置
os.environ["NANA_DB_PATH"] = "db/nana_test.db"

import pytest  # noqa: E402

from scripts.seed import seed  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def seeded_db():
    """整个测试 session 共享一次 seed 后的数据库。"""
    seed()
    yield
