"""Global pytest setup with a dedicated seeded test database.

Set NANA_DB_PATH before importing database modules to protect the development DB.
Seed two users, three products, and ten orders once per test session.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Set this before importing database modules.
os.environ["NANA_DB_PATH"] = "db/nana_test.db"

# All model responses are mocked; local tests need no real API credentials.
os.environ.setdefault("DASHSCOPE_API_KEY", "sk-test-placeholder")
os.environ.setdefault("DASHSCOPE_BASE_URL", "https://example.invalid/v1")
os.environ.setdefault("LLM_MODEL", "qwen3.7-plus")
os.environ.setdefault("LLM_LIGHT_MODEL", "qwen3.8-flash")
os.environ.setdefault("LLM_ENABLE_THINKING", "false")
os.environ.setdefault("LLM_MODEL_PRICING", "2.0,8.0")
os.environ.setdefault("LLM_LIGHT_MODEL_PRICING", "0.8,2.7")

import pytest  # noqa: E402

from scripts.seed import seed  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def seeded_db():
    """Share one seeded database across the test session."""
    seed()
    yield
