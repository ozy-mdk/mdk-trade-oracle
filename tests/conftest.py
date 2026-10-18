"""Shared pytest configuration."""

import os

import pytest


@pytest.fixture(scope="session", autouse=True)
def _drop_test_sandboxes():
    """Drop the PostgreSQL sandbox schemas created by this test session (tests often skip `close()`)."""
    yield
    try:
        from mdk_trading_oracle.core.db import SANDBOX_PREFIX, PostgresManager

        db = PostgresManager()
        conn = db.get_connection()
        pattern = f"{SANDBOX_PREFIX}{os.getpid()}_%"
        for (name,) in conn.execute(
            "SELECT schema_name FROM information_schema.schemata WHERE schema_name LIKE %s;", (pattern,)
        ).fetchall():
            conn.execute(f"DROP SCHEMA IF EXISTS {name} CASCADE;")
        db.close()
    except Exception:
        pass
