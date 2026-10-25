"""Incremental silver runs recompute only pending dates, never stray partial bronze days already in silver.

Rebuilding every bronze date overwrote complete silver history with partial bronze days and, through the
carried-forward FIFO ledger, changed every inventory feature after them (2026-10-25). Sandbox schema, no production
tables touched.
"""

from __future__ import annotations

import pytest

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.transformations import SilverTransformer


@pytest.fixture()
def db():
    try:
        manager = PostgresManager(in_memory=True)
        manager.get_connection()
    except Exception as e:  # pragma: no cover - database not available
        pytest.skip(f"PostgreSQL not available: {e}")
    yield manager
    manager.close()


def test_pending_dates_skip_stray_days_and_include_new_or_reingested(db) -> None:
    db.execute("""
        CREATE TABLE bronze_raw_trades (timestamp TIMESTAMP, symbol TEXT);
        CREATE TABLE bronze_ingestion_log (file_path TEXT, trade_date DATE, ingested_at TIMESTAMP);
        CREATE TABLE silver_daily_broker_summary (trade_date DATE, calculated_at TIMESTAMPTZ);
        INSERT INTO bronze_raw_trades VALUES
            ('2022-02-24 10:00', 'GARAN'),  -- stray partial day: in silver, no ingestion-log entry
            ('2026-10-16 10:00', 'GARAN'),  -- ingested before silver computed it: up to date
            ('2026-10-22 10:00', 'GARAN'),  -- re-ingested after silver computed it
            ('2026-10-23 10:00', 'GARAN');  -- new: not in silver yet
        INSERT INTO bronze_ingestion_log VALUES
            ('a', '2026-10-16', '2026-10-17 09:00'),
            ('b', '2026-10-22', '2026-10-25 12:00'),
            ('c', '2026-10-23', '2026-10-25 12:00');
        INSERT INTO silver_daily_broker_summary VALUES
            ('2022-02-24', '2026-09-17 04:56+03'),
            ('2026-10-16', '2026-10-17 10:00+03'),
            ('2026-10-22', '2026-10-23 10:00+03');
    """)
    assert SilverTransformer(db).pending_silver_dates() == ["2026-10-22", "2026-10-23"]
