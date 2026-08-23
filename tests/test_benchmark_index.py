"""Tests for BIST 30 Benchmark Index (Bronze -> Silver -> Gold Feature Extraction)."""

from datetime import date

import pytest

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze.ingestor import BronzeIngestor
from mdk_trading_oracle.data.bronze.schema import initialize_bronze_schema
from mdk_trading_oracle.data.silver.schema import initialize_silver_schema
from mdk_trading_oracle.data.silver.transformations import SilverTransformer


@pytest.fixture
def in_memory_db(tmp_path):
    """Create a temporary DuckDB database for test isolation."""
    db_file = tmp_path / "test_benchmark.duckdb"
    manager = DuckDBManager(db_path=db_file)
    initialize_bronze_schema(manager)
    initialize_silver_schema(manager)
    return manager


def test_bronze_benchmark_schema_and_sync(in_memory_db):
    """Test that bronze_bist_index_benchmarks can insert and forward-fill."""
    conn = in_memory_db.get_connection()

    # Insert mock benchmark data
    conn.execute("""
        INSERT INTO bronze_bist_index_benchmarks (
            trade_date, index_code, open_price, high_price, low_price, close_price, volume, daily_return_pct, price_range_pct, is_forward_filled
        ) VALUES 
        ('2026-03-01', 'XU030', 10000.0, 10100.0, 9950.0, 10050.0, 1000000.0, 0.005, 0.015, FALSE),
        ('2026-03-02', 'XU030', 10050.0, 10200.0, 10000.0, 10180.0, 1200000.0, 0.013, 0.020, FALSE);
    """)

    count = conn.execute("SELECT COUNT(*) FROM bronze_bist_index_benchmarks;").fetchone()[0]
    assert count == 2

    # Forward fill up to 2026-03-04
    ingestor = BronzeIngestor(in_memory_db)
    sync_res = ingestor.sync_bist30_benchmarks_to_market(target_end_date=date(2026, 3, 4))
    assert sync_res["status"] == "success"
    assert sync_res["forward_filled_count"] == 2

    total_count = conn.execute("SELECT COUNT(*) FROM bronze_bist_index_benchmarks;").fetchone()[0]
    assert total_count == 4


def test_silver_benchmark_transformation(in_memory_db):
    """Test silver_daily_benchmark_index computes rolling returns, volatility, and trend vs SMA."""
    conn = in_memory_db.get_connection()

    # Populate 25 days of mock benchmark data
    for i in range(1, 26):
        d_str = f"2026-01-{i:02d}"
        price = 10000.0 + i * 50.0
        ret = 0.005 if i > 1 else 0.0
        conn.execute(
            """
            INSERT INTO bronze_bist_index_benchmarks (
                trade_date, index_code, open_price, high_price, low_price, close_price, volume, daily_return_pct, price_range_pct, is_forward_filled
            ) VALUES (?, 'XU030', ?, ?, ?, ?, 1000000.0, ?, 0.01, FALSE);
        """,
            [d_str, price - 10, price + 20, price - 20, price, ret],
        )

    transformer = SilverTransformer(in_memory_db)
    res = transformer.transform_daily_benchmark_index()
    assert res["status"] == "success"
    assert res["rows"] == 25

    # Check rolling metrics
    row = conn.execute("""
        SELECT trade_date, close_price, rolling_5d_return_pct, rolling_20d_return_pct, rolling_20d_volatility, index_trend_vs_20d_sma
        FROM silver_daily_benchmark_index
        WHERE trade_date = '2026-01-25';
    """).fetchone()

    assert row is not None
    assert row[1] == 10000.0 + 25 * 50.0  # close_price
    assert row[2] is not None  # rolling_5d_return_pct
    assert row[3] is not None  # rolling_20d_return_pct
    assert row[4] is not None  # rolling_20d_volatility
    assert row[5] is not None  # index_trend_vs_20d_sma
