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


def test_silver_benchmark_shock_days(in_memory_db):
    """Test silver_daily_benchmark_index computes shock days (>= 3%) and days-since intervals."""
    conn = in_memory_db.get_connection()

    # Clear and insert 5 distinct sessions with positive, negative, and normal moves
    conn.execute("DELETE FROM bronze_bist_index_benchmarks;")
    test_days = [
        ("2026-02-01", 10000.0, 0.005),   # Normal
        ("2026-02-02", 10350.0, 0.035),   # POSITIVE_SHOCK (+3.5%)
        ("2026-02-03", 10250.0, -0.010),  # Normal
        ("2026-02-04", 9800.0, -0.042),   # NEGATIVE_SHOCK (-4.2%)
        ("2026-02-05", 9820.0, 0.002),    # Normal
    ]
    for d_str, price, ret in test_days:
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

    rows = conn.execute("""
        SELECT trade_date, daily_return_pct, is_shock_day, shock_type, days_since_last_shock, days_since_last_positive_shock, days_since_last_negative_shock
        FROM silver_daily_benchmark_index
        ORDER BY trade_date ASC;
    """).fetchall()

    assert len(rows) == 5

    # Day 1: 2026-02-01 (Normal)
    assert rows[0][2] is False
    assert rows[0][3] == "NONE"
    assert rows[0][4] is None

    # Day 2: 2026-02-02 (+3.5% POSITIVE_SHOCK)
    assert rows[1][2] is True
    assert rows[1][3] == "POSITIVE_SHOCK"

    # Day 3: 2026-02-03 (Normal - 1 day after positive shock)
    assert rows[2][2] is False
    assert rows[2][3] == "NONE"
    assert rows[2][4] == 1  # days_since_last_shock
    assert rows[2][5] == 1  # days_since_last_positive_shock
    assert rows[2][6] is None  # no negative shock yet

    # Day 4: 2026-02-04 (-4.2% NEGATIVE_SHOCK)
    assert rows[3][2] is True
    assert rows[3][3] == "NEGATIVE_SHOCK"
    assert rows[3][4] == 2  # days_since_last_shock (from day 2)
    assert rows[3][5] == 2  # days_since_last_positive_shock

    # Day 5: 2026-02-05 (Normal - 1 day after negative shock, 3 days after positive shock)
    assert rows[4][2] is False
    assert rows[4][3] == "NONE"
    assert rows[4][4] == 1  # days_since_last_shock (from day 4)
    assert rows[4][5] == 3  # days_since_last_positive_shock (from day 2)
    assert rows[4][6] == 1  # days_since_last_negative_shock (from day 4)

