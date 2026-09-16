"""Unit and integration tests for PostgreSQL raw trade export and candle generation."""


from mdk_trading_oracle.data.postgres.candle_engine import (
    TIMEFRAME_INTERVALS,
)
from mdk_trading_oracle.data.postgres.connection import PostgresConnectionManager


def test_postgres_connection_and_schema():
    """Test PostgresConnectionManager initialization and table inspection."""
    mgr = PostgresConnectionManager()
    params = mgr.get_connection_params()
    assert "dbname" in params
    assert "host" in params
    assert "port" in params
    assert "user" in params

    attach_str = mgr.get_duckdb_attach_string()
    assert "dbname=" in attach_str
    assert "host=" in attach_str

    # Test connecting
    conn = mgr.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1;")
            assert cur.fetchone()[0] == 1
    finally:
        conn.close()

    # Verify tables exist
    counts = mgr.get_table_counts()
    assert "raw_trades" in counts
    assert "market_candles" in counts
    assert "candle_broker_flows" in counts


def test_timeframe_intervals():
    """Verify all 8 requested timeframes are registered."""
    expected = {"1m", "5m", "15m", "30m", "60m", "120m", "240m", "8h"}
    assert set(TIMEFRAME_INTERVALS.keys()) == expected


def test_candle_math_consistency():
    """Verify high >= low, open/close within range, and non-negative volume."""
    mgr = PostgresConnectionManager()
    conn = mgr.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT count(*) 
                FROM market_candles 
                WHERE high < low OR high < open OR high < close OR low > open OR low > close OR volume < 0;
            """)
            violations = cur.fetchone()[0]
            assert violations == 0, f"Found {violations} candles violating OHLCV consistency!"
    finally:
        conn.close()


def test_bofa_view_queryable():
    """Verify view_market_candles_with_bofa returns data without errors."""
    mgr = PostgresConnectionManager()
    conn = mgr.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT count(*) 
                FROM view_market_candles_with_bofa;
            """)
            cnt = cur.fetchone()[0]
            assert cnt >= 0
    finally:
        conn.close()
