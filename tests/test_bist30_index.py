"""Unit tests for synthetic BIST 30 (XU030) index calculation and integration."""

import duckdb
import pytest

from mdk_trading_oracle.api.resolvers import get_candles, get_daily_fifo, get_event_study, get_instruments
from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.data.bist30.index_engine import BIST30IndexEngine
from mdk_trading_oracle.data.postgres.connection import PostgresConnectionManager


@pytest.fixture(scope="module")
def engine():
    return BIST30IndexEngine()


def test_get_constituents(engine):
    """Verify BIST 30 constituent resolution returns valid constituents."""
    constituents_sep = engine.get_constituents("2026-09-14")
    assert len(constituents_sep) == 30
    assert "XU030" not in constituents_sep
    assert "AKBNK" in constituents_sep
    assert "THYAO" in constituents_sep

    constituents_mar = engine.get_constituents("2026-03-02")
    assert len(constituents_mar) == 30
    assert "XU030" not in constituents_mar


def test_get_constituent_weights(engine):
    """Verify NNLS constituent weights calibration produces non-negative valid weights."""
    weights = engine.get_constituent_weights("2026-09-14")
    assert len(weights) == 30
    # All weights must be non-negative
    for sym, w in weights.items():
        assert w >= 0.0, f"Weight for {sym} is negative: {w}"

    # Weights cache should work on repeated calls
    cached_weights = engine.get_constituent_weights("2026-09-14")
    assert cached_weights == weights


def test_generate_xu030_candles(engine):
    """Verify synthetic multi-timeframe candles and broker flows are generated in PostgreSQL."""
    count = engine.generate_xu030_candles("2026-09-14", timeframe="60m")
    assert count > 0

    pg_mgr = PostgresConnectionManager()
    conn = pg_mgr.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT open, high, low, close, volume, turnover_tl
                FROM market_candles
                WHERE symbol = 'XU030' AND timeframe = '60m'
                  AND bucket_start >= '2026-09-14 00:00:00' AND bucket_start <= '2026-09-14 23:59:59';
            """)
            rows = cur.fetchall()
            assert len(rows) == count
            for o, h, low_p, c, v, val in rows:
                assert h >= max(o, c) - 1e-4
                assert low_p <= min(o, c) + 1e-4
                assert v > 0
                assert val > 0

            # Verify consolidated broker flows for MLB
            cur.execute("""
                SELECT count(*), sum(buy_turnover_tl), sum(sell_turnover_tl)
                FROM candle_broker_flows
                WHERE symbol = 'XU030' AND timeframe = '60m' AND broker_id = 'MLB'
                  AND bucket_start >= '2026-09-14 00:00:00' AND bucket_start <= '2026-09-14 23:59:59';
            """)
            b_cnt, buy_tl, sell_tl = cur.fetchone()
            assert b_cnt > 0
            assert buy_tl > 0
            assert sell_tl > 0
    finally:
        conn.close()


def test_generate_xu030_daily_summary(engine):
    """Verify DuckDB silver_daily_stock_summary contains consistent XU030 metrics."""
    cnt = engine.generate_xu030_daily_summary("2026-09-14")
    assert cnt >= 1

    settings = get_settings()
    conn = duckdb.connect(str(settings.duckdb_path), read_only=True)
    try:
        row = conn.execute("""
            SELECT symbol, close_price, adj_close_price, total_turnover_tl, bofa_net_flow_tl
            FROM silver_daily_stock_summary
            WHERE symbol = 'XU030' AND trade_date = '2026-09-14';
        """).fetchone()
        assert row is not None
        sym, close, adj_close, turnover, bofa_net = row
        assert sym == "XU030"
        assert close > 10000.0
        assert adj_close == close
        assert turnover > 0.0
    finally:
        conn.close()


def test_xu030_graphql_resolvers():
    """Verify GraphQL resolvers return XU030 correctly across instruments, candles, fifo, and event studies."""
    # 1. Instruments
    insts = get_instruments()
    xu030_inst = next((i for i in insts if i.symbol == "XU030"), None)
    assert xu030_inst is not None
    assert xu030_inst.sector == "ENDEKS"

    # 2. Candles
    candles = get_candles(symbol="XU030", timeframe="60m", date="2026-09-14", broker_id="MLB")
    assert len(candles) > 0
    assert candles[0].symbol == "XU030"
    assert candles[0].close > 10000.0

    # 3. Daily FIFO
    daily_fifo = get_daily_fifo(broker_id="MLB", symbol="XU030", limit=5)
    assert len(daily_fifo) > 0
    assert daily_fifo[0].symbol == "XU030"
    assert daily_fifo[0].buy_turnover_tl > 0

    # 4. Event Study
    study = get_event_study(symbol="XU030", condition_type="DAILY_RETURN", min_value=-2.0, max_value=-1.0, forward_days=5)
    assert study.symbol == "XU030"
    assert study.total_occurrences > 0
    assert len(study.horizon_stats) == 5
