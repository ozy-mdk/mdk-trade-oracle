"""PostgreSQL schema definitions and DDL scripts for raw trades and candle tables."""

DDL_CREATE_RAW_TRADES = """
CREATE TABLE IF NOT EXISTS raw_trades (
    trade_id VARCHAR(64),
    timestamp TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    symbol VARCHAR(16) NOT NULL,
    price DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    buyer_broker_id VARCHAR(16) NOT NULL,
    seller_broker_id VARCHAR(16) NOT NULL,
    raw_source VARCHAR(256),
    ingested_at TIMESTAMP WITHOUT TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_raw_trades_sym_time ON raw_trades (symbol, timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_trades_time ON raw_trades (timestamp);
CREATE INDEX IF NOT EXISTS idx_raw_trades_buyer ON raw_trades (buyer_broker_id);
CREATE INDEX IF NOT EXISTS idx_raw_trades_seller ON raw_trades (seller_broker_id);
"""

DDL_CREATE_MARKET_CANDLES = """
CREATE TABLE IF NOT EXISTS market_candles (
    timeframe VARCHAR(8) NOT NULL,
    bucket_start TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    bucket_end TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    symbol VARCHAR(16) NOT NULL,
    open DOUBLE PRECISION NOT NULL,
    high DOUBLE PRECISION NOT NULL,
    low DOUBLE PRECISION NOT NULL,
    close DOUBLE PRECISION NOT NULL,
    volume DOUBLE PRECISION NOT NULL,
    turnover_tl DOUBLE PRECISION NOT NULL,
    vwap DOUBLE PRECISION NOT NULL,
    trade_count INTEGER NOT NULL,
    PRIMARY KEY (timeframe, symbol, bucket_start)
);

CREATE INDEX IF NOT EXISTS idx_market_candles_lookup ON market_candles (symbol, timeframe, bucket_start);
"""

DDL_CREATE_CANDLE_BROKER_FLOWS = """
CREATE TABLE IF NOT EXISTS candle_broker_flows (
    timeframe VARCHAR(8) NOT NULL,
    bucket_start TIMESTAMP WITHOUT TIME ZONE NOT NULL,
    symbol VARCHAR(16) NOT NULL,
    broker_id VARCHAR(16) NOT NULL,
    buy_volume DOUBLE PRECISION NOT NULL DEFAULT 0,
    buy_turnover_tl DOUBLE PRECISION NOT NULL DEFAULT 0,
    buy_vwap DOUBLE PRECISION,
    sell_volume DOUBLE PRECISION NOT NULL DEFAULT 0,
    sell_turnover_tl DOUBLE PRECISION NOT NULL DEFAULT 0,
    sell_vwap DOUBLE PRECISION,
    net_volume DOUBLE PRECISION NOT NULL DEFAULT 0,
    net_flow_tl DOUBLE PRECISION NOT NULL DEFAULT 0,
    matched_volume DOUBLE PRECISION NOT NULL DEFAULT 0,
    matched_buy_value_tl DOUBLE PRECISION NOT NULL DEFAULT 0,
    matched_sell_value_tl DOUBLE PRECISION NOT NULL DEFAULT 0,
    realized_pnl_tl DOUBLE PRECISION NOT NULL DEFAULT 0,
    PRIMARY KEY (timeframe, symbol, broker_id, bucket_start)
);

CREATE INDEX IF NOT EXISTS idx_candle_broker_flows_lookup ON candle_broker_flows (broker_id, symbol, timeframe, bucket_start);
CREATE INDEX IF NOT EXISTS idx_candle_broker_flows_time ON candle_broker_flows (timeframe, bucket_start);
"""

DDL_CREATE_VIEWS = """
CREATE OR REPLACE VIEW view_market_candles_with_bofa AS
SELECT 
    c.timeframe,
    c.bucket_start,
    c.bucket_end,
    c.symbol,
    c.open,
    c.high,
    c.low,
    c.close,
    c.volume AS market_volume,
    c.turnover_tl AS market_turnover_tl,
    c.vwap AS market_vwap,
    c.trade_count,
    COALESCE(b.buy_volume, 0) AS bofa_buy_volume,
    COALESCE(b.buy_turnover_tl, 0) AS bofa_buy_turnover_tl,
    b.buy_vwap AS bofa_buy_vwap,
    COALESCE(b.sell_volume, 0) AS bofa_sell_volume,
    COALESCE(b.sell_turnover_tl, 0) AS bofa_sell_turnover_tl,
    b.sell_vwap AS bofa_sell_vwap,
    COALESCE(b.net_volume, 0) AS bofa_net_volume,
    COALESCE(b.net_flow_tl, 0) AS bofa_net_flow_tl,
    COALESCE(b.matched_volume, 0) AS bofa_matched_volume,
    COALESCE(b.realized_pnl_tl, 0) AS bofa_realized_pnl_tl,
    CASE 
        WHEN c.turnover_tl > 0 THEN (COALESCE(b.buy_turnover_tl, 0) + COALESCE(b.sell_turnover_tl, 0)) / (2.0 * c.turnover_tl)
        ELSE 0 
    END AS bofa_market_share
FROM market_candles c
LEFT JOIN candle_broker_flows b
    ON c.timeframe = b.timeframe 
    AND c.symbol = b.symbol 
    AND c.bucket_start = b.bucket_start 
    AND b.broker_id = 'MLB';
"""

ALL_DDL_SCRIPTS = [
    DDL_CREATE_RAW_TRADES,
    DDL_CREATE_MARKET_CANDLES,
    DDL_CREATE_CANDLE_BROKER_FLOWS,
    DDL_CREATE_VIEWS,
]
