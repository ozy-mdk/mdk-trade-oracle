"""Silver Layer schema definitions in DuckDB."""

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.silver.schema")


def initialize_silver_schema(db: DuckDBManager) -> None:
    """Initialize Silver layer aggregation and market summary tables in DuckDB."""
    conn = db.get_connection()

    # 1. Silver Table: Daily Broker Summary
    conn.execute("""
        CREATE TABLE IF NOT EXISTS silver_daily_broker_summary (
            trade_date DATE,
            symbol VARCHAR,
            broker_id VARCHAR,
            broker_name VARCHAR,
            is_primary_target BOOLEAN DEFAULT FALSE,
            buy_volume DOUBLE,
            buy_turnover_tl DOUBLE,
            buy_vwap DOUBLE,
            buy_trade_count BIGINT,
            sell_volume DOUBLE,
            sell_turnover_tl DOUBLE,
            sell_vwap DOUBLE,
            sell_trade_count BIGINT,
            total_volume DOUBLE,
            total_turnover_tl DOUBLE,
            net_volume DOUBLE,
            net_flow_tl DOUBLE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol, broker_id)
        );
    """)

    # 2. Silver Table: Daily Market OHLCV & Volume
    conn.execute("""
        CREATE TABLE IF NOT EXISTS silver_market_daily (
            trade_date DATE,
            symbol VARCHAR,
            open_price DOUBLE,
            high_price DOUBLE,
            low_price DOUBLE,
            close_price DOUBLE,
            total_volume DOUBLE,
            total_turnover_tl DOUBLE,
            market_vwap DOUBLE,
            total_trades BIGINT,
            active_brokers BIGINT,
            bofa_net_flow_tl DOUBLE,
            bofa_volume_share DOUBLE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol)
        );
    """)

    logger.info("DuckDB Silver schemas initialized.")
