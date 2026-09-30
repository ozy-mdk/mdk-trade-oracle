"""Gold Layer schema definitions for PostgreSQL."""

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.gold.schema")


def initialize_gold_schema(db: PostgresManager) -> None:
    """Initialize Gold layer feature tables and institutional signals."""

    # 1. Rolling Institutional Flow Signals & Multi-Day Accumulation
    db.execute("""
        CREATE TABLE IF NOT EXISTS gold_institutional_daily_signals (
            trade_date DATE,
            symbol VARCHAR,
            bofa_net_flow_tl DOUBLE PRECISION,
            bofa_volume_share DOUBLE PRECISION,
            bofa_flow_zscore_20d DOUBLE PRECISION,
            bofa_accum_5d_tl DOUBLE PRECISION,
            bofa_accum_20d_tl DOUBLE PRECISION,
            market_vwap DOUBLE PRECISION,
            close_price DOUBLE PRECISION,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol)
        );
    """)
