"""Gold Layer schema definitions in DuckDB."""

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.gold.schema")


def initialize_gold_schema(db: DuckDBManager) -> None:
    """Initialize Gold layer feature tables in DuckDB."""
    conn = db.get_connection()

    conn.execute("""
        CREATE TABLE IF NOT EXISTS gold_institutional_daily_signals (
            trade_date DATE,
            symbol VARCHAR,
            bofa_net_flow_tl DOUBLE,
            bofa_volume_share DOUBLE,
            bofa_flow_zscore_20d DOUBLE,
            bofa_accum_5d_tl DOUBLE,
            bofa_accum_20d_tl DOUBLE,
            market_vwap DOUBLE,
            close_price DOUBLE,
            calculated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (trade_date, symbol)
        );
    """)

    logger.info("DuckDB Gold schemas initialized.")
