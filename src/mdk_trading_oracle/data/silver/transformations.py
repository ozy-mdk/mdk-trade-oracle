"""Silver Layer Transformations: Computes clean broker summaries, VWAP, and daily market metrics."""

from typing import Any

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.silver.schema import initialize_silver_schema

logger = get_logger("mdk_oracle.data.silver.transformations")


class SilverTransformer:
    """Transforms raw Bronze tick data into clean, aggregated Silver tables in DuckDB."""

    def __init__(self, db: DuckDBManager):
        self.db = db
        self.settings = get_settings()

    def transform_daily_broker_summary(self) -> dict[str, Any]:
        """Aggregate buy/sell volume, turnover, and buy/sell VWAP per (trade_date, symbol, broker_id)."""
        conn = self.db.get_connection()
        logger.info("Computing `silver_daily_broker_summary` from `bronze_raw_trades`...")

        query = """
            CREATE OR REPLACE TABLE silver_daily_broker_summary AS
            WITH buys AS (
                SELECT 
                    CAST(timestamp AS DATE) AS trade_date,
                    symbol,
                    buyer_broker_id AS broker_id,
                    SUM(volume) AS buy_volume,
                    SUM(price * volume) AS buy_turnover_tl,
                    SUM(price * volume) / NULLIF(SUM(volume), 0) AS buy_vwap,
                    COUNT(*) AS buy_trade_count
                FROM bronze_raw_trades
                WHERE buyer_broker_id IS NOT NULL AND buyer_broker_id != ''
                GROUP BY CAST(timestamp AS DATE), symbol, buyer_broker_id
            ),
            sells AS (
                SELECT 
                    CAST(timestamp AS DATE) AS trade_date,
                    symbol,
                    seller_broker_id AS broker_id,
                    SUM(volume) AS sell_volume,
                    SUM(price * volume) AS sell_turnover_tl,
                    SUM(price * volume) / NULLIF(SUM(volume), 0) AS sell_vwap,
                    COUNT(*) AS sell_trade_count
                FROM bronze_raw_trades
                WHERE seller_broker_id IS NOT NULL AND seller_broker_id != ''
                GROUP BY CAST(timestamp AS DATE), symbol, seller_broker_id
            ),
            combined AS (
                SELECT 
                    COALESCE(b.trade_date, s.trade_date) AS trade_date,
                    COALESCE(b.symbol, s.symbol) AS symbol,
                    COALESCE(b.broker_id, s.broker_id) AS broker_id,
                    COALESCE(b.buy_volume, 0.0) AS buy_volume,
                    COALESCE(b.buy_turnover_tl, 0.0) AS buy_turnover_tl,
                    b.buy_vwap,
                    COALESCE(b.buy_trade_count, 0) AS buy_trade_count,
                    COALESCE(s.sell_volume, 0.0) AS sell_volume,
                    COALESCE(s.sell_turnover_tl, 0.0) AS sell_turnover_tl,
                    s.sell_vwap,
                    COALESCE(s.sell_trade_count, 0) AS sell_trade_count,
                    COALESCE(b.buy_volume, 0.0) + COALESCE(s.sell_volume, 0.0) AS total_volume,
                    COALESCE(b.buy_turnover_tl, 0.0) + COALESCE(s.sell_turnover_tl, 0.0) AS total_turnover_tl,
                    COALESCE(b.buy_volume, 0.0) - COALESCE(s.sell_volume, 0.0) AS net_volume,
                    COALESCE(b.buy_turnover_tl, 0.0) - COALESCE(s.sell_turnover_tl, 0.0) AS net_flow_tl
                FROM buys b
                FULL OUTER JOIN sells s
                    ON b.trade_date = s.trade_date 
                    AND b.symbol = s.symbol 
                    AND b.broker_id = s.broker_id
            )
            SELECT 
                c.trade_date,
                c.symbol,
                c.broker_id,
                COALESCE(brk.broker_name, c.broker_id) AS broker_name,
                COALESCE(brk.is_primary_target, FALSE) AS is_primary_target,
                c.buy_volume,
                c.buy_turnover_tl,
                c.buy_vwap,
                c.buy_trade_count,
                c.sell_volume,
                c.sell_turnover_tl,
                c.sell_vwap,
                c.sell_trade_count,
                c.total_volume,
                c.total_turnover_tl,
                c.net_volume,
                c.net_flow_tl,
                CURRENT_TIMESTAMP AS calculated_at
            FROM combined c
            LEFT JOIN bronze_brokers brk ON c.broker_id = brk.broker_id;
        """
        conn.execute(query)

        rows = conn.execute("SELECT COUNT(*) FROM silver_daily_broker_summary;").fetchone()[0]
        logger.info(f"Successfully populated `silver_daily_broker_summary`: {rows:,} rows.")
        return {"table": "silver_daily_broker_summary", "rows": rows, "status": "success"}

    def transform_market_daily(self) -> dict[str, Any]:
        """Compute Daily OHLCV, market turnover, active brokers, and BofA volume share."""
        conn = self.db.get_connection()
        logger.info("Computing `silver_market_daily` OHLCV and market metrics...")

        query = """
            CREATE OR REPLACE TABLE silver_market_daily AS
            WITH daily_trades AS (
                SELECT 
                    CAST(timestamp AS DATE) AS trade_date,
                    symbol,
                    MIN(price) AS low_price,
                    MAX(price) AS high_price,
                    SUM(volume) AS total_volume,
                    SUM(price * volume) AS total_turnover_tl,
                    SUM(price * volume) / NULLIF(SUM(volume), 0) AS market_vwap,
                    COUNT(*) AS total_trades
                FROM bronze_raw_trades
                GROUP BY CAST(timestamp AS DATE), symbol
            ),
            first_last_prices AS (
                SELECT 
                    trade_date,
                    symbol,
                    ARG_MIN(price, timestamp) AS open_price,
                    ARG_MAX(price, timestamp) AS close_price
                FROM (
                    SELECT 
                        CAST(timestamp AS DATE) AS trade_date,
                        symbol,
                        price,
                        timestamp
                    FROM bronze_raw_trades
                )
                GROUP BY trade_date, symbol
            ),
            bofa_metrics AS (
                SELECT 
                    trade_date,
                    symbol,
                    SUM(net_flow_tl) AS bofa_net_flow_tl,
                    SUM(total_volume) AS bofa_total_volume
                FROM silver_daily_broker_summary
                WHERE broker_id = 'MLB' OR is_primary_target = TRUE
                GROUP BY trade_date, symbol
            ),
            broker_counts AS (
                SELECT 
                    trade_date,
                    symbol,
                    COUNT(DISTINCT broker_id) AS active_brokers
                FROM silver_daily_broker_summary
                GROUP BY trade_date, symbol
            )
            SELECT 
                d.trade_date,
                d.symbol,
                f.open_price,
                d.high_price,
                d.low_price,
                f.close_price,
                d.total_volume,
                d.total_turnover_tl,
                d.market_vwap,
                d.total_trades,
                COALESCE(bc.active_brokers, 0) AS active_brokers,
                COALESCE(bm.bofa_net_flow_tl, 0.0) AS bofa_net_flow_tl,
                COALESCE(bm.bofa_total_volume, 0.0) / NULLIF(d.total_volume * 2.0, 0.0) AS bofa_volume_share,
                CURRENT_TIMESTAMP AS calculated_at
            FROM daily_trades d
            JOIN first_last_prices f ON d.trade_date = f.trade_date AND d.symbol = f.symbol
            LEFT JOIN bofa_metrics bm ON d.trade_date = bm.trade_date AND d.symbol = bm.symbol
            LEFT JOIN broker_counts bc ON d.trade_date = bc.trade_date AND d.symbol = bc.symbol;
        """
        conn.execute(query)

        rows = conn.execute("SELECT COUNT(*) FROM silver_market_daily;").fetchone()[0]
        logger.info(f"Successfully populated `silver_market_daily`: {rows:,} rows.")
        return {"table": "silver_market_daily", "rows": rows, "status": "success"}

    def run_all(self) -> dict[str, Any]:
        """Run full Silver transformation pipeline in dependency order."""
        initialize_silver_schema(self.db)
        res_broker = self.transform_daily_broker_summary()
        res_market = self.transform_market_daily()
        return {
            "silver_daily_broker_summary": res_broker,
            "silver_market_daily": res_market,
            "status": "success",
        }
