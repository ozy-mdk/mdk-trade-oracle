from typing import Any

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.gold.schema import initialize_gold_schema

logger = get_logger("mdk_oracle.data.gold.feature_engineering")


class GoldFeatureEngineer:
    """Computes high-alpha feature representations and institutional signals for Gold layer."""

    def __init__(self, db: PostgresManager):
        self.db = db
        self.settings = get_settings()

    def compute_institutional_signals(self) -> dict[str, Any]:
        """Compute rolling 5-day and 20-day BofA flows and Z-scores from `silver_daily_stock_summary`."""
        conn = self.db.get_connection()
        logger.info("Computing `gold_institutional_daily_signals` from Silver tables...")

        query = """
            CREATE OR REPLACE TABLE gold_institutional_daily_signals AS
            WITH base AS (
                SELECT 
                    trade_date,
                    symbol,
                    bofa_net_flow_tl,
                    bofa_stock_turnover_share AS bofa_volume_share,
                    market_vwap,
                    close_price,
                    SUM(bofa_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 4 PRECEDING AND CURRENT ROW
                    ) AS bofa_accum_5d_tl,
                    SUM(bofa_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS bofa_accum_20d_tl,
                    AVG(bofa_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS mean_flow_20d,
                    STDDEV(bofa_net_flow_tl) OVER (
                        PARTITION BY symbol ORDER BY trade_date ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
                    ) AS std_flow_20d
                FROM silver_daily_stock_summary
            )
            SELECT 
                trade_date,
                symbol,
                bofa_net_flow_tl,
                bofa_volume_share,
                CASE 
                    WHEN std_flow_20d > 0 THEN (bofa_net_flow_tl - mean_flow_20d) / std_flow_20d
                    ELSE 0.0
                END AS bofa_flow_zscore_20d,
                bofa_accum_5d_tl,
                bofa_accum_20d_tl,
                market_vwap,
                close_price,
                CURRENT_TIMESTAMP AS calculated_at
            FROM base;
        """
        conn.execute(query)

        rows = conn.execute("SELECT COUNT(*) FROM gold_institutional_daily_signals;").fetchone()[0]
        logger.info(f"Successfully populated `gold_institutional_daily_signals`: {rows:,} rows.")
        return {"table": "gold_institutional_daily_signals", "rows": rows, "status": "success"}

    def run_all(self, **kwargs: Any) -> dict[str, Any]:
        """Execute all Gold layer feature tables and institutional signals."""
        initialize_gold_schema(self.db)
        sig_res = self.compute_institutional_signals()
        return {
            "institutional_signals": sig_res,
            "status": "success",
        }
