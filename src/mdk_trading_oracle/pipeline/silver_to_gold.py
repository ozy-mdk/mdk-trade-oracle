"""Silver to Gold feature engineering pipeline."""

from typing import Dict
from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.pipeline.gold")


class SilverToGoldPipeline:
    """Transforms normalized Silver summaries into Gold BofA order flow features."""

    def __init__(self, db_manager: DuckDBManager):
        self.db = db_manager
        self.settings = get_settings()

    def run(self) -> Dict[str, int]:
        """Execute Silver -> Gold transformation."""
        conn = self.db.get_connection()
        logger.info("Running Silver -> Gold transformation...")

        primary_inst = self.settings.primary_institution

        # Build Gold BofA Flow table using window functions in DuckDB
        conn.execute(f"""
            WITH daily_market_totals AS (
                SELECT
                    date_val,
                    symbol,
                    SUM(total_buy_volume + total_sell_volume) / 2.0 AS total_symbol_volume,
                    SUM(total_buy_tl + total_sell_tl) / 2.0 AS total_symbol_tl
                FROM silver_daily_broker_summary
                GROUP BY date_val, symbol
            ),
            daily_prices AS (
                SELECT
                    date_val,
                    symbol,
                    -- Approximation of close price from last trade of the day
                    AVG(price) AS close_price
                FROM silver_broker_transactions
                GROUP BY date_val, symbol
            ),
            daily_bofa AS (
                SELECT
                    date_val,
                    symbol,
                    COALESCE(SUM(total_buy_tl), 0.0) AS bofa_buy_tl,
                    COALESCE(SUM(total_sell_tl), 0.0) AS bofa_sell_tl,
                    COALESCE(SUM(net_tl), 0.0) AS bofa_net_tl,
                    COALESCE(SUM(total_buy_volume + total_sell_volume), 0.0) AS bofa_total_vol
                FROM silver_daily_broker_summary
                WHERE broker_id = '{primary_inst}'
                GROUP BY date_val, symbol
            ),
            base_metrics AS (
                SELECT
                    m.date_val,
                    m.symbol,
                    COALESCE(p.close_price, 0.0) AS close_price,
                    m.total_symbol_volume,
                    m.total_symbol_tl,
                    COALESCE(b.bofa_buy_tl, 0.0) AS bofa_buy_tl,
                    COALESCE(b.bofa_sell_tl, 0.0) AS bofa_sell_tl,
                    COALESCE(b.bofa_net_tl, 0.0) AS bofa_net_tl,
                    CASE 
                        WHEN m.total_symbol_tl > 0 
                        THEN (COALESCE(b.bofa_buy_tl, 0.0) + COALESCE(b.bofa_sell_tl, 0.0)) / (2.0 * m.total_symbol_tl)
                        ELSE 0.0 
                    END AS bofa_volume_share,
                    CASE 
                        WHEN m.total_symbol_tl > 0 
                        THEN COALESCE(b.bofa_net_tl, 0.0) / m.total_symbol_tl
                        ELSE 0.0 
                    END AS bofa_net_share
                FROM daily_market_totals m
                LEFT JOIN daily_prices p ON m.date_val = p.date_val AND m.symbol = p.symbol
                LEFT JOIN daily_bofa b ON m.date_val = b.date_val AND m.symbol = b.symbol
            ),
            gold_calculated AS (
                SELECT
                    date_val,
                    symbol,
                    close_price,
                    total_symbol_volume,
                    total_symbol_tl,
                    bofa_buy_tl,
                    bofa_sell_tl,
                    bofa_net_tl,
                    bofa_volume_share,
                    bofa_net_share,
                    -- Rolling Window Net Flows
                    COALESCE(AVG(bofa_net_tl) OVER (PARTITION BY symbol ORDER BY date_val ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), bofa_net_tl) AS bofa_net_tl_roll_3d,
                    COALESCE(AVG(bofa_net_tl) OVER (PARTITION BY symbol ORDER BY date_val ROWS BETWEEN 4 PRECEDING AND CURRENT ROW), bofa_net_tl) AS bofa_net_tl_roll_5d,
                    COALESCE(AVG(bofa_net_tl) OVER (PARTITION BY symbol ORDER BY date_val ROWS BETWEEN 9 PRECEDING AND CURRENT ROW), bofa_net_tl) AS bofa_net_tl_roll_10d,
                    COALESCE(SUM(bofa_net_tl) OVER (PARTITION BY symbol ORDER BY date_val ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), bofa_net_tl) AS bofa_cum_net_tl_20d,
                    -- Acceleration: diff between current 3d roll vs previous 5d roll
                    (bofa_net_tl - COALESCE(LAG(bofa_net_tl, 5) OVER (PARTITION BY symbol ORDER BY date_val), 0.0)) AS bofa_flow_acceleration_5d,
                    -- Z-Score of BofA net flow relative to 20-day history
                    CASE 
                        WHEN COALESCE(STDDEV(bofa_net_tl) OVER (PARTITION BY symbol ORDER BY date_val ROWS BETWEEN 19 PRECEDING AND CURRENT ROW), 0.0) > 0
                        THEN (bofa_net_tl - AVG(bofa_net_tl) OVER (PARTITION BY symbol ORDER BY date_val ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)) / 
                             STDDEV(bofa_net_tl) OVER (PARTITION BY symbol ORDER BY date_val ROWS BETWEEN 19 PRECEDING AND CURRENT ROW)
                        ELSE 0.0
                    END AS bofa_flow_zscore_20d
                FROM base_metrics
            )
            INSERT OR REPLACE INTO gold_bofa_flow_metrics
            SELECT * FROM gold_calculated;
        """)

        # Export to Gold Parquet directory
        gold_parquet = self.settings.gold_dir / "bofa_flow_metrics.parquet"
        conn.execute(f"COPY gold_bofa_flow_metrics TO '{gold_parquet.as_posix()}' (FORMAT PARQUET);")

        gold_count = conn.execute("SELECT COUNT(*) FROM gold_bofa_flow_metrics").fetchone()[0]
        logger.info(f"Gold layer updated: {gold_count} flow metric records generated.")

        return {"gold_bofa_flow_metrics_count": gold_count}
