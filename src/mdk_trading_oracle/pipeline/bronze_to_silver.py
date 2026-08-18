"""Bronze to Silver normalization and daily broker aggregation pipeline."""

from typing import Dict
from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.pipeline.silver")


class BronzeToSilverPipeline:
    """Transforms raw bronze trades into normalized Silver daily summaries."""

    def __init__(self, db_manager: DuckDBManager):
        self.db = db_manager
        self.settings = get_settings()

    def run(self) -> Dict[str, int]:
        """Execute Bronze -> Silver transformation directly using streaming aggregations."""
        conn = self.db.get_connection()
        logger.info("Running Bronze -> Silver transformation...")

        # Build Daily Broker Summary table directly from bronze trades in a high-speed streaming query
        conn.execute("""
            WITH buyer_side AS (
                SELECT 
                    CAST(timestamp AS DATE) AS date_val,
                    symbol,
                    buyer_broker_id AS broker_id,
                    SUM(volume) AS total_buy_volume,
                    SUM(price * volume) AS total_buy_tl
                FROM bronze_raw_trades
                WHERE buyer_broker_id IS NOT NULL AND timestamp IS NOT NULL
                GROUP BY 1, 2, 3
            ),
            seller_side AS (
                SELECT 
                    CAST(timestamp AS DATE) AS date_val,
                    symbol,
                    seller_broker_id AS broker_id,
                    SUM(volume) AS total_sell_volume,
                    SUM(price * volume) AS total_sell_tl
                FROM bronze_raw_trades
                WHERE seller_broker_id IS NOT NULL AND timestamp IS NOT NULL
                GROUP BY 1, 2, 3
            ),
            combined_keys AS (
                SELECT date_val, symbol, broker_id FROM buyer_side
                UNION
                SELECT date_val, symbol, broker_id FROM seller_side
            ),
            daily_calculated AS (
                SELECT
                    k.date_val,
                    k.symbol,
                    k.broker_id,
                    COALESCE(b.total_buy_volume, 0.0) AS total_buy_volume,
                    COALESCE(s.total_sell_volume, 0.0) AS total_sell_volume,
                    (COALESCE(b.total_buy_volume, 0.0) - COALESCE(s.total_sell_volume, 0.0)) AS net_volume,
                    COALESCE(b.total_buy_tl, 0.0) AS total_buy_tl,
                    COALESCE(s.total_sell_tl, 0.0) AS total_sell_tl,
                    (COALESCE(b.total_buy_tl, 0.0) - COALESCE(s.total_sell_tl, 0.0)) AS net_tl,
                    CASE 
                        WHEN COALESCE(b.total_buy_volume, 0.0) > 0 
                        THEN b.total_buy_tl / b.total_buy_volume 
                        ELSE NULL 
                    END AS vwap_buy,
                    CASE 
                        WHEN COALESCE(s.total_sell_volume, 0.0) > 0 
                        THEN s.total_sell_tl / s.total_sell_volume 
                        ELSE NULL 
                    END AS vwap_sell
                FROM combined_keys k
                LEFT JOIN buyer_side b ON k.date_val = b.date_val AND k.symbol = b.symbol AND k.broker_id = b.broker_id
                LEFT JOIN seller_side s ON k.date_val = s.date_val AND k.symbol = s.symbol AND k.broker_id = s.broker_id
            )
            INSERT OR REPLACE INTO silver_daily_broker_summary
            SELECT * FROM daily_calculated;
        """)

        # Export to Silver Parquet directory for persistent local storage
        silver_sum_parquet = self.settings.silver_dir / "daily_broker_summary.parquet"
        conn.execute(f"COPY silver_daily_broker_summary TO '{silver_sum_parquet.as_posix()}' (FORMAT PARQUET);")

        sum_count = conn.execute("SELECT COUNT(*) FROM silver_daily_broker_summary").fetchone()[0]
        logger.info(f"Silver layer updated: {sum_count:,} daily broker summaries generated.")

        return {
            "silver_broker_transactions_count": sum_count,
            "silver_daily_broker_summary_count": sum_count,
        }
