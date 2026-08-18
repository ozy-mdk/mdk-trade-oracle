"""Bronze to Silver normalization pipeline."""

from typing import Dict
from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.pipeline.silver")


class BronzeToSilverPipeline:
    """Transforms raw bronze trades into normalized Silver transactions and summaries."""

    def __init__(self, db_manager: DuckDBManager):
        self.db = db_manager
        self.settings = get_settings()

    def run(self) -> Dict[str, int]:
        """Execute Bronze -> Silver transformation."""
        conn = self.db.get_connection()
        logger.info("Running Bronze -> Silver transformation...")

        # 1. Normalize two-sided trades into discrete broker transactions (BUY and SELL)
        conn.execute("""
            INSERT OR REPLACE INTO silver_broker_transactions (
                tx_id, timestamp, date_val, symbol, broker_id, side, price, volume, amount_tl, counterparty_broker_id
            )
            -- Buyer Side
            SELECT 
                trade_id || '_BUY' AS tx_id,
                timestamp,
                CAST(timestamp AS DATE) AS date_val,
                symbol,
                buyer_broker_id AS broker_id,
                'BUY' AS side,
                price,
                volume,
                (price * volume) AS amount_tl,
                seller_broker_id AS counterparty_broker_id
            FROM bronze_raw_trades
            WHERE buyer_broker_id IS NOT NULL

            UNION ALL

            -- Seller Side
            SELECT 
                trade_id || '_SELL' AS tx_id,
                timestamp,
                CAST(timestamp AS DATE) AS date_val,
                symbol,
                seller_broker_id AS broker_id,
                'SELL' AS side,
                price,
                volume,
                (price * volume) AS amount_tl,
                buyer_broker_id AS counterparty_broker_id
            FROM bronze_raw_trades
            WHERE seller_broker_id IS NOT NULL;
        """)

        # 2. Build Daily Broker Summary table
        conn.execute("""
            INSERT OR REPLACE INTO silver_daily_broker_summary (
                date_val, symbol, broker_id,
                total_buy_volume, total_sell_volume, net_volume,
                total_buy_tl, total_sell_tl, net_tl,
                vwap_buy, vwap_sell
            )
            SELECT
                date_val,
                symbol,
                broker_id,
                COALESCE(SUM(CASE WHEN side = 'BUY' THEN volume ELSE 0 END), 0.0) AS total_buy_volume,
                COALESCE(SUM(CASE WHEN side = 'SELL' THEN volume ELSE 0 END), 0.0) AS total_sell_volume,
                COALESCE(SUM(CASE WHEN side = 'BUY' THEN volume ELSE -volume END), 0.0) AS net_volume,
                COALESCE(SUM(CASE WHEN side = 'BUY' THEN amount_tl ELSE 0 END), 0.0) AS total_buy_tl,
                COALESCE(SUM(CASE WHEN side = 'SELL' THEN amount_tl ELSE 0 END), 0.0) AS total_sell_tl,
                COALESCE(SUM(CASE WHEN side = 'BUY' THEN amount_tl ELSE -amount_tl END), 0.0) AS net_tl,
                CASE 
                    WHEN SUM(CASE WHEN side = 'BUY' THEN volume ELSE 0 END) > 0 
                    THEN SUM(CASE WHEN side = 'BUY' THEN amount_tl ELSE 0 END) / SUM(CASE WHEN side = 'BUY' THEN volume ELSE 0 END)
                    ELSE NULL 
                END AS vwap_buy,
                CASE 
                    WHEN SUM(CASE WHEN side = 'SELL' THEN volume ELSE 0 END) > 0 
                    THEN SUM(CASE WHEN side = 'SELL' THEN amount_tl ELSE 0 END) / SUM(CASE WHEN side = 'SELL' THEN volume ELSE 0 END)
                    ELSE NULL 
                END AS vwap_sell
            FROM silver_broker_transactions
            GROUP BY date_val, symbol, broker_id;
        """)

        # 3. Export to Silver Parquet directory for persistent local storage
        silver_tx_parquet = self.settings.silver_dir / "broker_transactions.parquet"
        silver_sum_parquet = self.settings.silver_dir / "daily_broker_summary.parquet"

        conn.execute(f"COPY silver_broker_transactions TO '{silver_tx_parquet.as_posix()}' (FORMAT PARQUET);")
        conn.execute(f"COPY silver_daily_broker_summary TO '{silver_sum_parquet.as_posix()}' (FORMAT PARQUET);")

        tx_count = conn.execute("SELECT COUNT(*) FROM silver_broker_transactions").fetchone()[0]
        sum_count = conn.execute("SELECT COUNT(*) FROM silver_daily_broker_summary").fetchone()[0]

        logger.info(f"Silver layer updated: {tx_count} broker transactions, {sum_count} daily summaries.")
        return {
            "silver_broker_transactions_count": tx_count,
            "silver_daily_broker_summary_count": sum_count,
        }
