"""BofA flow and institutional concentration feature extractors."""

from typing import Optional
import polars as pl
from mdk_trading_oracle.core.db import DuckDBManager


class BofAFlowFeatureExtractor:
    """Feature extraction toolkit operating directly on DuckDB / Polars tables."""

    def __init__(self, db_manager: DuckDBManager):
        self.db = db_manager

    def get_symbol_flow_history(self, symbol: str, limit: Optional[int] = None) -> pl.DataFrame:
        """Fetch historical Gold flow metrics for a symbol ordered by date."""
        query = """
            SELECT *
            FROM gold_bofa_flow_metrics
            WHERE symbol = ?
            ORDER BY date_val ASC
        """
        if limit:
            query += f" LIMIT {int(limit)}"
        return self.db.query_pl(query, [symbol])

    def get_latest_market_snapshot(self) -> pl.DataFrame:
        """Fetch the most recent day's Gold metrics across all symbols."""
        query = """
            WITH latest_date AS (
                SELECT MAX(date_val) as max_date FROM gold_bofa_flow_metrics
            )
            SELECT g.*
            FROM gold_bofa_flow_metrics g
            JOIN latest_date l ON g.date_val = l.max_date
            ORDER BY g.bofa_net_tl DESC;
        """
        return self.db.query_pl(query)

    def get_broker_dominance_matrix(self, symbol: str, date_val: Optional[str] = None) -> pl.DataFrame:
        """Retrieve broker market share and net position breakdown for a given symbol."""
        if date_val:
            query = """
                SELECT 
                    b.broker_name,
                    s.broker_id,
                    s.total_buy_tl,
                    s.total_sell_tl,
                    s.net_tl,
                    (s.total_buy_tl + s.total_sell_tl) AS turnover_tl
                FROM silver_daily_broker_summary s
                LEFT JOIN silver_brokers b ON s.broker_id = b.broker_id
                WHERE s.symbol = ? AND s.date_val = ?
                ORDER BY turnover_tl DESC;
            """
            return self.db.query_pl(query, [symbol, date_val])
        else:
            query = """
                WITH latest_date AS (
                    SELECT MAX(date_val) as max_date FROM silver_daily_broker_summary WHERE symbol = ?
                )
                SELECT 
                    b.broker_name,
                    s.broker_id,
                    s.total_buy_tl,
                    s.total_sell_tl,
                    s.net_tl,
                    (s.total_buy_tl + s.total_sell_tl) AS turnover_tl
                FROM silver_daily_broker_summary s
                JOIN latest_date l ON s.date_val = l.max_date
                LEFT JOIN silver_brokers b ON s.broker_id = b.broker_id
                WHERE s.symbol = ?
                ORDER BY turnover_tl DESC;
            """
            return self.db.query_pl(query, [symbol, symbol])
