"""Multi-timeframe candlestick and broker execution & PnL engine for PostgreSQL."""

import datetime as dt
import logging
import time
from typing import Dict, List, Optional, Union

import duckdb

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.data.postgres.connection import PostgresConnectionManager

logger = logging.getLogger(__name__)

# Supported timeframes and their corresponding DuckDB INTERVAL representations
TIMEFRAME_INTERVALS: Dict[str, str] = {
    "1m": "INTERVAL '1 Minute'",
    "5m": "INTERVAL '5 Minutes'",
    "15m": "INTERVAL '15 Minutes'",
    "30m": "INTERVAL '30 Minutes'",
    "60m": "INTERVAL '60 Minutes'",
    "120m": "INTERVAL '120 Minutes'",
    "240m": "INTERVAL '240 Minutes'",
    "8h": "INTERVAL '8 Hours'",
}


class MultiTimeframeCandleEngine:
    """Computes multi-timeframe candles and broker PnL metrics from raw trades into PostgreSQL."""

    def __init__(
        self,
        pg_manager: Optional[PostgresConnectionManager] = None,
        duckdb_path: Optional[str] = None,
    ) -> None:
        self.settings = get_settings()
        self.pg_manager = pg_manager or PostgresConnectionManager()
        self.duckdb_path = duckdb_path or str(self.settings.duckdb_path)

    def _get_dual_connection(self) -> duckdb.DuckDBPyConnection:
        """Create in-memory DuckDB connection attaching source as READ_ONLY and pg as target."""
        conn = duckdb.connect(":memory:")
        conn.execute("SET preserve_insertion_order = false;")
        conn.execute(f"ATTACH '{self.duckdb_path}' AS source_db (READ_ONLY);")
        conn.execute("LOAD postgres;")
        attach_str = self.pg_manager.get_duckdb_attach_string()
        conn.execute(f"ATTACH '{attach_str}' AS pg (TYPE POSTGRES);")
        return conn

    def get_available_dates(
        self,
        start_date: Optional[Union[str, dt.date]] = None,
        end_date: Optional[Union[str, dt.date]] = None,
    ) -> List[str]:
        """Return list of distinct trading dates present in source_db.bronze_raw_trades."""
        conn = duckdb.connect(self.duckdb_path, read_only=True)
        try:
            query = """
                SELECT DISTINCT strftime(timestamp, '%Y-%m-%d') as t_date 
                FROM bronze_raw_trades
            """
            conditions = []
            if start_date:
                conditions.append(f"timestamp >= '{start_date} 00:00:00'")
            if end_date:
                conditions.append(f"timestamp <= '{end_date} 23:59:59'")
            if conditions:
                query += " WHERE " + " AND ".join(conditions)
            query += " ORDER BY t_date ASC;"

            rows = conn.execute(query).fetchall()
            return [r[0] for r in rows if r[0]]
        finally:
            conn.close()

    def process_date_timeframe(
        self,
        trade_date: str,
        timeframe: str,
        delete_existing: bool = True,
    ) -> Dict[str, int]:
        """Compute and insert market candles and broker flows for a specific date and timeframe.

        Args:
            trade_date: Date string 'YYYY-MM-DD'.
            timeframe: One of '1m', '5m', '15m', '30m', '60m', '120m', '240m', '8h'.
            delete_existing: If True, clear existing records for this date & timeframe first.

        Returns:
            Dictionary with 'market_candles' and 'candle_broker_flows' row counts.
        """
        if timeframe not in TIMEFRAME_INTERVALS:
            raise ValueError(f"Unsupported timeframe: {timeframe}. Must be one of {list(TIMEFRAME_INTERVALS.keys())}")

        interval_sql = TIMEFRAME_INTERVALS[timeframe]
        start_ts = f"{trade_date} 00:00:00"
        end_ts = f"{trade_date} 23:59:59"

        if delete_existing:
            pg_conn = self.pg_manager.get_connection()
            try:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM market_candles WHERE timeframe = %s AND bucket_start >= %s AND bucket_start <= %s;",
                        (timeframe, start_ts, end_ts),
                    )
                    cur.execute(
                        "DELETE FROM candle_broker_flows WHERE timeframe = %s AND bucket_start >= %s AND bucket_start <= %s;",
                        (timeframe, start_ts, end_ts),
                    )
                pg_conn.commit()
            finally:
                pg_conn.close()

        conn = self._get_dual_connection()
        try:
            # 1. Market OHLCV Candles
            t0 = time.time()
            candle_sql = f"""
                INSERT INTO pg.market_candles (
                    timeframe, bucket_start, bucket_end, symbol,
                    open, high, low, close, volume, turnover_tl, vwap, trade_count
                )
                SELECT 
                    '{timeframe}' as timeframe,
                    time_bucket({interval_sql}, timestamp) as bucket_start,
                    time_bucket({interval_sql}, timestamp) + {interval_sql} as bucket_end,
                    symbol,
                    first(price ORDER BY timestamp) as open,
                    max(price) as high,
                    min(price) as low,
                    last(price ORDER BY timestamp) as close,
                    sum(volume) as volume,
                    sum(price * volume) as turnover_tl,
                    sum(price * volume) / sum(volume) as vwap,
                    count(*) as trade_count
                FROM source_db.bronze_raw_trades
                WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
                GROUP BY bucket_start, symbol;
            """
            conn.execute(candle_sql)
            candle_dur = time.time() - t0

            candle_count = conn.execute(
                f"SELECT count(*) FROM pg.market_candles WHERE timeframe = '{timeframe}' AND bucket_start >= '{start_ts}' AND bucket_start <= '{end_ts}';"
            ).fetchone()[0]

            # 2. Broker Flows & Microstructure PnL
            t1 = time.time()
            broker_flow_sql = f"""
                INSERT INTO pg.candle_broker_flows (
                    timeframe, bucket_start, symbol, broker_id,
                    buy_volume, buy_turnover_tl, buy_vwap,
                    sell_volume, sell_turnover_tl, sell_vwap,
                    net_volume, net_flow_tl,
                    matched_volume, matched_buy_value_tl, matched_sell_value_tl, realized_pnl_tl
                )
                WITH buy_leg AS (
                    SELECT 
                        time_bucket({interval_sql}, timestamp) as bucket_start,
                        symbol,
                        buyer_broker_id as broker_id,
                        sum(volume) as buy_volume,
                        sum(price * volume) as buy_turnover_tl
                    FROM source_db.bronze_raw_trades
                    WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
                    GROUP BY bucket_start, symbol, broker_id
                ),
                sell_leg AS (
                    SELECT 
                        time_bucket({interval_sql}, timestamp) as bucket_start,
                        symbol,
                        seller_broker_id as broker_id,
                        sum(volume) as sell_volume,
                        sum(price * volume) as sell_turnover_tl
                    FROM source_db.bronze_raw_trades
                    WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}'
                    GROUP BY bucket_start, symbol, broker_id
                ),
                combined AS (
                    SELECT 
                        COALESCE(b.bucket_start, s.bucket_start) as bucket_start,
                        COALESCE(b.symbol, s.symbol) as symbol,
                        COALESCE(b.broker_id, s.broker_id) as broker_id,
                        COALESCE(b.buy_volume, 0.0) as buy_volume,
                        COALESCE(b.buy_turnover_tl, 0.0) as buy_turnover_tl,
                        COALESCE(s.sell_volume, 0.0) as sell_volume,
                        COALESCE(s.sell_turnover_tl, 0.0) as sell_turnover_tl
                    FROM buy_leg b
                    FULL OUTER JOIN sell_leg s
                        ON b.bucket_start = s.bucket_start AND b.symbol = s.symbol AND b.broker_id = s.broker_id
                )
                SELECT 
                    '{timeframe}' as timeframe,
                    bucket_start,
                    symbol,
                    broker_id,
                    buy_volume,
                    buy_turnover_tl,
                    CASE WHEN buy_volume > 0 THEN buy_turnover_tl / buy_volume ELSE NULL END as buy_vwap,
                    sell_volume,
                    sell_turnover_tl,
                    CASE WHEN sell_volume > 0 THEN sell_turnover_tl / sell_volume ELSE NULL END as sell_vwap,
                    buy_volume - sell_volume as net_volume,
                    buy_turnover_tl - sell_turnover_tl as net_flow_tl,
                    LEAST(buy_volume, sell_volume) as matched_volume,
                    CASE WHEN buy_volume > 0 THEN LEAST(buy_volume, sell_volume) * (buy_turnover_tl / buy_volume) ELSE 0.0 END as matched_buy_value_tl,
                    CASE WHEN sell_volume > 0 THEN LEAST(buy_volume, sell_volume) * (sell_turnover_tl / sell_volume) ELSE 0.0 END as matched_sell_value_tl,
                    (CASE WHEN sell_volume > 0 THEN LEAST(buy_volume, sell_volume) * (sell_turnover_tl / sell_volume) ELSE 0.0 END) -
                    (CASE WHEN buy_volume > 0 THEN LEAST(buy_volume, sell_volume) * (buy_turnover_tl / buy_volume) ELSE 0.0 END) as realized_pnl_tl
                FROM combined;
            """
            conn.execute(broker_flow_sql)
            broker_dur = time.time() - t1

            broker_count = conn.execute(
                f"SELECT count(*) FROM pg.candle_broker_flows WHERE timeframe = '{timeframe}' AND bucket_start >= '{start_ts}' AND bucket_start <= '{end_ts}';"
            ).fetchone()[0]

            logger.info(
                f"[CandleEngine] {trade_date} [{timeframe}]: {candle_count:,} market candles ({candle_dur:.2f}s) | "
                f"{broker_count:,} broker flows with PnL ({broker_dur:.2f}s)"
            )
            return {
                "market_candles": candle_count,
                "candle_broker_flows": broker_count,
            }
        finally:
            conn.close()

    def process_range(
        self,
        timeframes: Optional[List[str]] = None,
        start_date: Optional[Union[str, dt.date]] = None,
        end_date: Optional[Union[str, dt.date]] = None,
        delete_existing: bool = True,
    ) -> Dict[str, int]:
        """Compute candles and broker flows across all requested timeframes and date range."""
        tfs = timeframes or list(TIMEFRAME_INTERVALS.keys())
        dates = self.get_available_dates(start_date, end_date)

        logger.info(
            f"[CandleEngine] Processing {len(tfs)} timeframes ({', '.join(tfs)}) across {len(dates)} dates "
            f"({dates[0] if dates else 'N/A'} to {dates[-1] if dates else 'N/A'})..."
        )

        total_candles = 0
        total_broker_rows = 0
        start_time = time.time()

        for d_idx, d in enumerate(dates, 1):
            logger.info(f"[{d_idx}/{len(dates)}] Date: {d}...")
            for tf in tfs:
                res = self.process_date_timeframe(d, tf, delete_existing=delete_existing)
                total_candles += res["market_candles"]
                total_broker_rows += res["candle_broker_flows"]

        total_elapsed = time.time() - start_time
        logger.info(
            f"[CandleEngine] Finished: {total_candles:,} market candles and {total_broker_rows:,} broker flows "
            f"generated in {total_elapsed:.2f}s"
        )
        return {
            "total_market_candles": total_candles,
            "total_candle_broker_flows": total_broker_rows,
        }
