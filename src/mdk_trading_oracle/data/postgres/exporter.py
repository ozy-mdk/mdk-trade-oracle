"""High-performance non-destructive exporter of raw trade ticks to PostgreSQL."""

import datetime as dt
import logging
import time
from typing import List, Optional, Union

import duckdb

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.data.postgres.connection import PostgresConnectionManager

logger = logging.getLogger(__name__)


class RawTradeExporter:
    """Exports raw trade ticks from DuckDB to PostgreSQL without modifying source data."""

    def __init__(
        self,
        pg_manager: Optional[PostgresConnectionManager] = None,
        duckdb_path: Optional[str] = None,
    ) -> None:
        self.settings = get_settings()
        self.pg_manager = pg_manager or PostgresConnectionManager()
        self.duckdb_path = duckdb_path or str(self.settings.duckdb_path)

    def _get_dual_connection(self) -> duckdb.DuckDBPyConnection:
        """Create an in-memory DuckDB connection attaching source as READ_ONLY and pg as target."""
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

    def export_date(
        self,
        trade_date: str,
        delete_existing: bool = True,
    ) -> int:
        """Export a single day's raw trades into PostgreSQL raw_trades table.

        Args:
            trade_date: Date string in 'YYYY-MM-DD' format.
            delete_existing: If True, delete any existing records for this date in PG first.

        Returns:
            Number of rows exported.
        """
        start_ts = f"{trade_date} 00:00:00"
        end_ts = f"{trade_date} 23:59:59"

        if delete_existing:
            pg_conn = self.pg_manager.get_connection()
            try:
                with pg_conn.cursor() as cur:
                    cur.execute(
                        "DELETE FROM raw_trades WHERE timestamp >= %s AND timestamp <= %s;",
                        (start_ts, end_ts),
                    )
                pg_conn.commit()
            finally:
                pg_conn.close()

        conn = self._get_dual_connection()
        try:
            t0 = time.time()
            insert_sql = f"""
                INSERT INTO pg.raw_trades (
                    trade_id, timestamp, symbol, price, volume,
                    buyer_broker_id, seller_broker_id, raw_source, ingested_at
                )
                SELECT 
                    trade_id, timestamp, symbol, price, volume,
                    buyer_broker_id, seller_broker_id, raw_source, ingested_at
                FROM source_db.bronze_raw_trades
                WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}';
            """
            conn.execute(insert_sql)
            duration = time.time() - t0

            # Count rows inserted
            row_count = conn.execute(
                f"SELECT count(*) FROM pg.raw_trades WHERE timestamp >= '{start_ts}' AND timestamp <= '{end_ts}';"
            ).fetchone()[0]

            rate = (row_count / duration) if duration > 0 else 0
            logger.info(
                f"[RawTradeExporter] Exported {row_count:,} trades for {trade_date} in {duration:.2f}s ({rate:,.0f} rows/s)"
            )
            return row_count
        finally:
            conn.close()

    def export_range(
        self,
        start_date: Optional[Union[str, dt.date]] = None,
        end_date: Optional[Union[str, dt.date]] = None,
        delete_existing: bool = True,
    ) -> int:
        """Export raw trades for a date range in day-by-day chunks."""
        dates = self.get_available_dates(start_date, end_date)
        logger.info(f"Starting raw trade export for {len(dates)} trading dates ({dates[0] if dates else 'N/A'} to {dates[-1] if dates else 'N/A'})...")

        total_rows = 0
        total_start = time.time()

        for idx, d in enumerate(dates, 1):
            logger.info(f"[{idx}/{len(dates)}] Exporting {d}...")
            count = self.export_date(d, delete_existing=delete_existing)
            total_rows += count

        total_dur = time.time() - total_start
        avg_rate = (total_rows / total_dur) if total_dur > 0 else 0
        logger.info(
            f"[RawTradeExporter] Completed raw trade export: {total_rows:,} rows across {len(dates)} dates in {total_dur:.2f}s (Average: {avg_rate:,.0f} rows/s)"
        )
        return total_rows
