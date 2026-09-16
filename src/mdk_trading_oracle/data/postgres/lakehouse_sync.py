"""Synchronize DuckDB Medallion Lakehouse tables to PostgreSQL for 100% PostgreSQL API serving."""

import logging
import time
from typing import Optional

import duckdb

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.data.postgres.connection import PostgresConnectionManager

logger = logging.getLogger(__name__)


class LakehousePostgresSynchronizer:
    """Syncs lakehouse Silver and Bronze analytical tables to PostgreSQL."""

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

    def sync_all(self) -> dict[str, int]:
        """Synchronize all core analytical tables to PostgreSQL.

        Returns:
            Dictionary with table names and copied row counts.
        """
        conn = self._get_dual_connection()
        counts = {}
        try:
            # 1. instruments
            t0 = time.time()
            conn.execute("CREATE TABLE IF NOT EXISTS pg.instruments AS SELECT * FROM source_db.bronze_instruments WITH NO DATA;")
            conn.execute("TRUNCATE pg.instruments;")
            conn.execute("INSERT INTO pg.instruments SELECT * FROM source_db.bronze_instruments;")
            c = conn.execute("SELECT count(*) FROM pg.instruments;").fetchone()[0]
            counts["instruments"] = c
            logger.info("Synced instruments: %d rows in %.2fs", c, time.time() - t0)

            # 2. brokers
            t0 = time.time()
            conn.execute("CREATE TABLE IF NOT EXISTS pg.brokers AS SELECT * FROM source_db.bronze_brokers WITH NO DATA;")
            conn.execute("TRUNCATE pg.brokers;")
            conn.execute("INSERT INTO pg.brokers SELECT * FROM source_db.bronze_brokers;")
            c = conn.execute("SELECT count(*) FROM pg.brokers;").fetchone()[0]
            counts["brokers"] = c
            logger.info("Synced brokers: %d rows in %.2fs", c, time.time() - t0)

            # 3. daily_stock_summary
            t0 = time.time()
            conn.execute("CREATE TABLE IF NOT EXISTS pg.daily_stock_summary AS SELECT * FROM source_db.silver_daily_stock_summary WITH NO DATA;")
            conn.execute("TRUNCATE pg.daily_stock_summary;")
            conn.execute("INSERT INTO pg.daily_stock_summary SELECT * FROM source_db.silver_daily_stock_summary;")
            c = conn.execute("SELECT count(*) FROM pg.daily_stock_summary;").fetchone()[0]
            counts["daily_stock_summary"] = c
            logger.info("Synced daily_stock_summary: %d rows in %.2fs", c, time.time() - t0)

            # 4. broker_fifo_lots
            t0 = time.time()
            conn.execute("CREATE TABLE IF NOT EXISTS pg.broker_fifo_lots AS SELECT * FROM source_db.silver_broker_fifo_lots WITH NO DATA;")
            conn.execute("TRUNCATE pg.broker_fifo_lots;")
            conn.execute("INSERT INTO pg.broker_fifo_lots SELECT * FROM source_db.silver_broker_fifo_lots;")
            c = conn.execute("SELECT count(*) FROM pg.broker_fifo_lots;").fetchone()[0]
            counts["broker_fifo_lots"] = c
            logger.info("Synced broker_fifo_lots: %d rows in %.2fs", c, time.time() - t0)

            # 5. broker_fifo_lot_lifecycle
            t0 = time.time()
            conn.execute("CREATE TABLE IF NOT EXISTS pg.broker_fifo_lot_lifecycle AS SELECT * FROM source_db.silver_broker_fifo_lot_lifecycle WITH NO DATA;")
            conn.execute("TRUNCATE pg.broker_fifo_lot_lifecycle;")
            conn.execute("INSERT INTO pg.broker_fifo_lot_lifecycle SELECT * FROM source_db.silver_broker_fifo_lot_lifecycle;")
            c = conn.execute("SELECT count(*) FROM pg.broker_fifo_lot_lifecycle;").fetchone()[0]
            counts["broker_fifo_lot_lifecycle"] = c
            logger.info("Synced broker_fifo_lot_lifecycle: %d rows in %.2fs", c, time.time() - t0)

            # 6. broker_fifo_daily
            t0 = time.time()
            conn.execute("CREATE TABLE IF NOT EXISTS pg.broker_fifo_daily AS SELECT * FROM source_db.silver_broker_fifo_daily WITH NO DATA;")
            conn.execute("TRUNCATE pg.broker_fifo_daily;")
            conn.execute("INSERT INTO pg.broker_fifo_daily SELECT * FROM source_db.silver_broker_fifo_daily;")
            c = conn.execute("SELECT count(*) FROM pg.broker_fifo_daily;").fetchone()[0]
            counts["broker_fifo_daily"] = c
            logger.info("Synced broker_fifo_daily: %d rows in %.2fs", c, time.time() - t0)

        finally:
            conn.close()

        self._ensure_indexes()
        return counts

    def _ensure_indexes(self) -> None:
        """Create B-tree indexes on PostgreSQL tables for high-performance querying."""
        index_queries = [
            "CREATE INDEX IF NOT EXISTS idx_dss_symbol_date ON daily_stock_summary(symbol, trade_date);",
            "CREATE INDEX IF NOT EXISTS idx_dss_date ON daily_stock_summary(trade_date);",
            "CREATE INDEX IF NOT EXISTS idx_bfd_broker_date_sym ON broker_fifo_daily(broker_id, trade_date, symbol);",
            "CREATE INDEX IF NOT EXISTS idx_bfd_trade_date ON broker_fifo_daily(trade_date);",
            "CREATE INDEX IF NOT EXISTS idx_bfl_broker_sym ON broker_fifo_lots(broker_id, symbol);",
            "CREATE INDEX IF NOT EXISTS idx_bfll_broker_status ON broker_fifo_lot_lifecycle(broker_id, status, open_date);",
        ]
        with self.pg_manager.get_connection() as conn:
            with conn.cursor() as cur:
                for q in index_queries:
                    cur.execute(q)
            conn.commit()
