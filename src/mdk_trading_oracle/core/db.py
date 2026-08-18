"""DuckDB connection and Bronze schema management."""

from pathlib import Path
from typing import Optional, Union

import duckdb
import polars as pl

from mdk_trading_oracle.core.config import Settings, get_settings
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.db")


class DuckDBManager:
    """Manages DuckDB connection, Bronze schema, and analytical queries."""

    def __init__(self, db_path: Optional[Union[str, Path]] = None, in_memory: bool = False):
        self.settings: Settings = get_settings()
        if in_memory:
            self.db_path = ":memory:"
        elif db_path:
            self.db_path = str(db_path)
        else:
            self.db_path = str(self.settings.database_path)

        self._conn: Optional[duckdb.DuckDBPyConnection] = None

    def get_connection(self) -> duckdb.DuckDBPyConnection:
        """Return an active DuckDB connection configured for analytical queries."""
        if self._conn is None:
            logger.debug(f"Connecting to DuckDB: {self.db_path}")
            self._conn = duckdb.connect(self.db_path)

            # Performance and memory spill pragmas
            tmp_dir = self.settings.database_dir / "tmp"
            tmp_dir.mkdir(parents=True, exist_ok=True)
            self._conn.execute("PRAGMA preserve_insertion_order=false;")
            self._conn.execute(f"PRAGMA temp_directory='{tmp_dir.as_posix()}';")
            self._conn.execute("PRAGMA max_temp_directory_size='30GiB';")

        return self._conn

    def close(self) -> None:
        """Close connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def initialize_schema(self) -> None:
        """Initialize Bronze DuckDB tables."""
        conn = self.get_connection()

        logger.info("Initializing DuckDB Bronze schema tables...")

        # 1. Reference Table: Brokers
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bronze_brokers (
                broker_id VARCHAR PRIMARY KEY,
                broker_name VARCHAR,
                category VARCHAR,
                is_primary_target BOOLEAN,
                description VARCHAR
            );
        """)

        # 2. Reference Table: Instruments
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bronze_instruments (
                symbol VARCHAR PRIMARY KEY,
                name VARCHAR,
                sector VARCHAR,
                index_name VARCHAR,
                lot_multiplier DOUBLE
            );
        """)

        # 3. Bronze Table: Raw Trades
        conn.execute("""
            CREATE TABLE IF NOT EXISTS bronze_raw_trades (
                trade_id VARCHAR,
                timestamp TIMESTAMP,
                symbol VARCHAR,
                price DOUBLE,
                volume DOUBLE,
                buyer_broker_id VARCHAR,
                seller_broker_id VARCHAR,
                raw_source VARCHAR,
                ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Sync reference data from YAML configs
        self.sync_reference_data()
        logger.info("DuckDB Bronze schema initialized successfully.")

    def sync_reference_data(self) -> None:
        """Sync YAML broker and instrument reference data into Bronze tables."""
        conn = self.get_connection()

        brokers = self.settings.get_brokers()
        for b in brokers:
            conn.execute("""
                INSERT OR REPLACE INTO bronze_brokers (broker_id, broker_name, category, is_primary_target, description)
                VALUES (?, ?, ?, ?, ?);
            """, [
                b["broker_id"],
                b.get("broker_name", b["broker_id"]),
                b.get("category", "unknown"),
                b.get("is_primary_target", False),
                b.get("description", ""),
            ])

        instruments = self.settings.get_instruments()
        for inst in instruments:
            conn.execute("""
                INSERT OR REPLACE INTO bronze_instruments (symbol, name, sector, index_name, lot_multiplier)
                VALUES (?, ?, ?, ?, ?);
            """, [
                inst["symbol"],
                inst.get("name", inst["symbol"]),
                inst.get("sector", "unknown"),
                inst.get("index", "BIST"),
                inst.get("lot_multiplier", 1.0),
            ])

    def query_pl(self, query: str, params: Optional[list] = None) -> pl.DataFrame:
        """Execute query and return as a Polars DataFrame."""
        conn = self.get_connection()
        if params:
            arrow_table = conn.execute(query, params).fetch_arrow_table()
        else:
            arrow_table = conn.execute(query).fetch_arrow_table()
        return pl.from_arrow(arrow_table)

    def execute(self, query: str, params: Optional[list] = None) -> duckdb.DuckDBPyConnection:
        """Execute a SQL statement."""
        conn = self.get_connection()
        if params:
            return conn.execute(query, params)
        return conn.execute(query)
