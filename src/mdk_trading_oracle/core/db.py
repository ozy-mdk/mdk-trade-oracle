"""DuckDB connection lifecycle and query execution management."""

from pathlib import Path
from typing import Any, Optional, Union

import duckdb
import polars as pl

from mdk_trading_oracle.core.config import Settings, get_settings
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.core.db")


class DuckDBManager:
    """Manages DuckDB connection lifecycle, pragmas, and analytical queries."""

    def __init__(
        self,
        db_path: Optional[Union[str, Path]] = None,
        in_memory: bool = False,
        read_only: bool = False,
    ):
        self.settings: Settings = get_settings()
        self.read_only = read_only
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
            logger.debug(f"Connecting to DuckDB: {self.db_path} (read_only={self.read_only})")
            try:
                self._conn = duckdb.connect(self.db_path, read_only=self.read_only)
            except duckdb.IOException as e:
                if "Could not set lock on file" in str(e):
                    logger.error(
                        f"DuckDB lock conflict: {self.db_path} is currently locked by another process "
                        f"(e.g. an active Jupyter notebook kernel). Please restart/close the notebook kernel "
                        f"or connect with read_only=True for querying."
                    )
                raise e

            # Enforce Turkish Time (Europe/Istanbul / TRT / UTC+3) across all DuckDB operations
            self._conn.execute(f"SET TimeZone = '{self.settings.timezone}';")

            # Performance and memory spill pragmas (only if read-write and not in-memory)
            if not self.read_only and self.db_path != ":memory:":
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
        """Initialize all Medallion Lakehouse layers (Bronze, Silver, Gold)."""
        from mdk_trading_oracle.data.bronze.schema import initialize_bronze_schema
        from mdk_trading_oracle.data.gold.schema import initialize_gold_schema
        from mdk_trading_oracle.data.silver.schema import initialize_silver_schema

        logger.info("Initializing DuckDB Medallion Lakehouse schemas (Bronze, Silver, Gold)...")
        initialize_bronze_schema(self)
        initialize_silver_schema(self)
        initialize_gold_schema(self)
        logger.info("All Medallion Lakehouse schemas initialized successfully.")

    def sync_reference_data(self) -> None:
        """Sync YAML broker and instrument reference data into Bronze tables."""
        from mdk_trading_oracle.data.bronze.schema import sync_reference_data

        sync_reference_data(self)

    def query_pl(self, query: str, params: Optional[list[Any]] = None) -> pl.DataFrame:
        """Execute query and return as a Polars DataFrame."""
        conn = self.get_connection()
        if params:
            arrow_table = conn.execute(query, params).fetch_arrow_table()
        else:
            arrow_table = conn.execute(query).fetch_arrow_table()
        return pl.from_arrow(arrow_table)

    def execute(self, query: str, params: Optional[list[Any]] = None) -> duckdb.DuckDBPyConnection:
        """Execute a SQL statement."""
        conn = self.get_connection()
        if params:
            return conn.execute(query, params)
        return conn.execute(query)
