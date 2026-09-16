"""PostgreSQL connection management and schema lifecycle."""

import logging
from typing import Any, Dict, Optional

import psycopg2
from psycopg2.extensions import connection as PgConnection

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.data.postgres.schema import ALL_DDL_SCRIPTS

logger = logging.getLogger(__name__)


class PostgresConnectionManager:
    """Manages connections to PostgreSQL and ensures schema integrity."""

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        database: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
    ) -> None:
        settings = get_settings()
        self.host = host or settings.pg_host
        self.port = port or settings.pg_port
        self.database = database or settings.pg_database
        self.user = user or settings.pg_user
        self.password = password if password is not None else settings.pg_password

    def get_connection_params(self) -> Dict[str, Any]:
        """Return dict of connection parameters."""
        params: Dict[str, Any] = {
            "host": self.host,
            "port": self.port,
            "dbname": self.database,
            "user": self.user,
        }
        if self.password:
            params["password"] = self.password
        return params

    def get_duckdb_attach_string(self) -> str:
        """Return connection string suitable for DuckDB ATTACH (TYPE POSTGRES)."""
        parts = [
            f"dbname={self.database}",
            f"host={self.host}",
            f"port={self.port}",
            f"user={self.user}",
        ]
        if self.password:
            parts.append(f"password={self.password}")
        return " ".join(parts)

    def get_connection(self) -> PgConnection:
        """Create and return a raw psycopg2 connection."""
        return psycopg2.connect(**self.get_connection_params())

    def init_schema(self, drop_first: bool = False) -> None:
        """Execute DDL scripts to create all required tables, indexes, and views."""
        conn = self.get_connection()
        conn.autocommit = True
        try:
            with conn.cursor() as cur:
                if drop_first:
                    logger.warning("Dropping existing PostgreSQL tables...")
                    cur.execute("DROP VIEW IF EXISTS view_market_candles_with_bofa CASCADE;")
                    cur.execute("DROP TABLE IF EXISTS candle_broker_flows CASCADE;")
                    cur.execute("DROP TABLE IF EXISTS market_candles CASCADE;")
                    cur.execute("DROP TABLE IF EXISTS raw_trades CASCADE;")

                for ddl in ALL_DDL_SCRIPTS:
                    cur.execute(ddl)
            logger.info("PostgreSQL schema initialized successfully.")
        finally:
            conn.close()

    def get_table_counts(self) -> Dict[str, int]:
        """Return row counts for all PostgreSQL tables."""
        conn = self.get_connection()
        counts = {}
        tables = ["raw_trades", "market_candles", "candle_broker_flows"]
        try:
            with conn.cursor() as cur:
                for tbl in tables:
                    try:
                        cur.execute(f"SELECT count(*) FROM {tbl};")
                        counts[tbl] = cur.fetchone()[0]
                    except Exception:
                        counts[tbl] = -1
        finally:
            conn.close()
        return counts
