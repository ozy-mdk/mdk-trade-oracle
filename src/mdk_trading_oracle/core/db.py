"""DuckDB connection and schema management."""

from pathlib import Path
from typing import Optional, Union
import duckdb
import polars as pl
from mdk_trading_oracle.core.config import Settings, get_settings
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.db")


class DuckDBManager:
    """Manages DuckDB connection, schema migrations, and Parquet table operations."""

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
        """Return an active DuckDB connection."""
        if self._conn is None:
            logger.debug(f"Connecting to DuckDB: {self.db_path}")
            self._conn = duckdb.connect(self.db_path)
        return self._conn

    def close(self) -> None:
        """Close connection if open."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def initialize_schema(self) -> None:
        """Initialize core DuckDB tables across all layers."""
        conn = self.get_connection()

        logger.info("Initializing DuckDB schema tables...")

        # 0. Reference Tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS silver_brokers (
                broker_id VARCHAR PRIMARY KEY,
                broker_name VARCHAR,
                category VARCHAR,
                is_primary_target BOOLEAN,
                description VARCHAR
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS silver_instruments (
                symbol VARCHAR PRIMARY KEY,
                name VARCHAR,
                sector VARCHAR,
                index_name VARCHAR,
                lot_multiplier DOUBLE
            );
        """)

        # 1. Bronze Stage
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

        # 2. Silver Tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS silver_broker_transactions (
                tx_id VARCHAR PRIMARY KEY,
                timestamp TIMESTAMP,
                date_val DATE,
                symbol VARCHAR,
                broker_id VARCHAR,
                side VARCHAR,
                price DOUBLE,
                volume DOUBLE,
                amount_tl DOUBLE,
                counterparty_broker_id VARCHAR
            );
        """)

        conn.execute("""
            CREATE TABLE IF NOT EXISTS silver_daily_broker_summary (
                date_val DATE,
                symbol VARCHAR,
                broker_id VARCHAR,
                total_buy_volume DOUBLE,
                total_sell_volume DOUBLE,
                net_volume DOUBLE,
                total_buy_tl DOUBLE,
                total_sell_tl DOUBLE,
                net_tl DOUBLE,
                vwap_buy DOUBLE,
                vwap_sell DOUBLE,
                PRIMARY KEY (date_val, symbol, broker_id)
            );
        """)

        # 3. Gold Tables
        conn.execute("""
            CREATE TABLE IF NOT EXISTS gold_bofa_flow_metrics (
                date_val DATE,
                symbol VARCHAR,
                close_price DOUBLE,
                total_symbol_volume DOUBLE,
                total_symbol_tl DOUBLE,
                bofa_buy_tl DOUBLE,
                bofa_sell_tl DOUBLE,
                bofa_net_tl DOUBLE,
                bofa_volume_share DOUBLE,
                bofa_net_share DOUBLE,
                bofa_net_tl_roll_3d DOUBLE,
                bofa_net_tl_roll_5d DOUBLE,
                bofa_net_tl_roll_10d DOUBLE,
                bofa_cum_net_tl_20d DOUBLE,
                bofa_flow_acceleration_5d DOUBLE,
                bofa_flow_zscore_20d DOUBLE,
                PRIMARY KEY (date_val, symbol)
            );
        """)

        # 4. Oracle Signals
        conn.execute("""
            CREATE TABLE IF NOT EXISTS oracle_decision_signals (
                signal_id VARCHAR PRIMARY KEY,
                date_val DATE,
                symbol VARCHAR,
                signal VARCHAR,
                confidence DOUBLE,
                bofa_net_tl DOUBLE,
                bofa_net_share DOUBLE,
                bofa_flow_zscore DOUBLE,
                summary VARCHAR,
                reasons VARCHAR, -- JSON list
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)

        # Sync reference data from YAML configs
        self.sync_reference_data()
        logger.info("DuckDB schema initialized successfully.")

    def sync_reference_data(self) -> None:
        """Sync YAML broker and instrument reference data into DuckDB."""
        conn = self.get_connection()

        brokers = self.settings.get_brokers()
        for b in brokers:
            conn.execute("""
                INSERT OR REPLACE INTO silver_brokers (broker_id, broker_name, category, is_primary_target, description)
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
                INSERT OR REPLACE INTO silver_instruments (symbol, name, sector, index_name, lot_multiplier)
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
