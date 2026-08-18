"""Bronze layer schema definitions and metadata synchronizers."""

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.bronze.schema")


def initialize_bronze_schema(db: DuckDBManager) -> None:
    """Initialize all Bronze layer tables and indexes in DuckDB."""
    conn = db.get_connection()

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
    sync_reference_data(db)
    logger.info("DuckDB Bronze schemas initialized.")


def sync_reference_data(db: DuckDBManager) -> None:
    """Sync broker and instrument reference YAML data into DuckDB Bronze tables."""
    conn = db.get_connection()
    settings = get_settings()

    brokers = settings.get_brokers()
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

    instruments = settings.get_instruments()
    for inst in instruments:
        conn.execute("""
            INSERT OR REPLACE INTO bronze_instruments (symbol, name, sector, index_name, lot_multiplier)
            VALUES (?, ?, ?, ?, ?);
        """, [
            inst["symbol"],
            inst.get("name", inst["symbol"]),
            inst.get("sector", "unknown"),
            inst.get("index_name", "BIST100"),
            float(inst.get("lot_multiplier", 1.0)),
        ])

    logger.debug(f"Synced {len(brokers)} brokers and {len(instruments)} instruments into Bronze tables.")
