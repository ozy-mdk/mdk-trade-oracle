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

    # 4. Bronze Tracking Table: Ingestion Log (for incremental & partition-aware ingestion)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bronze_ingestion_log (
            file_path VARCHAR PRIMARY KEY,
            file_name VARCHAR,
            file_size_bytes BIGINT,
            file_mtime_epoch DOUBLE,
            trade_date DATE,
            year_month VARCHAR,
            rows_ingested BIGINT,
            raw_source_label VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 5. Bronze Macro Table: Central Bank Interest Rates (TCMB 1-Week Repo & Policy Rates)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS bronze_central_bank_rates (
            rate_date DATE,
            rate_type VARCHAR DEFAULT '1_week_repo',
            interest_rate DOUBLE NOT NULL,
            rate_change DOUBLE DEFAULT 0.0,
            is_rate_change_day BOOLEAN DEFAULT FALSE,
            is_forward_filled BOOLEAN DEFAULT FALSE,
            raw_source VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (rate_date, rate_type)
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
        broker_id = b.get("code") or b.get("broker_id")
        if not broker_id:
            continue
        conn.execute("""
            INSERT OR REPLACE INTO bronze_brokers (broker_id, broker_name, category, is_primary_target, description)
            VALUES (?, ?, ?, ?, ?);
        """, [
            broker_id,
            b.get("name") or b.get("broker_name", broker_id),
            b.get("type") or b.get("category", "unknown"),
            b.get("is_primary_target", False),
            b.get("description", ""),
        ])

    instruments = settings.get_instruments()
    for inst in instruments:
        symbol = inst.get("symbol")
        if not symbol:
            continue
        conn.execute("""
            INSERT OR REPLACE INTO bronze_instruments (symbol, name, sector, index_name, lot_multiplier)
            VALUES (?, ?, ?, ?, ?);
        """, [
            symbol,
            inst.get("name", symbol),
            inst.get("sector", "unknown"),
            inst.get("index") or inst.get("index_name", "BIST100"),
            float(inst.get("lot_multiplier", 1.0)),
        ])

    logger.debug(f"Synced {len(brokers)} brokers and {len(instruments)} instruments into Bronze tables.")
