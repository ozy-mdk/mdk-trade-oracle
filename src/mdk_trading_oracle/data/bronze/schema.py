"""Bronze layer schema definitions and metadata synchronizers for PostgreSQL and TimescaleDB."""

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.core.logger import get_logger

logger = get_logger("mdk_oracle.data.bronze.schema")


def initialize_bronze_schema(db: PostgresManager) -> None:
    """Initialize all Bronze layer tables, hypertables, and indexes."""
    # Enable TimescaleDB extension if running on PostgreSQL
    try:
        db.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
    except Exception as e:
        logger.debug(f"TimescaleDB extension notice (e.g. running in test/in-memory mode): {e}")

    # 1. Reference Table: Brokers
    db.execute("""
        CREATE TABLE IF NOT EXISTS bronze_brokers (
            broker_id VARCHAR PRIMARY KEY,
            broker_name VARCHAR,
            category VARCHAR,
            is_primary_target BOOLEAN,
            description VARCHAR
        );
    """)

    # 2. Reference Table: Instruments
    db.execute("""
        CREATE TABLE IF NOT EXISTS bronze_instruments (
            symbol VARCHAR PRIMARY KEY,
            name VARCHAR,
            sector VARCHAR,
            index_name VARCHAR,
            lot_multiplier DOUBLE PRECISION
        );
    """)

    # 3. Bronze Table: Raw Trades (Partitioned by timestamp)
    db.execute("""
        CREATE TABLE IF NOT EXISTS bronze_raw_trades (
            trade_id VARCHAR,
            timestamp TIMESTAMP NOT NULL,
            symbol VARCHAR NOT NULL,
            price DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION NOT NULL,
            buyer_broker_id VARCHAR,
            seller_broker_id VARCHAR,
            raw_source VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # Convert to Timescale Hypertable and enable compression if TimescaleDB is available
    try:
        db.execute("SELECT create_hypertable('bronze_raw_trades', 'timestamp', chunk_time_interval => INTERVAL '7 days', if_not_exists => TRUE);")
        db.execute("""
            ALTER TABLE bronze_raw_trades SET (
                timescaledb.compress,
                timescaledb.compress_segmentby = 'symbol',
                timescaledb.compress_orderby = 'timestamp ASC'
            );
        """)
    except Exception as e:
        logger.debug(f"Timescale hypertable setup notice: {e}")

    # Indexes on bronze_raw_trades
    try:
        db.execute("CREATE INDEX IF NOT EXISTS idx_bronze_raw_trades_symbol_time ON bronze_raw_trades (symbol, timestamp ASC);")
        db.execute("CREATE INDEX IF NOT EXISTS idx_bronze_raw_trades_buyer ON bronze_raw_trades (buyer_broker_id, timestamp ASC);")
        db.execute("CREATE INDEX IF NOT EXISTS idx_bronze_raw_trades_seller ON bronze_raw_trades (seller_broker_id, timestamp ASC);")
    except Exception as e:
        logger.debug(f"Index creation notice: {e}")

    # 4. Bronze Tracking Table: Ingestion Log (for incremental & partition-aware ingestion)
    db.execute("""
        CREATE TABLE IF NOT EXISTS bronze_ingestion_log (
            file_path VARCHAR PRIMARY KEY,
            file_name VARCHAR,
            file_size_bytes BIGINT,
            file_mtime_epoch DOUBLE PRECISION,
            trade_date DATE,
            year_month VARCHAR,
            rows_ingested BIGINT,
            raw_source_label VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 5. Bronze Macro Table: Central Bank Interest Rates (TCMB 1-Week Repo & Policy Rates)
    db.execute("""
        CREATE TABLE IF NOT EXISTS bronze_central_bank_rates (
            rate_date DATE,
            rate_type VARCHAR DEFAULT '1_week_repo',
            interest_rate DOUBLE PRECISION NOT NULL,
            rate_change DOUBLE PRECISION DEFAULT 0.0,
            is_rate_change_day BOOLEAN DEFAULT FALSE,
            is_forward_filled BOOLEAN DEFAULT FALSE,
            raw_source VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (rate_date, rate_type)
        );
    """)

    # 6. Bronze Benchmark Table: Official BIST Index Benchmarks (XU030, etc.)
    db.execute("""
        CREATE TABLE IF NOT EXISTS bronze_bist_index_benchmarks (
            trade_date DATE PRIMARY KEY,
            index_code VARCHAR DEFAULT 'XU030',
            open_price DOUBLE PRECISION,
            high_price DOUBLE PRECISION,
            low_price DOUBLE PRECISION,
            close_price DOUBLE PRECISION NOT NULL,
            volume DOUBLE PRECISION,
            daily_return_pct DOUBLE PRECISION,
            price_range_pct DOUBLE PRECISION,
            is_forward_filled BOOLEAN DEFAULT FALSE,
            source VARCHAR DEFAULT 'yfinance_XU030.IS',
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 7. Bronze Corporate Actions Table: Stock Splits, Bonus Issues & Ticker Renames
    db.execute("""
        CREATE TABLE IF NOT EXISTS bronze_corporate_actions (
            action_date DATE NOT NULL,
            symbol VARCHAR NOT NULL,
            target_symbol VARCHAR,
            quantity_multiplier DOUBLE PRECISION NOT NULL,
            note VARCHAR,
            raw_source VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (action_date, symbol)
        );
    """)

    # 8. Bronze BIST 30 Index Membership Snapshots
    db.execute("""
        CREATE TABLE IF NOT EXISTS bronze_bist30_membership (
            start_date DATE NOT NULL,
            end_date DATE,
            symbol VARCHAR NOT NULL,
            is_active BOOLEAN DEFAULT FALSE,
            source_type VARCHAR,
            confidence VARCHAR,
            source_url VARCHAR,
            raw_source VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (start_date, symbol)
        );
    """)

    # 9. Bronze BIST 30 Periodic Rebalancing Changes
    db.execute("""
        CREATE TABLE IF NOT EXISTS bronze_bist30_changes (
            effective_date DATE PRIMARY KEY,
            end_date DATE,
            in_symbols VARCHAR,
            in_count INTEGER DEFAULT 0,
            out_symbols VARCHAR,
            out_count INTEGER DEFAULT 0,
            description VARCHAR,
            raw_source VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
    """)

    # 10. Bronze BIST 30 Stock Membership Periods (Aggregated Spans)
    db.execute("""
        CREATE TABLE IF NOT EXISTS bronze_bist30_stock_periods (
            symbol VARCHAR NOT NULL,
            start_date DATE NOT NULL,
            end_date DATE,
            status VARCHAR DEFAULT 'Aktif',
            is_active BOOLEAN DEFAULT FALSE,
            calendar_days INTEGER,
            membership_period_no INTEGER DEFAULT 1,
            raw_source VARCHAR,
            ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, start_date)
        );
    """)

    # Sync reference data from YAML configs
    sync_reference_data(db)
    logger.info("Bronze schemas initialized successfully.")


def sync_reference_data(db: PostgresManager) -> None:
    """Sync broker and instrument reference YAML data into Bronze tables."""
    settings = get_settings()

    brokers = settings.get_brokers()
    for b in brokers:
        broker_id = b.get("code") or b.get("broker_id")
        if not broker_id:
            continue
        db.execute(
            """
            INSERT INTO bronze_brokers (broker_id, broker_name, category, is_primary_target, description)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (broker_id) DO UPDATE SET
                broker_name = EXCLUDED.broker_name,
                category = EXCLUDED.category,
                is_primary_target = EXCLUDED.is_primary_target,
                description = EXCLUDED.description;
        """,
            [
                broker_id,
                b.get("name") or b.get("broker_name", broker_id),
                b.get("type") or b.get("category", "unknown"),
                b.get("is_primary_target", False),
                b.get("description", ""),
            ],
        )

    instruments = settings.get_instruments()
    for inst in instruments:
        symbol = inst.get("symbol")
        if not symbol:
            continue
        db.execute(
            """
            INSERT INTO bronze_instruments (symbol, name, sector, index_name, lot_multiplier)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (symbol) DO UPDATE SET
                name = EXCLUDED.name,
                sector = EXCLUDED.sector,
                index_name = EXCLUDED.index_name,
                lot_multiplier = EXCLUDED.lot_multiplier;
        """,
            [
                symbol,
                inst.get("name", symbol),
                inst.get("sector", "unknown"),
                inst.get("index") or inst.get("index_name", "BIST100"),
                float(inst.get("lot_multiplier", 1.0)),
            ],
        )

    logger.debug(f"Synced {len(brokers)} brokers and {len(instruments)} instruments into Bronze tables.")
