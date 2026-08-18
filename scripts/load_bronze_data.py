#!/usr/bin/env python3
"""Script to load raw BIST data into the DuckDB Bronze Layer."""

from datetime import datetime
from pathlib import Path
from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.ingestion.file_loader import FileIngestor

logger = get_logger("mdk_oracle.load_bronze")


def main():
    start_time = datetime.now()
    logger.info("Starting Bronze Layer Data Loading into DuckDB...")

    settings = get_settings()
    db = DuckDBManager()
    db.initialize_schema()

    ingestor = FileIngestor(db)
    
    # Check March 2026 raw CSV path
    raw_march_path = settings.bronze_dir / "2026/03_march/raw_csv/**/*.csv"
    logger.info(f"Targeting Bronze Raw CSVs: {raw_march_path}")

    res = ingestor.ingest_bist_raw_csv_glob(
        glob_pattern=raw_march_path.as_posix(),
        raw_source_label="bist_2026_03_march"
    )

    conn = db.get_connection()
    total_trades = conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0]
    total_brokers = conn.execute("SELECT COUNT(*) FROM bronze_brokers;").fetchone()[0]
    total_instruments = conn.execute("SELECT COUNT(*) FROM bronze_instruments;").fetchone()[0]

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Bronze Loading Complete in {elapsed:.1f}s | "
        f"bronze_raw_trades: {total_trades:,} rows | "
        f"bronze_brokers: {total_brokers} | "
        f"bronze_instruments: {total_instruments}"
    )


if __name__ == "__main__":
    main()
