#!/usr/bin/env python3
"""Script to load raw BIST data into the DuckDB Bronze Layer with incremental and selective partition support."""

import argparse
from datetime import datetime

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.bronze import BronzeIngestor

logger = get_logger("mdk_oracle.load_bronze")


def main():
    parser = argparse.ArgumentParser(description="Load raw BIST trade CSV/Parquet feeds into the DuckDB Bronze Layer.")
    parser.add_argument(
        "--date",
        "-d",
        type=str,
        default=None,
        help="Target a specific trading date partition for selective update (e.g. '2026-03-09')",
    )
    parser.add_argument(
        "--month",
        "-m",
        type=str,
        default=None,
        help="Target a specific month partition for selective update (e.g. '2026-03')",
    )
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        default=None,
        help="Target a specific raw CSV or Parquet file for ingestion",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force full rebuild of Bronze layer (clears tables and re-ingests all raw files)",
    )

    args = parser.parse_args()

    start_time = datetime.now()
    logger.info("Starting Bronze Layer Data Loading into DuckDB...")

    db = DuckDBManager()
    db.initialize_schema()

    ingestor = BronzeIngestor(db)

    if args.date:
        res = ingestor.ingest_date(args.date)
    elif args.month:
        res = ingestor.ingest_month(args.month)
    elif args.file:
        res = ingestor.ingest_file(args.file, force=args.force)
    elif args.force:
        res = ingestor.ingest_all(force=True)
    else:
        res = ingestor.ingest_incremental()

    logger.info(f"Ingestion result status: {res.get('status')}")

    conn = db.get_connection()
    total_trades = conn.execute("SELECT COUNT(*) FROM bronze_raw_trades;").fetchone()[0]
    total_logged = conn.execute("SELECT COUNT(*) FROM bronze_ingestion_log;").fetchone()[0]
    total_brokers = conn.execute("SELECT COUNT(*) FROM bronze_brokers;").fetchone()[0]
    total_instruments = conn.execute("SELECT COUNT(*) FROM bronze_instruments;").fetchone()[0]

    elapsed = (datetime.now() - start_time).total_seconds()
    logger.info(
        f"Bronze Loading Complete in {elapsed:.1f}s | "
        f"bronze_raw_trades: {total_trades:,} rows | "
        f"bronze_ingestion_log: {total_logged:,} files | "
        f"bronze_brokers: {total_brokers} | "
        f"bronze_instruments: {total_instruments}"
    )


if __name__ == "__main__":
    main()
