#!/usr/bin/env python
"""CLI runner to export raw trade ticks and generate multi-timeframe candles in PostgreSQL."""

import argparse
import logging
import sys

from mdk_trading_oracle.data.postgres.candle_engine import (
    TIMEFRAME_INTERVALS,
    MultiTimeframeCandleEngine,
)
from mdk_trading_oracle.data.postgres.connection import PostgresConnectionManager
from mdk_trading_oracle.data.postgres.exporter import RawTradeExporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("sync_postgres")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync raw trades and generate multi-timeframe broker candles into PostgreSQL."
    )
    parser.add_argument(
        "--target",
        choices=["all", "raw", "candles", "schema", "status"],
        default="all",
        help="Target action: 'all' (raw + candles), 'raw' (only raw trades), 'candles' (only candles), 'schema' (init DDL), 'status' (row counts)",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="1m,5m,15m,30m,60m,120m,240m,8h",
        help="Comma-separated list of timeframes (e.g. '1m,5m,15m,30m,60m,120m,240m,8h')",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="Specific trading date to process (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date for range processing (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=None,
        help="End date for range processing (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--month",
        type=str,
        default=None,
        help="Process an entire month (e.g. '2026-03')",
    )
    parser.add_argument(
        "--drop-first",
        action="store_true",
        help="Drop existing tables before recreating schema (CAUTION: clears PG tables)",
    )
    parser.add_argument(
        "--no-delete-existing",
        action="store_true",
        help="Do not delete existing date records in PG prior to insertion",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pg_mgr = PostgresConnectionManager()

    if args.target == "schema":
        logger.info(f"Initializing PostgreSQL schema (drop_first={args.drop_first})...")
        pg_mgr.init_schema(drop_first=args.drop_first)
        counts = pg_mgr.get_table_counts()
        print("\nPostgreSQL Table Status:")
        for t, c in counts.items():
            print(f"  - {t}: {c:,} rows")
        return

    if args.target == "status":
        counts = pg_mgr.get_table_counts()
        print("\nPostgreSQL Table Status:")
        for t, c in counts.items():
            print(f"  - {t}: {c:,} rows")
        return

    # Ensure schema exists
    pg_mgr.init_schema(drop_first=args.drop_first)

    # Determine date range
    start_date = args.start_date
    end_date = args.end_date

    if args.date:
        start_date = args.date
        end_date = args.date
    elif args.month:
        start_date = f"{args.month}-01"
        # Approximate end of month
        end_date = f"{args.month}-31"

    timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]
    invalid_tfs = [tf for tf in timeframes if tf not in TIMEFRAME_INTERVALS]
    if invalid_tfs:
        logger.error(f"Invalid timeframes: {invalid_tfs}. Valid: {list(TIMEFRAME_INTERVALS.keys())}")
        sys.exit(1)

    delete_existing = not args.no_delete_existing

    # 1. Export Raw Trades
    if args.target in ["all", "raw"]:
        exporter = RawTradeExporter(pg_manager=pg_mgr)
        if args.date:
            logger.info(f"Exporting raw trades for date {args.date}...")
            exporter.export_date(args.date, delete_existing=delete_existing)
        else:
            logger.info(f"Exporting raw trades for range {start_date or 'ALL'} to {end_date or 'ALL'}...")
            exporter.export_range(start_date=start_date, end_date=end_date, delete_existing=delete_existing)

    # 2. Compute Candles & Broker Flows
    if args.target in ["all", "candles"]:
        engine = MultiTimeframeCandleEngine(pg_manager=pg_mgr)
        logger.info(
            f"Generating candles for timeframes {timeframes} across range {start_date or 'ALL'} to {end_date or 'ALL'}..."
        )
        engine.process_range(
            timeframes=timeframes,
            start_date=start_date,
            end_date=end_date,
            delete_existing=delete_existing,
        )

    # Final Summary
    counts = pg_mgr.get_table_counts()
    print("\n" + "=" * 50)
    print("PostgreSQL Synchronization Complete!")
    print("=" * 50)
    for t, c in counts.items():
        print(f"  - {t}: {c:,} rows")
    print("=" * 50)


if __name__ == "__main__":
    main()
