#!/usr/bin/env python3
"""Unified CLI Runner for the Medallion Data Lakehouse Pipeline (Bronze -> Silver -> Gold).

Examples:
  # 1. Incremental Pipeline (only ingests new/modified files, runs Silver & Gold):
  .venv/bin/python scripts/run_pipeline.py --target all

  # 2. Pipeline with Catalog Discovery & Sync:
  .venv/bin/python scripts/run_pipeline.py --target all --sync-catalog

  # 3. Selective Single-Date Re-ingestion & Pipeline Update:
  .venv/bin/python scripts/run_pipeline.py --target all --date 2026-03-09

  # 4. Selective Month Re-ingestion:
  .venv/bin/python scripts/run_pipeline.py --target all --month 2026-03

  # 5. Full Force Rebuild:
  .venv/bin/python scripts/run_pipeline.py --target all --force
"""

import argparse
import sys
from pathlib import Path

# Add project root to sys.path if invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.pipeline import MedallionPipeline

logger = get_logger("mdk_oracle.scripts.pipeline")


def main():
    parser = argparse.ArgumentParser(
        description="MDK Trading Oracle - Medallion Lakehouse Pipeline Runner"
    )
    parser.add_argument(
        "--target",
        "-t",
        choices=["catalog", "bronze", "silver", "gold", "all"],
        default="all",
        help="Target layer to run: catalog, bronze, silver, gold, or all (default: 'all')",
    )
    parser.add_argument(
        "--sync-catalog",
        "-s",
        action="store_true",
        help="Discover entities and synchronize YAML catalogs before executing pipeline",
    )
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
    parser.add_argument(
        "--glob",
        "-g",
        type=str,
        default=None,
        help="Custom raw CSV glob pattern (for Bronze ingestion)",
    )
    parser.add_argument(
        "--no-deps",
        action="store_true",
        help="Disable automatic dependency resolution (e.g., run only Silver without verifying Bronze)",
    )
    parser.add_argument(
        "--backfill-dates",
        type=str,
        default=None,
        help="Comma-separated list of historical dates to point-in-time backfill into performance ledger (e.g. '2026-03-10,2026-03-18')",
    )
    parser.add_argument(
        "--backfill-missing",
        action="store_true",
        help="Auto-discover all completed historical sessions in Silver missing from performance ledger and backfill point-in-time",
    )
    parser.add_argument(
        "--backfill-lookback-months",
        type=int,
        default=None,
        help="Number of trailing months to look back when running --backfill-missing (default: 2 months from config)",
    )
    parser.add_argument(
        "--backfill-lookback-days",
        type=int,
        default=None,
        help="Number of trailing days to look back when running --backfill-missing (e.g. 60 days)",
    )

    args = parser.parse_args()

    db = DuckDBManager()
    pipeline = MedallionPipeline(db)

    backfill_dates_list = [d.strip() for d in args.backfill_dates.split(",")] if args.backfill_dates else None

    logger.info(
        f"Triggering Medallion Lakehouse Pipeline ("
        f"Target: {args.target}, Date: {args.date}, Month: {args.month}, File: {args.file}, "
        f"Force: {args.force}, Sync Catalog: {args.sync_catalog}, "
        f"Backfill: {backfill_dates_list or args.backfill_missing} "
        f"[Lookback: {args.backfill_lookback_months or args.backfill_lookback_days or 'default 2 months'}])..."
    )
    pipeline.run(
        target=args.target,
        raw_glob=args.glob,
        target_date=args.date,
        target_month=args.month,
        target_file=args.file,
        force=args.force,
        sync_catalog=args.sync_catalog,
        resolve_dependencies=not args.no_deps,
        print_summary=True,
        backfill_dates=backfill_dates_list,
        all_missing=args.backfill_missing,
        backfill_lookback_months=args.backfill_lookback_months,
        backfill_lookback_days=args.backfill_lookback_days,
    )


if __name__ == "__main__":
    main()

