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

  # 6. Model 3: Focused session — forecast only AKBNK + GARAN (W2 and W5):
  .venv/bin/python scripts/run_pipeline.py --target gold --symbols AKBNK,GARAN --windows w2,w5

  # 7. Model 3: Load symbol list from file (one ticker per line):
  .venv/bin/python scripts/run_pipeline.py --target gold --symbols-file watchlist.txt
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
    parser = argparse.ArgumentParser(description="MDK Trading Oracle - Medallion Lakehouse Pipeline Runner")
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
    parser.add_argument(
        "--exclude-features",
        type=str,
        default=None,
        help="Comma-separated list of specific feature column names to exclude (e.g. 'feat_macro_rate_shock_decay,feat_bofa_holding_flow_prev_day')",
    )
    parser.add_argument(
        "--include-features",
        type=str,
        default=None,
        help="Comma-separated list of specific feature column names to force include",
    )
    parser.add_argument(
        "--disabled-clusters",
        type=str,
        default=None,
        help="Comma-separated list of semantic feature clusters to disable (e.g. 'macro_rates,calendar_dynamics')",
    )
    parser.add_argument(
        "--enabled-clusters",
        type=str,
        default=None,
        help="Comma-separated list of semantic feature clusters to exclusively enable",
    )
    # ── Model 3: Stock Intraday Reaction symbol/window filtering ───────────────────────────────
    parser.add_argument(
        "--symbols",
        type=str,
        default=None,
        help="Comma-separated BIST tickers to run Model 3 for (e.g. 'AKBNK,GARAN'). Default: all BIST30.",
    )
    parser.add_argument(
        "--symbols-file",
        type=str,
        default=None,
        help="Path to a file with one ticker per line for Model 3 symbol filtering.",
    )
    parser.add_argument(
        "--windows",
        type=str,
        default=None,
        help="Comma-separated windows to run Model 3 for (e.g. 'w2,w5'). Default: w2,w3,w5.",
    )
    parser.add_argument(
        "--run-stock-backtest",
        action="store_true",
        help="Also run full walk-forward backtests for Model 3 (slow — typically run once on setup).",
    )

    args = parser.parse_args()

    db = DuckDBManager()
    pipeline = MedallionPipeline(db)

    backfill_dates_list = [d.strip() for d in args.backfill_dates.split(",")] if args.backfill_dates else None
    exclude_features_list = [f.strip() for f in args.exclude_features.split(",")] if args.exclude_features else None
    include_features_list = [f.strip() for f in args.include_features.split(",")] if args.include_features else None
    disabled_clusters_list = [c.strip() for c in args.disabled_clusters.split(",")] if args.disabled_clusters else None
    enabled_clusters_list = [c.strip() for c in args.enabled_clusters.split(",")] if args.enabled_clusters else None

    # Resolve Model 3 symbol list
    symbols_list = None
    if args.symbols:
        symbols_list = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    elif args.symbols_file:
        sf_path = Path(args.symbols_file)
        if sf_path.exists():
            symbols_list = [line.strip().upper() for line in sf_path.read_text().splitlines() if line.strip()]
        else:
            logger.warning(f"--symbols-file path not found: {args.symbols_file}")

    windows_list = [w.strip() for w in args.windows.split(",")] if args.windows else None

    logger.info(
        f"Triggering Medallion Lakehouse Pipeline ("
        f"Target: {args.target}, Date: {args.date}, Month: {args.month}, File: {args.file}, "
        f"Force: {args.force}, Sync Catalog: {args.sync_catalog}, "
        f"Backfill: {backfill_dates_list or args.backfill_missing} "
        f"[Lookback: {args.backfill_lookback_months or args.backfill_lookback_days or 'default 2 months'}], "
        f"Excluded Features: {exclude_features_list}, Disabled Clusters: {disabled_clusters_list}, "
        f"Symbols: {symbols_list or 'ALL'}, Windows: {windows_list or 'ALL'})..."
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
        disabled_clusters=disabled_clusters_list,
        enabled_clusters=enabled_clusters_list,
        include_features=include_features_list,
        exclude_features=exclude_features_list,
        stock_reaction_symbols=symbols_list,
        stock_reaction_windows=windows_list,
        stock_reaction_backtest=args.run_stock_backtest,
    )


if __name__ == "__main__":
    main()
