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

After --target all|silver|gold the session-signal forward log runs (scripts/experiments/session_triggers.py
--forward): it publishes frozen triggers for the next session and grades earlier ones. Writes only under
~/data/mdk_oracle/experiments/; failures are logged and never fail the pipeline. Disable with --skip-forward-log.
"""

import argparse
import subprocess
import sys
from pathlib import Path

# Add project root to sys.path if invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdk_trading_oracle.core.db import PostgresManager
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
        "--skip-forward-log",
        action="store_true",
        help="Do not run the session-signal forward log after the pipeline",
    )
    args = parser.parse_args()

    db = PostgresManager()
    pipeline = MedallionPipeline(db)

    logger.info(
        f"Triggering Medallion Lakehouse Pipeline ("
        f"Target: {args.target}, Date: {args.date}, Month: {args.month}, File: {args.file}, "
        f"Force: {args.force}, Sync Catalog: {args.sync_catalog})..."
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
    )

    if args.target in ("all", "silver", "gold") and not args.skip_forward_log:
        run_forward_log()


def run_forward_log() -> None:
    """Publish next-session triggers and grade earlier ones (isolated experiment; never fails the pipeline)."""
    script = Path(__file__).resolve().parent / "experiments" / "session_triggers.py"
    logger.info("Running session-signal forward log...")
    try:
        res = subprocess.run([sys.executable, str(script), "--forward"], capture_output=True, text=True, timeout=1800)
        out = "\n".join(line for line in res.stdout.splitlines() if "Warning" not in line)
        if res.returncode == 0:
            logger.info(f"Forward log done:\n{out}")
        else:
            logger.warning(f"Forward log failed (exit {res.returncode}):\n{res.stderr[-2000:]}")
    except Exception as e:  # pragma: no cover - defensive: the pipeline result must not depend on the experiment
        logger.warning(f"Forward log could not run: {e}")


if __name__ == "__main__":
    main()
