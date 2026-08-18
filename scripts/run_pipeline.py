#!/usr/bin/env python3
"""Unified CLI Runner for the Medallion Data Lakehouse Pipeline (Bronze -> Silver -> Gold)."""

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
        choices=["bronze", "silver", "gold", "all"],
        default="all",
        help="Target layer to run (default: 'all')",
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

    args = parser.parse_args()

    db = DuckDBManager()
    pipeline = MedallionPipeline(db)

    logger.info(f"Triggering Medallion Lakehouse Pipeline (Target: {args.target})...")
    pipeline.run(
        target=args.target,
        raw_glob=args.glob,
        resolve_dependencies=not args.no_deps,
        print_summary=True,
    )


if __name__ == "__main__":
    main()
