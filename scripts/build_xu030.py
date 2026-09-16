#!/usr/bin/env python3
"""Build and synchronize synthetic BIST 30 (XU030) index candles, flows, and daily summaries.

Usage:
    .venv/bin/python scripts/build_xu030.py --date 2026-09-14 --all-timeframes
    .venv/bin/python scripts/build_xu030.py --all-dates --daily-summary
"""

import argparse
import sys
from pathlib import Path
from typing import List

# Add project root to sys.path if invoked directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdk_trading_oracle.core.logger import get_logger
from mdk_trading_oracle.data.bist30.index_engine import BIST30IndexEngine
from mdk_trading_oracle.data.postgres.connection import PostgresConnectionManager

logger = get_logger("mdk_oracle.scripts.build_xu030")

ALL_TIMEFRAMES = ["1m", "5m", "15m", "30m", "60m", "120m", "240m", "8h"]


def get_available_pg_dates() -> List[str]:
    """Retrieve distinct dates available in PostgreSQL market_candles."""
    mgr = PostgresConnectionManager()
    conn = mgr.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT DISTINCT substring(bucket_start::text from 1 for 10) as dt
                FROM market_candles
                WHERE symbol != 'XU030'
                ORDER BY dt ASC;
            """)
            return [r[0] for r in cur.fetchall() if r[0]]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Build synthetic BIST 30 (XU030) index candles and daily summaries."
    )
    parser.add_argument(
        "--date",
        type=str,
        help="Target trade date (YYYY-MM-DD). If omitted and --all-dates not set, uses latest available date.",
    )
    parser.add_argument(
        "--all-dates",
        action="store_true",
        help="Process all distinct dates currently stored in PostgreSQL market_candles.",
    )
    parser.add_argument(
        "--timeframes",
        type=str,
        default="60m,5m,8h",
        help="Comma-separated timeframes (e.g., '1m,5m,15m,30m,60m,120m,240m,8h').",
    )
    parser.add_argument(
        "--all-timeframes",
        action="store_true",
        help="Process all 8 supported timeframes ('1m,5m,15m,30m,60m,120m,240m,8h').",
    )
    parser.add_argument(
        "--daily-summary",
        action="store_true",
        default=True,
        help="Populate DuckDB silver_daily_stock_summary for XU030 (default: True).",
    )
    args = parser.parse_args()

    engine = BIST30IndexEngine()

    if args.all_timeframes:
        timeframes = ALL_TIMEFRAMES
    else:
        timeframes = [tf.strip() for tf in args.timeframes.split(",") if tf.strip()]

    if args.all_dates:
        dates = get_available_pg_dates()
    elif args.date:
        dates = [args.date[:10]]
    else:
        avail = get_available_pg_dates()
        dates = [avail[-1]] if avail else ["2026-09-14"]

    logger.info(f"Starting XU030 build for {len(dates)} dates: {dates} across timeframes {timeframes}")

    total_candles = 0
    for d in dates:
        logger.info(f"--- Processing XU030 for date: {d} ---")
        weights = engine.get_constituent_weights(d)
        logger.info(f"Loaded {len(weights)} constituent weights for {d}.")

        for tf in timeframes:
            count = engine.generate_xu030_candles(d, tf)
            total_candles += count
            logger.info(f"  [{tf}] Generated/updated {count} candles.")

        if args.daily_summary:
            d_cnt = engine.generate_xu030_daily_summary(trade_date=d)
            logger.info(f"  [Daily Summary] DuckDB silver_daily_stock_summary updated ({d_cnt} records).")

    logger.info(f"Finished XU030 build. Total candles generated/updated: {total_candles}.")


if __name__ == "__main__":
    main()
