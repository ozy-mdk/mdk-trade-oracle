#!/usr/bin/env python3
"""Stream annual BIST ZIP archives into a resumable DuckDB Bronze table."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import DuckDBManager
from mdk_trading_oracle.data.bronze import BronzeIngestor, initialize_bronze_schema


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Load yearly BIST ZIP archives into DuckDB Bronze without permanent extraction."
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=settings.raw_data_dir / "yearly_archives",
        help="Directory containing annual ZIP archives",
    )
    parser.add_argument(
        "--temp-dir",
        type=Path,
        default=settings.database_dir / "tmp" / "zip_ingest",
        help="Temporary batch directory",
    )
    parser.add_argument("--batch-mb", type=int, default=512)
    parser.add_argument("--limit-files", type=int)
    args = parser.parse_args()

    db = DuckDBManager()
    try:
        initialize_bronze_schema(db)
        result = BronzeIngestor(db).ingest_bist_zip_archives(
            archive_dir=args.archive_dir,
            temp_dir=args.temp_dir,
            batch_mb=args.batch_mb,
            limit_files=args.limit_files,
        )
        conn = db.get_connection()
        coverage = conn.execute("""
            SELECT
                COUNT(*) AS row_count,
                MIN(timestamp) AS min_timestamp,
                MAX(timestamp) AS max_timestamp,
                COUNT(DISTINCT CAST(timestamp AS DATE)) AS trading_days,
                COUNT(DISTINCT symbol) AS symbols
            FROM bronze_raw_trades;
        """).fetchone()
        result["coverage"] = {
            "row_count": coverage[0],
            "min_timestamp": str(coverage[1]),
            "max_timestamp": str(coverage[2]),
            "trading_days": coverage[3],
            "symbols": coverage[4],
        }
        print(json.dumps(result, ensure_ascii=False, indent=2))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
