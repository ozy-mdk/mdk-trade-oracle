#!/usr/bin/env python
"""Replace stray partial bronze days with the complete days from the raw parquet export, then rebuild silver.

Bronze held a few partial historical days (blocks of 32,768 rows, no bronze_ingestion_log entry); complete history
for them lives in the raw export ~/data/mdk_oracle/00_raw_data/mdk_parquet_part_*/bronze_raw_trades.part-*.parquet.
An incremental silver run that rebuilt every bronze date overwrote those silver days with partial data, and the
carried-forward FIFO ledger then changed every inventory feature after them (fixed in SilverTransformer.run_all,
which now recomputes only pending dates).

Usage:
    .venv/bin/python scripts/repair_stray_bronze_days.py            # dry run: report stray days and raw row counts
    .venv/bin/python scripts/repair_stray_bronze_days.py --apply    # replace them, log the repair, rebuild silver
"""

from __future__ import annotations

import argparse
import glob
import time

import polars as pl

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import PostgresManager

RAW_GLOB = "mdk_parquet_part_*/bronze_raw_trades.part-*.parquet"
COLUMNS = ["trade_id", "timestamp", "symbol", "price", "volume", "buyer_broker_id", "seller_broker_id", "raw_source", "ingested_at"]


def stray_days(db: PostgresManager) -> list[tuple[str, int]]:
    rows = db.execute("""
        SELECT CAST(timestamp AS DATE) AS d, COUNT(*) AS n FROM bronze_raw_trades
        WHERE CAST(timestamp AS DATE) NOT IN (SELECT trade_date FROM bronze_ingestion_log WHERE trade_date IS NOT NULL)
        GROUP BY 1 ORDER BY 1;
    """).fetchall()
    return [(str(d), int(n)) for d, n in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()
    db = PostgresManager()
    files = sorted(glob.glob(str(get_settings().data_dir / "00_raw_data" / RAW_GLOB)))
    days = stray_days(db)
    if not days:
        print("no stray bronze days")
        return
    raw = (
        pl.scan_parquet(files)
        .filter(pl.col("timestamp").cast(pl.Date).cast(pl.Utf8).is_in([d for d, _ in days]))
        .with_columns(pl.col("timestamp").cast(pl.Date).cast(pl.Utf8).alias("_d"))
    )
    counts = dict(raw.group_by("_d").agg(pl.len()).collect().iter_rows())
    print(f"{len(days)} stray bronze day(s); raw export files: {len(files)}")
    repair = []
    for d, n in days:
        r = int(counts.get(d, 0))
        ok = r > n
        print(f"  {d}: bronze {n:,} rows, raw {r:,} rows -> {'REPLACE' if ok else 'keep (raw not larger)'}")
        if ok:
            repair.append(d)
    if not args.apply or not repair:
        print("dry run" if not args.apply else "nothing to repair")
        return

    t0 = time.time()
    for d in repair:
        day = raw.filter(pl.col("_d") == d).select(COLUMNS).collect()
        db.execute("DELETE FROM bronze_raw_trades WHERE CAST(timestamp AS DATE) = %s;", params=[d])
        n = db.copy_df_to_table(day, "bronze_raw_trades")
        db.execute(
            """
            INSERT INTO bronze_ingestion_log (file_path, file_name, trade_date, year_month, rows_ingested, raw_source_label, ingested_at)
            VALUES (%s, %s, %s, %s, %s, %s, NOW());
            """,
            params=[f"repair://{RAW_GLOB}#{d}", "mdk_parquet_parts", d, d[:7], n, "parquet_export_repair"],
        )
        print(f"  {d}: {n:,} rows restored ({time.time() - t0:.0f}s)", flush=True)

    from mdk_trading_oracle.data.pipeline import MedallionPipeline

    print(f"rebuilding silver for {len(repair)} date(s) + full FIFO ledger ...", flush=True)
    MedallionPipeline().run_silver(target_dates=repair)
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
