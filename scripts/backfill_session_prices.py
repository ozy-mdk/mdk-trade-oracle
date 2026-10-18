#!/usr/bin/env python
"""Back-fill `silver_daily_session_prices` from the tick archive (dates older than the bronze retention window).

Same definitions as `SilverTransformer.transform_daily_session_prices`: each reference price is the volume-weighted
average of the prints in the decisive second (first second of the day, last second before 10:30 / 11:30, last second of
the day); VWAPs over [11:30, 11:35) and [11:30, 11:45). Every archive date is (re)written; dates after the archive come
from the silver transform on `bronze_raw_trades`.

Usage:
    .venv/bin/python scripts/backfill_session_prices.py                 # every symbol that has been in BIST30
    .venv/bin/python scripts/backfill_session_prices.py --symbols ASELS,GARAN
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mdk_trading_oracle.core.config import get_settings  # noqa: E402
from mdk_trading_oracle.core.db import PostgresManager  # noqa: E402
from mdk_trading_oracle.data.silver.schema import initialize_silver_schema  # noqa: E402

BATCH = 3


def universe(db: PostgresManager) -> list[str]:
    rows = db.query_df("SELECT DISTINCT symbol FROM silver_daily_stock_summary WHERE index_name = 'BIST30'")
    return sorted(rows["symbol"].tolist())


def session_prices(files: list[str], symbols: list[str]) -> pl.DataFrame:
    t = pl.col("timestamp").dt.time()
    lf = (pl.scan_parquet(files).filter(pl.col("symbol").is_in(symbols) & (pl.col("volume") > 0))
          .select(["timestamp", "symbol", "price", "volume"])
          .with_columns(pl.col("timestamp").dt.date().alias("trade_date"), t.alias("t"), (pl.col("price") * pl.col("volume")).alias("tl")))
    ts = pl.col("timestamp")

    def second_vwap(at: pl.Expr) -> pl.Expr:
        m = ts == at
        return pl.col("tl").filter(m).sum() / pl.col("volume").filter(m).sum()

    w5 = pl.col("t").is_between(pl.time(11, 30), pl.time(11, 35), closed="left")
    w15 = pl.col("t").is_between(pl.time(11, 30), pl.time(11, 45), closed="left")
    out = (
        lf.group_by(["trade_date", "symbol"])
        .agg(
            second_vwap(ts.min()).alias("open_price"),
            second_vwap(ts.filter(pl.col("t") < pl.time(10, 30)).max()).alias("price_1030"),
            second_vwap(ts.filter(pl.col("t") < pl.time(11, 30)).max()).alias("price_1130"),
            (pl.col("tl").filter(w5).sum() / pl.col("volume").filter(w5).sum()).alias("vwap_1130_1135"),
            (pl.col("tl").filter(w15).sum() / pl.col("volume").filter(w15).sum()).alias("vwap_1130_1145"),
            second_vwap(ts.max()).alias("close_price"),
            pl.col("t").max().alias("last_trade_time"),
            pl.len().alias("trade_count"),
        )
        .with_columns((pl.col("last_trade_time") < pl.time(17, 0)).alias("is_half_day"))
        .collect()
    )
    return out.with_columns([pl.col(c).fill_nan(None) for c in ("open_price", "price_1030", "price_1130", "vwap_1130_1135", "vwap_1130_1145", "close_price")])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    a = ap.parse_args()
    db = PostgresManager()
    initialize_silver_schema(db)
    syms = [s.strip().upper() for s in a.symbols.split(",") if s.strip()] or universe(db)
    files = sorted(glob.glob(str(get_settings().raw_data_dir / "mdk_parquet_part_0*" / "*.parquet")))
    print(f"{len(syms)} symbols, {len(files)} archive files", flush=True)
    total = 0
    for i in range(0, len(syms), BATCH):  # batches keep memory bounded
        batch = syms[i:i + BATCH]
        df = session_prices(files, batch).select(["trade_date", "symbol", "open_price", "price_1030", "price_1130", "vwap_1130_1135",
                                                  "vwap_1130_1145", "close_price", "last_trade_time", "is_half_day", "trade_count"])
        syms_in = ", ".join(f"'{x}'" for x in batch)
        db.execute(f"DELETE FROM silver_daily_session_prices WHERE symbol IN ({syms_in}) AND trade_date <= '{df['trade_date'].max()}';")
        total += db.copy_df_to_table(df, "silver_daily_session_prices")
        print(f"  {', '.join(batch)}: {df.height:,} rows", flush=True)
    print(f"back-filled {total:,} rows", flush=True)

if __name__ == "__main__":
    main()
