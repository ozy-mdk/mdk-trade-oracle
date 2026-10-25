#!/usr/bin/env python
"""Load the event calendar into bronze (idempotent upserts).

Raw files under <DATA_DIR>/00_raw_data/event_calendar/:
    tcmb_mpc_meetings.csv           TCMB Monetary Policy Committee meeting dates 2022-2026 (scheduled + interim),
                                    compiled from the published annual calendars (sources in `source_note`)
    bist_dividends_yfinance.csv     ex-dividend dates and gross dividend per share (Yahoo Finance via yfinance)

Tables: bronze_tcmb_mpc_meetings (meeting_date PK), bronze_dividends (symbol, ex_date PK).
BIST 30 rebalancing dates already live in bronze_bist30_changes (quarterly effective dates, in / out symbols).

Usage:
    .venv/bin/python scripts/load_event_calendar.py
    .venv/bin/python scripts/load_event_calendar.py --refresh-dividends   # re-download dividends first
"""

from __future__ import annotations

import argparse

import pandas as pd

from mdk_trading_oracle.core.config import get_settings
from mdk_trading_oracle.core.db import PostgresManager


def refresh_dividends(db: PostgresManager, path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    import yfinance as yf

    syms = db.query_df("SELECT DISTINCT symbol FROM bronze_bist30_stock_periods ORDER BY 1")["symbol"].tolist()

    def get(s: str) -> pd.DataFrame:
        try:
            d = yf.Ticker(f"{s}.IS").dividends
        except Exception:
            return pd.DataFrame()
        return pd.DataFrame({"symbol": s, "ex_date": d.index.tz_localize(None).strftime("%Y-%m-%d"), "gross_dividend_tl": d.values})

    with ThreadPoolExecutor(12) as ex:
        out = pd.concat([x for x in ex.map(get, syms) if not x.empty], ignore_index=True)
    out = out[out["ex_date"] >= "2021-01-01"].assign(source="yfinance Ticker.dividends (Yahoo Finance, split-adjusted per share)")
    out.to_csv(path, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh-dividends", action="store_true")
    args = ap.parse_args()
    raw = get_settings().data_dir / "00_raw_data" / "event_calendar"
    db = PostgresManager()
    db.execute("""
        CREATE TABLE IF NOT EXISTS bronze_tcmb_mpc_meetings (
            meeting_date DATE PRIMARY KEY,
            meeting_type VARCHAR(16) NOT NULL,
            source_note TEXT,
            ingested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE IF NOT EXISTS bronze_dividends (
            symbol VARCHAR(16) NOT NULL,
            ex_date DATE NOT NULL,
            gross_dividend_tl DOUBLE PRECISION,
            source TEXT,
            ingested_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (symbol, ex_date)
        );
    """)
    mpc = pd.read_csv(raw / "tcmb_mpc_meetings.csv")
    for r in mpc.itertuples(index=False):
        db.execute(
            """INSERT INTO bronze_tcmb_mpc_meetings (meeting_date, meeting_type, source_note) VALUES (%s, %s, %s)
               ON CONFLICT (meeting_date) DO UPDATE SET meeting_type = EXCLUDED.meeting_type,
               source_note = EXCLUDED.source_note, ingested_at = CURRENT_TIMESTAMP;""",
            params=[r.meeting_date, r.meeting_type, r.source_note],
        )
    div_path = raw / "bist_dividends_yfinance.csv"
    if args.refresh_dividends or not div_path.exists():
        refresh_dividends(db, div_path)
    div = pd.read_csv(div_path)
    for r in div.itertuples(index=False):
        db.execute(
            """INSERT INTO bronze_dividends (symbol, ex_date, gross_dividend_tl, source) VALUES (%s, %s, %s, %s)
               ON CONFLICT (symbol, ex_date) DO UPDATE SET gross_dividend_tl = EXCLUDED.gross_dividend_tl,
               source = EXCLUDED.source, ingested_at = CURRENT_TIMESTAMP;""",
            params=[r.symbol, r.ex_date, float(r.gross_dividend_tl), r.source],
        )
    print(f"bronze_tcmb_mpc_meetings: {len(mpc)} meetings; bronze_dividends: {len(div)} ex-dates ({div['symbol'].nunique()} symbols)")


if __name__ == "__main__":
    main()
