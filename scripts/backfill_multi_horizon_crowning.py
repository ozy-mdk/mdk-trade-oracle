#!/usr/bin/env python
"""Backfill the honest monthly re-crowning tables for the 1D and 3D-30D forecasters, then publish gold.

For every active BIST 30 symbol (in parallel): extend gold_tertip_multi_horizon_candidates over the trailing
CANDIDATE_HISTORY_SESSIONS origins (75 candidates x 6 horizons, point-in-time walk-forward) and crown every recent
month start in gold_tertip_multi_horizon_crowns using only outcomes realized before it. The nightly gold job then only
adds the newest sessions. Finally publishes gold: update_gold_tertip_forecasts (1D) and
update_gold_tertip_multi_horizon_forecasts (3D-30D), each live forecast + 90-session track.

Usage:
    .venv/bin/python scripts/backfill_multi_horizon_crowning.py                  # all active BIST 30
    .venv/bin/python scripts/backfill_multi_horizon_crowning.py --symbols ASELS,GARAN
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import time  # noqa: E402
from multiprocessing import Pool  # noqa: E402

from mdk_trading_oracle.core.db import PostgresManager  # noqa: E402


def backfill_symbol(sym: str) -> tuple[str, int, int, float]:
    from mdk_trading_oracle.data.gold.tertip_multi_horizon_crowning import (
        _dates,
        load_candidates,
        update_candidate_ledger,
        update_crowns,
        update_t1_candidate_ledger,
    )
    from mdk_trading_oracle.data.silver.tertip_ml_forecaster import extract_3pillar_time_series

    t0 = time.time()
    db = PostgresManager()
    try:
        df = extract_3pillar_time_series(db, sym, lookback_days=550)
        if df.empty or len(df) < 120:
            return sym, 0, 0, time.time() - t0
        df = df.sort_values("trade_date").reset_index(drop=True)
        rows = update_t1_candidate_ledger(db, sym, df) + update_candidate_ledger(db, sym, df)
        crowns = update_crowns(db, sym, load_candidates(db, sym), _dates(df))
        return sym, rows, crowns, time.time() - t0
    finally:
        db.close()


def main() -> None:
    from mdk_trading_oracle.data.gold.tertip_forecasts import update_gold_tertip_forecasts
    from mdk_trading_oracle.data.gold.tertip_multi_horizon import update_gold_tertip_multi_horizon_forecasts
    from mdk_trading_oracle.data.gold.tertip_multi_horizon_crowning import ensure_crowning_tables

    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--skip-gold", action="store_true", help="only build candidates and crowns")
    args = ap.parse_args()
    db = PostgresManager()
    ensure_crowning_tables(db)
    if args.symbols:
        symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    else:
        symbols = db.query_df("SELECT symbol FROM bronze_bist30_membership WHERE is_active = true ORDER BY symbol")["symbol"].tolist()
    t0 = time.time()
    with Pool(min(args.workers, len(symbols))) as pool:
        for sym, rows, crowns, secs in pool.imap_unordered(backfill_symbol, symbols):
            print(f"[{sym}] {rows} candidate rows, {crowns} crowns in {secs:.0f}s ({time.time() - t0:.0f}s total)", flush=True)
    if not args.skip_gold:
        t1 = update_gold_tertip_forecasts(db, symbols=symbols)
        print(f"gold 1D: {t1['equities_updated']} symbols", flush=True)
        res = update_gold_tertip_multi_horizon_forecasts(db, symbols=symbols)
        print(f"gold 3D-30D: {res['updated_symbols']}/{res['total_requested']} symbols, errors {res['errors'][:3]}", flush=True)
    print(f"done in {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
