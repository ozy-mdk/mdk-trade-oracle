import os
import sys

# Critical: Prevent OpenMP / BLAS thread contention across multiple processes on Apple Silicon
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import time
import concurrent.futures
from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.gold.tertip_multi_horizon import update_gold_tertip_multi_horizon_forecasts

def process_symbol(sym: str):
    db = PostgresManager()
    t0 = time.time()
    res = update_gold_tertip_multi_horizon_forecasts(db, symbols=[sym])
    elapsed = time.time() - t0
    msg = f"[{sym}] Finished in {elapsed:.1f}s: {res.get('status')}"
    print(msg, flush=True)
    return sym, res

def main():
    db = PostgresManager()
    # Find all active BIST 30 symbols that do NOT yet have 900 records (180 sessions x 5 horizons)
    df = db.query_pl("""
        SELECT m.symbol, COALESCE(b.cnt, 0) as cnt
        FROM bronze_bist30_membership m
        LEFT JOIN (
            SELECT symbol, count(*) as cnt 
            FROM gold_tertip_multi_horizon_backtests 
            GROUP BY symbol
        ) b ON m.symbol = b.symbol
        WHERE m.is_active = true
        ORDER BY m.symbol ASC;
    """).to_pandas()
    
    needed = df[df["cnt"] < 900]["symbol"].tolist()
    already_done = df[df["cnt"] >= 900]["symbol"].tolist()
    print(f"Already hydrated (900 rows): {already_done}", flush=True)
    print(f"Symbols needing 180-session hydration ({len(needed)}): {needed}", flush=True)
    
    if not needed:
        print("All symbols already fully hydrated to 180 sessions!", flush=True)
        return

    t_start = time.time()
    # Run 4 parallel workers with OMP_NUM_THREADS=1 (each takes ~12-15s)
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_symbol, sym): sym for sym in needed}
        for future in concurrent.futures.as_completed(futures):
            sym = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"[{sym}] Error: {e}", flush=True)

    print(f"All {len(needed)} symbols hydrated in {time.time() - t_start:.1f}s.", flush=True)

if __name__ == "__main__":
    main()
