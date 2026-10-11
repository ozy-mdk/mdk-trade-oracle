import concurrent.futures
import time
from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.gold.tertip_multi_horizon import update_gold_tertip_multi_horizon_forecasts

def process_symbol(sym: str):
    db = PostgresManager()
    t0 = time.time()
    res = update_gold_tertip_multi_horizon_forecasts(db, symbols=[sym])
    elapsed = time.time() - t0
    print(f"[{sym}] Finished in {elapsed:.1f}s: {res.get('status')}")
    return sym, res

def main():
    db = PostgresManager()
    rows = db.query_pl("SELECT symbol FROM bronze_bist30_membership WHERE is_active = true ORDER BY symbol ASC;").to_pandas()
    symbols = [s for s in rows["symbol"].tolist() if s != "THYAO"]
    print(f"Starting parallel hydration of 180 backtest sessions for {len(symbols)} symbols...")
    
    t_start = time.time()
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(process_symbol, symbols))
    
    print(f"All {len(symbols)} symbols completed in {time.time() - t_start:.1f}s.")

if __name__ == "__main__":
    main()
