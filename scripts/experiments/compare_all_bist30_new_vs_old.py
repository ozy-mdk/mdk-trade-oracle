"""Parallel Comparison: NEW 39-feature 3-Pillar suite vs OLD 25-feature suite across all 30 BIST 30 equities."""

import time

import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    FEATURE_COLS,
    extract_3pillar_time_series,
    run_30d_walk_forward_arena,
)

BIST30_SYMBOLS = [
    "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "DOAS", "EKGYO", "ENKAI",
    "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL", "KONTR", "KOZAL", "KRDMD",
    "OYAKC", "PETKM", "PGSUS", "SAHOL", "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS",
]

OLD_FEATURES = [
    "feat_cost_spread_pct",
    "feat_ret_today_pct",
    "feat_ret_yesterday_pct",
    "feat_mlb_w5_share",
    "feat_prophet_ret_today_pct",
    "feat_bist30_ret_today_pct",
    "feat_days_since_pos_shock",
    "feat_days_since_neg_shock",
    "feat_mlb_tertip_3m_ratio",
    "feat_big5_tertip_3m_ratio",
    "feat_kamu_tertip_3m_ratio",
    "feat_mlb_buy_share_today",
    "feat_mlb_sell_share_today",
    "feat_mlb_pnl_share_today",
    "feat_big5_buy_share_today",
    "feat_big5_sell_share_today",
    "feat_big5_pnl_share_today",
    "feat_kamu_buy_share_today",
    "feat_kamu_sell_share_today",
    "feat_kamu_pnl_share_today",
    "feat_total_inst_net_share_today",
    "feat_bofa_aggression_ratio",
    "feat_tertip_squeeze_delta",
    "feat_volatility_pinch_5d_20d",
    "feat_tertip_inventory_zscore",
]

def eval_stock(db: PostgresManager, sym: str) -> dict:
    df = extract_3pillar_time_series(db, sym, lookback_days=550)
    if df.empty or len(df) < 65:
        return {"symbol": sym, "error": "Insufficient data"}
    
    # 1. Run OLD Feature Suite
    df_old = df.copy()
    if "feat_cost_spread_pct" not in df_old.columns:
        df_old["feat_cost_spread_pct"] = df_old.get("feat_mlb_cost_spread_pct", 0.0)
    if "feat_prophet_ret_today_pct" not in df_old.columns:
        df_old["feat_prophet_ret_today_pct"] = 0.0
    if "feat_mlb_tertip_3m_ratio" not in df_old.columns:
        df_old["feat_mlb_tertip_3m_ratio"] = 0.0
    if "feat_big5_tertip_3m_ratio" not in df_old.columns:
        df_old["feat_big5_tertip_3m_ratio"] = 0.0
    if "feat_kamu_tertip_3m_ratio" not in df_old.columns:
        df_old["feat_kamu_tertip_3m_ratio"] = 0.0
        
    _, s_old, _ = run_30d_walk_forward_arena(df_old, n_sessions=30, feature_cols=OLD_FEATURES)
    old_model = s_old.get("ml_champion_type", "N/A")
    old_hits = s_old.get("champion_dir_hits", 0)
    old_hit_rate = s_old.get("champion_dir_hit_rate_pct", 0.0) * 100.0
    old_loss = s_old.get("champion_30d_penalty_loss", 999.0)
    old_big_hits = s_old.get("champion_30d_big_move_hits", 0)
    old_big_total = s_old.get("champion_30d_big_move_total", 0)
    
    # 2. Run NEW 39-Feature Suite
    _, s_new, _ = run_30d_walk_forward_arena(df, n_sessions=30, feature_cols=FEATURE_COLS)
    new_model = s_new.get("ml_champion_type", "N/A")
    new_hits = s_new.get("champion_dir_hits", 0)
    new_hit_rate = s_new.get("champion_dir_hit_rate_pct", 0.0) * 100.0
    new_loss = s_new.get("champion_30d_penalty_loss", 999.0)
    new_big_hits = s_new.get("champion_30d_big_move_hits", 0)
    new_big_total = s_new.get("champion_30d_big_move_total", 0)
    
    delta_hits = new_hits - old_hits
    delta_loss = new_loss - old_loss
    
    return {
        "symbol": sym,
        "old_model": old_model,
        "old_hits": old_hits,
        "old_hit_rate_pct": round(old_hit_rate, 1),
        "old_loss": round(old_loss, 2),
        "old_big_hits": old_big_hits,
        "old_big_total": old_big_total,
        "new_model": new_model,
        "new_hits": new_hits,
        "new_hit_rate_pct": round(new_hit_rate, 1),
        "new_loss": round(new_loss, 2),
        "new_big_hits": new_big_hits,
        "new_big_total": new_big_total,
        "delta_hits": delta_hits,
        "delta_loss": round(delta_loss, 2),
    }

def main():
    t_start = time.time()
    db = PostgresManager()
    print(f"Starting comparison across {len(BIST30_SYMBOLS)} BIST 30 stocks...", flush=True)
    print("=" * 115, flush=True)
    print(f"{'#':<3} {'Symbol':<7} | {'OLD Model':<10} {'OLD Hits':<11} {'OLD Loss':<9} | {'NEW Model':<10} {'NEW Hits':<11} {'NEW Loss':<9} | {'Δ Hits':<8} {'Δ Loss'}", flush=True)
    print("=" * 115, flush=True)
    
    results = []
    for idx, sym in enumerate(BIST30_SYMBOLS, 1):
        t0 = time.time()
        r = eval_stock(db, sym)
        if "error" not in r:
            results.append(r)
            sign_hits = f"+{r['delta_hits']}" if r['delta_hits'] > 0 else f"{r['delta_hits']}"
            sign_loss = f"{r['delta_loss']:+.2f}"
            print(
                f"[{idx:2d}/30] {r['symbol']:<5} | "
                f"{r['old_model']:<10} {r['old_hits']:2d}/30 ({r['old_hit_rate_pct']:.0f}%) {r['old_loss']:<9.2f} | "
                f"{r['new_model']:<10} {r['new_hits']:2d}/30 ({r['new_hit_rate_pct']:.0f}%) {r['new_loss']:<9.2f} | "
                f"{sign_hits:<8} {sign_loss} ({time.time()-t0:.1f}s)",
                flush=True,
            )
        else:
            print(f"[{idx:2d}/30] {sym:<5} | Error: {r['error']}", flush=True)

    df_res = pd.DataFrame(results)
    print("=" * 115, flush=True)
    avg_old_hits = df_res["old_hits"].mean()
    avg_new_hits = df_res["new_hits"].mean()
    avg_old_loss = df_res["old_loss"].mean()
    avg_new_loss = df_res["new_loss"].mean()
    improved_stocks = (df_res["delta_loss"] < 0).sum()
    same_or_improved = (df_res["delta_loss"] <= 0).sum()
    
    print(f"SUMMARY ({len(df_res)} Stocks):", flush=True)
    print(f"Average Hits: OLD {avg_old_hits:.2f}/30 ({avg_old_hits/30*100:.1f}%) -> NEW {avg_new_hits:.2f}/30 ({avg_new_hits/30*100:.1f}%) [Δ {avg_new_hits - avg_old_hits:+.2f} hits]", flush=True)
    print(f"Average Loss: OLD {avg_old_loss:.2f} -> NEW {avg_new_loss:.2f} [Δ {avg_new_loss - avg_old_loss:+.2f} pts]", flush=True)
    print(f"Stocks Strictly Improved: {improved_stocks}/{len(df_res)} ({improved_stocks/len(df_res)*100:.1f}%)", flush=True)
    print(f"Stocks Improved or Equal: {same_or_improved}/{len(df_res)} ({same_or_improved/len(df_res)*100:.1f}%)", flush=True)
    print(f"Total Execution Time: {time.time() - t_start:.1f}s", flush=True)
    
    # Save CSV
    df_res.to_csv("scripts/experiments/bist30_new_vs_old_comparison.csv", index=False)
    print("Saved comparison table to scripts/experiments/bist30_new_vs_old_comparison.csv", flush=True)

if __name__ == "__main__":
    main()
