"""Zero-Lookahead Substitution Experiment: YTD PnL vs FIFO Features.

Tests whether substituting FIFO features with Year-to-Date (YTD) PnL features
improves or degrades predictive performance across the 6 focus equities:
- ASELS (Aselsan)
- GARAN (Garanti BBVA)
- AKBNK (Akbank)
- ISCTR (İş Bankası)
- YKBNK (Yapı Kredi)
- TUPRS (Tüpraş)

Experimental Setups Benchmarked:
1. BASELINE: Production Feature Suite (39 features, including all FIFO metrics)
2. SWAP_PNL: Replace overlapping daily FIFO PnL (feat_mlb_pnl_share_today,
   feat_big5_pnl_share_today, feat_kamu_pnl_share_today) with YTD PnL equivalents.
3. SWAP_ALL_FIFO: Replace all FIFO cost basis spreads, profitability shares,
   and daily PnL with YTD PnL and cumulative flow metrics.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    FEATURE_COLS,
    LOOKBACK_MAP,
    attach_prophet_rolling_features,
    extract_3pillar_time_series,
    run_30d_walk_forward_arena,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ytd_vs_fifo_experiment")

TARGET_SYMBOLS = ["ASELS", "GARAN", "AKBNK", "ISCTR", "YKBNK", "TUPRS"]
CONFIG_PATH = Path("config/tertip_crowned_models.yaml")

# Overlapping FIFO PnL features
OVERLAPPING_PNL_FEATURES = [
    "feat_mlb_pnl_share_today",
    "feat_big5_pnl_share_today",
    "feat_kamu_pnl_share_today",
]

# Broader FIFO Cost & PnL features
ALL_FIFO_COST_PNL_FEATURES = [
    "feat_mlb_cost_spread_pct",
    "feat_big5_cost_spread_pct",
    "feat_kamu_cost_spread_pct",
    "feat_mlb_pnl_share_today",
    "feat_big5_pnl_share_today",
    "feat_kamu_pnl_share_today",
    "feat_mlb_profitable_tertip_share",
    "feat_big5_profitable_tertip_share",
    "feat_kamu_profitable_tertip_share",
]


def add_zero_leakage_ytd_pnl_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute strictly zero-leakage, stationary Year-to-Date PnL features for MLB, BIG5, and KAMU."""
    df = df.copy()
    if "trade_date" not in df.columns:
        return df, []

    df["trade_date_dt"] = pd.to_datetime(df["trade_date"])
    df["year"] = df["trade_date_dt"].dt.year

    # 1. Cumulative Economic YTD PnL within each calendar year
    df["mlb_ytd_pnl_raw"] = df.groupby("year")["mlb_daily_pnl_tl"].cumsum()
    df["big5_ytd_pnl_raw"] = df.groupby("year")["big5_daily_pnl_tl"].cumsum()
    df["kamu_ytd_pnl_raw"] = df.groupby("year")["kamu_daily_pnl_tl"].cumsum()

    # 2. Cumulative Net Flow within each calendar year
    df["mlb_ytd_flow_raw"] = df.groupby("year")["mlb_flow"].cumsum()
    df["big5_ytd_flow_raw"] = df.groupby("year")["big5_flow"].cumsum()
    df["kamu_ytd_flow_raw"] = df.groupby("year")["kamu_flow"].cumsum()

    # 3. Rolling 20-day median turnover as of T-1 (normalization anchor)
    tt_series = df["total_turnover_tl"] if "total_turnover_tl" in df.columns else df["total_turnover"]
    median_tt_20d = tt_series.shift(1).rolling(20, min_periods=5).median().replace(0, np.nan)

    # Shift by 1: Information available strictly as of session T-1 Close
    mlb_ytd_pnl_prev = df["mlb_ytd_pnl_raw"].shift(1)
    big5_ytd_pnl_prev = df["big5_ytd_pnl_raw"].shift(1)
    kamu_ytd_pnl_prev = df["kamu_ytd_pnl_raw"].shift(1)
    mlb_ytd_flow_prev = df["mlb_ytd_flow_raw"].shift(1)

    df["feat_mlb_ytd_pnl_share_20d_to"] = (mlb_ytd_pnl_prev / median_tt_20d * 100.0).fillna(0.0).clip(-500.0, 500.0)
    df["feat_big5_ytd_pnl_share_20d_to"] = (big5_ytd_pnl_prev / median_tt_20d * 100.0).fillna(0.0).clip(-500.0, 500.0)
    df["feat_kamu_ytd_pnl_share_20d_to"] = (kamu_ytd_pnl_prev / median_tt_20d * 100.0).fillna(0.0).clip(-500.0, 500.0)
    df["feat_inst_ytd_pnl_divergence"] = (df["feat_mlb_ytd_pnl_share_20d_to"] - df["feat_big5_ytd_pnl_share_20d_to"]).clip(-500.0, 500.0)
    df["feat_mlb_ytd_flow_share_20d_to"] = (mlb_ytd_flow_prev / median_tt_20d * 100.0).fillna(0.0).clip(-500.0, 500.0)

    ytd_features = [
        "feat_mlb_ytd_pnl_share_20d_to",
        "feat_big5_ytd_pnl_share_20d_to",
        "feat_kamu_ytd_pnl_share_20d_to",
        "feat_inst_ytd_pnl_divergence",
        "feat_mlb_ytd_flow_share_20d_to",
    ]

    return df, ytd_features


def load_crowned_lookback(symbol: str) -> int:
    """Load the crowned training lookback horizon for the symbol from tertip_crowned_models.yaml."""
    if not CONFIG_PATH.exists():
        return 63
    try:
        with open(CONFIG_PATH, "r") as f:
            data = yaml.safe_load(f) or {}
        sym_meta = data.get("models", {}).get(symbol, {})
        horizon = str(sym_meta.get("crowned_horizon") or "3m").lower()
        return LOOKBACK_MAP.get(horizon, 63)
    except Exception:
        return 63


def run_substitution_experiment(eval_sessions: int = 30) -> None:
    """Run walk-forward substitution experiment across TARGET_SYMBOLS."""
    db = PostgresManager()
    print("=" * 115)
    print(f"  MDK TRADING ORACLE — YTD P&L INSTEAD OF FIFO SUBSTITUTION EXPERIMENT ({eval_sessions} SESSIONS)")
    print("=" * 115)
    print(f"Target Equities: {', '.join(TARGET_SYMBOLS)}")
    print("Testing 3 Configurations:")
    print("  1. BASELINE:      Current Production Suite (39 features with full FIFO)")
    print("  2. SWAP_PNL:      Replace overlapping daily FIFO PnL with YTD PnL equivalents")
    print("  3. SWAP_ALL_FIFO: Replace all FIFO cost & PnL features with YTD PnL equivalents")
    print("-" * 115)

    comparison_results = []

    for sym in TARGET_SYMBOLS:
        print(f"\n>>> Running Substitution Experiment for {sym}...")
        raw_df = extract_3pillar_time_series(db, sym)
        if raw_df.empty or len(raw_df) < eval_sessions + 63:
            print(f"    [SKIP] Insufficient history for {sym}: {len(raw_df)} sessions")
            continue

        df_base = attach_prophet_rolling_features(raw_df, sym)
        df_enriched, _ = add_zero_leakage_ytd_pnl_features(df_base)

        train_lb = load_crowned_lookback(sym)
        print(f"    Anchored Training Lookback: {train_lb} sessions | Total Available History: {len(df_enriched)}")

        # 1. BASELINE: Production Feature Suite (39 features)
        _, baseline_summary, _ = run_30d_walk_forward_arena(
            df_enriched.copy(),
            n_sessions=eval_sessions,
            model_type="auto",
            feature_cols=list(FEATURE_COLS),
            train_lookback_sessions=train_lb,
        )

        # 2. SWAP_PNL: Replace overlapping daily FIFO PnL with YTD PnL
        swap_pnl_cols = [c for c in FEATURE_COLS if c not in OVERLAPPING_PNL_FEATURES] + [
            "feat_mlb_ytd_pnl_share_20d_to",
            "feat_big5_ytd_pnl_share_20d_to",
            "feat_kamu_ytd_pnl_share_20d_to",
            "feat_inst_ytd_pnl_divergence",
        ]
        _, swap_pnl_summary, _ = run_30d_walk_forward_arena(
            df_enriched.copy(),
            n_sessions=eval_sessions,
            model_type="auto",
            feature_cols=swap_pnl_cols,
            train_lookback_sessions=train_lb,
        )

        # 3. SWAP_ALL_FIFO: Replace all FIFO cost basis spreads, profitability shares, and daily PnL
        swap_all_cols = [c for c in FEATURE_COLS if c not in ALL_FIFO_COST_PNL_FEATURES] + [
            "feat_mlb_ytd_pnl_share_20d_to",
            "feat_big5_ytd_pnl_share_20d_to",
            "feat_kamu_ytd_pnl_share_20d_to",
            "feat_inst_ytd_pnl_divergence",
            "feat_mlb_ytd_flow_share_20d_to",
        ]
        _, swap_all_summary, _ = run_30d_walk_forward_arena(
            df_enriched.copy(),
            n_sessions=eval_sessions,
            model_type="auto",
            feature_cols=swap_all_cols,
            train_lookback_sessions=train_lb,
        )

        def get_metrics(s: dict[str, Any]) -> tuple[str, float, float, float]:
            hr = s.get("champion_dir_hit_rate_pct", 0.0)
            mae = s.get("champion_mae_pct", 0.0)
            hit_mae = s.get("champion_hit_mae_pct", 0.0)
            miss_mae = s.get("champion_miss_mae_pct", 0.0)
            loss = (100.0 - hr) + 1.0 * hit_mae + 2.5 * miss_mae
            champ = s.get("grand_champion_key", "Auto")
            return champ, hr, mae, loss

        b_champ, b_hr, b_mae, b_loss = get_metrics(baseline_summary)
        pnl_champ, pnl_hr, pnl_mae, pnl_loss = get_metrics(swap_pnl_summary)
        all_champ, all_hr, all_mae, all_loss = get_metrics(swap_all_summary)

        print(f"    1. BASELINE:      {b_champ:<12} | Hit Rate: {b_hr:.1f}% | MAE: {b_mae:.2f}% | Loss: {b_loss:.2f}")
        print(f"    2. SWAP_PNL:      {pnl_champ:<12} | Hit Rate: {pnl_hr:.1f}% | MAE: {pnl_mae:.2f}% | Loss: {pnl_loss:.2f} (Δ HR: {pnl_hr - b_hr:+.1f}%, Δ Loss: {pnl_loss - b_loss:+.2f})")
        print(f"    3. SWAP_ALL_FIFO: {all_champ:<12} | Hit Rate: {all_hr:.1f}% | MAE: {all_mae:.2f}% | Loss: {all_loss:.2f} (Δ HR: {all_hr - b_hr:+.1f}%, Δ Loss: {all_loss - b_loss:+.2f})")

        comparison_results.append({
            "symbol": sym,
            "lookback": train_lb,
            "b_champ": b_champ,
            "b_hr": b_hr,
            "b_loss": b_loss,
            "pnl_champ": pnl_champ,
            "pnl_hr": pnl_hr,
            "pnl_loss": pnl_loss,
            "all_champ": all_champ,
            "all_hr": all_hr,
            "all_loss": all_loss,
        })

    # Summary Table
    print("\n" + "=" * 125)
    print(f"{'SYMBOL':<8} {'LOOKBACK':<9} {'BASELINE (FULL FIFO)':<25} {'SWAP PNL (YTD INSTEAD)':<26} {'SWAP ALL FIFO (YTD ONLY)':<26} {'BEST SETUP':<14}")
    print(f"{'':<8} {'':<9} {'CHAMP / HR% / LOSS':<25} {'CHAMP / HR% / LOSS':<26} {'CHAMP / HR% / LOSS':<26} {''}")
    print("-" * 125)
    for r in comparison_results:
        b_str = f"{r['b_champ']} ({r['b_hr']:.1f}% / {r['b_loss']:.1f})"
        pnl_str = f"{r['pnl_champ']} ({r['pnl_hr']:.1f}% / {r['pnl_loss']:.1f})"
        all_str = f"{r['all_champ']} ({r['all_hr']:.1f}% / {r['all_loss']:.1f})"

        # Determine winner by lowest tournament loss
        losses = [("BASELINE", r["b_loss"]), ("SWAP_PNL", r["pnl_loss"]), ("SWAP_ALL_FIFO", r["all_loss"])]
        best = min(losses, key=lambda x: x[1])[0]

        print(f"{r['symbol']:<8} {str(r['lookback']) + 'd':<9} {b_str:<25} {pnl_str:<26} {all_str:<26} {best:<14}")
    print("=" * 125)


if __name__ == "__main__":
    eval_sess = 30
    if len(sys.argv) > 1:
        try:
            eval_sess = int(sys.argv[1])
        except ValueError:
            pass
    run_substitution_experiment(eval_sessions=eval_sess)
