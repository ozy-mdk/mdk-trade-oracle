"""Zero-Lookahead YTD PnL Feature Ablation Tournament.

Benchmarks 6 key institutional equities:
- ASELS (Aselsan)
- GARAN (Garanti BBVA)
- AKBNK (Akbank)
- ISCTR (İş Bankası)
- YKBNK (Yapı Kredi)
- TUPRS (Tüpraş)

Compares:
- Baseline: Existing 39 Microstructure Feature Suite
- Challenger: Baseline + Zero-Leakage Normalized YTD PnL Features (BofA, BIG5, KAMU)

Evaluation Criteria:
1. Directional Hit Rate % (±0.25% market consolidation deadband)
2. Hit MAE % & Miss MAE %
3. Composite Tournament Loss: (100 - hit_rate) + 1.0 * hit_mae + 2.5 * miss_mae
4. Significant Move Accuracy % (|Return| >= 1.0%)
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
logger = logging.getLogger("ytd_ablation")

TARGET_SYMBOLS = ["ASELS", "GARAN", "AKBNK", "ISCTR", "YKBNK", "TUPRS"]
CONFIG_PATH = Path("config/tertip_crowned_models.yaml")


def add_zero_leakage_ytd_pnl_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Compute strictly zero-leakage, stationary Year-to-Date PnL features for MLB, BIG5, and KAMU.
    
    Features use strictly data up to T-1 Close (via shift(1)) and are normalized
    by rolling 20-day median turnover to achieve scale invariance across time and stocks.
    """
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

    # 4. Zero-Leakage (T-1) Normalized Features
    new_features = [
        "feat_mlb_ytd_pnl_share_20d_to",
        "feat_big5_ytd_pnl_share_20d_to",
        "feat_kamu_ytd_pnl_share_20d_to",
        "feat_inst_ytd_pnl_divergence",
        "feat_mlb_ytd_flow_share_20d_to",
    ]

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

    return df, new_features


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


def run_ablation_experiment(eval_sessions: int = 60) -> None:
    """Run walk-forward ablation tournament across TARGET_SYMBOLS."""
    db = PostgresManager()
    print("=" * 105)
    print(f"  MDK TRADING ORACLE — ZERO-LEAKAGE YTD P&L ABLATION TOURNAMENT ({eval_sessions} SESSIONS)")
    print("=" * 105)
    print(f"Target Equities: {', '.join(TARGET_SYMBOLS)}")
    print(f"Baseline Feature Count: {len(FEATURE_COLS)}")
    print("-" * 105)

    comparison_results = []

    for sym in TARGET_SYMBOLS:
        print(f"\n>>> Running Ablation for {sym}...")
        raw_df = extract_3pillar_time_series(db, sym)
        if raw_df.empty or len(raw_df) < eval_sessions + 63:
            print(f"    [SKIP] Insufficient history for {sym}: {len(raw_df)} sessions")
            continue

        # Attach Prophet baseline rolling forecasts
        df_base = attach_prophet_rolling_features(raw_df, sym)

        # Enrich with YTD PnL features
        df_enriched, ytd_features = add_zero_leakage_ytd_pnl_features(df_base)

        train_lb = load_crowned_lookback(sym)
        print(f"    Anchored Training Lookback: {train_lb} sessions | Total Available History: {len(df_enriched)}")

        # 1. Run Baseline (without YTD PnL features)
        _, baseline_summary, _ = run_30d_walk_forward_arena(
            df_enriched.copy(),
            n_sessions=eval_sessions,
            model_type="auto",
            feature_cols=list(FEATURE_COLS),
            train_lookback_sessions=train_lb,
        )

        # 2. Run Challenger (with YTD PnL features)
        challenger_features = list(FEATURE_COLS) + ytd_features
        _, challenger_summary, _ = run_30d_walk_forward_arena(
            df_enriched.copy(),
            n_sessions=eval_sessions,
            model_type="auto",
            feature_cols=challenger_features,
            train_lookback_sessions=train_lb,
        )

        base_hits = baseline_summary.get("champion_dir_hits", 0)
        base_hr = baseline_summary.get("champion_dir_hit_rate_pct", 0.0)
        base_mae = baseline_summary.get("champion_mae_pct", 0.0)
        base_hit_mae = baseline_summary.get("champion_hit_mae_pct", 0.0)
        base_miss_mae = baseline_summary.get("champion_miss_mae_pct", 0.0)
        base_loss = (100.0 - base_hr) + 1.0 * base_hit_mae + 2.5 * base_miss_mae
        base_champ = baseline_summary.get("grand_champion_key", "Auto")

        chal_hits = challenger_summary.get("champion_dir_hits", 0)
        chal_hr = challenger_summary.get("champion_dir_hit_rate_pct", 0.0)
        chal_mae = challenger_summary.get("champion_mae_pct", 0.0)
        chal_hit_mae = challenger_summary.get("champion_hit_mae_pct", 0.0)
        chal_miss_mae = challenger_summary.get("champion_miss_mae_pct", 0.0)
        chal_loss = (100.0 - chal_hr) + 1.0 * chal_hit_mae + 2.5 * chal_miss_mae
        chal_champ = challenger_summary.get("grand_champion_key", "Auto")

        hr_delta = chal_hr - base_hr
        loss_delta = chal_loss - base_loss

        outcome = "BETTER" if loss_delta < -0.5 else ("WORSE" if loss_delta > 0.5 else "PARITY")

        print(f"    Baseline ({base_champ}):   Hit Rate: {base_hr:.1f}% ({base_hits}/{eval_sessions}) | MAE: {base_mae:.2f}% | Loss: {base_loss:.2f}")
        print(f"    Challenger ({chal_champ}): Hit Rate: {chal_hr:.1f}% ({chal_hits}/{eval_sessions}) | MAE: {chal_mae:.2f}% | Loss: {chal_loss:.2f}")
        print(f"    Outcome: {outcome} (Δ Hit Rate: {hr_delta:+.1f}%, Δ Tournament Loss: {loss_delta:+.2f})")

        comparison_results.append({
            "symbol": sym,
            "lookback": train_lb,
            "base_champ": base_champ,
            "base_hr": base_hr,
            "base_mae": base_mae,
            "base_loss": base_loss,
            "chal_champ": chal_champ,
            "chal_hr": chal_hr,
            "chal_mae": chal_mae,
            "chal_loss": chal_loss,
            "hr_delta": hr_delta,
            "loss_delta": loss_delta,
            "outcome": outcome,
        })

    # Summary Table
    print("\n" + "=" * 115)
    print(f"{'SYMBOL':<8} {'LOOKBACK':<10} {'BASE CHAMP':<14} {'BASE HR':<10} {'BASE LOSS':<12} {'CHAL CHAMP':<14} {'CHAL HR':<10} {'CHAL LOSS':<12} {'OUTCOME':<10}")
    print("-" * 115)
    for r in comparison_results:
        print(
            f"{r['symbol']:<8} "
            f"{str(r['lookback']) + 'd':<10} "
            f"{r['base_champ']:<14} "
            f"{r['base_hr']:.1f}%{'':<5} "
            f"{r['base_loss']:.2f}{'':<6} "
            f"{r['chal_champ']:<14} "
            f"{r['chal_hr']:.1f}%{'':<5} "
            f"{r['chal_loss']:.2f}{'':<6} "
            f"{r['outcome']:<10}"
        )
    print("=" * 115)


if __name__ == "__main__":
    eval_sess = 60
    if len(sys.argv) > 1:
        try:
            eval_sess = int(sys.argv[1])
        except ValueError:
            pass
    run_ablation_experiment(eval_sessions=eval_sess)
