"""Experiment: Test Institutional Flow Divergence & Momentum Rollover on 10D Multi-Horizon Forecaster.

Hypothesis:
At cyclical tops (e.g. ASELS topping at 404 TL before cascading to 336 TL), flat rolling 10-day
features carry stale accumulation memory. Adding:
1. BofA flow divergence (3d vs 10d): feat_mlb_flow_divergence_3d_10d
2. Institutional flow divergence (5d vs 10d): feat_inst_flow_divergence_5d_10d
3. Momentum rollover delta (3d vs 10d): feat_ret_rollover_3d_10d
4. Exponentially-decayed flow share (EWMA 5d): feat_mlb_ewma_flow_share_5d
will immediately detect when institutions flip to distribution, breaking the 10-day lag and
significantly improving directional hit rate on ASELS and AKBNK.
"""

from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import extract_3pillar_time_series
from mdk_trading_oracle.data.silver.tertip_multi_horizon_forecaster import (
    engineer_multi_horizon_features,
    evaluate_horizon_walk_forward,
)

CANDIDATES = ["LightGBM", "XGBoost", "BayesianRidge", "Huber", "Ridge"]
LOOKBACKS = [("3m", 63), ("6m", 126), ("12m", 252), ("18m", 378), ("24m", 504)]


def enrich_divergence_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Enrich with divergence and momentum rollover features."""
    df, f_cols = engineer_multi_horizon_features(df)
    df = df.copy()

    turnover_10d = df["total_turnover_tl"].rolling(10, min_periods=1).sum().replace(0, np.nan)
    mlb_10d_flow = df["mlb_flow"].rolling(10, min_periods=1).sum()
    big5_10d_flow = df["big5_flow"].rolling(10, min_periods=1).sum()
    inst_10d_flow = mlb_10d_flow + big5_10d_flow
    inst_10d_share = (inst_10d_flow / turnover_10d * 100.0).fillna(0.0)

    # 1. BofA Flow Divergence (3d vs 10d)
    df["feat_mlb_flow_divergence_3d_10d"] = (df["feat_mlb_flow_3d_share"] - df["feat_mlb_flow_10d_share"]).clip(-100.0, 100.0).fillna(0.0)

    # 2. Total Institutional Divergence (5d vs 10d)
    df["feat_inst_flow_divergence_5d_10d"] = (df["feat_total_inst_5d_share"] - inst_10d_share).clip(-100.0, 100.0).fillna(0.0)

    # 3. Momentum Rollover Delta (3d vs 10d)
    df["feat_ret_rollover_3d_10d"] = (df["feat_close_ret_3d"] - df["feat_close_ret_10d"]).clip(-50.0, 50.0).fillna(0.0)

    # 4. Exponentially-Decayed Flow Share (EWMA 5d)
    ewma_to = df["total_turnover_tl"].ewm(span=5, min_periods=1).mean().replace(0, np.nan)
    ewma_mlb = df["mlb_flow"].ewm(span=5, min_periods=1).mean()
    df["feat_mlb_ewma_flow_share_5d"] = (ewma_mlb / ewma_to * 100.0).clip(-100.0, 100.0).fillna(0.0)

    new_f_cols = [c for c in df.columns if c.startswith("feat_")]
    return df, new_f_cols


def run_benchmark(db: PostgresManager, symbol: str, use_divergence: bool = False) -> dict:
    df_raw = extract_3pillar_time_series(db, symbol, lookback_days=1100)
    if use_divergence:
        df_feat, f_cols = enrich_divergence_features(df_raw)
    else:
        df_feat, f_cols = engineer_multi_horizon_features(df_raw)

    best_loss = 999.0
    best_hit_rate = 0.0
    best_cfg = None

    for m in CANDIDATES:
        for lb_name, lb_s in LOOKBACKS:
            res = evaluate_horizon_walk_forward(
                df=df_feat,
                horizon_key="10d",
                model_name=m,
                train_lookback_sessions=lb_s,
                n_eval_sessions=60,
                feature_cols=f_cols,
            )
            loss = res.get("loss", 999.0)
            hit_rate = res.get("hit_rate_pct", 0.0)
            if loss < best_loss or (abs(loss - best_loss) < 0.05 and hit_rate > best_hit_rate):
                best_loss = loss
                best_hit_rate = hit_rate
                best_cfg = {
                    "model": m,
                    "lookback": lb_name,
                    "sessions": lb_s,
                    "hit_rate_pct": hit_rate,
                    "hits": res.get("hits", 0),
                    "total": res.get("total_evals", 60),
                    "mae_pct": res.get("mae_pct", 0.0),
                    "loss": loss,
                }
    return best_cfg


def main():
    db = PostgresManager()
    
    print("=" * 80)
    print("MULTI-HORIZON 10D EXPERIMENT: FLOW DIVERGENCE & MOMENTUM ROLLOVER")
    print("=" * 80)

    for sym in ["ASELS", "AKBNK"]:
        print(f"\n--- Testing {sym} 10-Day Horizon ---")
        
        # 1. Baseline Run
        base_cfg = run_benchmark(db, sym, use_divergence=False)
        print(f"BASELINE  [{sym} 10d]: Champion = {base_cfg['model']} ({base_cfg['lookback']}) | "
              f"Win Rate: {base_cfg['hit_rate_pct']:.1f}% ({base_cfg['hits']}/{base_cfg['total']}) | "
              f"MAE: {base_cfg['mae_pct']:.2f}% | Loss: {base_cfg['loss']:.2f}")

        # 2. Enhanced Divergence Run
        enh_cfg = run_benchmark(db, sym, use_divergence=True)
        delta_win = enh_cfg['hit_rate_pct'] - base_cfg['hit_rate_pct']
        delta_loss = enh_cfg['loss'] - base_cfg['loss']
        print(f"ENHANCED  [{sym} 10d]: Champion = {enh_cfg['model']} ({enh_cfg['lookback']}) | "
              f"Win Rate: {enh_cfg['hit_rate_pct']:.1f}% ({enh_cfg['hits']}/{enh_cfg['total']}) | "
              f"MAE: {enh_cfg['mae_pct']:.2f}% | Loss: {enh_cfg['loss']:.2f}")
        print(f"DELTA     [{sym} 10d]: Win Rate: {delta_win:+.1f}% | Loss: {delta_loss:+.2f}")


if __name__ == "__main__":
    main()
