"""Experiment: Test Recency-Weighted Tournament Crowning & Lookback Capping.

Hypothesis:
Equally-weighted 60-session tournament loss allows models like LightGBM (24M)
to win crowns based on stale market regimes from 2-3 months ago, even while
failing recent sessions (e.g. YKBNK 10D getting only 5/15 hits in September).

We test:
1. Baseline: Unweighted 60-session tournament loss.
2. Recency-Weighted Loss: Linear evaluation weights (0.5x on day 1 up to 1.5x on day 60).
3. Lookback Capped (Max 12M / 252 sessions on 10D).
4. Combined Recency-Weighted + Lookback Capped.

Tested on YKBNK, ASELS, AKBNK, and ISCTR on 10D (and 5D / 15D).
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

CANDIDATES = ["BayesianRidge", "Huber", "Ridge", "XGBoost", "LightGBM"]
LOOKBACKS = [("3m", 63), ("6m", 126), ("12m", 252), ("18m", 378), ("24m", 504)]
SYMBOLS = ["YKBNK", "ASELS", "AKBNK", "ISCTR"]
HORIZON = "10d"


def compute_metrics_from_ledger(ledger: list[dict], recency_weighted: bool = False) -> dict:
    if not ledger:
        return {"hit_rate_pct": 0.0, "mae_pct": 999.0, "loss": 999.0, "recent_hit_rate_pct": 0.0}

    n = len(ledger)
    hits = np.array([1.0 if row.get("is_hit") else 0.0 for row in ledger])
    errors = np.array([float(row.get("err_pct", 0.0)) for row in ledger])

    # Recent 20 sessions metrics
    recent_20 = ledger[-20:] if n >= 20 else ledger
    recent_hits = sum(1 for r in recent_20 if r.get("is_hit"))
    recent_hit_rate_pct = recent_hits / len(recent_20) * 100.0

    if recency_weighted and n > 1:
        weights = np.linspace(0.5, 1.5, n)
        weights /= weights.mean()  # normalize so mean weight = 1.0
    else:
        weights = np.ones(n)

    weighted_hit_rate = float(np.sum(hits * weights) / np.sum(weights) * 100.0)
    raw_hit_rate = float(np.mean(hits) * 100.0)

    # MAEs
    hit_mask = hits == 1.0
    miss_mask = hits == 0.0

    mae = float(np.average(errors, weights=weights))
    h_mae = float(np.average(errors[hit_mask], weights=weights[hit_mask])) if np.any(hit_mask) else mae
    m_mae = float(np.average(errors[miss_mask], weights=weights[miss_mask])) if np.any(miss_mask) else mae

    loss = float((100.0 - weighted_hit_rate) + 1.0 * h_mae + 2.5 * m_mae)

    return {
        "hit_rate_pct": raw_hit_rate,
        "weighted_hit_rate_pct": weighted_hit_rate,
        "recent_hit_rate_pct": recent_hit_rate_pct,
        "recent_hits": recent_hits,
        "recent_total": len(recent_20),
        "mae_pct": mae,
        "hit_mae_pct": h_mae,
        "miss_mae_pct": m_mae,
        "loss": loss,
    }


def run_experiment_for_symbol(db: PostgresManager, symbol: str) -> dict:
    df_raw = extract_3pillar_time_series(db, symbol, lookback_days=1100)
    df_feat, _ = engineer_multi_horizon_features(df_raw)

    eval_cache: dict[tuple[str, str, int], list[dict]] = {}

    for m in CANDIDATES:
        for lb_name, lb_s in LOOKBACKS:
            res = evaluate_horizon_walk_forward(
                df=df_feat,
                horizon_key=HORIZON,
                model_name=m,
                train_lookback_sessions=lb_s,
                n_eval_sessions=60,
            )
            eval_cache[(m, lb_name, lb_s)] = res.get("ledger", [])

    def find_champion(recency: bool, max_lookback_sessions: int | None = None) -> dict:
        best_loss = 999.0
        best_cfg = None

        for (m, lb_name, lb_s), ledger in eval_cache.items():
            if max_lookback_sessions and lb_s > max_lookback_sessions:
                continue

            metrics = compute_metrics_from_ledger(ledger, recency_weighted=recency)
            if metrics["loss"] < best_loss:
                best_loss = metrics["loss"]
                best_cfg = {
                    "model": m,
                    "lookback": lb_name,
                    "sessions": lb_s,
                    **metrics,
                }
        return best_cfg

    # 1. Baseline: Unweighted, All Lookbacks
    cfg_base = find_champion(recency=False, max_lookback_sessions=None)

    # 2. Recency-Weighted: All Lookbacks
    cfg_recency = find_champion(recency=True, max_lookback_sessions=None)

    # 3. Lookback Capped (Max 12M / 252 sessions), Unweighted
    cfg_capped = find_champion(recency=False, max_lookback_sessions=252)

    # 4. Combined: Recency-Weighted + Capped (Max 12M)
    cfg_combined = find_champion(recency=True, max_lookback_sessions=252)

    return {
        "symbol": symbol,
        "base": cfg_base,
        "recency": cfg_recency,
        "capped": cfg_capped,
        "combined": cfg_combined,
    }


def main():
    db = PostgresManager()

    print("=" * 96)
    print("EXPERIMENT: RECENCY-WEIGHTED CROWNING & LOOKBACK CAPPING (10D SWING)")
    print("Testing YKBNK, ASELS, AKBNK, ISCTR")
    print("=" * 96)

    results = []

    for sym in SYMBOLS:
        print(f"\nEvaluating {sym}...")
        res = run_experiment_for_symbol(db, sym)
        b = res["base"]
        r = res["recency"]
        c = res["capped"]
        comb = res["combined"]

        print(f"\n--- {sym} [10D] Results ---")
        print(f"1. BASELINE (Equal Weights)        : {b['model']} ({b['lookback']}) | "
              f"Full Win Rate: {b['hit_rate_pct']:.1f}% | Recent 20 Win Rate: {b['recent_hit_rate_pct']:.1f}% ({b['recent_hits']}/{b['recent_total']}) | Loss: {b['loss']:.2f}")
        print(f"2. RECENCY-WEIGHTED (Linear 0.5-1.5): {r['model']} ({r['lookback']}) | "
              f"Full Win Rate: {r['hit_rate_pct']:.1f}% | Recent 20 Win Rate: {r['recent_hit_rate_pct']:.1f}% ({r['recent_hits']}/{r['recent_total']}) | Loss: {r['loss']:.2f}")
        print(f"3. LOOKBACK CAPPED (Max 12M)       : {c['model']} ({c['lookback']}) | "
              f"Full Win Rate: {c['hit_rate_pct']:.1f}% | Recent 20 Win Rate: {c['recent_hit_rate_pct']:.1f}% ({c['recent_hits']}/{c['recent_total']}) | Loss: {c['loss']:.2f}")
        print(f"4. COMBINED (Recency + Max 12M)    : {comb['model']} ({comb['lookback']}) | "
              f"Full Win Rate: {comb['hit_rate_pct']:.1f}% | Recent 20 Win Rate: {comb['recent_hit_rate_pct']:.1f}% ({comb['recent_hits']}/{comb['recent_total']}) | Loss: {comb['loss']:.2f}")

        results.append({
            "symbol": sym,
            "base_champ": f"{b['model']} ({b['lookback']})",
            "base_full_win": f"{b['hit_rate_pct']:.1f}%",
            "base_recent_win": f"{b['recent_hit_rate_pct']:.1f}%",
            "comb_champ": f"{comb['model']} ({comb['lookback']})",
            "comb_full_win": f"{comb['hit_rate_pct']:.1f}%",
            "comb_recent_win": f"{comb['recent_hit_rate_pct']:.1f}%",
            "recent_delta": f"{(comb['recent_hit_rate_pct'] - b['recent_hit_rate_pct']):+.1f}%",
        })

    print("\n" + "=" * 96)
    print("FINAL SUMMARY SCORECARD (BASELINE vs COMBINED RECENCY + 12M CAP)")
    print("=" * 96)
    df_sum = pd.DataFrame(results)
    print(df_sum.to_string(index=False))


if __name__ == "__main__":
    main()
