"""Experiment: Testing XGBoost with different sample weighting schemes and hyperparameters on ASELSAN."""

import numpy as np
import pandas as pd
import xgboost as xgb

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    FEATURE_COLS,
    _check_hit,
    extract_3pillar_time_series,
)


def get_weights(y: np.ndarray, scheme: str) -> np.ndarray:
    abs_y = np.abs(y)
    n = len(y)
    
    if scheme == "unweighted":
        return np.ones(n, dtype=np.float32)
    
    elif scheme == "current_mild":
        # Quiet days (<1%): 0.75; >=1%: 1.0 + 0.5 * min(|y|-1, 3)
        return np.where(abs_y < 1.0, 0.75, 1.0 + 0.5 * np.minimum(abs_y - 1.0, 3.0)).astype(np.float32)
    
    elif scheme == "moderate_booster":
        # Quiet days (<1%): 0.60; 1% to 2%: 1.5; >=2%: 2.5 + 0.5 * min(|y|-2, 3)
        w = np.where(abs_y < 1.0, 0.60, np.where(abs_y < 2.0, 1.5, 2.5 + 0.5 * np.minimum(abs_y - 2.0, 3.0)))
        return w.astype(np.float32)
    
    elif scheme == "aggressive_booster":
        # Quiet days (<1%): 0.40; 1% to 2%: 2.0; >=2%: 3.5 + 0.75 * min(|y|-2, 4)
        w = np.where(abs_y < 1.0, 0.40, np.where(abs_y < 2.0, 2.0, 3.5 + 0.75 * np.minimum(abs_y - 2.0, 4.0)))
        return w.astype(np.float32)
    
    elif scheme == "linear_magnitude":
        # Weight = 0.5 + 0.75 * |y|
        return (0.5 + 0.75 * abs_y).astype(np.float32)
    
    elif scheme == "cubic_threshold":
        # Quiet days: 0.5; >=1%: 1.0 + (|y|-1)^1.2
        return np.where(abs_y < 1.0, 0.50, 1.0 + np.power(np.maximum(abs_y - 1.0, 0.0), 1.2)).astype(np.float32)

    return np.ones(n, dtype=np.float32)


def run_asels_walk_forward(
    df: pd.DataFrame,
    features: list[str],
    scheme: str,
    lookback_sessions: int = 63,
    n_eval: int = 30,
    max_depth: int = 2,
    lr: float = 0.05,
    n_estimators: int = 100,
    reg_alpha: float = 0.1,
    reg_lambda: float = 1.0,
    subsample: float = 0.8,
) -> dict:
    T = len(df)
    results = []
    
    for i in range(n_eval):
        eval_idx = T - n_eval + i
        train_start = max(0, eval_idx - lookback_sessions)
        
        train_df = df.iloc[train_start:eval_idx]
        test_row = df.iloc[eval_idx]
        
        X_train = train_df[features].fillna(0.0).values
        y_train = (train_df["daily_return_pct"].fillna(0.0).values) * 100.0
        
        X_test = test_row[features].fillna(0.0).values.reshape(1, -1)
        actual_ret = float(test_row["daily_return_pct"] or 0.0) * 100.0
        
        weights = get_weights(y_train, scheme)
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            subsample=subsample,
            colsample_bytree=0.8,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=42,
            verbosity=0,
            n_jobs=1,
        )
        model.fit(X_train, y_train, sample_weight=weights)
        pred_ret = float(model.predict(X_test)[0])
        
        is_hit = bool(_check_hit(pred_ret, actual_ret))
        err = abs(pred_ret - actual_ret)
        
        results.append({
            "trade_date": str(test_row["trade_date"]).split(" ")[0],
            "actual_ret": actual_ret,
            "pred_ret": pred_ret,
            "is_hit": is_hit,
            "err": err,
        })
        
    res_df = pd.DataFrame(results)
    
    total = len(res_df)
    hits = res_df["is_hit"].sum()
    hit_rate = (hits / total) * 100.0
    mae = res_df["err"].mean()
    
    hit_mae = res_df[res_df["is_hit"]]["err"].mean() if hits > 0 else 0.0
    miss_mae = res_df[~res_df["is_hit"]]["err"].mean() if (total - hits) > 0 else 0.0
    
    # Significant moves (|actual| >= 1.0%)
    sig_df = res_df[np.abs(res_df["actual_ret"]) >= 1.0]
    sig_total = len(sig_df)
    sig_hits = sig_df["is_hit"].sum()
    sig_rate = (sig_hits / sig_total * 100.0) if sig_total > 0 else 0.0
    
    # Big moves (|actual| >= 2.0%)
    big_df = res_df[np.abs(res_df["actual_ret"]) >= 2.0]
    big_total = len(big_df)
    big_hits = big_df["is_hit"].sum()
    big_rate = (big_hits / big_total * 100.0) if big_total > 0 else 0.0
    
    # Actionable big move call: |pred| >= 1.5% and same direction
    actionable_big_hits = big_df[(np.abs(big_df["pred_ret"]) >= 1.5) & big_df["is_hit"]].shape[0]
    
    # Quiet False Alarms (|pred| >= 1.0% when |actual| < 0.5%)
    q_fa = res_df[(np.abs(res_df["pred_ret"]) >= 1.0) & (np.abs(res_df["actual_ret"]) < 0.5)].shape[0]
    
    # Average predicted magnitude on >=1% days vs <1% days
    avg_pred_sig = np.abs(sig_df["pred_ret"]).mean() if sig_total > 0 else 0.0
    avg_pred_quiet = np.abs(res_df[np.abs(res_df["actual_ret"]) < 1.0]["pred_ret"]).mean()
    
    # Composite tournament loss
    loss = (100.0 - hit_rate) + 1.0 * (100.0 - sig_rate) + 0.5 * (100.0 - big_rate) + (q_fa * 5.0) + 1.0 * hit_mae + 2.5 * miss_mae
    
    return {
        "scheme": scheme,
        "lb": lookback_sessions,
        "depth": max_depth,
        "lr": lr,
        "hits": hits,
        "total": total,
        "hit_rate": hit_rate,
        "sig_hits": sig_hits,
        "sig_total": sig_total,
        "sig_rate": sig_rate,
        "big_hits": big_hits,
        "big_total": big_total,
        "big_rate": big_rate,
        "act_big_hits": actionable_big_hits,
        "q_fa": q_fa,
        "mae": mae,
        "hit_mae": hit_mae,
        "miss_mae": miss_mae,
        "avg_pred_sig": avg_pred_sig,
        "avg_pred_quiet": avg_pred_quiet,
        "loss": loss,
    }


def main():
    db = PostgresManager()
    df = extract_3pillar_time_series(db, "ASELS")
    features = [c for c in FEATURE_COLS if c in df.columns]
    
    print(f"Running experiments on ASELS (Rows: {len(df)}, Features: {len(features)})")
    print("=" * 135)
    print(f"{'Scheme':<20} | {'LB':<4} | {'D':<2} | {'LR':<4} | {'Hits':<6} | {'Sig≥1%':<9} | {'Big≥2%':<9} | {'ActBig':<6} | {'Q-FA':<4} | {'MAE%':<5} | {'AvgPredSig':<10} | {'Loss':<6}")
    print("-" * 135)
    
    schemes = ["unweighted", "current_mild", "moderate_booster", "aggressive_booster", "linear_magnitude", "cubic_threshold"]
    rows = []
    
    # 1. Compare weighting schemes across horizons
    for lb in [63, 126, 252]:
        for sch in schemes:
            r = run_asels_walk_forward(df, features, scheme=sch, lookback_sessions=lb, max_depth=2, lr=0.05)
            rows.append(r)
            print(f"{r['scheme']:<20} | {r['lb']:<4} | {r['depth']:<2} | {r['lr']:<4} | {r['hits']}/{r['total']} ({r['hit_rate']:.0f}%) | {r['sig_hits']}/{r['sig_total']} ({r['sig_rate']:.0f}%) | {r['big_hits']}/{r['big_total']} ({r['big_rate']:.0f}%) | {r['act_big_hits']}/{r['big_total']} | {r['q_fa']:<4} | {r['mae']:.2f}% | {r['avg_pred_sig']:.2f}%     | {r['loss']:.1f}")
        print("-" * 135)
        
    # 2. Hyperparameter sweep on best weighting scheme
    print("\nHYPERPARAMETER TUNING (Depth, LR, Subsample):")
    print("-" * 135)
    for depth in [2, 3, 4]:
        for lr in [0.03, 0.05, 0.08]:
            for lb in [63, 126]:
                r = run_asels_walk_forward(df, features, scheme="moderate_booster", lookback_sessions=lb, max_depth=depth, lr=lr)
                rows.append(r)
                print(f"{'mod_boost':<20} | {r['lb']:<4} | {r['depth']:<2} | {r['lr']:<4} | {r['hits']}/{r['total']} ({r['hit_rate']:.0f}%) | {r['sig_hits']}/{r['sig_total']} ({r['sig_rate']:.0f}%) | {r['big_hits']}/{r['big_total']} ({r['big_rate']:.0f}%) | {r['act_big_hits']}/{r['big_total']} | {r['q_fa']:<4} | {r['mae']:.2f}% | {r['avg_pred_sig']:.2f}%     | {r['loss']:.1f}")

    df_out = pd.DataFrame(rows)
    best = df_out.sort_values("loss").iloc[0]
    print("=" * 135)
    print("CHAMPION FOR ASELSAN:")
    print(f"Scheme: {best['scheme']}, Lookback: {best['lb']}d, Depth: {best['depth']}, LR: {best['lr']}")
    print(f"Hits: {best['hits']}/{best['total']} ({best['hit_rate']:.1f}%), Sig≥1%: {best['sig_hits']}/{best['sig_total']} ({best['sig_rate']:.1f}%), Big≥2%: {best['big_hits']}/{best['big_total']} ({best['big_rate']:.1f}%), Actionable Big Hits: {best['act_big_hits']}/{best['big_total']}, Loss: {best['loss']:.2f}")


if __name__ == "__main__":
    main()
