"""Test sample weighting schemes on ASELSAN with exact zero-leakage walk-forward arena."""

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    FEATURE_COLS,
    _check_hit,
    extract_3pillar_time_series,
)


def evaluate_asels_weights(df: pd.DataFrame, f_cols: list[str], train_lookback_sessions: int, weight_fn_name: str, n_eval: int = 30):
    T = len(df)
    records = []
    
    for i in range(n_eval):
        eval_idx = T - n_eval + i
        eval_row = df.iloc[eval_idx]
        actual_price = float(eval_row["close_price"])
        prev_row = df.iloc[eval_idx - 1]
        prev_price = float(prev_row["close_price"])
        actual_ret = (actual_price - prev_price) / prev_price * 100.0
        
        train_data = df.iloc[:eval_idx]
        tr_clean = train_data.iloc[-train_lookback_sessions:].dropna(subset=f_cols).copy()
        y_tr = (tr_clean["close_price"].shift(-1) - tr_clean["close_price"]) / tr_clean["close_price"] * 100.0
        X_tr = tr_clean[f_cols].iloc[:-1]
        y_tr = y_tr.iloc[:-1]
        X_prev = pd.DataFrame([prev_row[f_cols]])
        
        abs_y = y_tr.abs()
        
        if weight_fn_name == "unweighted":
            weights = np.ones(len(y_tr), dtype=np.float32)
        elif weight_fn_name == "current_mild":
            weights = np.where(abs_y < 1.0, 0.75, 1.0 + 0.5 * (abs_y - 1.0).clip(upper=3.0))
        elif weight_fn_name == "moderate_booster":
            weights = np.where(abs_y < 1.0, 0.60, np.where(abs_y < 2.0, 1.5, 2.5 + 0.5 * (abs_y - 2.0).clip(upper=3.0)))
        elif weight_fn_name == "aggressive_booster":
            weights = np.where(abs_y < 1.0, 0.40, np.where(abs_y < 2.0, 2.0, 3.5 + 0.75 * (abs_y - 2.0).clip(upper=4.0)))
        elif weight_fn_name == "power_booster":
            weights = np.where(abs_y < 1.0, 0.50, 1.0 + 0.8 * np.power(np.maximum(abs_y - 1.0, 0.0), 1.3))
        elif weight_fn_name == "big_move_2x":
            # Weight is 1.0 on normal, but 2.5 on >=1% and 4.0 on >=2%
            weights = np.where(abs_y < 1.0, 0.70, np.where(abs_y < 2.0, 2.5, 4.0))
        elif weight_fn_name == "linear_continuous":
            weights = 0.5 + 0.5 * abs_y
        else:
            weights = np.ones(len(y_tr), dtype=np.float32)

        xgb = XGBRegressor(
            n_estimators=60,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
            verbosity=0,
        )
        xgb.fit(X_tr, y_tr, sample_weight=weights)
        pred_ret = float(xgb.predict(X_prev)[0])
        pred_ret = max(-10.0, min(10.0, pred_ret))
        
        hit = _check_hit(pred_ret, actual_ret)
        err = abs(pred_ret - actual_ret)
        
        records.append({
            "date": str(eval_row["trade_date"]).split(" ")[0],
            "actual_ret": actual_ret,
            "pred_ret": pred_ret,
            "hit": hit,
            "err": err,
        })
        
    rdf = pd.DataFrame(records)
    total = len(rdf)
    hits = rdf["hit"].sum()
    hit_rate = (hits / total) * 100.0
    mae = rdf["err"].mean()
    
    hit_mae = rdf[rdf["hit"]]["err"].mean() if hits > 0 else 0.0
    miss_mae = rdf[~rdf["hit"]]["err"].mean() if (total - hits) > 0 else 0.0
    
    # Significant moves (>= 1.0%)
    sig_df = rdf[np.abs(rdf["actual_ret"]) >= 1.0]
    sig_total = len(sig_df)
    sig_hits = sig_df["hit"].sum()
    sig_rate = (sig_hits / sig_total * 100.0) if sig_total > 0 else 0.0
    
    # Big moves (>= 2.0%)
    big_df = rdf[np.abs(rdf["actual_ret"]) >= 2.0]
    big_total = len(big_df)
    big_hits = big_df["hit"].sum()
    big_rate = (big_hits / big_total * 100.0) if big_total > 0 else 0.0
    
    # Actionable big move calls (|pred| >= 1.5% and hit)
    act_big_hits = big_df[(np.abs(big_df["pred_ret"]) >= 1.5) & big_df["hit"]].shape[0]
    
    # Quiet False Alarms (|pred| >= 1.0% when |actual| < 0.5%)
    q_fa = rdf[(np.abs(rdf["pred_ret"]) >= 1.0) & (np.abs(rdf["actual_ret"]) < 0.5)].shape[0]
    
    # Magnitudes
    avg_pred_sig = np.abs(sig_df["pred_ret"]).mean() if sig_total > 0 else 0.0
    avg_pred_quiet = np.abs(rdf[np.abs(rdf["actual_ret"]) < 1.0]["pred_ret"]).mean()
    
    loss = (100.0 - hit_rate) + 1.0 * (100.0 - sig_rate) + 0.5 * (100.0 - big_rate) + (q_fa * 5.0) + 1.0 * hit_mae + 2.5 * miss_mae
    
    return {
        "scheme": weight_fn_name,
        "lb": train_lookback_sessions,
        "hits": hits,
        "total": total,
        "hit_rate": hit_rate,
        "sig_hits": sig_hits,
        "sig_total": sig_total,
        "sig_rate": sig_rate,
        "big_hits": big_hits,
        "big_total": big_total,
        "big_rate": big_rate,
        "act_big": act_big_hits,
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
    f_cols = [c for c in FEATURE_COLS if c in df.columns]
    
    schemes = [
        "unweighted",
        "current_mild",
        "moderate_booster",
        "aggressive_booster",
        "power_booster",
        "big_move_2x",
        "linear_continuous",
    ]
    
    print("=" * 135)
    print(f"{'Weight Scheme':<20} | {'LB':<4} | {'Hits':<8} | {'Sig≥1%':<10} | {'Big≥2%':<10} | {'ActBig':<6} | {'Q-FA':<4} | {'MAE%':<5} | {'AvgPredSig':<10} | {'Loss':<6}")
    print("-" * 135)
    
    all_res = []
    for lb in [63, 126, 252]:
        for sch in schemes:
            r = evaluate_asels_weights(df, f_cols, lb, sch)
            all_res.append(r)
            print(f"{r['scheme']:<20} | {r['lb']:<4} | {r['hits']}/{r['total']} ({r['hit_rate']:.0f}%) | {r['sig_hits']}/{r['sig_total']} ({r['sig_rate']:.0f}%) | {r['big_hits']}/{r['big_total']} ({r['big_rate']:.0f}%) | {r['act_big']}/{r['big_total']} | {r['q_fa']:<4} | {r['mae']:.2f}% | {r['avg_pred_sig']:.2f}%     | {r['loss']:.1f}")
        print("-" * 135)
        
    df_res = pd.DataFrame(all_res)
    best = df_res.sort_values("loss").iloc[0]
    print("\nTOP CHAMPION FOR ASELSAN:")
    print(f"Scheme: {best['scheme']}, Horizon: {best['lb']}d ({'3M' if best['lb']==63 else ('6M' if best['lb']==126 else '12M')})")
    print(f"Hits: {best['hits']}/{best['total']} ({best['hit_rate']:.1f}%), Sig≥1%: {best['sig_hits']}/{best['sig_total']} ({best['sig_rate']:.1f}%), Big≥2%: {best['big_hits']}/{best['big_total']} ({best['big_rate']:.1f}%)")
    print(f"Actionable Big Move Calls: {best['act_big']}/{best['big_total']}, Quiet False Alarms: {best['q_fa']}, MAE: {best['mae']:.2f}%, Loss: {best['loss']:.2f}")


if __name__ == "__main__":
    main()
