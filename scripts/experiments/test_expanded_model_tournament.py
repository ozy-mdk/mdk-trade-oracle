"""Expanded Candidate Model Tournament: 180-Day Out-of-Sample Walk-Forward Arena.

Evaluates 8 diverse, mathematically sound model paradigms for institutional flow prediction:
1. Ridge (L2 Linear Regularization)
2. XGBoost (Shallow Gradient Boosted Trees)
3. LightGBM (Histogram-based Gradient Boosting)
4. Huber Regressor (Outlier-Immune Robust Loss)
5. Epsilon-SVR (RBF Kernel with 25 bps epsilon-insensitive deadband tube)
6. Linear-SVR (Linear Support Vector Regression with 25 bps epsilon-tube)
7. Bayesian Ridge (Analytical Evidence Maximization / Empirical Bayes)
8. Blended Ensemble (Shrinkage average of Ridge + Huber + XGBoost)
9. Prophet Baseline (Univariate Price Momentum Prior)

Zero data leakage protocol. Evaluated on calibrated ±0.25% (25 bps) deadband.
"""

from __future__ import annotations

import logging
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge, ElasticNet, HuberRegressor, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from mdk_trading_oracle.core.db import PostgresManager
from mdk_trading_oracle.data.silver.tertip_ml_forecaster import (
    DEADBAND_PCT,
    FEATURE_COLS,
    attach_prophet_rolling_features,
    extract_3pillar_time_series,
)

warnings.filterwarnings("ignore")
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("lightgbm").setLevel(logging.ERROR)

CACHE_DIR = Path.home() / "data" / "mdk_oracle" / "cache" / "expanded_models_180d"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _check_hit(pred: float, actual: float, deadband: float = DEADBAND_PCT) -> bool:
    """Directional hit check against calibrated deadband."""
    if abs(actual) <= deadband:
        return (abs(pred) <= deadband) or ((pred * actual) >= 0.0)
    elif actual > deadband:
        return pred > 0.0
    else:
        return pred < 0.0


def build_candidate_models() -> dict[str, Any]:
    """Instantiate the candidate model registry."""
    return {
        "Ridge": Ridge(alpha=10.0, random_state=42),
        "XGBoost": XGBRegressor(
            n_estimators=45,
            max_depth=2,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=1,
        ),
        "LightGBM": lgb.LGBMRegressor(
            n_estimators=45,
            max_depth=2,
            num_leaves=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1,
            n_jobs=1,
        ),
        "Huber": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", HuberRegressor(epsilon=1.35, alpha=10.0, max_iter=300)),
        ]),
        "BayesianRidge": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", BayesianRidge(max_iter=300)),
        ]),
        "Epsilon_SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(kernel="rbf", C=1.0, epsilon=DEADBAND_PCT)),
        ]),
        "Linear_SVR": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", SVR(kernel="linear", C=0.5, epsilon=DEADBAND_PCT)),
        ]),
        "ElasticNet": Pipeline([
            ("scaler", StandardScaler()),
            ("reg", ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42)),
        ]),
    }


def evaluate_symbol_models(
    df: pd.DataFrame,
    symbol: str,
    lookback_sessions: int = 252,
    n_eval_sessions: int = 180,
    feature_cols: list[str] | None = None,
) -> dict[str, Any]:
    """Execute 180-day walk-forward arena evaluating all candidate models."""
    f_cols = feature_cols if feature_cols is not None else list(FEATURE_COLS)
    N = len(df)
    start_eval_idx = N - n_eval_sessions

    models = build_candidate_models()
    model_names = list(models.keys()) + ["Blended_Ensemble", "Prophet_Base"]

    hits_counter: dict[str, int] = {m: 0 for m in model_names}
    mae_counter: dict[str, list[float]] = {m: [] for m in model_names}
    daily_predictions: list[dict[str, Any]] = []

    for idx in range(start_eval_idx, N):
        train_start = max(0, idx - lookback_sessions)
        train_data = df.iloc[train_start:idx]
        test_row = df.iloc[idx]
        prev_row = df.iloc[idx - 1]

        actual_price = float(test_row["close_price"])
        prev_price = float(prev_row["close_price"])
        actual_ret = (actual_price - prev_price) / prev_price * 100.0

        # Training slice
        tr_clean = train_data.dropna(subset=f_cols)
        y_tr = (tr_clean["close_price"].shift(-1) - tr_clean["close_price"]) / tr_clean["close_price"] * 100.0
        X_tr = tr_clean[f_cols].iloc[:-1]
        y_tr = y_tr.iloc[:-1]

        valid_m = (~X_tr.isna().any(axis=1)) & (~y_tr.isna())
        X_tr = X_tr[valid_m]
        y_tr = y_tr[valid_m]

        if len(X_tr) < 25:
            continue

        x_test = pd.DataFrame([prev_row[f_cols]])
        step_preds: dict[str, float] = {}

        # 1. Fit individual ML models
        for m_name, model in models.items():
            try:
                model.fit(X_tr, y_tr)
                p_ret = float(model.predict(x_test)[0])
                p_ret = max(-10.0, min(10.0, p_ret))
            except Exception:
                p_ret = 0.0

            step_preds[m_name] = p_ret
            hit = _check_hit(p_ret, actual_ret)
            err = abs(p_ret - actual_ret)
            if hit:
                hits_counter[m_name] += 1
            mae_counter[m_name].append(err)

        # 2. Blended Ensemble (Ridge + Huber + XGBoost)
        p_blend = (step_preds["Ridge"] + step_preds["Huber"] + step_preds["XGBoost"]) / 3.0
        step_preds["Blended_Ensemble"] = p_blend
        b_hit = _check_hit(p_blend, actual_ret)
        b_err = abs(p_blend - actual_ret)
        if b_hit:
            hits_counter["Blended_Ensemble"] += 1
        mae_counter["Blended_Ensemble"].append(b_err)

        # 3. Prophet Baseline
        p_prophet = float(test_row.get("feat_prophet_ret_today_pct", 0.0))
        step_preds["Prophet_Base"] = p_prophet
        pr_hit = _check_hit(p_prophet, actual_ret)
        pr_err = abs(p_prophet - actual_ret)
        if pr_hit:
            hits_counter["Prophet_Base"] += 1
        mae_counter["Prophet_Base"].append(pr_err)

        daily_predictions.append({
            "symbol": symbol,
            "trade_date": str(test_row["trade_date"]).split(" ")[0],
            "actual_ret": actual_ret,
            **{f"pred_{m}": step_preds[m] for m in model_names},
        })

    total_sessions = len(daily_predictions)
    scorecard: dict[str, dict[str, float]] = {}

    for m in model_names:
        hits = hits_counter[m]
        rate = (hits / total_sessions * 100.0) if total_sessions > 0 else 0.0
        mae = float(np.mean(mae_counter[m])) if mae_counter[m] else 0.0
        scorecard[m] = {
            "hits": hits,
            "hit_rate_pct": round(rate, 1),
            "mae_pct": round(mae, 2),
        }

    # Crown champion among candidate ML models (excluding pure Prophet base)
    ml_candidates = [m for m in model_names if m != "Prophet_Base"]
    champion = max(
        ml_candidates,
        key=lambda m: (scorecard[m]["hit_rate_pct"], -scorecard[m]["mae_pct"]),
    )

    return {
        "symbol": symbol,
        "lookback_sessions": lookback_sessions,
        "total_sessions": total_sessions,
        "champion": champion,
        "scorecard": scorecard,
        "daily_predictions": daily_predictions,
    }


def main() -> None:
    symbols = ["THYAO", "AKBNK", "ASELS", "EREGL", "TUPRS"]
    n_eval = 180

    if len(sys.argv) > 1:
        symbols = [s.strip().upper() for s in sys.argv[1].split(",") if s.strip()]

    print("\n" + "=" * 100)
    print("        EXPANDED 8-MODEL TOURNAMENT: 180-DAY WALK-FORWARD REALITY ARENA")
    print(f"        Evaluating {len(symbols)} Equities | 180 Sessions | Deadband: ±{DEADBAND_PCT}% (25 bps)")
    print("=" * 100)

    db = PostgresManager()
    all_symbol_results: dict[str, Any] = {}

    for idx, sym in enumerate(symbols, 1):
        sym_t0 = time.time()
        print(f"\n[{idx}/{len(symbols)}] Benchmarking 8 Candidate Models for {sym}...")

        df = extract_3pillar_time_series(db, sym, lookback_days=850)
        df = attach_prophet_rolling_features(df, sym, n_history_needed=220)

        res = evaluate_symbol_models(df, sym, lookback_sessions=252, n_eval_sessions=n_eval)
        all_symbol_results[sym] = res

        champ = res["champion"]
        c_rate = res["scorecard"][champ]["hit_rate_pct"]
        c_hits = res["scorecard"][champ]["hits"]
        tot = res["total_sessions"]
        pr_rate = res["scorecard"]["Prophet_Base"]["hit_rate_pct"]

        print(
            f"[{idx}/{len(symbols)}] {sym} Complete in {time.time() - sym_t0:.1f}s | "
            f"Champion: {champ} ({c_hits}/{tot} -> {c_rate:.1f}%) | "
            f"Prophet: {pr_rate:.1f}% | Alpha: {c_rate - pr_rate:+.1f}%"
        )

    # Print Full Comparative Table
    print("\n" + "=" * 100)
    print("                     EXPANDED MODEL ARENA SCORECARD (180 SESSIONS)")
    print("=" * 100)

    sample_res = next(iter(all_symbol_results.values()))
    models_evaluated = list(sample_res["scorecard"].keys())

    header = f"{'Symbol':<8}" + "".join(f"{m[:10]:<12}" for m in models_evaluated) + f"{'Champion':<16}"
    print(header)
    print("-" * len(header))

    portfolio_hits: dict[str, list[float]] = {m: [] for m in models_evaluated}
    portfolio_maes: dict[str, list[float]] = {m: [] for m in models_evaluated}

    for sym, r in all_symbol_results.items():
        row = f"{sym:<8}"
        champ = r["champion"]
        for m in models_evaluated:
            sc = r["scorecard"][m]
            rate = sc["hit_rate_pct"]
            portfolio_hits[m].append(rate)
            portfolio_maes[m].append(sc["mae_pct"])
            mark = "*" if m == champ else " "
            row += f"{rate:4.1f}%{mark}      "
        row += f"{champ:<16}"
        print(row)

    print("-" * len(header))
    port_row = f"{'PORTFOLIO':<8}"
    for m in models_evaluated:
        m_avg = np.mean(portfolio_hits[m]) if portfolio_hits[m] else 0.0
        port_row += f"{m_avg:4.1f}%       "
    print(port_row)

    port_mae_row = f"{'MAE %':<8}"
    for m in models_evaluated:
        mae_avg = np.mean(portfolio_maes[m]) if portfolio_maes[m] else 0.0
        port_mae_row += f"{mae_avg:4.2f}%       "
    print(port_mae_row)
    print("=" * 100)


if __name__ == "__main__":
    main()
